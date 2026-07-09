"""Lightweight 2-D visual artifacts for the pipeline.

Glanceable PNGs, written alongside the `.usdz` scene so a run produces
something you can *look at* without opening a USD viewer:

  save_occupancy_png  — top-down occupancy grid (floor plan) from the TSDF
  save_points_preview — the Gaussian cloud drawn as crisp points
  save_splat_preview  — the same cloud drawn as soft, alpha-composited splats

Both cloud previews **auto-frame** the point set: they fit a virtual camera to
the cloud's own extent so the scene fills the frame regardless of where it sits
in world space. This matters once pose tracking is on — the accumulated cloud
lives in world coordinates, not the current camera's, so projecting through a
camera nailed to the origin would shove everything into a corner.

All pure functions that depend only on numpy + OpenCV (no `pxr`), so they run on
any machine and regardless of whether USD export is available.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

# BGR palette for the three occupancy states (OpenCV is BGR-ordered).
_UNKNOWN_BGR = (128, 128, 128)   # gray
_FREE_BGR = (245, 245, 245)      # near-white
_OCCUPIED_BGR = (40, 40, 220)    # red


def save_occupancy_png(
    grid: np.ndarray,
    path: str,
    cell_px: int = 8,
    crop: bool = False,
    crop_margin: int = 2,
) -> str:
    """Render a top-down occupancy grid to a color PNG.

    Parameters
    ----------
    grid : ndarray (X, Z) int
        Values {-1 unknown, 0 free, 1 occupied} as produced by
        ``TSDFVolume.occupancy_grid_2d``.
    path : str
        Output PNG path.
    cell_px : int
        Edge length in pixels of one voxel column (nearest-neighbour upscale).
    crop : bool
        Trim the map to the observed region (a bbox around all cells that are not
        unknown), plus ``crop_margin`` cells of context. A large TSDF volume is
        mostly unobserved, so without this the actual floor plan is a small patch
        lost in a sea of gray. Off by default to preserve the raw-grid contract;
        the live pipeline turns it on.
    crop_margin : int
        Cells of padding kept around the observed bbox when ``crop`` is set.

    Returns
    -------
    The path written.

    The image is oriented as a map: X increases to the right, and depth (Z,
    away from the camera) increases upward, so the camera sits at the bottom.
    """
    grid = np.asarray(grid)
    if crop:
        observed = grid >= 0
        if np.any(observed):
            xs = np.where(observed.any(axis=1))[0]
            zs = np.where(observed.any(axis=0))[0]
            x0 = max(int(xs.min()) - crop_margin, 0)
            x1 = min(int(xs.max()) + crop_margin + 1, grid.shape[0])
            z0 = max(int(zs.min()) - crop_margin, 0)
            z1 = min(int(zs.max()) + crop_margin + 1, grid.shape[1])
            grid = grid[x0:x1, z0:z1]
    # grid is [X, Z]; an image is [row=y, col=x]. Put Z on rows then flip so
    # "away from camera" is up rather than down.
    disp = np.flipud(grid.T)   # (Z, X)

    img = np.empty((*disp.shape, 3), dtype=np.uint8)
    img[disp == -1] = _UNKNOWN_BGR
    img[disp == 0] = _FREE_BGR
    img[disp == 1] = _OCCUPIED_BGR

    if cell_px > 1:
        img = cv2.resize(
            img,
            (disp.shape[1] * cell_px, disp.shape[0] * cell_px),
            interpolation=cv2.INTER_NEAREST,
        )
    cv2.imwrite(path, img)
    return path


def _auto_frame_project(
    points: Sequence[Sequence[float]],
    width: int,
    height: int,
    fill: float = 0.85,
):
    """Fit a virtual pinhole to the cloud and project it to fill the frame.

    Places the camera behind (in front of, in +Z terms) the nearest points,
    centred on the cloud's median, and picks an isotropic focal length so the
    lateral extent spans ``fill`` of the frame. Returns ``(u, v, z, mask)`` where
    ``u, v`` are float pixel coords of the on-screen points, ``z`` their
    virtual-camera depth, and ``mask`` the boolean selector back into ``points``
    (so a parallel colour array can be subset identically). Returns None when the
    cloud is empty or nothing lands on screen.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return None

    centre = np.median(pts, axis=0)
    x = pts[:, 0] - centre[0]
    y = pts[:, 1] - centre[1]
    zc = pts[:, 2]

    z_lo, z_hi = np.percentile(zc, [2.0, 98.0])
    depth_span = max(float(z_hi - z_lo), 1e-3)
    z_cam = float(z_lo) - 0.30 * depth_span - 1e-3       # camera in front of all pts
    z = zc - z_cam                                       # strictly > 0

    rx = float(np.percentile(np.abs(x), 98.0)) + 1e-6
    ry = float(np.percentile(np.abs(y), 98.0)) + 1e-6
    z_rep = float(np.median(z))
    f = min(fill * 0.5 * width * z_rep / rx,
            fill * 0.5 * height * z_rep / ry)

    u = f * x / z + width * 0.5
    v = f * y / z + height * 0.5
    on = (
        np.isfinite(u) & np.isfinite(v)
        & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    )
    if not np.any(on):
        return None
    return u[on], v[on], z[on], on


def _resolve_colors(
    colors: Optional[Sequence[Sequence[float]]],
    mask: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Per-point BGR uint8 colours: real per-splat RGB if given, else a depth ramp.

    ``colors`` (when supplied) is the full-cloud (N, 3) RGB in [0, 1], subset by
    ``mask`` to match the projected points; otherwise points are coloured by depth
    (near = warm) via TURBO. Falls back to the depth ramp on any shape mismatch.
    """
    if colors is not None:
        rgb = np.asarray(colors, dtype=np.float32).reshape(-1, 3)
        if rgb.shape[0] == mask.shape[0]:
            bgr = np.clip(rgb[mask][:, ::-1], 0.0, 1.0) * 255.0
            return bgr.astype(np.uint8)
    z_min, z_max = float(z.min()), float(z.max())
    span = z_max - z_min
    norm = np.zeros_like(z) if span < 1e-6 else (z - z_min) / span
    depth_u8 = (255 * (1.0 - norm)).astype(np.uint8)     # near = high
    return cv2.applyColorMap(depth_u8.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


def save_points_preview(
    points: Sequence[Sequence[float]],
    path: str,
    width: int = 518,
    height: int = 518,
    colors: Optional[Sequence[Sequence[float]]] = None,
    point_radius: int = 2,
    background_bgr: tuple = (18, 18, 18),
) -> Optional[str]:
    """Auto-framed point-cloud preview: each Gaussian centre as a crisp dot.

    Points are drawn far-to-near so nearer ones land on top. Coloured by real
    per-splat RGB when ``colors`` is given, else by depth. Returns the path, or
    None if the cloud is empty / off-screen.

    Parameters
    ----------
    points : (N, 3) array-like
        World- (or camera-) space XYZ; the frame is fitted to them automatically.
    colors : (N, 3) array-like, optional
        Per-point RGB in [0, 1], parallel to ``points``.
    """
    proj = _auto_frame_project(points, width, height)
    if proj is None:
        return None
    u, v, z, mask = proj
    col = _resolve_colors(colors, mask, z)

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    img = np.full((height, width, 3), background_bgr, dtype=np.uint8)
    # Painter's order: draw farthest first so nearest land on top.
    for i in np.argsort(-z):
        cv2.circle(
            img,
            (int(ui[i]), int(vi[i])),
            point_radius,
            (int(col[i, 0]), int(col[i, 1]), int(col[i, 2])),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(path, img)
    return path


def save_splat_preview(
    points: Sequence[Sequence[float]],
    path: str,
    width: int = 518,
    height: int = 518,
    colors: Optional[Sequence[Sequence[float]]] = None,
    splat_radius: int = 3,
    sigma: float = 1.6,
    background_bgr: tuple = (18, 18, 18),
) -> Optional[str]:
    """Auto-framed splat preview: each Gaussian as a soft, alpha-composited blob.

    Unlike :func:`save_points_preview` (hard dots), every point is stamped as a
    small Gaussian footprint and the footprints are weight-averaged — the fuzzy,
    overlapping look of a real splat render. Nearer points carry more weight so
    they dominate where splats overlap. Coloured by real per-splat RGB when
    ``colors`` is given, else by depth. Returns the path, or None if empty.

    Parameters
    ----------
    splat_radius : int
        Footprint half-width in pixels (kernel is ``2r+1`` square).
    sigma : float
        Gaussian falloff of the footprint, in pixels.
    """
    proj = _auto_frame_project(points, width, height)
    if proj is None:
        return None
    u, v, z, mask = proj
    col = _resolve_colors(colors, mask, z).astype(np.float64)

    # Nearness weight (near = 1) so foreground splats win the weighted average.
    z_min, z_max = float(z.min()), float(z.max())
    span = z_max - z_min
    near = np.ones_like(z) if span < 1e-6 else 1.0 - (z - z_min) / span
    alpha = 0.35 + 0.65 * near

    ui = np.rint(u).astype(np.intp)
    vi = np.rint(v).astype(np.intp)
    acc = np.zeros((height, width, 3), dtype=np.float64)
    wsum = np.zeros((height, width), dtype=np.float64)

    r = int(splat_radius)
    two_sig2 = 2.0 * sigma * sigma
    # Scatter-add the whole cloud per kernel tap: the loop is over the (2r+1)^2
    # footprint (a few dozen iterations), each fully vectorised over all points.
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            g = math.exp(-(di * di + dj * dj) / two_sig2)
            vv = vi + di
            uu = ui + dj
            m = (vv >= 0) & (vv < height) & (uu >= 0) & (uu < width)
            if not np.any(m):
                continue
            w = g * alpha[m]
            np.add.at(wsum, (vv[m], uu[m]), w)
            np.add.at(acc, (vv[m], uu[m]), w[:, None] * col[m])

    img = np.full((height, width, 3), background_bgr, dtype=np.uint8)
    hit = wsum > 1e-6
    img[hit] = np.clip(acc[hit] / wsum[hit, None], 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path


# ANSI color codes for terminal rendering.
_ANSI_RESET = "\033[0m"
_ANSI_OCCUPIED = "\033[91m█\033[0m"   # bright red block
_ANSI_FREE = "\033[90m·\033[0m"       # dim gray dot
_ANSI_UNKNOWN = " "


def occupancy_to_ascii(
    grid: np.ndarray,
    max_cols: int = 60,
    color: bool = True,
) -> str:
    """Render a top-down occupancy grid as an ANSI/ASCII string for the terminal.

    Lets you watch the map form over SSH on a headless box without exporting or
    copying a PNG. Down-samples the grid so it fits `max_cols` columns; a column
    is occupied if it contains any occupied cell, free if any free cell, else
    unknown (occupied wins, so obstacles never vanish under down-sampling).

    Same orientation as ``save_occupancy_png``: X to the right, depth upward.

    Parameters
    ----------
    grid : ndarray (X, Z) int   values {-1 unknown, 0 free, 1 occupied}
    max_cols : int              target width in characters
    color : bool                emit ANSI colors (set False for plain ASCII)
    """
    grid = np.asarray(grid)
    disp = np.flipud(grid.T)   # (Z, X), depth up — matches the PNG

    rows, cols = disp.shape
    if cols > max_cols:
        step = int(np.ceil(cols / max_cols))
        # Block-reduce: a block is occupied if any occupied, else free if any
        # free (max over {-1,0,1} gives exactly that precedence).
        rr = int(np.ceil(rows / step))
        cc = int(np.ceil(cols / step))
        reduced = np.full((rr, cc), -1, dtype=np.int8)
        for i in range(rr):
            for j in range(cc):
                block = disp[i * step:(i + 1) * step, j * step:(j + 1) * step]
                if block.size:
                    reduced[i, j] = block.max()
        disp = reduced

    if color:
        occ, free, unk = _ANSI_OCCUPIED, _ANSI_FREE, _ANSI_UNKNOWN
    else:
        occ, free, unk = "#", ".", " "

    lines = []
    for row in disp:
        lines.append("".join(occ if v == 1 else free if v == 0 else unk for v in row))
    return "\n".join(lines)
