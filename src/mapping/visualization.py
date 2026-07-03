"""Lightweight 2-D visual artifacts for the pipeline.

Two glanceable PNGs, written alongside the `.usdz` scene so a run produces
something you can *look at* without opening a USD viewer:

  save_occupancy_png  — top-down occupancy grid (floor plan) from the TSDF
  save_splat_preview  — depth-colored projection of the Gaussian point cloud

Both are pure functions that depend only on numpy + OpenCV (no `pxr`), so they
run on any machine and regardless of whether USD export is available.
"""

from __future__ import annotations

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

    Returns
    -------
    The path written.

    The image is oriented as a map: X increases to the right, and depth (Z,
    away from the camera) increases upward, so the camera sits at the bottom.
    """
    grid = np.asarray(grid)
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


def save_splat_preview(
    points: Sequence[Sequence[float]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    path: str,
    point_radius: int = 2,
    background_bgr: tuple = (18, 18, 18),
) -> Optional[str]:
    """Project a 3-D point cloud through a pinhole camera into a preview PNG.

    Points are colored by depth (near = warm, far = cool via the TURBO map) and
    drawn far-to-near so nearer splats occlude farther ones. Returns the path,
    or None if there is nothing in front of the camera to draw.

    Parameters
    ----------
    points : (N, 3) array-like
        Camera-space XYZ (Z forward), e.g. the pipeline's Gaussian centres.
    fx, fy, cx, cy : float
        Pinhole intrinsics.
    width, height : int
        Output image size in pixels.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.shape[0] == 0:
        return None

    z = pts[:, 2]
    in_front = z > 1e-3
    if not np.any(in_front):
        return None
    pts = pts[in_front]
    z = z[in_front]

    u = np.rint(fx * pts[:, 0] / z + cx).astype(np.int32)
    v = np.rint(fy * pts[:, 1] / z + cy).astype(np.int32)
    on_screen = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(on_screen):
        return None
    u, v, z = u[on_screen], v[on_screen], z[on_screen]

    # Depth → 0..255 (near bright), then a perceptual colormap.
    z_min, z_max = float(z.min()), float(z.max())
    span = z_max - z_min
    norm = np.zeros_like(z) if span < 1e-6 else (z - z_min) / span
    depth_u8 = (255 * (1.0 - norm)).astype(np.uint8)            # near = high
    colors = cv2.applyColorMap(depth_u8.reshape(-1, 1), cv2.COLORMAP_TURBO)
    colors = colors.reshape(-1, 3)

    img = np.full((height, width, 3), background_bgr, dtype=np.uint8)
    # Painter's order: draw farthest first so nearest land on top.
    for i in np.argsort(-z):
        cv2.circle(
            img,
            (int(u[i]), int(v[i])),
            point_radius,
            (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(path, img)
    return path
