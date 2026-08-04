"""Turn a reconstructed driving-scene *surface mesh* into a nav occupancy grid — the real twin.

NVIDIA's NuRec (and any real-to-sim reconstruction) ships a recorded drive as 3D Gaussian splats
+ a surface mesh (+ an `.xodr` road map), ~20 s clips as USDZ. For *navigation* we don't need the
splats — we need the drivable-space occupancy. This projects the surface mesh **top-down** into a
`GridWorld`: a cell is occupied if the mesh has obstacle geometry in a height band above the ground
at that cell (walls, parked cars, curbs, poles), free if only the road surface is there. The
shielded DWA car then drives the *real reconstructed scene* — all on CPU, no GPU/box.

The reconstruction (splats) and its physics render stay a later, optional box/Isaac step; the nav
path is deliberately box-free. Pure NumPy + OpenCV for the projection; `trimesh` / OpenUSD (`pxr`)
are lazy-imported only when actually loading a mesh/USDZ file, so this module imports anywhere.

Coordinates: a mesh has an `up_axis` (NuRec/Omniverse AV scenes are Z-up); the other two axes are
the ground plane and map straight to the nav world's planar (x, y).
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from .grid_world import GridWorld


def estimate_ground(up_coords: np.ndarray, percentile: float = 5.0) -> float:
    """Robust ground level = a low percentile of the up-axis coordinates (ignores outliers)."""
    return float(np.percentile(np.asarray(up_coords, float), percentile))


def mesh_to_occupancy(vertices: np.ndarray, faces: np.ndarray, *, resolution: float = 0.2,
                      up_axis: int = 2, ground: Optional[float] = None,
                      band: tuple[float, float] = (0.3, 2.5),
                      bounds: Optional[tuple[float, float, float, float]] = None,
                      pad: float = 1.0, dilate_cells: int = 0) -> GridWorld:
    """Project a triangle surface mesh top-down into a `GridWorld` occupancy grid.

    A triangle marks its XY footprint occupied iff its up-axis span overlaps the obstacle band
    `[ground + band[0], ground + band[1]]` — i.e. it is a wall / car / curb a vehicle would hit,
    not the road surface below the band nor an overhead gantry above it. Footprints are filled
    (not just vertices stamped), so a large sparse triangle still marks its whole interior.

    `up_axis` is the mesh's vertical axis (2 = Z-up, NuRec/Omniverse). The two ground axes map to
    the nav world's (x, y). `bounds` overrides the auto XY extent; `pad` metres ring the auto
    bounds. `dilate_cells` grows the occupancy by that many cells (morphological) — recommended
    ~1 for a real *surface* mesh, whose vertical walls project to thin outlines that can otherwise
    leave a one-cell gap a car could tunnel through. Pure NumPy + OpenCV.
    """
    V = np.asarray(vertices, float)
    F = np.asarray(faces, int).reshape(-1, 3)
    ax = [a for a in (0, 1, 2) if a != up_axis]          # the two ground axes, ascending
    gx_all, gy_all = V[:, ax[0]], V[:, ax[1]]

    if ground is None:
        ground = estimate_ground(V[:, up_axis])
    band_lo, band_hi = ground + band[0], ground + band[1]

    if bounds is None:
        x_min, x_max = float(gx_all.min()) - pad, float(gx_all.max()) + pad
        y_min, y_max = float(gy_all.min()) - pad, float(gy_all.max()) + pad
    else:
        x_min, y_min, x_max, y_max = bounds
    w = max(1, int(np.ceil((x_max - x_min) / resolution)))
    h = max(1, int(np.ceil((y_max - y_min) / resolution)))
    occ = np.zeros((h, w), np.uint8)

    if len(F):
        tri = V[F]                                        # (nF, 3, 3)
        z = tri[:, :, up_axis]                            # (nF, 3)
        in_band = (z.max(axis=1) >= band_lo) & (z.min(axis=1) <= band_hi)
        tri = tri[in_band]
        if len(tri):
            # Project each triangle's 3 vertices to integer grid pixels (col=x, row=y) and fill.
            cols = ((tri[:, :, ax[0]] - x_min) / resolution)
            rows = ((tri[:, :, ax[1]] - y_min) / resolution)
            polys = np.stack([cols, rows], axis=-1).astype(np.int32)   # (nTri, 3, 2)
            cv2.fillPoly(occ, list(polys), 1)
    if dilate_cells > 0:
        k = 2 * dilate_cells + 1
        occ = cv2.dilate(occ, np.ones((k, k), np.uint8))
    return GridWorld(occ, origin=(x_min, y_min), resolution=resolution)


def pick_start_goal(grid: GridWorld, robot_radius: float = 0.25, margin: float = 0.4):
    """Pick a solvable (start, goal) as a long traverse along the free-space centreline.

    Among cells with clearance ≥ `robot_radius + margin`, keeps those near the lateral median of
    the free space (staying off the edges / disconnected padding corners), then takes the −x end
    as start (heading +x) and the +x end as goal. Falls back to the grid centre if the free space
    is too thin. Returns `((x, y, heading), (x, y))`; a real drive would instead seed these from
    the recorded ego trajectory.
    """
    edt = grid.distance_field()
    x_min, y_min, _, _ = grid.bounds
    res = grid.resolution
    h, w = grid.shape
    # Inset from the grid edges too: the env has rectangular bounds walls there that the grid's
    # own clearance (obstacle-cells only) doesn't see, so an edge cell would collide on spawn.
    inset = int(np.ceil((robot_radius + margin) / res))
    border = np.zeros_like(edt, bool)
    border[inset:h - inset, inset:w - inset] = True
    ys, xs = np.nonzero((edt >= (robot_radius + margin)) & border)
    if len(xs) == 0:
        cx = (grid.bounds[0] + grid.bounds[2]) / 2
        cy = (grid.bounds[1] + grid.bounds[3]) / 2
        return (cx, cy, 0.0), (cx, cy)
    # Keep candidates near the lateral (y) median so start/goal sit on the road, not in a corner.
    y_med = np.median(ys)
    y_spread = max(1.0, 0.25 * (ys.max() - ys.min()))
    central = np.abs(ys - y_med) <= y_spread
    xs_c, ys_c = xs[central], ys[central]
    def cell_world(xc, yc):
        return (x_min + (xc + 0.5) * res, y_min + (yc + 0.5) * res)
    i0, i1 = int(np.argmin(xs_c)), int(np.argmax(xs_c))
    sx = cell_world(xs_c[i0], ys_c[i0])
    gx = cell_world(xs_c[i1], ys_c[i1])
    return (sx[0], sx[1], 0.0), (gx[0], gx[1])


def load_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a surface mesh as `(vertices (N,3), faces (M,3))`.

    Dispatches by extension: `.obj/.ply/.stl/.glb/...` via `trimesh`; `.usd/.usda/.usdc/.usdz`
    (NuRec) via OpenUSD (`pxr`) — every `UsdGeom.Mesh` triangulated and concatenated into world
    space. Both are lazy-imported so this module needs neither unless you actually load a file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".usd", ".usda", ".usdc", ".usdz"):
        return _load_usd_mesh(path)
    try:
        import trimesh
    except ImportError as e:                              # pragma: no cover - env-dependent
        raise ImportError("loading this mesh format needs trimesh (`pip install trimesh`)") from e
    m = trimesh.load(path, force="mesh")
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def _load_usd_mesh(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all triangulated UsdGeom.Mesh prims (world-space) from a USD/USDZ stage."""
    try:
        from pxr import Usd, UsdGeom, Vt  # noqa: F401
    except ImportError as e:                              # pragma: no cover - env-dependent
        raise ImportError("reading USD/USDZ needs OpenUSD (`pip install usd-core`)") from e
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise FileNotFoundError(f"could not open USD stage {path}")
    all_v, all_f, base = [], [], 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), float)
        if pts is None or not len(pts):
            continue
        xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                        float)                            # 4x4 row-vector convention (USD)
        pts_h = np.c_[pts, np.ones(len(pts))] @ xf
        pts_w = pts_h[:, :3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
        faces, o = [], 0
        for c in counts:                                  # fan-triangulate each polygon
            for k in range(1, c - 1):
                faces.append((idx[o], idx[o + k], idx[o + k + 1]))
            o += c
        if faces:
            all_v.append(pts_w)
            all_f.append(np.asarray(faces, int) + base)
            base += len(pts_w)
    if not all_v:
        raise ValueError(f"no mesh geometry found in {path}")
    return np.concatenate(all_v), np.concatenate(all_f)


def nurec_to_gridworld(path: str, *, resolution: float = 0.2, up_axis: int = 2,
                       band: tuple[float, float] = (0.3, 2.5), **kw):
    """Load a reconstructed-scene mesh/USDZ and return `(grid, start, goal)` ready for the car env.

    Convenience wrapper: `load_mesh` → `mesh_to_occupancy` → `pick_start_goal`. Point it at a NuRec
    clip's surface mesh (or any `.obj`/`.usdz`) and drive the shielded car through it with
    `driving_scene.driving_scene_config` + `car_sim.BicycleNavEnv`.
    """
    verts, faces = load_mesh(path)
    grid = mesh_to_occupancy(verts, faces, resolution=resolution, up_axis=up_axis, band=band, **kw)
    start, goal = pick_start_goal(grid)
    return grid, start, goal


# ============================================================================================
# NuRec ground-truth (clipgt) path: build a scene from the annotated road boundaries + tracked
# obstacles + ego trajectory — richer and cleaner than the raw surface mesh. A real NuRec clip's
# `.usdz` is a container zip; unpack it and point at the `clipgt/` directory of `.parquet` files.
# ============================================================================================

def quat_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Heading (yaw about +z, rad) of a quaternion — the planar orientation the car env uses."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def gt_to_occupancy(boundaries, obstacle_boxes, bounds, resolution: float = 0.3,
                    boundary_thickness: int = 2, dilate_cells: int = 1) -> GridWorld:
    """Rasterise NuRec GT into a `GridWorld`: boundary polylines → walls, obstacle boxes → filled.

    `boundaries` is a list of `(N, 2)` world polylines (road edges / barriers / curbs); each is
    drawn as an occupied line `boundary_thickness` px wide. `obstacle_boxes` is a list of
    `(cx, cy, sx, sy, yaw)` oriented rectangles (tracked vehicles/people); each is filled. A
    `dilate_cells` pass closes one-cell gaps. `bounds` is `(x_min, y_min, x_max, y_max)`.
    """
    x_min, y_min, x_max, y_max = bounds
    w = max(1, int(np.ceil((x_max - x_min) / resolution)))
    h = max(1, int(np.ceil((y_max - y_min) / resolution)))
    occ = np.zeros((h, w), np.uint8)

    def to_px(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        return np.stack([(pts[:, 0] - x_min) / resolution,
                         (pts[:, 1] - y_min) / resolution], axis=-1).astype(np.int32)

    for poly in boundaries:
        p = to_px(poly)
        if len(p) >= 2:
            cv2.polylines(occ, [p], False, 1, boundary_thickness)
    for cx, cy, sx, sy, yaw in obstacle_boxes:
        hx, hy = sx / 2.0, sy / 2.0
        c, s = np.cos(yaw), np.sin(yaw)
        corners = [(cx + dx * c - dy * s, cy + dx * s + dy * c)
                   for dx, dy in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
        cv2.fillPoly(occ, [to_px(corners)], 1)
    if dilate_cells > 0:
        k = 2 * dilate_cells + 1
        occ = cv2.dilate(occ, np.ones((k, k), np.uint8))
    return GridWorld(occ, origin=(x_min, y_min), resolution=resolution)


def path_arclength(path: np.ndarray) -> np.ndarray:
    """Cumulative arc length (m) along an `(N, 2)` polyline — `[0, ..., total]`, length N."""
    seg = np.diff(np.asarray(path, float), axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])


def pursuit_target(path: np.ndarray, arc: np.ndarray, pos: np.ndarray,
                   lookahead: float) -> np.ndarray:
    """Pure-pursuit goal: the path point ~`lookahead` m ahead (by arc length) of the nearest one.

    A local planner (DWA) can't follow a long curved route from a single far goal — it cuts the
    corner and wedges. Feeding a moving look-ahead point along the recorded ego path turns it into
    path-following, so the car tracks the actual driven route. Pure geometry (testable).
    """
    path = np.asarray(path, float)
    i = int(np.argmin(np.hypot(path[:, 0] - pos[0], path[:, 1] - pos[1])))
    j = min(int(np.searchsorted(arc, arc[i] + lookahead)), len(path) - 1)
    return path[j].copy()


def load_nurec_gt(clipgt_dir: str, drop_on_path_m: float = 2.8) -> dict:
    """Read a NuRec clip's `clipgt/` GT into `{path, yaw0, boundaries, obstacle_boxes}`.

    Parses `egomotion_estimate` (the ego trajectory → start/goal + heading), `road_boundary`
    (drivable-area edges → walls) and `obstacle` (tracked boxes; one per `trackline_id`). Obstacles
    whose centre is within `drop_on_path_m` of the ego path are dropped: with a static snapshot they
    are in-lane traffic the ego passed by *timing*, which would falsely block the route (roadside
    parked cars stay). Needs `pyarrow` (lazy-imported); a real NuRec `.usdz` unpacks to this dir.
    """
    import os as _os
    try:
        import pyarrow.parquet as pq
    except ImportError as e:                              # pragma: no cover - env-dependent
        raise ImportError("reading NuRec GT needs pyarrow (`pip install pyarrow`)") from e

    def col(name, key):
        return pq.read_table(_os.path.join(clipgt_dir, name)).column(key).to_pylist()

    ego = col("egomotion_estimate.parquet", "egomotion_estimate")
    path = np.array([[e["location"]["x"], e["location"]["y"]] for e in ego], float)
    q0 = ego[0]["orientation"]
    yaw0 = quat_yaw(q0["x"], q0["y"], q0["z"], q0["w"])

    boundaries = [np.array([[p["x"], p["y"]] for p in b["location"]], float)
                  for b in col("road_boundary.parquet", "road_boundary") if b.get("location")]

    tracks = {}
    for o in col("obstacle.parquet", "obstacle"):
        tracks.setdefault(o["trackline_id"], o)
    boxes = []
    for o in tracks.values():
        c, s, q = o["center"], o["size"], o["orientation"]
        if len(path) and np.min(np.hypot(path[:, 0] - c["x"], path[:, 1] - c["y"])) < drop_on_path_m:
            continue
        boxes.append((c["x"], c["y"], s["x"], s["y"],
                      quat_yaw(q["x"], q["y"], q["z"], q["w"])))
    return {"path": path, "yaw0": yaw0, "boundaries": boundaries, "obstacle_boxes": boxes}


def nurec_gt_to_scene(clipgt_dir: str, *, resolution: float = 0.3, margin: float = 15.0,
                      drop_on_path_m: float = 2.8, **occ_kw):
    """NuRec `clipgt/` → `(grid, start, goal, ego_path)` ready to drive with pure-pursuit.

    `start`/`goal`/heading come from the recorded ego trajectory; the returned `ego_path` is the
    route to follow via `pursuit_target`. `grid` bounds are the ego path's extent + `margin`.
    """
    gt = load_nurec_gt(clipgt_dir, drop_on_path_m=drop_on_path_m)
    p = gt["path"]
    bounds = (float(p[:, 0].min()) - margin, float(p[:, 1].min()) - margin,
              float(p[:, 0].max()) + margin, float(p[:, 1].max()) + margin)
    grid = gt_to_occupancy(gt["boundaries"], gt["obstacle_boxes"], bounds,
                           resolution=resolution, **occ_kw)
    start = (float(p[0, 0]), float(p[0, 1]), gt["yaw0"])
    goal = (float(p[-1, 0]), float(p[-1, 1]))
    return grid, start, goal, p
