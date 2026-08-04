"""Driving-scale scenes for the car env — a stand-in "recorded drive", and the real path in.

The digital-twin idea is: reconstruct a recorded drive into geometry, then test a driving policy
*inside* it, closed-loop. Until a real reconstruction is on hand, `make_driving_scene` authors a
plausible driving-scale occupancy scene — a straight road corridor between curbs/buildings with
parked cars intruding into alternating lanes (a slalom the car must weave through) — as a
`GridWorld`, so the exact same car env + braking shield run on it.

`load_occupancy_grid` is the real path in: point it at a saved top-down occupancy map (the
pipeline's `*_occupancy` array, a KITTI BEV, or any authored PNG) and it becomes the world the
car drives. Same `GridWorld`, so nothing downstream changes when you swap synthetic for real.

Pure NumPy + OpenCV. Metric: metres. World +x = along the road, +y = across it (left).
"""

from __future__ import annotations

import os

import numpy as np

from .car_sim import CarSimConfig
from .grid_world import GridWorld
from .nav_task import NavTaskConfig


def _rasterise(bounds, resolution, lane_half, circles):
    """Occupancy grid: curbs where |y| > lane_half, plus each (cx, cy, r) circular obstacle."""
    x_min, y_min, x_max, y_max = bounds
    w = int(np.ceil((x_max - x_min) / resolution))
    h = int(np.ceil((y_max - y_min) / resolution))
    gx = x_min + (np.arange(w) + 0.5) * resolution        # cell-centre x per column
    gy = y_min + (np.arange(h) + 0.5) * resolution        # cell-centre y per row
    GX, GY = np.meshgrid(gx, gy)                           # (H, W)
    occ = (np.abs(GY) > lane_half)                        # curbs / buildings flank the road
    for cx, cy, r in circles:
        occ |= (GX - cx) ** 2 + (GY - cy) ** 2 <= r * r
    return occ.astype(np.uint8)


def make_driving_scene(resolution: float = 0.1, length: float = 20.0, lane_half: float = 2.5,
                       obstacle_radius: float = 1.0, cross: float = 0.35):
    """A straight road with alternating round obstacles — a threadable, flowing driving slalom.

    Returns `(grid, start, goal)`: a `GridWorld`, a `(x, y, heading)` start, and an `(x, y)`
    goal at the far end. Obstacles are circles (think roundabout planters / stalled cars) that
    reach `cross` metres *past* the centreline from alternating sides, so the route requires a
    real lane-change weave — but their curved faces (no flat head-on wall) let a car, which can't
    pivot in place, arc smoothly around each rather than ramming and stalling at a gap mouth.
    """
    bounds = (0.0, -lane_half - 1.0, length, lane_half + 1.0)   # 1 m of curb/building each side
    # A circle at |cy| = r - cross reaches `cross` past the centreline (its near edge at ∓cross),
    # leaving a clear (lane_half - cross)-wide lane against the opposite curb.
    off = obstacle_radius - cross
    r = obstacle_radius
    circles = [
        (0.26 * length,  off, r),                          # upper -> weave DOWN
        (0.50 * length, -off, r),                          # lower -> weave UP
        (0.74 * length,  off, r),                          # upper -> weave DOWN
    ]
    occ = _rasterise(bounds, resolution, lane_half, circles)
    grid = GridWorld(occ, origin=(bounds[0], bounds[1]), resolution=resolution)
    start = (1.0, 0.0, 0.0)
    goal = (length - 1.0, 0.0)
    return grid, start, goal


def driving_scene_config(grid: GridWorld, start, goal, **overrides) -> CarSimConfig:
    """A `CarSimConfig` wired to a scene grid with sensible car + sensor defaults.

    Bounds come from the grid; start/goal are fixed; a 16-beam lidar and a braking-friendly
    safety margin are set up. `overrides` patch any field (e.g. `max_speed`, `n_lidar_beams`).
    """
    cfg = dict(
        bounds=grid.bounds, grid_world=grid,
        fixed_start=start, fixed_goal=goal,
        robot_radius=0.25, safety_margin=0.12,
        max_speed=1.4, max_steer=0.6, wheelbase=0.45, max_ang_vel=3.0,
        n_lidar_beams=16, lidar_fov=np.pi, lidar_range=6.0,
        task=NavTaskConfig(max_steps=600, goal_radius=0.6),
    )
    cfg.update(overrides)
    return CarSimConfig(**cfg)


def load_occupancy_grid(path: str, resolution: float, origin=(0.0, 0.0),
                        occupied_when: str = "positive") -> GridWorld:
    """Load a saved occupancy map into a `GridWorld` — the real reconstructed-scene path.

    Supports `.npy` (e.g. the pipeline's `{-1 unknown, 0 free, 1 occupied}` array) and common
    image formats (PNG/JPG; read grayscale, **dark pixels = occupied** by convention, matching
    a printed occupancy map). `occupied_when` (`"positive"`/`"nonzero"`) applies to `.npy`.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        grid = np.load(path)
    else:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read occupancy image {path}")
        grid = (img < 128).astype(np.int8)            # dark = occupied
        occupied_when = "nonzero"
    return GridWorld.from_occupancy_image(grid, origin=origin, resolution=resolution,
                                          occupied_when=occupied_when)
