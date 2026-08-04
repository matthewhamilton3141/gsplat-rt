"""Occupancy-grid world for the nav env — navigate reconstructed *shape*, not just circles.

The circle+wall world in `nav_sim` is enough for synthetic curricula, but a robot dropped into
a *reconstructed* scene (the pipeline's top-down occupancy map, a driving-scene BEV, ...) has to
collide against and sense arbitrary geometry. `GridWorld` wraps a 2-D occupancy grid in
world-metric coordinates and exposes the same three geometric primitives the env and the safety
shield need — signed clearance, disc collision, and lidar raycast — so the *exact same* policy
and one-step-lookahead shield run unchanged over reconstructed geometry.

This is the bridge that turns "reconstruct a scene" into "test a policy *inside* it": the
digital-twin closed-loop idea, at whatever scale the grid is authored (a room, an intersection,
a KITTI BEV tile). It is deliberately additive — an env may carry circle obstacles *and* a grid;
the env folds both via `min` clearance / `or` collision / `min` ray distance.

Pure NumPy + OpenCV. The clearance field is `cv2.distanceTransform` (cv2 is a core dependency,
so this stays importable on a fresh install and the box); no torch/GPU. Metric throughout:
metres, radians.

Coordinate convention (image-like, y-up in world):
  - `occupancy` is `(H, W)` with 1.0 = occupied, 0.0 = free.
  - World +x maps to columns, world +y maps to rows. Cell `(row=i, col=j)` centre is at
    world `(x0 + (j+0.5)*res, y0 + (i+0.5)*res)`, where `origin=(x0, y0)` is the world coord
    of the grid's minimum corner and `res` is the (square) cell size in metres.
  - The grid covers world `x in [x0, x0 + W*res]`, `y in [y0, y0 + H*res]`. Queries outside the
    grid see *no* obstacle from it (clearance +inf, no ray hit) — the env's rectangular `bounds`
    own the outer walls, so grid = interior geometry, bounds = perimeter, combined additively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class GridWorld:
    """A metric 2-D occupancy grid with clearance / collision / raycast queries.

    Build directly from a boolean/float occupancy array, or via :meth:`from_obstacles`
    (rasterise a circle field — handy for tests and for matching the analytic world) or
    :meth:`from_occupancy_image` (a saved top-down map).
    """

    occupancy: np.ndarray                 # (H, W) float/bool, 1 = occupied
    origin: tuple[float, float] = (0.0, 0.0)   # world (x, y) of the grid's min corner
    resolution: float = 0.1               # metres per cell (square)

    # cached signed-distance field (metres), computed lazily on first query
    _edt: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        occ = np.asarray(self.occupancy)
        # Store as a contiguous uint8 {0,1} grid so cv2 and indexing are predictable.
        self.occupancy = (occ > 0.5).astype(np.uint8)
        if self.occupancy.ndim != 2:
            raise ValueError(f"occupancy must be 2-D (H, W); got shape {occ.shape}")

    # -- shape / coordinate helpers ------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.occupancy.shape

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """World `(x_min, y_min, x_max, y_max)` the grid spans."""
        h, w = self.occupancy.shape
        x0, y0 = self.origin
        return (x0, y0, x0 + w * self.resolution, y0 + h * self.resolution)

    def world_to_cell(self, xy: np.ndarray) -> tuple[int, int]:
        """Integer `(row, col)` containing world point `xy` (may be out of range)."""
        x0, y0 = self.origin
        col = int(np.floor((xy[0] - x0) / self.resolution))
        row = int(np.floor((xy[1] - y0) / self.resolution))
        return row, col

    # -- distance field ------------------------------------------------------------------
    def distance_field(self) -> np.ndarray:
        """Per-cell distance (metres) from a free cell to the nearest occupied cell.

        Occupied cells are 0. Computed once with `cv2.distanceTransform` (Euclidean) on the
        free-space mask and cached. Resolution-limited: the true obstacle boundary sits within
        ~one cell of the nearest occupied cell centre, so clearances carry an O(`resolution`)
        error — fine for a shield whose safety margin is set well above the cell size.
        """
        if self._edt is None:
            free = (self.occupancy == 0).astype(np.uint8)   # 255-scale not needed for DIST_L2
            if free.all():                                  # no obstacles at all
                self._edt = np.full(self.occupancy.shape, np.inf, np.float32)
            elif not free.any():                            # everything occupied
                self._edt = np.zeros(self.occupancy.shape, np.float32)
            else:
                # DIST_MASK_PRECISE = true Euclidean (not the 3×3 chamfer approximation), so a
                # clearance is exact to the cell grid and the shield's margin isn't eaten by
                # distance-transform error.
                dt = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                self._edt = dt.astype(np.float32) * self.resolution
        return self._edt

    # -- primitives the env / shield consume ---------------------------------------------
    def clearance(self, xy: np.ndarray, robot_radius: float) -> float:
        """Signed distance (m) from the robot disc edge at `xy` to the nearest occupied cell.

        Positive is free space ahead of contact, negative under overlap. Points outside the
        grid return `+inf` (the grid imposes no obstacle there; the env's `bounds` do).
        """
        row, col = self.world_to_cell(xy)
        h, w = self.occupancy.shape
        if not (0 <= row < h and 0 <= col < w):
            return float("inf")
        return float(self.distance_field()[row, col]) - robot_radius

    def collides(self, xy: np.ndarray, robot_radius: float) -> bool:
        """True if the robot disc at `xy` overlaps an occupied cell."""
        return self.clearance(xy, robot_radius) < 0.0

    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_range: float) -> float:
        """Distance (m) along a unit `direction` from `origin` to the first occupied cell.

        Returns `max_range` if the ray clears the grid without a hit. Marches at half-cell
        steps (vectorised) — resolution-limited but continuous enough for a lidar fan; a cell
        the ray only clips at a corner may be missed, which is acceptable at these cell sizes.
        """
        origin = np.asarray(origin, float)
        direction = np.asarray(direction, float)
        step = self.resolution * 0.5
        n = max(1, int(np.ceil(max_range / step)))
        ts = np.arange(1, n + 1) * step
        ts = ts[ts <= max_range]
        if ts.size == 0:
            return float(max_range)
        pts = origin[None, :] + ts[:, None] * direction[None, :]
        x0, y0 = self.origin
        cols = np.floor((pts[:, 0] - x0) / self.resolution).astype(int)
        rows = np.floor((pts[:, 1] - y0) / self.resolution).astype(int)
        h, w = self.occupancy.shape
        inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        occ = np.zeros(ts.shape, bool)
        occ[inb] = self.occupancy[rows[inb], cols[inb]] > 0
        if occ.any():
            return float(ts[np.argmax(occ)])
        return float(max_range)

    def occupied_at(self, xy: np.ndarray) -> bool:
        """Whether the cell containing world point `xy` is occupied (out-of-grid = free)."""
        row, col = self.world_to_cell(xy)
        h, w = self.occupancy.shape
        if not (0 <= row < h and 0 <= col < w):
            return False
        return bool(self.occupancy[row, col])

    def occupied_mask(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorised occupancy at broadcastable world-coord arrays (out-of-grid = free).

        Used to stamp the grid into the env's egocentric occupancy observation so a policy
        that consumes the top-down map sees the reconstructed geometry, not just circles.
        """
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        x0, y0 = self.origin
        cols = np.floor((xs - x0) / self.resolution).astype(int)
        rows = np.floor((ys - y0) / self.resolution).astype(int)
        h, w = self.occupancy.shape
        inb = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.zeros(np.broadcast(xs, ys).shape, bool)
        out[inb] = self.occupancy[rows[inb], cols[inb]] > 0
        return out

    # -- constructors --------------------------------------------------------------------
    @classmethod
    def from_obstacles(cls, bounds: tuple[float, float, float, float],
                       obstacles: np.ndarray, resolution: float = 0.1,
                       wall_thickness: int = 0) -> "GridWorld":
        """Rasterise a circle-obstacle field (and optionally the walls) into a GridWorld.

        Lets a grid world be built to *match* the analytic circle world — the basis of the
        equivalence tests, and a way to promote a synthetic curriculum scene into a grid.
        `obstacles` is an `(N, 3)` array of `(cx, cy, r)`; `wall_thickness` (in cells) rings
        the border with occupied cells if > 0.
        """
        x_min, y_min, x_max, y_max = bounds
        w = int(np.ceil((x_max - x_min) / resolution))
        h = int(np.ceil((y_max - y_min) / resolution))
        occ = np.zeros((h, w), np.uint8)
        obstacles = np.asarray(obstacles, float).reshape(-1, 3)
        # Cell-centre world coords, vectorised over the whole grid.
        cx_world = x_min + (np.arange(w) + 0.5) * resolution
        cy_world = y_min + (np.arange(h) + 0.5) * resolution
        gx, gy = np.meshgrid(cx_world, cy_world)             # (H, W)
        for ox, oy, r in obstacles:
            occ |= ((gx - ox) ** 2 + (gy - oy) ** 2 <= r * r).astype(np.uint8)
        if wall_thickness > 0:
            t = wall_thickness
            occ[:t, :] = 1; occ[-t:, :] = 1; occ[:, :t] = 1; occ[:, -t:] = 1
        return cls(occupancy=occ, origin=(x_min, y_min), resolution=resolution)

    @classmethod
    def from_occupancy_image(cls, grid: np.ndarray, origin: tuple[float, float],
                             resolution: float, occupied_when="positive") -> "GridWorld":
        """Wrap a saved top-down occupancy grid (e.g. the pipeline's `*_occupancy` array).

        The pipeline emits `{-1 unknown, 0 free, 1 occupied}`; pass `occupied_when="positive"`
        to treat only 1 as blocking (unknown = free, the permissive choice) or `"nonzero"` to
        also block unknown. Any 2-D array works for a hand-authored or reconstructed map.
        """
        g = np.asarray(grid)
        if occupied_when == "positive":
            occ = (g > 0)
        elif occupied_when == "nonzero":
            occ = (g != 0)
        else:
            raise ValueError("occupied_when must be 'positive' or 'nonzero'")
        return cls(occupancy=occ.astype(np.uint8), origin=origin, resolution=resolution)
