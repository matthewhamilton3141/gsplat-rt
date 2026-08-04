"""Background car-navigation runner the web viewer polls — the in-browser digital-twin loop.

Runs a `BicycleNavEnv` (kinematic-bicycle car) driven by the DWA local planner behind the
braking safety shield, over a reconstructed-style occupancy scene, in a daemon thread stepping
at real time. The web viewer reads two JSON views off it: a *static* scene (`scene_json` — the
occupancy geometry + goal, built once by the client) and a *dynamic* snapshot (`snapshot` — the
car pose, trail, lidar fan, step, outcome), polled fast for smooth motion. On reaching / colliding
/ timing out it holds briefly then resets to a fresh episode, so the demo loops forever.

This closes the arc: reconstruct a scene into an occupancy grid, then watch a shielded car drive
*inside* that reconstructed geometry, live in the browser. Pure NumPy + OpenCV; no GPU.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import numpy as np

# The viz package lives under src/; isaac is a sibling package. Ensure src is importable whether
# this is imported as viz.nav_runner or run from a script that already added src to the path.
sys.path.insert(0, __file__.rsplit("/viz/", 1)[0])

from isaac.car_sim import (  # noqa: E402
    BicycleNavEnv, CarSimConfig, car_dwa_action, car_safety_shield,
)
from isaac.driving_scene import (  # noqa: E402
    driving_scene_config, load_occupancy_grid, make_driving_scene,
)


class NavRunner:
    """Steps a shielded DWA car over an occupancy scene in a background thread.

    Build with the default procedural driving scene, or pass a `CarSimConfig` already wired to a
    reconstructed occupancy (see `driving_scene.load_occupancy_grid`). Thread-safe snapshots.
    """

    def __init__(self, cfg: Optional[CarSimConfig] = None, *, render_cell: float = 0.2,
                 trail_len: int = 400, use_shield: bool = True, hold_s: float = 1.5,
                 seed: int = 0):
        if cfg is None:
            grid, start, goal = make_driving_scene()
            cfg = driving_scene_config(grid, start, goal)
        self.cfg = cfg
        self.use_shield = use_shield
        self.hold_s = hold_s
        self._trail_len = trail_len
        self._seed = seed

        self.env = BicycleNavEnv(cfg)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._trail: list[np.ndarray] = []
        self._outcome = "run"
        self._step = 0
        self.env.reset(seed=self._seed)
        self._trail.append(self.env.robot_xy.copy())

        # Static render geometry (occupied cells, coarsened) — computed once.
        self._scene = self._build_scene_json(render_cell)

    # -- scene geometry (static) ---------------------------------------------------------
    def _build_scene_json(self, render_cell: float) -> dict:
        grid = self.cfg.grid_world
        x_min, y_min, x_max, y_max = self.cfg.bounds
        cells, cell_size = [], render_cell
        if grid is not None:
            occ = grid.occupancy
            s = max(1, int(round(render_cell / grid.resolution)))
            cell_size = s * grid.resolution
            h, w = occ.shape
            # Max-pool into coarse cells so a coarse cell is occupied if any fine cell is; emit
            # the world centre of each occupied coarse cell (tiles cleanly as boxes in 3-D).
            for i in range(0, h, s):
                for j in range(0, w, s):
                    if occ[i:i + s, j:j + s].any():
                        cx = grid.origin[0] + (j + s / 2.0) * grid.resolution
                        cy = grid.origin[1] + (i + s / 2.0) * grid.resolution
                        cells.append([round(float(cx), 3), round(float(cy), 3)])
        rr = self.cfg.robot_radius
        return {
            "bounds": [x_min, y_min, x_max, y_max],
            "cell_size": round(float(cell_size), 3),
            "cells": cells,
            "goal": [round(float(self.env._goal[0]), 3), round(float(self.env._goal[1]), 3)],
            "car": {"radius": rr, "length": round(rr * 2.6, 3), "width": round(rr * 1.6, 3)},
        }

    def scene_json(self) -> dict:
        return self._scene

    # -- dynamic snapshot ----------------------------------------------------------------
    def _lidar_endpoints(self) -> list[list[float]]:
        cfg = self.cfg
        n = cfg.n_lidar_beams
        if not n:
            return []
        angles = (self.env._heading + np.linspace(-cfg.lidar_fov / 2, cfg.lidar_fov / 2, n)
                  if n > 1 else np.array([self.env._heading]))
        dists = self.env._lidar() * cfg.lidar_range
        o = self.env.robot_xy
        return [[round(float(o[0] + d * np.cos(a)), 3), round(float(o[1] + d * np.sin(a)), 3)]
                for a, d in zip(angles, dists)]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "car": [round(float(self.env.robot_xy[0]), 3),
                        round(float(self.env.robot_xy[1]), 3)],
                "heading": round(float(self.env._heading), 4),
                "trail": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in self._trail],
                "lidar": self._lidar_endpoints(),
                "goal": self._scene["goal"],
                "step": self._step,
                "outcome": self._outcome,
            }

    # -- stepping ------------------------------------------------------------------------
    def _reset_episode(self):
        self._seed += 1
        self.env.reset(seed=self._seed)
        self._trail = [self.env.robot_xy.copy()]
        self._outcome = "run"
        self._step = 0

    def step_once(self) -> bool:
        """Advance one control step. Returns True if the episode is still running."""
        env, cfg = self.env, self.cfg
        raw = car_dwa_action(env.robot_xy, env._heading, env._goal, env._obstacles, cfg)
        act = car_safety_shield(raw, env.robot_xy, env._heading, env._obstacles, cfg) \
            if self.use_shield else raw
        _, _, term, trunc, info = env.step(act)
        with self._lock:
            self._step = info["step"]
            self._trail.append(env.robot_xy.copy())
            if len(self._trail) > self._trail_len:
                self._trail = self._trail[-self._trail_len:]
            if term or trunc:
                self._outcome = ("reached" if info["reached"]
                                 else "collided" if info["collided"] else "timeout")
        return not (term or trunc)

    def _loop(self):
        while not self._stop.is_set():
            running = self.step_once()
            if not running:
                # hold the final frame so the outcome is visible, then start a fresh episode
                if self._stop.wait(self.hold_s):
                    break
                with self._lock:
                    self._reset_episode()
            else:
                if self._stop.wait(self.cfg.dt):     # real-time pacing
                    break

    def start(self) -> "NavRunner":
        if self._thread is not None:
            raise RuntimeError("NavRunner already started")
        self._thread = threading.Thread(target=self._loop, name="NavRunner", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None


def make_nav_runner(occupancy: Optional[str] = None, resolution: float = 0.1,
                    start=None, goal=None, **kw) -> NavRunner:
    """Convenience builder: default procedural driving scene, or a reconstructed occupancy map.

    `occupancy` (a `.npy`/`.png` path) loads a real reconstructed scene via
    `driving_scene.load_occupancy_grid`; `start`/`goal` are then required (world x,y[,heading]).
    """
    if occupancy is None:
        return NavRunner(**kw)
    grid = load_occupancy_grid(occupancy, resolution)
    if start is None or goal is None:
        raise ValueError("a reconstructed occupancy needs explicit start and goal")
    cfg = driving_scene_config(grid, tuple(start), tuple(goal))
    return NavRunner(cfg, **kw)
