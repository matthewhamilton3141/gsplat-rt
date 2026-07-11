"""Pure-NumPy differential-drive navigation simulator — the durable Phase-4 flagship core.

Per the roadmap ("build the flagship in PyBullet first; Isaac is a port, not a start"), the
*durable* part of the nav flagship is the simulator loop + env contract, which is independent
of any physics backend. This module implements that loop in plain NumPy so it runs and is
fully unit-tested on a laptop (no GPU / torch / gymnasium / Isaac), then a PyBullet or Isaac
Lab binding becomes a thin port on top.

Design:
  - Reward / observation (goal part) / termination all come from the already-tested
    `nav_task` module — the single source of truth shared with the Isaac Lab adapter. This
    env only adds the *world*: kinematics, obstacles, collision, and optional range sensing.
  - Kinematic unicycle (differential-drive) model. State is planar: (x, y) metres on the
    ground plane, `heading` yaw in rad (0 = +x). Actions are (linear_vel_cmd, angular_vel_cmd)
    in m/s and rad/s, clipped to the config limits and applied as first-order velocity targets.
  - Obstacles are axis-symmetric circles (cx, cy, radius) plus the rectangular world bounds.
    Collision = the robot disc (radius `robot_radius`) touching any obstacle or leaving bounds.
  - Optional lidar: `n_lidar_beams` rays fanned across `lidar_fov` around the heading, each
    returning the normalised free distance to the nearest obstacle/wall (1.0 = clear to
    `lidar_range`). With beams > 0 the policy observation is `[nav_task obs (7)] + [lidar]`,
    so the goal contract stays authoritative while obstacle sensing is added cleanly.

The API is duck-typed to Gymnasium (`reset(seed) -> (obs, info)`,
`step(action) -> (obs, reward, terminated, truncated, info)`) so wrapping it for
stable-baselines3 / rsl_rl later is trivial, without taking a hard dependency here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .nav_task import (
    ACT_DIM,
    OBS_DIM,
    NavTaskConfig,
    distance_to_goal,
    observation,
    reached_goal,
    reward as nav_reward,
    terminated as nav_terminated,
    truncated as nav_truncated,
)

# An obstacle is a circle on the ground plane: (center_x, center_y, radius) in metres.
Circle = np.ndarray  # shape (3,)


@dataclass
class NavSimConfig:
    """Simulator + world parameters (metric: metres, seconds, rad).

    The task-reward parameters live in the shared `task` (NavTaskConfig); everything here is
    world/kinematics that a physics backend would otherwise own.
    """

    task: NavTaskConfig = field(default_factory=NavTaskConfig)

    # --- kinematics ---
    dt: float = 0.1                  # integration step (s)
    max_lin_vel: float = 1.0         # |linear velocity| clip (m/s)
    max_ang_vel: float = 2.0         # |angular velocity| clip (rad/s)
    robot_radius: float = 0.18       # robot disc radius for collision (m)

    # --- world ---
    # Bounds as (x_min, y_min, x_max, y_max). The robot collides with these walls.
    bounds: tuple[float, float, float, float] = (-5.0, -5.0, 5.0, 5.0)
    # Obstacles as an (N, 3) array of circles; default is empty (open field).
    obstacles: Optional[np.ndarray] = None

    # --- start / goal sampling ---
    # If set, used verbatim each reset; else sampled uniformly in bounds (collision-free).
    fixed_start: Optional[tuple[float, float, float]] = None   # (x, y, heading)
    fixed_goal: Optional[tuple[float, float]] = None
    min_start_goal_dist: float = 1.5   # rejection-sample starts this far from the goal

    # --- optional range sensor ---
    n_lidar_beams: int = 0             # 0 disables lidar (obs is just the 7-dim goal vector)
    lidar_fov: float = np.pi           # angular span of the fan (rad), centred on heading
    lidar_range: float = 5.0           # max sensed distance (m); readings normalised by this

    def obstacle_array(self) -> np.ndarray:
        """Obstacles as a contiguous (N, 3) float array (empty (0, 3) if none)."""
        if self.obstacles is None:
            return np.zeros((0, 3), float)
        arr = np.asarray(self.obstacles, float).reshape(-1, 3)
        return arr


def _ray_min_distance(
    origin: np.ndarray, direction: np.ndarray, obstacles: np.ndarray,
    bounds: tuple[float, float, float, float], max_range: float,
) -> float:
    """Nearest positive hit distance of a ray against all circles and the bounding walls.

    `direction` must be a unit vector. Returns `max_range` if nothing is hit within range.
    """
    best = max_range

    # Ray vs circles: solve |o + t d - c|^2 = r^2 for the smallest t > 0.
    if len(obstacles):
        oc = origin[None, :] - obstacles[:, :2]           # (N, 2)
        b = oc @ direction                                # (N,) since |d| = 1, a = 1
        c = np.einsum("ij,ij->i", oc, oc) - obstacles[:, 2] ** 2
        disc = b * b - c
        hit = disc >= 0.0
        if np.any(hit):
            sqrt_disc = np.sqrt(disc[hit])
            t = -b[hit] - sqrt_disc                       # near root
            t = np.where(t > 1e-9, t, -b[hit] + sqrt_disc)  # fall back to far root
            t = t[t > 1e-9]
            if t.size:
                best = min(best, float(t.min()))

    # Ray vs axis-aligned walls (x = x_min/x_max, y = y_min/y_max).
    x_min, y_min, x_max, y_max = bounds
    for axis, planes in ((0, (x_min, x_max)), (1, (y_min, y_max))):
        d = direction[axis]
        if abs(d) < 1e-12:
            continue
        for plane in planes:
            t = (plane - origin[axis]) / d
            if t > 1e-9:
                best = min(best, float(t))
    return best


class DiffDriveNavEnv:
    """Kinematic differential-drive robot navigating to a goal amid circular obstacles.

    Reward / termination / the goal part of the observation are delegated to `nav_task`
    (the tested single source of truth). Duck-typed to the Gymnasium API.
    """

    def __init__(self, cfg: Optional[NavSimConfig] = None):
        self.cfg = cfg or NavSimConfig()
        self._obstacles = self.cfg.obstacle_array()
        self.act_dim = ACT_DIM
        self.obs_dim = OBS_DIM + self.cfg.n_lidar_beams
        # Symmetric action limits, handy for a Gymnasium Box space when wrapped later.
        self.action_low = np.array([-self.cfg.max_lin_vel, -self.cfg.max_ang_vel], np.float32)
        self.action_high = np.array([self.cfg.max_lin_vel, self.cfg.max_ang_vel], np.float32)

        self._rng = np.random.default_rng()
        self._x = self._y = self._heading = 0.0
        self._lin_vel = self._ang_vel = 0.0
        self._goal = np.zeros(2, float)
        self._prev_dist = 0.0
        self._step = 0

    # -- helpers -------------------------------------------------------------------------
    @property
    def robot_xy(self) -> np.ndarray:
        return np.array([self._x, self._y], float)

    def _collides(self, xy: np.ndarray) -> bool:
        """True if the robot disc at `xy` overlaps an obstacle or crosses the world bounds."""
        r = self.cfg.robot_radius
        x_min, y_min, x_max, y_max = self.cfg.bounds
        if xy[0] - r < x_min or xy[0] + r > x_max or xy[1] - r < y_min or xy[1] + r > y_max:
            return True
        if len(self._obstacles):
            d = np.linalg.norm(self._obstacles[:, :2] - xy[None, :], axis=1)
            if np.any(d <= self._obstacles[:, 2] + r):
                return True
        return False

    def _lidar(self) -> np.ndarray:
        """Normalised free-distance readings for the beam fan (empty if disabled)."""
        n = self.cfg.n_lidar_beams
        if n == 0:
            return np.zeros(0, np.float32)
        if n == 1:
            angles = np.array([self._heading])
        else:
            angles = self._heading + np.linspace(-self.cfg.lidar_fov / 2,
                                                  self.cfg.lidar_fov / 2, n)
        origin = self.robot_xy
        out = np.empty(n, np.float32)
        for i, a in enumerate(angles):
            direction = np.array([np.cos(a), np.sin(a)], float)
            dist = _ray_min_distance(origin, direction, self._obstacles,
                                     self.cfg.bounds, self.cfg.lidar_range)
            out[i] = dist / self.cfg.lidar_range
        return out

    def _make_obs(self) -> np.ndarray:
        base = observation(self.robot_xy, self._heading, self._goal,
                           self._lin_vel, self._ang_vel)
        if self.cfg.n_lidar_beams == 0:
            return base
        return np.concatenate([base, self._lidar()]).astype(np.float32)

    def _sample_free_xy(self) -> np.ndarray:
        """Uniformly sample a collision-free point inside the (radius-shrunk) bounds."""
        x_min, y_min, x_max, y_max = self.cfg.bounds
        r = self.cfg.robot_radius
        for _ in range(1000):
            xy = np.array([self._rng.uniform(x_min + r, x_max - r),
                           self._rng.uniform(y_min + r, y_max - r)])
            if not self._collides(xy):
                return xy
        raise RuntimeError("could not sample a collision-free point; world too crowded")

    # -- Gymnasium-style API -------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        """Start a new episode. Returns (observation, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.cfg.fixed_goal is not None:
            self._goal = np.asarray(self.cfg.fixed_goal, float)
        else:
            self._goal = self._sample_free_xy()

        if self.cfg.fixed_start is not None:
            self._x, self._y, self._heading = map(float, self.cfg.fixed_start)
        else:
            for _ in range(1000):
                start = self._sample_free_xy()
                if distance_to_goal(start, self._goal) >= self.cfg.min_start_goal_dist:
                    break
            self._x, self._y = float(start[0]), float(start[1])
            self._heading = float(self._rng.uniform(-np.pi, np.pi))

        self._lin_vel = self._ang_vel = 0.0
        self._prev_dist = distance_to_goal(self.robot_xy, self._goal)
        self._step = 0
        return self._make_obs(), self._info(collided=False, reached=False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one control step. Returns (obs, reward, terminated, truncated, info)."""
        action = np.asarray(action, float).reshape(-1)
        lin = float(np.clip(action[0], -self.cfg.max_lin_vel, self.cfg.max_lin_vel))
        ang = float(np.clip(action[1], -self.cfg.max_ang_vel, self.cfg.max_ang_vel))

        # Unicycle integration (midpoint heading keeps curved motion accurate at large dt).
        dt = self.cfg.dt
        new_heading = self._heading + ang * dt
        mid = 0.5 * (self._heading + new_heading)
        new_xy = self.robot_xy + lin * dt * np.array([np.cos(mid), np.sin(mid)])

        collided = self._collides(new_xy)
        if not collided:
            self._x, self._y = float(new_xy[0]), float(new_xy[1])
            self._heading = float((new_heading + np.pi) % (2 * np.pi) - np.pi)  # wrap to (-pi, pi]
        # On collision the robot stops where it was; the episode terminates below.
        self._lin_vel, self._ang_vel = lin, ang

        self._step += 1
        curr_dist = distance_to_goal(self.robot_xy, self._goal)
        reached = reached_goal(self.robot_xy, self._goal, self.cfg.task)
        r = nav_reward(self._prev_dist, curr_dist, collided, reached, self.cfg.task)
        self._prev_dist = curr_dist

        term = nav_terminated(reached, collided)
        trunc = nav_truncated(self._step, self.cfg.task)
        return self._make_obs(), r, term, trunc, self._info(collided, reached)

    def _info(self, collided: bool, reached: bool) -> dict:
        return {
            "robot_xy": self.robot_xy,
            "heading": self._heading,
            "goal_xy": self._goal.copy(),
            "distance": distance_to_goal(self.robot_xy, self._goal),
            "collided": collided,
            "reached": reached,
            "step": self._step,
        }


def heuristic_action(obs: np.ndarray, cfg: Optional[NavSimConfig] = None) -> np.ndarray:
    """A simple obs-only go-to-goal controller — proves the env is solvable, seeds tests.

    Uses only the goal-in-robot-frame channels of the observation (so it works for any
    lidar setting): turn toward the goal, drive forward when roughly aligned. Not an
    obstacle avoider — that is what the learned policy is for; this just demonstrates the
    reward/termination loop reaches the goal on an open field.
    """
    cfg = cfg or NavSimConfig()
    from .nav_task import OBS_GOAL_X, OBS_GOAL_Y

    gx, gy = float(obs[OBS_GOAL_X]), float(obs[OBS_GOAL_Y])
    bearing = np.arctan2(gy, gx)                          # goal angle in robot frame
    ang = float(np.clip(2.0 * bearing, -cfg.max_ang_vel, cfg.max_ang_vel))
    # Slow down when the goal is off to the side so we turn in place rather than arc wide.
    align = max(np.cos(bearing), 0.0)
    lin = float(np.clip(cfg.max_lin_vel * align, 0.0, cfg.max_lin_vel))
    return np.array([lin, ang], np.float32)
