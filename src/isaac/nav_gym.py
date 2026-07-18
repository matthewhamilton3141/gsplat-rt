"""Gymnasium adapter for `DiffDriveNavEnv` — the thin RL-facing shell over the durable core.

Per the roadmap ("build the flagship in PyBullet first; Isaac is a port"), the simulator loop
+ task contract live in the backend-agnostic `nav_sim` / `nav_task` (pure NumPy, fully tested
on a laptop). This module is the *only* piece that takes a hard dependency on `gymnasium`, so
stable-baselines3 / rsl_rl can consume the env without pulling that dep into the core or its
tests. It adds nothing but the `gym.Env` interface (spaces + the standard 5-tuple `step`); all
dynamics, reward, and termination stay in the tested core.

Box-only (gymnasium isn't in the Mac dev env); the core it wraps runs and is tested anywhere.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:                                # pragma: no cover - box-only dep
    raise ImportError(
        "nav_gym requires gymnasium (install in the training env); the pure-NumPy core in "
        "nav_sim runs without it."
    ) from e

from .nav_sim import DiffDriveNavEnv, NavSimConfig, safety_shield


class NavGymEnv(gym.Env):
    """`gymnasium.Env` view of `DiffDriveNavEnv` (continuous 2-D action, flat float32 obs)."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[NavSimConfig] = None):
        super().__init__()
        self.sim = DiffDriveNavEnv(cfg)
        self.action_space = spaces.Box(
            low=self.sim.action_low, high=self.sim.action_high, dtype=np.float32)
        # Obs channels have mixed natural ranges (goal-frame metres, ±1 heading, velocities,
        # 0..1 lidar/occupancy). A single generous finite box covers them all and keeps
        # sb3's obs-normalisation happy (tight per-channel bounds aren't required).
        bound = np.full(self.sim.obs_dim, 50.0, np.float32)
        self.observation_space = spaces.Box(low=-bound, high=bound, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        obs, info = self.sim.reset(seed=seed)
        return np.asarray(obs, np.float32), info

    def step(self, action):
        # Optional shield-in-the-loop: filter the commanded action through the safety shield
        # *before* stepping, so a policy trained with `use_safety_shield` learns through the
        # shielded dynamics (and adapts to it) rather than only being wrapped at eval time.
        # Applied here at the RL boundary so the pure-NumPy `nav_sim` core stays untouched.
        action = np.asarray(action, np.float32)
        if self.sim.cfg.use_safety_shield:
            action = safety_shield(action, self.sim.robot_xy, self.sim._heading,
                                   self.sim._obstacles, self.sim.cfg)
        obs, reward, terminated, truncated, info = self.sim.step(action)
        return np.asarray(obs, np.float32), float(reward), bool(terminated), bool(truncated), info
