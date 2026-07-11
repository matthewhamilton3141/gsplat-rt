"""Framework-agnostic navigation-task logic for the reconstructed-scene RL milestone (M7).

Per the roadmap ("build the flagship in PyBullet first; Isaac is a port, not a start"),
the *durable* part of the nav task is its reward / observation / termination math — which
is independent of the simulator. This module keeps that logic as pure NumPy functions so
it can be:
  - unit-tested on a laptop (no GPU / Isaac), and
  - shared by a PyBullet env (Phase 4) and the Isaac Lab adapter (`isaac_nav_env.py`).

Task: a differential-drive robot must reach a goal in the room the pipeline reconstructed,
using the exported collision mesh as the world (obstacle avoidance is the whole task).

Conventions: planar navigation on the ground plane. All positions are 2-D (x, y) in the
world's ground plane in metres; `heading` is the robot yaw in radians (0 = +x). The caller
(the sim adapter) is responsible for mapping the stage's up-axis to this ground plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Observation layout (indices), so the env and any downstream code agree on the vector.
OBS_GOAL_X, OBS_GOAL_Y, OBS_DIST, OBS_HEAD_COS, OBS_HEAD_SIN, OBS_LIN_VEL, OBS_ANG_VEL = range(7)
OBS_DIM = 7
ACT_DIM = 2  # (linear velocity command, angular velocity command)


@dataclass
class NavTaskConfig:
    """Tunable task parameters. Defaults are metric (metres, seconds, rad)."""
    goal_radius: float = 0.30        # success threshold (m)
    max_steps: int = 500             # episode step budget -> truncation
    progress_weight: float = 1.0     # reward per metre of progress toward the goal
    time_penalty: float = 0.01       # per-step penalty (encourages efficiency)
    collision_penalty: float = 5.0   # one-off penalty on contact with an obstacle
    success_bonus: float = 10.0      # one-off reward for reaching the goal


def distance_to_goal(robot_xy: np.ndarray, goal_xy: np.ndarray) -> float:
    """Euclidean ground-plane distance (m) from robot to goal."""
    return float(np.linalg.norm(np.asarray(goal_xy, float) - np.asarray(robot_xy, float)))


def goal_in_robot_frame(robot_xy: np.ndarray, heading: float, goal_xy: np.ndarray) -> np.ndarray:
    """Goal offset expressed in the robot's local frame (x forward, y left).

    Rotating world→robot makes the policy translation- and rotation-invariant: the same
    'goal 2 m ahead' looks identical wherever the robot is or however it's turned.
    """
    d = np.asarray(goal_xy, float) - np.asarray(robot_xy, float)
    c, s = np.cos(-heading), np.sin(-heading)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]], float)


def observation(robot_xy: np.ndarray, heading: float, goal_xy: np.ndarray,
                lin_vel: float, ang_vel: float) -> np.ndarray:
    """Assemble the policy observation vector (see OBS_* indices)."""
    gx, gy = goal_in_robot_frame(robot_xy, heading, goal_xy)
    obs = np.empty(OBS_DIM, np.float32)
    obs[OBS_GOAL_X] = gx
    obs[OBS_GOAL_Y] = gy
    obs[OBS_DIST] = np.hypot(gx, gy)
    obs[OBS_HEAD_COS] = np.cos(heading)
    obs[OBS_HEAD_SIN] = np.sin(heading)
    obs[OBS_LIN_VEL] = lin_vel
    obs[OBS_ANG_VEL] = ang_vel
    return obs


def reached_goal(robot_xy: np.ndarray, goal_xy: np.ndarray, cfg: NavTaskConfig) -> bool:
    """True once the robot is within `goal_radius` of the goal."""
    return distance_to_goal(robot_xy, goal_xy) <= cfg.goal_radius


def reward(prev_dist: float, curr_dist: float, collided: bool, reached: bool,
           cfg: NavTaskConfig) -> float:
    """Dense progress reward + shaping.

    progress: `progress_weight * (prev_dist - curr_dist)` — positive when the robot got
    closer this step, negative when it backed away (this is potential-based and sums to a
    bounded total, which trains far more stably than a raw −distance reward).
    """
    r = cfg.progress_weight * (prev_dist - curr_dist)
    r -= cfg.time_penalty
    if collided:
        r -= cfg.collision_penalty
    if reached:
        r += cfg.success_bonus
    return float(r)


def terminated(reached: bool, collided: bool) -> bool:
    """Episode ended by outcome (goal reached or crashed) — a real MDP terminal state."""
    return bool(reached or collided)


def truncated(step: int, cfg: NavTaskConfig) -> bool:
    """Episode cut off by the time budget (not a terminal state — bootstrap value here)."""
    return step >= cfg.max_steps
