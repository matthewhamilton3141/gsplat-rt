"""Tests for the framework-agnostic nav-task core (src/isaac/nav_task.py).

Pure CPU/NumPy — no Isaac, no GPU. Locks the reward shaping, observation frame, and
termination logic so they're correct before either the PyBullet or Isaac Lab env is
wired on top.

Run:
    pytest tests/test_nav_task.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isaac.nav_task import (  # noqa: E402
    NavTaskConfig, OBS_DIM, OBS_DIST, OBS_GOAL_X, OBS_GOAL_Y,
    distance_to_goal, goal_in_robot_frame, observation, reached_goal,
    reward, terminated, truncated,
)


def test_distance_and_reached():
    cfg = NavTaskConfig(goal_radius=0.3)
    assert distance_to_goal([0, 0], [3, 4]) == 5.0
    assert reached_goal([1.0, 1.0], [1.1, 1.1], cfg)         # within 0.3 m
    assert not reached_goal([0, 0], [1, 1], cfg)


def test_goal_in_robot_frame_is_heading_relative():
    # goal 2 m straight ahead in world +x; robot at origin facing +x -> forward, no lateral
    fwd = goal_in_robot_frame([0, 0], 0.0, [2, 0])
    assert np.allclose(fwd, [2, 0], atol=1e-6)
    # rotate the robot +90deg (facing +y): the same world goal is now 2 m to its RIGHT (-y local)
    right = goal_in_robot_frame([0, 0], np.pi / 2, [2, 0])
    assert np.allclose(right, [0, -2], atol=1e-6)


def test_observation_shape_and_distance_invariance():
    obs = observation([1, 2], 0.7, [4, 6], lin_vel=0.5, ang_vel=-0.1)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    # the distance channel must equal true world distance regardless of heading
    d = distance_to_goal([1, 2], [4, 6])
    assert abs(float(obs[OBS_DIST]) - d) < 1e-5
    for h in (0.0, 1.0, -2.5, np.pi):
        o = observation([1, 2], h, [4, 6], 0.0, 0.0)
        assert abs(float(o[OBS_DIST]) - d) < 1e-5
        # robot-frame goal components always reconstruct the same range
        assert abs(np.hypot(o[OBS_GOAL_X], o[OBS_GOAL_Y]) - d) < 1e-5


def test_reward_rewards_progress_penalizes_regress():
    cfg = NavTaskConfig(progress_weight=1.0, time_penalty=0.01)
    closer = reward(prev_dist=5.0, curr_dist=4.0, collided=False, reached=False, cfg=cfg)
    farther = reward(prev_dist=4.0, curr_dist=5.0, collided=False, reached=False, cfg=cfg)
    assert closer > 0 > farther
    assert abs(closer - (1.0 - 0.01)) < 1e-6           # +1 m progress − time penalty


def test_reward_collision_and_success_terms():
    cfg = NavTaskConfig(collision_penalty=5.0, success_bonus=10.0, time_penalty=0.0)
    base = reward(2.0, 2.0, collided=False, reached=False, cfg=cfg)   # no progress
    assert abs(base) < 1e-9
    assert reward(2.0, 2.0, collided=True, reached=False, cfg=cfg) == -5.0
    assert reward(2.0, 2.0, collided=False, reached=True, cfg=cfg) == 10.0


def test_termination_vs_truncation():
    cfg = NavTaskConfig(max_steps=500)
    assert terminated(reached=True, collided=False)
    assert terminated(reached=False, collided=True)
    assert not terminated(reached=False, collided=False)
    assert truncated(step=500, cfg=cfg)
    assert not truncated(step=499, cfg=cfg)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
