"""Contract + behaviour tests for the PyBullet rigid-body nav backend (src/isaac/nav_pybullet.py).

Needs `pybullet` (a prebuilt wheel on Linux/the A10G box; the macOS source build is flaky), so
the whole module SKIPS when pybullet isn't importable — exactly like the GPU/dataset rows. The
task math it reuses (`nav_task`, scene sampling) is already locked by the pure-NumPy tests; here
we assert the physics backend honours the same contract and that real contacts behave.

Run (box): pytest tests/test_nav_pybullet.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("pybullet", reason="pybullet not installed (Linux/box only)")

from isaac.nav_pybullet import PyBulletNavEnv, shielded_action   # noqa: E402
from isaac.nav_sim import DiffDriveNavEnv, NavSimConfig, heuristic_action   # noqa: E402
from isaac.nav_task import OBS_DIM, ACT_DIM, NavTaskConfig       # noqa: E402


def _open_field(**kw) -> NavSimConfig:
    base = dict(fixed_start=(0.0, 0.0, 0.0), fixed_goal=(3.0, 0.0),
                task=NavTaskConfig(max_steps=300))
    base.update(kw)
    return NavSimConfig(**base)


def test_reset_and_step_contract():
    env = PyBulletNavEnv(_open_field(n_lidar_beams=16))
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM + 16,) and obs.dtype == np.float32
        assert env.obs_dim == OBS_DIM + 16 and env.act_dim == ACT_DIM
        assert {"robot_xy", "goal_xy", "distance", "collided", "reached"} <= set(info)
        obs, r, term, trunc, info = env.step([1.0, 0.0])
        assert obs.shape == (OBS_DIM + 16,)
        assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)
    finally:
        env.close()


def test_scene_matches_kinematic_for_same_seed():
    # The PyBullet env samples its scene via the internal kinematic env, so a given seed must
    # yield the identical goal + obstacle field — the premise of a fair kinematic↔physics compare.
    cfg = NavSimConfig(randomize_obstacles=5, n_lidar_beams=16)
    kin = DiffDriveNavEnv(cfg)
    kin.reset(seed=7)
    pb = PyBulletNavEnv(cfg)
    try:
        pb.reset(seed=7)
        assert np.allclose(pb._kin._goal, kin._goal)
        assert pb._kin._obstacles.shape == kin._obstacles.shape
        assert np.allclose(pb._kin._obstacles, kin._obstacles)
    finally:
        pb.close()


def test_physics_robot_moves_forward():
    env = PyBulletNavEnv(_open_field())
    try:
        env.reset(seed=0)
        x0 = env._kin.robot_xy[0]
        for _ in range(10):
            env.step([1.0, 0.0])           # drive straight along +x
        assert env._kin.robot_xy[0] > x0 + 0.3, "robot should advance under forward command"
    finally:
        env.close()


def test_collision_is_detected_by_contacts():
    # Obstacle straight ahead; driving into it must register a physics contact + terminate.
    cfg = _open_field(obstacles=np.array([[1.0, 0.0, 0.4]]))
    env = PyBulletNavEnv(cfg)
    try:
        env.reset(seed=0)
        hit = False
        for _ in range(80):
            _, _, term, _, info = env.step([1.0, 0.0])
            if term and info["collided"]:
                hit = True
                break
        assert hit, "driving into the obstacle should be caught by contact detection"
    finally:
        env.close()


def test_shielded_heuristic_reaches_on_open_field():
    # On an open field the go-to-goal heuristic (shielded) should reach the goal in the rigid-body
    # sim — proving the physics backend is navigable end to end and the shield doesn't block a
    # clear path. (Going *around* an obstacle in the robot's path is the learned policy's job, not
    # this non-avoider heuristic's — with an obstacle dead ahead the shield correctly stops it and
    # the heuristic, having no avoidance, would just deadlock. See the transfer eval for that.)
    cfg = _open_field(n_lidar_beams=16, fixed_goal=(3.0, 0.0),
                      task=NavTaskConfig(max_steps=400))
    env = PyBulletNavEnv(cfg)
    try:
        obs, _ = env.reset(seed=0)
        reached = collided = False
        for _ in range(400):
            obs, r, term, trunc, info = env.step(shielded_action(env, heuristic_action(obs, cfg)))
            collided = collided or info["collided"]
            if term or trunc:
                reached = info["reached"]
                break
        assert reached and not collided, "shielded heuristic should reach the goal on an open field"
    finally:
        env.close()


def test_shield_prevents_collision_in_physics():
    # The safety guarantee must survive real contacts: a reckless full-forward policy driven
    # through the shield into an obstacle dead ahead must never register a physics collision (it
    # may deadlock at the margin — that's fine; the invariant is *no contact*).
    cfg = _open_field(obstacles=np.array([[1.2, 0.0, 0.4]]), n_lidar_beams=16,
                      fixed_goal=(3.0, 0.0), task=NavTaskConfig(max_steps=250))
    env = PyBulletNavEnv(cfg)
    try:
        obs, _ = env.reset(seed=0)
        for _ in range(250):
            act = shielded_action(env, np.array([1.0, 0.4]))   # reckless: full speed ahead
            obs, r, term, trunc, info = env.step(act)
            assert not info["collided"], "shield must prevent every physics collision"
            if term or trunc:
                break
    finally:
        env.close()
