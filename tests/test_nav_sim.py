"""Tests for the pure-NumPy diff-drive nav simulator (src/isaac/nav_sim.py).

CPU/NumPy only — no Isaac, no torch, no gymnasium, no GPU. Locks the sim loop, collision,
lidar, and the delegation to the tested `nav_task` reward/termination so a PyBullet or
Isaac Lab port can be validated against known-good behaviour.

Run:
    pytest tests/test_nav_sim.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isaac.nav_sim import (  # noqa: E402
    DiffDriveNavEnv, NavSimConfig, avoidance_action, heuristic_action,
    random_obstacle_field,
)
from isaac.nav_task import (  # noqa: E402
    ACT_DIM, OBS_DIM, NavTaskConfig, reward as nav_reward,
)


def _open_field(**kw) -> NavSimConfig:
    """Obstacle-free config with a fixed, solvable start/goal for deterministic tests."""
    base = dict(fixed_start=(0.0, 0.0, 0.0), fixed_goal=(3.0, 0.0),
                task=NavTaskConfig(max_steps=300))
    base.update(kw)
    return NavSimConfig(**base)


def test_reset_shapes_and_info():
    env = DiffDriveNavEnv(_open_field())
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert env.obs_dim == OBS_DIM and env.act_dim == ACT_DIM
    assert set(("robot_xy", "goal_xy", "distance", "collided", "reached")) <= set(info)
    assert np.allclose(info["robot_xy"], [0.0, 0.0])
    assert np.allclose(info["goal_xy"], [3.0, 0.0])


def test_step_tuple_and_reward_matches_nav_task():
    env = DiffDriveNavEnv(_open_field())
    env.reset(seed=0)
    prev = env._prev_dist
    obs, r, term, trunc, info = env.step([1.0, 0.0])   # drive straight toward goal (+x)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)
    # One dt of forward motion at 1 m/s closes 0.1 m of distance; reward must equal nav_task.
    expected = nav_reward(prev, info["distance"], collided=False, reached=False,
                          cfg=env.cfg.task)
    assert abs(r - expected) < 1e-9
    assert info["distance"] < prev            # got closer


def test_determinism_under_seed():
    def rollout():
        env = DiffDriveNavEnv(NavSimConfig(task=NavTaskConfig(max_steps=50)))
        obs, _ = env.reset(seed=123)
        rng = np.random.default_rng(7)
        traj = [obs.copy()]
        for _ in range(20):
            a = rng.uniform(-1, 1, ACT_DIM)
            obs, *_ = env.step(a)
            traj.append(obs.copy())
        return np.array(traj)
    assert np.allclose(rollout(), rollout())


def test_heuristic_reaches_goal_on_open_field():
    env = DiffDriveNavEnv(_open_field())
    obs, _ = env.reset(seed=0)
    reached = False
    for _ in range(env.cfg.task.max_steps):
        obs, r, term, trunc, info = env.step(heuristic_action(obs, env.cfg))
        if term:
            reached = info["reached"]
            break
    assert reached, "go-to-goal controller should reach the goal on an obstacle-free field"


def test_collision_terminates_with_penalty():
    # Obstacle straight ahead between start (0,0) and goal (3,0); drive forward into it.
    cfg = _open_field(obstacles=np.array([[1.0, 0.0, 0.4]]))
    env = DiffDriveNavEnv(cfg)
    obs, _ = env.reset(seed=0)
    hit = False
    for _ in range(100):
        obs, r, term, trunc, info = env.step([1.0, 0.0])
        if term:
            hit = info["collided"]
            assert r < 0, "collision step should carry the collision penalty"
            break
    assert hit, "robot driving into the obstacle should register a collision + terminate"
    # Robot must have stopped short of the obstacle surface, never inside it.
    assert info["robot_xy"][0] < 1.0 - 0.4


def test_out_of_bounds_is_collision():
    cfg = _open_field(bounds=(-1.0, -1.0, 1.0, 1.0), fixed_goal=(0.5, 0.0),
                      task=NavTaskConfig(max_steps=300, goal_radius=0.05))
    env = DiffDriveNavEnv(cfg)
    env.reset(seed=0)
    term = False
    for _ in range(100):
        _, r, term, trunc, info = env.step([1.0, 0.0])   # drive toward +x wall past the goal
        if term and info["collided"]:
            break
    assert term and info["collided"]


def test_truncation_when_goal_unreachable():
    # Goal outside a tiny walled box: never reached, so the episode must truncate, not terminate.
    cfg = NavSimConfig(fixed_start=(0.0, 0.0, np.pi), fixed_goal=(4.0, 0.0),
                       bounds=(-2.0, -2.0, 2.0, 2.0),
                       task=NavTaskConfig(max_steps=10, goal_radius=0.1))
    env = DiffDriveNavEnv(cfg)
    env.reset(seed=0)
    trunc = False
    for _ in range(10):
        _, _, term, trunc, info = env.step([0.0, 0.0])   # sit still, burn the step budget
        assert not term
    assert trunc and info["step"] == 10


def test_lidar_shape_and_senses_obstacle():
    cfg = _open_field(n_lidar_beams=5, lidar_range=5.0,
                      obstacles=np.array([[1.0, 0.0, 0.4]]))
    env = DiffDriveNavEnv(cfg)
    obs, _ = env.reset(seed=0)          # start (0,0) facing +x, obstacle 1 m ahead
    assert env.obs_dim == OBS_DIM + 5
    assert obs.shape == (OBS_DIM + 5,)
    lidar = obs[OBS_DIM:]
    assert np.all((lidar >= 0.0) & (lidar <= 1.0))
    centre = lidar[2]                    # middle beam points along the heading (+x)
    # Nearest hit is the obstacle surface at ~0.6 m -> normalised ~0.12, well under 1.0.
    assert centre < 0.2
    assert abs(centre * cfg.lidar_range - 0.6) < 0.05


def test_action_clipping_respects_limits():
    cfg = _open_field(max_lin_vel=0.5, max_ang_vel=1.0)
    env = DiffDriveNavEnv(cfg)
    env.reset(seed=0)
    env.step([10.0, 10.0])              # way over the limits
    assert env._lin_vel == 0.5 and env._ang_vel == 1.0


def test_avoidance_falls_back_to_heuristic_without_lidar():
    # No lidar -> the gap-follower has nothing to sense, must equal the go-to-goal controller.
    cfg = _open_field(n_lidar_beams=0)
    env = DiffDriveNavEnv(cfg)
    obs, _ = env.reset(seed=0)
    assert np.allclose(avoidance_action(obs, cfg), heuristic_action(obs, cfg))


def test_avoidance_does_not_ram_head_on_obstacle():
    # Same head-on obstacle the naive heuristic crashes into: the gap-follower must steer clear.
    cfg = _open_field(n_lidar_beams=9, lidar_fov=np.pi, lidar_range=5.0,
                      obstacles=np.array([[1.0, 0.0, 0.4]]))
    env = DiffDriveNavEnv(cfg)
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        obs, r, term, trunc, info = env.step(avoidance_action(obs, cfg))
        if term and info["collided"]:
            raise AssertionError("gap-follower drove into an obstacle it could see")
        if term and info["reached"]:
            break
    assert not info["collided"]


def test_avoidance_reaches_goal_around_obstacle():
    # Obstacle squarely on the straight-line path; the gap-follower should detour and arrive.
    cfg = NavSimConfig(
        fixed_start=(0.0, 0.0, 0.0), fixed_goal=(4.0, 0.0),
        bounds=(-6.0, -6.0, 6.0, 6.0), n_lidar_beams=11, lidar_fov=np.pi, lidar_range=5.0,
        obstacles=np.array([[2.0, 0.0, 0.5]]), task=NavTaskConfig(max_steps=1000),
    )
    env = DiffDriveNavEnv(cfg)
    obs, _ = env.reset(seed=0)
    reached = False
    for _ in range(env.cfg.task.max_steps):
        obs, r, term, trunc, info = env.step(avoidance_action(obs, cfg))
        assert not info["collided"], "should never collide while following the gap"
        if term:
            reached = info["reached"]
            break
    assert reached, "gap-follower should route around a single obstacle to the goal"


def test_random_obstacle_field_non_overlapping_and_in_bounds():
    bounds = (-5.0, -5.0, 5.0, 5.0)
    field = random_obstacle_field(8, bounds, radius_range=(0.3, 0.6), clearance=0.4, seed=1)
    assert field.ndim == 2 and field.shape[1] == 3 and len(field) <= 8
    x_min, y_min, x_max, y_max = bounds
    for cx, cy, r in field:
        assert x_min <= cx - r and cx + r <= x_max
        assert y_min <= cy - r and cy + r <= y_max
    for i in range(len(field)):
        for j in range(i + 1, len(field)):
            (xi, yi, ri), (xj, yj, rj) = field[i], field[j]
            assert np.hypot(xi - xj, yi - yj) > ri + rj, "obstacles must not overlap"


def test_random_obstacle_field_keeps_points_clear_and_is_deterministic():
    bounds = (-5.0, -5.0, 5.0, 5.0)
    start, goal = (0.0, 0.0), (4.0, 0.0)
    a = random_obstacle_field(10, bounds, keep_clear=(start, goal), seed=42)
    b = random_obstacle_field(10, bounds, keep_clear=(start, goal), seed=42)
    assert np.array_equal(a, b), "same seed must reproduce the field"
    for cx, cy, r in a:
        assert np.hypot(cx - start[0], cy - start[1]) > r, "start must stay outside obstacles"
        assert np.hypot(cx - goal[0], cy - goal[1]) > r, "goal must stay outside obstacles"


def test_avoidance_beats_blind_heuristic_on_random_fields():
    # Over a set of procedurally generated fields, the lidar gap-follower must be a genuine
    # obstacle-aware baseline: no collisions, and it reaches the goal more often than the
    # blind go-to-goal controller (which drives straight into obstacles).
    bounds = (-5.0, -5.0, 5.0, 5.0)
    start, goal = (-4.0, -4.0, 0.0), (4.0, 4.0)

    def run(policy, seed):
        field = random_obstacle_field(5, bounds, keep_clear=(start[:2], goal), seed=seed)
        cfg = NavSimConfig(fixed_start=start, fixed_goal=goal, bounds=bounds, obstacles=field,
                           n_lidar_beams=15, lidar_fov=np.pi, lidar_range=5.0,
                           task=NavTaskConfig(max_steps=700))
        env = DiffDriveNavEnv(cfg)
        obs, _ = env.reset(seed=seed)
        for _ in range(700):
            obs, r, term, trunc, info = env.step(policy(obs, cfg))
            if term:
                return bool(info["reached"]), bool(info["collided"])
        return False, False

    seeds = range(15)
    avoid = [run(avoidance_action, s) for s in seeds]
    blind = [run(heuristic_action, s) for s in seeds]
    avoid_reached = sum(r for r, _ in avoid)
    avoid_collided = sum(c for _, c in avoid)
    blind_reached = sum(r for r, _ in blind)

    assert avoid_collided == 0, "gap-follower should not collide on solvable fields"
    assert avoid_reached > blind_reached, "gap-follower should reach the goal more than blind"


def test_random_field_env_is_collision_free_at_reset():
    # A field generated with the start/goal kept clear must yield a valid (non-colliding) reset.
    bounds = (-5.0, -5.0, 5.0, 5.0)
    start, goal = (-4.0, -4.0, 0.0), (4.0, 4.0)
    field = random_obstacle_field(12, bounds, keep_clear=(start[:2], goal), seed=3)
    cfg = NavSimConfig(fixed_start=start, fixed_goal=goal, bounds=bounds, obstacles=field)
    env = DiffDriveNavEnv(cfg)
    _, info = env.reset(seed=0)
    assert not env._collides(env.robot_xy) and not info["collided"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
