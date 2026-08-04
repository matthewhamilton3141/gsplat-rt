"""Tests for the kinematic-bicycle car env + braking shield + driving scene.

CPU/NumPy + OpenCV only — no torch, no gymnasium, no GPU. Covers:
  - bicycle kinematics (lookahead == integration, straight/curved motion, no pivot-in-place),
  - the braking safety shield (passes safe, brakes when blocked, never enters the margin,
    prevents every collision in-the-loop over reconstructed grid geometry),
  - the driving scene + gap-follower solving the slalom collision-free (the digital-twin loop:
    a car threading a reconstructed-style scene under the shield).

Run:
    pytest tests/test_car_sim.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isaac.car_sim import (  # noqa: E402
    BicycleNavEnv, CarSimConfig, bicycle_yaw_rate, car_dwa_action, car_safety_shield,
    predict_pose_bicycle,
)
from isaac.driving_scene import (  # noqa: E402
    driving_scene_config, load_occupancy_grid, make_driving_scene,
)
from isaac.nav_task import NavTaskConfig, OBS_DIM  # noqa: E402


def _car_cfg(**kw) -> CarSimConfig:
    base = dict(fixed_start=(0.0, 0.0, 0.0), fixed_goal=(4.0, 0.0),
                bounds=(-6.0, -6.0, 6.0, 6.0), task=NavTaskConfig(max_steps=400))
    base.update(kw)
    return CarSimConfig(**base)


# ----------------------------------------------------------------------------------------
# bicycle kinematics
# ----------------------------------------------------------------------------------------

def test_lookahead_matches_step_integration():
    # The shield's guarantee is void unless its lookahead integrates exactly as step() does.
    env = BicycleNavEnv(_car_cfg())
    env.reset(seed=0)
    xy0, h0 = env.robot_xy.copy(), env._heading
    speed, steer = 1.0, 0.3
    predicted = predict_pose_bicycle(xy0, h0, speed, steer, env.cfg.wheelbase, env.cfg.dt)
    env.step([speed, steer])
    assert np.allclose(env.robot_xy, predicted, atol=1e-9)


def test_zero_speed_cannot_pivot():
    # A car can't turn in place: full steering at zero speed changes neither pose nor heading.
    env = BicycleNavEnv(_car_cfg())
    env.reset(seed=0)
    xy0, h0 = env.robot_xy.copy(), env._heading
    env.step([0.0, env.cfg.max_steer])
    assert np.allclose(env.robot_xy, xy0, atol=1e-12)
    assert abs(env._heading - h0) < 1e-12


def test_straight_when_steer_zero():
    env = BicycleNavEnv(_car_cfg(fixed_start=(0.0, 0.0, 0.0)))
    env.reset(seed=0)
    for _ in range(10):
        env.step([1.0, 0.0])
    assert abs(env.robot_xy[1]) < 1e-9 and env.robot_xy[0] > 0.0   # moved along +x, no drift


def test_positive_steer_turns_left():
    # +steering -> +yaw rate -> heading increases (turns left / CCW).
    env = BicycleNavEnv(_car_cfg(fixed_start=(0.0, 0.0, 0.0)))
    env.reset(seed=0)
    env.step([1.0, 0.4])
    assert env._heading > 0.0
    assert bicycle_yaw_rate(1.0, 0.4, env.cfg.wheelbase) > 0.0


def test_yaw_rate_clip_guard():
    # If max_ang_vel can't bound the max yaw rate, cornering would be silently clipped -> raise.
    try:
        BicycleNavEnv(CarSimConfig(max_speed=3.0, max_steer=1.0, wheelbase=0.3, max_ang_vel=1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for an unbounded yaw rate")


# ----------------------------------------------------------------------------------------
# braking safety shield
# ----------------------------------------------------------------------------------------

def test_car_shield_passes_safe_action():
    cfg = _car_cfg(bounds=(-6.0, -6.0, 6.0, 6.0))
    out = car_safety_shield([1.0, 0.2], np.array([0.0, 0.0]), 0.0, np.zeros((0, 3)), cfg)
    assert abs(out[0] - 1.0) < 1e-6 and abs(out[1] - 0.2) < 1e-6


def test_car_shield_brakes_into_obstacle():
    # Obstacle right ahead: no safe forward speed -> shield brakes to 0, steering passes through.
    cfg = _car_cfg(obstacles=np.array([[0.35, 0.0, 0.20]]), robot_radius=0.18, safety_margin=0.05)
    out = car_safety_shield([1.0, 0.3], np.array([0.0, 0.0]), 0.0, cfg.obstacle_array(), cfg)
    assert out[0] == 0.0
    assert abs(out[1] - 0.3) < 1e-6


def test_car_shield_never_enters_margin():
    rng = np.random.default_rng(0)
    cfg = CarSimConfig(bounds=(-5.0, -5.0, 5.0, 5.0), robot_radius=0.18, safety_margin=0.05,
                       max_speed=1.4, max_steer=0.5, wheelbase=0.5, max_ang_vel=3.0)
    from isaac.nav_sim import world_clearance
    for _ in range(500):
        obstacles = np.column_stack([rng.uniform(-4, 4, 4), rng.uniform(-4, 4, 4),
                                     rng.uniform(0.3, 0.6, 4)])
        xy = rng.uniform(-4, 4, 2)
        if world_clearance(xy, obstacles, cfg) < cfg.safety_margin:
            continue                                  # already inside the margin; shield only not-worse
        heading = rng.uniform(-np.pi, np.pi)
        action = [rng.uniform(-1.4, 1.4), rng.uniform(-0.5, 0.5)]
        out = car_safety_shield(action, xy, heading, obstacles, cfg)
        nxt = predict_pose_bicycle(xy, heading, out[0], out[1], cfg.wheelbase, cfg.dt)
        assert world_clearance(nxt, obstacles, cfg) >= cfg.safety_margin - 1e-9


def test_car_shield_prevents_collision_in_loop_over_grid():
    # Reckless "always full throttle" car driven THROUGH the braking shield must never hit the
    # reconstructed grid geometry between start and goal.
    from isaac.grid_world import GridWorld
    obstacles = np.array([[1.0, 0.0, 0.4], [1.6, 0.6, 0.3], [1.6, -0.6, 0.3]])
    g = GridWorld.from_obstacles((-6.0, -6.0, 6.0, 6.0), obstacles, resolution=0.04)
    cfg = _car_cfg(grid_world=g, robot_radius=0.2, safety_margin=0.08)
    env = BicycleNavEnv(cfg)
    env.reset(seed=0)
    for _ in range(400):
        raw = np.array([1.4, 0.4])
        safe = car_safety_shield(raw, env.robot_xy, env._heading, env._obstacles, cfg)
        _, _, term, trunc, info = env.step(safe)
        assert not info["collided"], "braking shield must prevent every collision"
        if term or trunc:
            break


# ----------------------------------------------------------------------------------------
# driving scene + gap-follower (the digital-twin closed loop)
# ----------------------------------------------------------------------------------------

def test_make_driving_scene_valid():
    grid, start, goal = make_driving_scene()
    cfg = driving_scene_config(grid, start, goal)
    env = BicycleNavEnv(cfg)
    _, info = env.reset(seed=0)
    assert not env._collides(env.robot_xy), "start must be collision-free on the road"
    assert not grid.occupied_at(np.array(goal)), "goal must sit in free road"
    assert grid.occupied_at(np.array([0.0, 2.9])), "curb/building flanks the road"


def test_dwa_car_threads_the_slalom():
    # THE integration test: the DWA-planned car, behind the braking shield, weaves the alternating
    # obstacle slalom of a reconstructed-style driving scene and reaches the goal, never colliding.
    grid, start, goal = make_driving_scene()
    cfg = driving_scene_config(grid, start, goal)
    env = BicycleNavEnv(cfg)
    env.reset(seed=0)
    reached = False
    for _ in range(cfg.task.max_steps):
        raw = car_dwa_action(env.robot_xy, env._heading, env._goal, env._obstacles, cfg)
        safe = car_safety_shield(raw, env.robot_xy, env._heading, env._obstacles, cfg)
        _, _, term, trunc, info = env.step(safe)
        assert not info["collided"], "car must thread the scene without colliding"
        if term or trunc:
            reached = info["reached"]
            break
    assert reached, "DWA-planned shielded car should reach the far end of the driving scene"


def test_dwa_reaches_around_single_obstacle():
    # DWA solves what the reactive gap-follower could not: a car arcing around an obstacle
    # squarely on the straight-line path, without wedging at the shield's keep-out shell.
    from isaac.grid_world import GridWorld
    B = (-1.0, -4.0, 20.0, 4.0)
    g = GridWorld.from_obstacles(B, np.array([[8.0, 0.0, 1.0]]), resolution=0.05)
    cfg = CarSimConfig(bounds=B, grid_world=g, fixed_start=(0.0, 0.0, 0.0), fixed_goal=(18.0, 0.0),
                       robot_radius=0.25, safety_margin=0.12, max_speed=1.1, max_steer=0.6,
                       wheelbase=0.45, max_ang_vel=3.0, n_lidar_beams=16, lidar_fov=np.pi,
                       lidar_range=6.0, task=NavTaskConfig(max_steps=500, goal_radius=0.6))
    env = BicycleNavEnv(cfg)
    env.reset(seed=0)
    reached = False
    for _ in range(cfg.task.max_steps):
        raw = car_dwa_action(env.robot_xy, env._heading, env._goal, env._obstacles, cfg)
        safe = car_safety_shield(raw, env.robot_xy, env._heading, env._obstacles, cfg)
        _, _, term, trunc, info = env.step(safe)
        assert not info["collided"]
        if term or trunc:
            reached = info["reached"]
            break
    assert reached, "DWA should route the car around a single blocking obstacle to the goal"


def test_car_lidar_senses_scene_geometry():
    grid, start, goal = make_driving_scene()
    cfg = driving_scene_config(grid, start, goal, n_lidar_beams=16)
    env = BicycleNavEnv(cfg)
    obs, _ = env.reset(seed=0)
    lidar = obs[OBS_DIM:OBS_DIM + 16]
    assert lidar.shape == (16,)
    assert np.all((lidar >= 0.0) & (lidar <= 1.0))
    assert lidar.min() < 1.0, "some beam should see the curb / a parked car"


def test_load_occupancy_grid_npy_roundtrip(tmp_path):
    arr = np.zeros((10, 12), np.int8)
    arr[3:6, 4:7] = 1                                  # an occupied block
    p = os.path.join(tmp_path, "occ.npy")
    np.save(p, arr)
    g = load_occupancy_grid(p, resolution=0.2, origin=(-1.0, -1.0))
    assert g.shape == (10, 12)
    # cell (row=4, col=5) is occupied -> its world centre reads occupied.
    x = -1.0 + (5 + 0.5) * 0.2
    y = -1.0 + (4 + 0.5) * 0.2
    assert g.occupied_at(np.array([x, y]))
    assert not g.occupied_at(np.array([-0.9, -0.9]))   # corner cell (0,0) is free


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
