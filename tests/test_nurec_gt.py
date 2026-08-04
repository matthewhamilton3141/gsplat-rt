"""Tests for the NuRec ground-truth (clipgt) nav path (src/isaac/nurec_scene.py).

CPU/NumPy + OpenCV only — the GT *parsing* (`load_nurec_gt`, needs a real clip + pyarrow) is not
unit-tested; the pure builders it feeds are: quaternion yaw, GT→occupancy rasterisation, arc length,
pure-pursuit target selection, and a full synthetic end-to-end (boundaries + obstacle box + ego
path → occupancy → the shielded DWA car follows the route to the goal).

Run:
    pytest tests/test_nurec_gt.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isaac.car_sim import BicycleNavEnv, car_dwa_action, car_safety_shield  # noqa: E402
from isaac.driving_scene import driving_scene_config  # noqa: E402
from isaac.nurec_scene import (  # noqa: E402
    gt_to_occupancy, path_arclength, pursuit_target, quat_yaw,
)


def test_quat_yaw():
    assert abs(quat_yaw(0, 0, 0, 1)) < 1e-9                       # identity -> 0
    assert abs(quat_yaw(0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)) - np.pi / 2) < 1e-6  # +90° z
    assert abs(quat_yaw(0, 0, np.sin(-np.pi / 6), np.cos(-np.pi / 6)) + np.pi / 3) < 1e-6


def test_gt_to_occupancy_boundaries_and_box():
    bounds = (0.0, -5.0, 40.0, 5.0)
    boundaries = [np.array([[0.0, 3.0], [40.0, 3.0]]),        # top wall
                  np.array([[0.0, -3.0], [40.0, -3.0]])]      # bottom wall
    boxes = [(20.0, 1.5, 3.0, 1.5, 0.0)]                      # one obstacle
    g = gt_to_occupancy(boundaries, boxes, bounds, resolution=0.25, dilate_cells=0)
    assert g.occupied_at(np.array([20.0, 1.5]))              # inside the obstacle box
    assert g.occupied_at(np.array([10.0, 3.0]))              # on the top wall
    assert not g.occupied_at(np.array([10.0, 0.0]))          # clear corridor
    assert not g.occupied_at(np.array([35.0, 0.0]))


def test_path_arclength():
    path = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])     # 3 + 4 = 7 m
    arc = path_arclength(path)
    assert arc.shape == (3,)
    assert abs(arc[-1] - 7.0) < 1e-9 and arc[0] == 0.0


def test_pursuit_target_straight_and_advances():
    path = np.stack([np.arange(0, 31, 1.0), np.zeros(31)], axis=1)   # along +x
    arc = path_arclength(path)
    t0 = pursuit_target(path, arc, np.array([5.0, 0.0]), lookahead=10.0)
    assert np.allclose(t0, [15.0, 0.0], atol=1e-6)                   # 5 + 10 ahead
    t1 = pursuit_target(path, arc, np.array([20.0, 0.0]), lookahead=10.0)
    assert t1[0] > t0[0]                                             # advances with the car
    # clamps to the final point near the end
    tend = pursuit_target(path, arc, np.array([29.0, 0.0]), lookahead=10.0)
    assert np.allclose(tend, path[-1])


def test_pursuit_target_follows_a_turn():
    # L-shaped path: east then north. From the corner, a look-ahead points up the second leg.
    east = np.stack([np.arange(0, 11, 1.0), np.zeros(11)], axis=1)
    north = np.stack([np.full(10, 10.0), np.arange(1, 11, 1.0)], axis=1)
    path = np.vstack([east, north])
    arc = path_arclength(path)
    tgt = pursuit_target(path, arc, np.array([10.0, 0.0]), lookahead=5.0)   # at the corner
    assert tgt[0] == 10.0 and tgt[1] > 0.0                          # turned up the north leg


def test_synthetic_gt_drive_follows_route_to_goal():
    # Full committed path on synthetic GT: a corridor (two boundary walls) + one obstacle box, a
    # straight ego route; the shielded DWA car pursues the route and reaches the goal, no collision.
    bounds = (-2.0, -6.0, 42.0, 6.0)
    boundaries = [np.array([[0.0, 3.5], [40.0, 3.5]]), np.array([[0.0, -3.5], [40.0, -3.5]])]
    boxes = [(20.0, 1.6, 3.0, 1.6, 0.0)]                     # car in the upper lane -> dip under it
    grid = gt_to_occupancy(boundaries, boxes, bounds, resolution=0.25, dilate_cells=1)
    ego = np.stack([np.linspace(0.5, 38.0, 60), np.zeros(60)], axis=1)
    arc = path_arclength(ego)
    start, goal = (0.5, 0.0, 0.0), (38.0, 0.0)

    cfg = driving_scene_config(grid, start, goal, robot_radius=0.7, safety_margin=0.3,
                               max_speed=4.0, max_steer=0.5, wheelbase=2.2, max_ang_vel=2.5,
                               n_lidar_beams=16, lidar_fov=np.pi, lidar_range=20.0)
    cfg.task.goal_radius = 3.0
    cfg.task.max_steps = 2500
    env = BicycleNavEnv(cfg)
    env.reset(seed=0)
    reached = False
    for _ in range(cfg.task.max_steps):
        tgt = pursuit_target(ego, arc, env.robot_xy, lookahead=8.0)
        raw = car_dwa_action(env.robot_xy, env._heading, tgt, env._obstacles, cfg, horizon=20)
        safe = car_safety_shield(raw, env.robot_xy, env._heading, env._obstacles, cfg)
        _, _, term, trunc, info = env.step(safe)
        assert not info["collided"], "car must not hit the reconstructed GT geometry"
        if term or trunc:
            reached = info["reached"]
            break
    assert reached, "shielded car should pursue the ego route to the goal"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
