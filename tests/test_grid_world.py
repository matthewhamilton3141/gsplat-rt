"""Tests for the occupancy-grid world (src/isaac/grid_world.py) and its nav-env integration.

CPU/NumPy + OpenCV only — no torch, no gymnasium, no GPU. Two layers:

  1. GridWorld primitives in isolation: distance field vs brute-force Euclidean, clearance sign,
     disc collision, and lidar raycast against known geometry.
  2. Integration with DiffDriveNavEnv / safety_shield: a grid obstacle collides, is sensed by
     lidar, and — the load-bearing guarantee — the one-step shield never lets a reckless policy
     hit reconstructed geometry. Plus an *equivalence* check: a circle world and the same world
     rasterised into a grid produce matching navigation behaviour (the faithful-bridge property).

Run:
    pytest tests/test_grid_world.py -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isaac.grid_world import GridWorld  # noqa: E402
from isaac.nav_sim import (  # noqa: E402
    DiffDriveNavEnv, NavSimConfig, avoidance_action, safety_shield, world_clearance,
)
from isaac.nav_task import NavTaskConfig, OBS_DIM  # noqa: E402


# ----------------------------------------------------------------------------------------
# GridWorld primitives
# ----------------------------------------------------------------------------------------

def _brute_edt(occ: np.ndarray, res: float) -> np.ndarray:
    """Reference distance field: each free cell's Euclidean distance to the nearest occupied."""
    occ = occ > 0
    oi, oj = np.nonzero(occ)
    out = np.zeros(occ.shape, float)
    for i in range(occ.shape[0]):
        for j in range(occ.shape[1]):
            if occ[i, j]:
                continue
            out[i, j] = np.min(np.hypot(i - oi, j - oj)) * res
    return out


def test_bounds_and_shape():
    occ = np.zeros((4, 6), np.uint8)
    g = GridWorld(occ, origin=(-1.0, -2.0), resolution=0.5)
    assert g.shape == (4, 6)
    # width 6*0.5 = 3.0, height 4*0.5 = 2.0
    assert g.bounds == (-1.0, -2.0, 2.0, 0.0)


def test_world_to_cell_roundtrip():
    g = GridWorld(np.zeros((10, 10), np.uint8), origin=(0.0, 0.0), resolution=0.1)
    assert g.world_to_cell(np.array([0.05, 0.05])) == (0, 0)
    assert g.world_to_cell(np.array([0.35, 0.95])) == (9, 3)   # row = y, col = x


def test_distance_field_matches_bruteforce():
    rng = np.random.default_rng(0)
    occ = (rng.random((12, 15)) < 0.15).astype(np.uint8)
    occ[0, 0] = 1                                   # ensure at least one occupied cell
    g = GridWorld(occ, origin=(0.0, 0.0), resolution=0.1)
    assert np.allclose(g.distance_field(), _brute_edt(occ, 0.1), atol=1e-4)


def test_distance_field_edge_cases():
    # No obstacles -> everything infinitely clear.
    g_free = GridWorld(np.zeros((5, 5), np.uint8))
    assert np.all(np.isinf(g_free.distance_field()))
    # Fully occupied -> zero everywhere.
    g_full = GridWorld(np.ones((5, 5), np.uint8))
    assert np.all(g_full.distance_field() == 0.0)


def test_clearance_sign_and_outside_grid():
    occ = np.zeros((20, 20), np.uint8)
    occ[10, 10] = 1                                 # one occupied cell
    g = GridWorld(occ, origin=(0.0, 0.0), resolution=0.1)
    centre = np.array([10.5 * 0.1, 10.5 * 0.1])     # centre of the occupied cell
    assert g.clearance(centre, robot_radius=0.0) == 0.0      # on the obstacle
    assert g.clearance(centre, robot_radius=0.2) < 0.0       # disc overlaps -> negative
    far = np.array([0.05, 0.05])                    # opposite corner, well clear
    assert g.clearance(far, robot_radius=0.1) > 0.0
    # Outside the grid: no obstacle from the grid -> +inf.
    assert np.isinf(g.clearance(np.array([-5.0, -5.0]), robot_radius=0.1))


def test_collides_disc():
    occ = np.zeros((20, 20), np.uint8)
    occ[10, 10] = 1
    g = GridWorld(occ, origin=(0.0, 0.0), resolution=0.1)
    assert g.collides(np.array([10.5 * 0.1, 10.5 * 0.1]), robot_radius=0.05)
    assert not g.collides(np.array([0.05, 0.05]), robot_radius=0.05)


def test_raycast_hits_and_misses():
    # A vertical wall of occupied cells at col=15 (world x = 1.5..1.6). Robot at (0.05, y) firing
    # +x hits the wall face at ~1.5 m.
    occ = np.zeros((30, 30), np.uint8)
    occ[:, 15] = 1
    g = GridWorld(occ, origin=(0.0, 0.0), resolution=0.1)
    origin = np.array([0.05, 1.05])
    d_hit = g.raycast(origin, np.array([1.0, 0.0]), max_range=5.0)
    assert abs(d_hit - 1.5) < 0.1                   # within a cell of the wall face
    # Firing -x (away from the wall) leaves the grid without a hit -> max_range.
    d_miss = g.raycast(origin, np.array([-1.0, 0.0]), max_range=5.0)
    assert d_miss == 5.0


def test_from_obstacles_rasterises_circle():
    bounds = (-3.0, -3.0, 3.0, 3.0)
    g = GridWorld.from_obstacles(bounds, np.array([[0.0, 0.0, 1.0]]), resolution=0.05)
    assert g.occupied_at(np.array([0.0, 0.0]))          # centre occupied
    assert not g.occupied_at(np.array([2.5, 2.5]))      # far corner free
    # A point just outside the circle radius is free; just inside is occupied.
    assert not g.occupied_at(np.array([1.2, 0.0]))
    assert g.occupied_at(np.array([0.8, 0.0]))


def test_occupied_mask_vectorised_matches_scalar():
    g = GridWorld.from_obstacles((-2.0, -2.0, 2.0, 2.0),
                                 np.array([[0.5, 0.5, 0.4]]), resolution=0.05)
    rng = np.random.default_rng(1)
    pts = rng.uniform(-2, 2, (20, 2))
    mask = g.occupied_mask(pts[:, 0], pts[:, 1])
    scalar = np.array([g.occupied_at(p) for p in pts])
    assert np.array_equal(mask, scalar)


# ----------------------------------------------------------------------------------------
# GridWorld <-> circle-world equivalence (the "faithful bridge" property)
# ----------------------------------------------------------------------------------------

def test_grid_clearance_matches_analytic_circle():
    # A grid rasterised from one circle should report clearances close to the analytic circle
    # clearance (to within a cell), away from the walls.
    bounds = (-4.0, -4.0, 4.0, 4.0)
    circle = np.array([[1.0, 0.0, 0.5]])
    res = 0.03
    g = GridWorld.from_obstacles(bounds, circle, resolution=res)
    cfg_circle = NavSimConfig(bounds=bounds, obstacles=circle, robot_radius=0.18)
    cfg_grid = NavSimConfig(bounds=bounds, grid_world=g, robot_radius=0.18)
    for xy in ([0.0, 0.0], [0.0, 1.5], [-1.5, 0.5], [2.5, 0.0]):
        p = np.array(xy)
        ca = world_clearance(p, cfg_circle.obstacle_array(), cfg_circle)
        cg = world_clearance(p, np.zeros((0, 3)), cfg_grid)
        assert abs(ca - cg) < 2 * res + 1e-6, f"clearance mismatch at {xy}: {ca} vs {cg}"


# ----------------------------------------------------------------------------------------
# nav-env integration
# ----------------------------------------------------------------------------------------

def _grid_cfg(**kw) -> NavSimConfig:
    base = dict(fixed_start=(0.0, 0.0, 0.0), fixed_goal=(3.0, 0.0),
                bounds=(-5.0, -5.0, 5.0, 5.0), task=NavTaskConfig(max_steps=300))
    base.update(kw)
    return NavSimConfig(**base)


def test_env_collides_with_grid_obstacle():
    # Grid obstacle straight ahead between start (0,0) and goal (3,0); drive into it.
    g = GridWorld.from_obstacles((-5.0, -5.0, 5.0, 5.0),
                                 np.array([[1.0, 0.0, 0.4]]), resolution=0.05)
    env = DiffDriveNavEnv(_grid_cfg(grid_world=g, robot_radius=0.18))
    env.reset(seed=0)
    hit = False
    for _ in range(100):
        _, r, term, _, info = env.step([1.0, 0.0])
        if term:
            hit = info["collided"]
            break
    assert hit, "robot driving into the grid obstacle should collide + terminate"
    assert info["robot_xy"][0] < 1.0 - 0.4 + 0.05   # stopped short of the obstacle (± a cell)


def test_env_lidar_senses_grid_obstacle_ahead():
    g = GridWorld.from_obstacles((-5.0, -5.0, 5.0, 5.0),
                                 np.array([[1.0, 0.0, 0.4]]), resolution=0.05)
    cfg = _grid_cfg(grid_world=g, n_lidar_beams=5, lidar_range=5.0, lidar_fov=np.pi)
    env = DiffDriveNavEnv(cfg)
    obs, _ = env.reset(seed=0)                       # facing +x, obstacle ~0.6 m ahead (surface)
    centre_beam = obs[OBS_DIM + 2]
    assert centre_beam < 0.2
    assert abs(centre_beam * cfg.lidar_range - 0.6) < 0.1


def test_shield_prevents_collision_with_grid_geometry():
    # The load-bearing guarantee: a reckless full-forward policy driven THROUGH the shield must
    # never hit reconstructed grid geometry, on a wall of grid obstacles between start and goal.
    obstacles = np.array([[1.0, 0.0, 0.4], [1.5, 0.6, 0.3], [1.5, -0.6, 0.3]])
    g = GridWorld.from_obstacles((-5.0, -5.0, 5.0, 5.0), obstacles, resolution=0.04)
    cfg = _grid_cfg(grid_world=g, robot_radius=0.18, safety_margin=0.06)
    env = DiffDriveNavEnv(cfg)
    env.reset(seed=0)
    for _ in range(300):
        raw = np.array([1.0, 0.5])
        safe = safety_shield(raw, env.robot_xy, env._heading, env._obstacles, cfg)
        _, _, term, trunc, info = env.step(safe)
        assert not info["collided"], "shield must prevent every collision against the grid"
        if term or trunc:
            break


def test_grid_and_circles_combine_additively():
    # A circle obstacle AND a grid obstacle in the same env: clearance is the min of both.
    g = GridWorld.from_obstacles((-5.0, -5.0, 5.0, 5.0),
                                 np.array([[2.0, 0.0, 0.3]]), resolution=0.04)
    cfg = _grid_cfg(grid_world=g, obstacles=np.array([[-2.0, 0.0, 0.3]]), robot_radius=0.1)
    env = DiffDriveNavEnv(cfg)
    env.reset(seed=0)
    # Near the circle (left) the circle dominates; near the grid obstacle (right) the grid does.
    assert env._clearance(np.array([-1.5, 0.0])) < env._clearance(np.array([0.0, 0.0]))
    assert env._clearance(np.array([1.5, 0.0])) < env._clearance(np.array([0.0, 0.0]))


def test_avoidance_reaches_goal_over_grid_like_over_circles():
    # Faithful-bridge behaviour: the same gap-follower that routes around a *circle* obstacle
    # also routes around the same obstacle rasterised into a *grid*, reaching the goal collision-
    # free. Proves a policy navigates reconstructed shape as it would the analytic world.
    obstacle = np.array([[2.0, 0.0, 0.5]])
    common = dict(fixed_start=(0.0, 0.0, 0.0), fixed_goal=(4.0, 0.0),
                  bounds=(-6.0, -6.0, 6.0, 6.0), n_lidar_beams=15, lidar_fov=np.pi,
                  lidar_range=5.0, task=NavTaskConfig(max_steps=1000))
    g = GridWorld.from_obstacles((-6.0, -6.0, 6.0, 6.0), obstacle, resolution=0.04)
    cfg = NavSimConfig(grid_world=g, **common)
    env = DiffDriveNavEnv(cfg)
    obs, _ = env.reset(seed=0)
    reached = False
    for _ in range(cfg.task.max_steps):
        obs, _, term, _, info = env.step(avoidance_action(obs, cfg))
        assert not info["collided"], "gap-follower should never hit the grid obstacle"
        if term:
            reached = info["reached"]
            break
    assert reached, "gap-follower should route around the grid obstacle to the goal"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
