"""Tests for the reconstructed-mesh -> occupancy loader (src/isaac/nurec_scene.py).

CPU/NumPy + OpenCV only for the core; `trimesh` and OpenUSD (`pxr`) are exercised where available
(skipped otherwise) to de-risk loading a real NuRec `.usdz`. Verifies the top-down projection, the
obstacle height-band filtering (road below / gantry above are NOT obstacles), gap-closing dilation,
up-axis handling, start/goal selection, and the full chain: a synthetic *reconstructed* scene ->
occupancy grid -> the shielded DWA car drives it to the goal.

Run:
    pytest tests/test_nurec_scene.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from isaac.car_sim import BicycleNavEnv, car_dwa_action, car_safety_shield  # noqa: E402
from isaac.driving_scene import driving_scene_config  # noqa: E402
from isaac.nurec_scene import (  # noqa: E402
    load_mesh, mesh_to_occupancy, nurec_to_gridworld, pick_start_goal,
)


# -- synthetic mesh builders (z-up unless noted) -----------------------------------------
def _quad(p0, p1, p2, p3):
    """Two triangles (0,1,2)+(0,2,3) for a quad; returns (verts(4,3), faces(2,3))."""
    return np.array([p0, p1, p2, p3], float), np.array([[0, 1, 2], [0, 2, 3]], int)


def _slab(x0, x1, y0, y1, z, up_axis=2):
    """A horizontal quad at height `z` spanning [x0,x1]x[y0,y1] — fills its footprint."""
    if up_axis == 2:
        pts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    else:                                            # up_axis == 1 (y-up): ground is (x, z)
        pts = [(x0, z, y0), (x1, z, y0), (x1, z, y1), (x0, z, y1)]
    return _quad(*pts)


def _wall(x0, x1, y, z0, z1):
    """A vertical quad along y=const spanning x in [x0,x1], height z in [z0,z1]."""
    return _quad((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1))


def _combine(*meshes):
    verts, faces, base = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(np.asarray(f) + base)
        base += len(v)
    return np.concatenate(verts), np.concatenate(faces)


# -- projection + band filtering ---------------------------------------------------------
def test_ground_only_is_free():
    # A flat road at z=0 with the obstacle band starting at 0.3 -> nothing occupied.
    v, f = _slab(-5, 25, -4, 4, z=0.0)
    grid = mesh_to_occupancy(v, f, resolution=0.2, band=(0.3, 2.5))
    assert grid.occupancy.sum() == 0


def test_overhead_gantry_not_occupied():
    # A slab high above the band (a bridge / sign gantry) is not an obstacle for the car.
    v, f = _combine(_slab(-5, 25, -4, 4, 0.0), _slab(8, 12, -4, 4, 5.0))
    grid = mesh_to_occupancy(v, f, resolution=0.2, band=(0.3, 2.5))
    assert grid.occupancy.sum() == 0


def test_in_band_slab_fills_footprint():
    # A solid obstacle (slab at z=1, inside the band) marks its whole XY footprint occupied.
    v, f = _combine(_slab(-5, 25, -4, 4, 0.0), _slab(4, 6, -1, 1, 1.0))
    grid = mesh_to_occupancy(v, f, resolution=0.2, band=(0.3, 2.5))
    assert grid.occupied_at(np.array([5.0, 0.0]))        # inside the obstacle footprint
    assert not grid.occupied_at(np.array([0.0, 0.0]))    # open road
    assert not grid.occupied_at(np.array([5.0, 3.0]))    # beside it


def test_vertical_wall_is_in_band_and_marked():
    # A wall spanning z[0,3] overlaps the band -> its footprint line is occupied.
    v, f = _combine(_slab(-5, 25, -4, 4, 0.0), _wall(4, 16, 2.0, 0.0, 3.0))
    grid = mesh_to_occupancy(v, f, resolution=0.2, band=(0.3, 2.5))
    assert grid.occupied_at(np.array([10.0, 2.0]))       # on the wall
    assert not grid.occupied_at(np.array([10.0, 0.0]))   # road below the wall line


def test_dilate_closes_gaps():
    # Dilation grows a thin wall's footprint (closing one-cell tunnels in real surface meshes).
    v, f = _wall(0, 20, 0.0, 0.5, 3.0)
    base = mesh_to_occupancy(v, f, resolution=0.2, band=(0.3, 2.5), dilate_cells=0)
    grown = mesh_to_occupancy(v, f, resolution=0.2, band=(0.3, 2.5), dilate_cells=1)
    assert grown.occupancy.sum() > base.occupancy.sum()


def test_up_axis_y():
    # A y-up scene: the ground plane is (x, z); an in-band slab still fills its footprint.
    v, f = _combine(_slab(-5, 25, -4, 4, 0.0, up_axis=1), _slab(4, 6, -1, 1, 1.0, up_axis=1))
    grid = mesh_to_occupancy(v, f, resolution=0.2, up_axis=1, band=(0.3, 2.5))
    assert grid.occupied_at(np.array([5.0, 0.0]))
    assert not grid.occupied_at(np.array([0.0, 0.0]))


# -- start / goal ------------------------------------------------------------------------
def test_pick_start_goal_free_and_ordered():
    # Corridor between two curbs; start/goal land in free space with the start at the -x end.
    v, f = _combine(_slab(0, 20, -4, 4, 0.0),
                    _slab(0, 20, 2.5, 4, 1.0), _slab(0, 20, -4, -2.5, 1.0))
    grid = mesh_to_occupancy(v, f, resolution=0.2, band=(0.3, 2.5))
    (sx, sy, sh), (gx, gy) = pick_start_goal(grid, robot_radius=0.25, margin=0.3)
    assert sx < gx
    assert grid.clearance(np.array([sx, sy]), 0.25) >= 0.0
    assert grid.clearance(np.array([gx, gy]), 0.25) >= 0.0


# -- full chain: reconstructed mesh -> occupancy -> car drives it -------------------------
def _corridor_scene():
    """Ground + two curbs (free corridor y in [-2.5, 2.5]) + one offset obstacle block."""
    return _combine(
        _slab(0, 20, -4, 4, 0.0),                    # road
        _slab(0, 20, 2.5, 4, 1.0),                   # upper curb/building
        _slab(0, 20, -4, -2.5, 1.0),                 # lower curb/building
        _slab(9, 11, 0.2, 2.5, 1.0),                 # obstacle intruding from the upper side
    )


def test_dwa_car_drives_reconstructed_scene():
    v, f = _corridor_scene()
    grid = mesh_to_occupancy(v, f, resolution=0.2, band=(0.3, 2.5), dilate_cells=1)
    cfg = driving_scene_config(grid, (1.0, 0.0, 0.0), (19.0, 0.0))
    cfg.task.max_steps = 900
    env = BicycleNavEnv(cfg)
    env.reset(seed=0)
    reached = False
    for _ in range(cfg.task.max_steps):
        raw = car_dwa_action(env.robot_xy, env._heading, env._goal, env._obstacles, cfg)
        safe = car_safety_shield(raw, env.robot_xy, env._heading, env._obstacles, cfg)
        _, _, term, trunc, info = env.step(safe)
        assert not info["collided"], "car must not hit the reconstructed geometry"
        if term or trunc:
            reached = info["reached"]
            break
    assert reached, "shielded DWA car should drive the reconstructed corridor to the goal"


# -- loaders (env-dependent; skip cleanly if the lib is absent) ---------------------------
def test_load_mesh_obj_roundtrip(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    v, f = _combine(_slab(-5, 25, -4, 4, 0.0), _slab(4, 6, -1, 1, 1.0))
    p = os.path.join(tmp_path, "scene.obj")
    trimesh.Trimesh(vertices=v, faces=f).export(p)
    vv, ff = load_mesh(p)
    grid = mesh_to_occupancy(vv, ff, resolution=0.2, band=(0.3, 2.5))
    assert grid.occupied_at(np.array([5.0, 0.0]))


def test_load_usd_mesh_roundtrip(tmp_path):
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, Vt, Gf
    p = os.path.join(tmp_path, "scene.usda")
    stage = Usd.Stage.CreateNew(p)
    # ground + one in-band slab obstacle, authored as two USD meshes.
    for name, (x0, x1, y0, y1, z) in {"road": (-5, 25, -4, 4, 0.0),
                                      "obs": (4, 6, -1, 1, 1.0)}.items():
        m = UsdGeom.Mesh.Define(stage, f"/World/{name}")
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x1, y0, z),
                                          Gf.Vec3f(x1, y1, z), Gf.Vec3f(x0, y1, z)]))
        m.CreateFaceVertexCountsAttr([4])
        m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    stage.GetRootLayer().Save()

    grid, start, goal = nurec_to_gridworld(p, resolution=0.2, band=(0.3, 2.5))
    assert grid.occupied_at(np.array([5.0, 0.0]))        # the USD obstacle came through
    assert not grid.occupied_at(np.array([0.0, 0.0]))
    assert start[0] < goal[0]


if __name__ == "__main__":
    import pytest as _p
    sys.exit(_p.main([__file__, "-v"]))
