"""Tests for the Isaac-readiness USD validator (src/mapping/usd_isaac_check.py).

The important one is `test_real_bridge_export_is_isaac_ready`: it runs the *actual*
`UsdBridge` export and asserts the validator finds zero ERRORs — so the reconstruct→Isaac
bridge is checked on the laptop before any GPU/Isaac session. The rest are negative cases
built with raw pxr, one per invariant, so a regression in the validator is caught.

Skips cleanly where OpenUSD (pxr) isn't installed.

Run:
    pytest tests/test_usd_isaac_check.py -v
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("pxr")  # whole module needs OpenUSD

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mapping.usd_isaac_check import validate_isaac_stage, summarize  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _unit_cube():
    """8 verts + 12 triangles of a 1 m axis-aligned cube (a valid closed volume)."""
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    f = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # sides
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ], dtype=np.int32)
    return v, f


def _errors(issues):
    return [i for i in issues if i.severity == "ERROR"]


def _codes(issues):
    return {i.code for i in issues}


def _raw_mesh_stage(path, *, verts, faces, meters_per_unit=1.0, up="Z",
                    collision_api=True, mesh_collision_api=True, approximation="convexHull"):
    """Build a minimal stage with one (optionally physics-enabled) mesh via raw pxr,
    for precise control in negative tests."""
    from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt, Sdf

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up == "Z" else UsdGeom.Tokens.y)
    if meters_per_unit is not None:
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/World/CollisionMesh")
    mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in verts]))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([int(i) for i in faces.ravel()]))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    prim = mesh.GetPrim()
    if collision_api:
        UsdPhysics.CollisionAPI.Apply(prim)
    if mesh_collision_api:
        UsdPhysics.MeshCollisionAPI.Apply(prim)
        if approximation is not None:
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set(approximation)
    stage.GetRootLayer().Save()
    return path


# ---------------------------------------------------------------------------
# The headline test: the real export passes
# ---------------------------------------------------------------------------

def test_real_bridge_export_is_isaac_ready(tmp_path):
    from mapping.usd_bridge import UsdBridge

    v, f = _unit_cube()
    usd = str(tmp_path / "scene.usd")
    bridge = UsdBridge(usd)
    bridge.update_collision_mesh(v, f, approximation="convexDecomposition")
    # a tiny splat layer so the render-layer WARN doesn't fire
    n = 10
    bridge.update_gaussian_splats(
        means=np.random.rand(n, 3).astype(np.float32),
        scales=np.zeros((n, 3), np.float32),
        rotations=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=np.ones(n, np.float32),
    )
    bridge.save()

    issues = validate_isaac_stage(usd)
    assert not _errors(issues), f"real export not Isaac-ready: {summarize(issues)}\n" + \
        "\n".join(str(i) for i in issues if i.severity == "ERROR")
    # bridge deliberately writes Y-up; validator should surface that as a WARN, not error
    assert "up_axis_mismatch" in _codes(issues)


def test_usdz_package_also_validates(tmp_path):
    """The packaged .usdz (what actually ships to the box) validates too."""
    from mapping.usd_bridge import UsdBridge

    v, f = _unit_cube()
    usd = str(tmp_path / "scene.usd")
    bridge = UsdBridge(usd)
    bridge.update_collision_mesh(v, f)
    bridge.save()
    usdz = bridge.export_usdz(str(tmp_path / "scene.usdz"))

    issues = validate_isaac_stage(usdz)
    assert not _errors(issues), f"usdz not Isaac-ready: {summarize(issues)}"


# ---------------------------------------------------------------------------
# Negative cases — one invariant each
# ---------------------------------------------------------------------------

def test_missing_collision_api_errors(tmp_path):
    v, f = _unit_cube()
    p = _raw_mesh_stage(str(tmp_path / "s.usd"), verts=v, faces=f, collision_api=False)
    issues = validate_isaac_stage(p)
    assert "no_collision_mesh" in _codes(issues)
    assert _errors(issues)


def test_missing_meters_per_unit_errors(tmp_path):
    v, f = _unit_cube()
    p = _raw_mesh_stage(str(tmp_path / "s.usd"), verts=v, faces=f, meters_per_unit=None)
    issues = validate_isaac_stage(p)
    assert "no_meters_per_unit" in _codes(issues)


def test_extent_too_large_warns_metric_scale(tmp_path):
    v, f = _unit_cube()
    v = v * 500.0            # a 500 m "room" -> relative-depth-not-aligned smell
    p = _raw_mesh_stage(str(tmp_path / "s.usd"), verts=v, faces=f)
    issues = validate_isaac_stage(p)
    assert "extent_too_large" in _codes(issues)
    assert not _errors(issues)  # it's a WARN, not a hard error


def test_face_index_out_of_range_errors(tmp_path):
    v, f = _unit_cube()
    f = f.copy(); f[0, 0] = 999      # dangling index
    p = _raw_mesh_stage(str(tmp_path / "s.usd"), verts=v, faces=f)
    issues = validate_isaac_stage(p)
    assert "face_index_oob" in _codes(issues)


def test_non_finite_vertex_errors(tmp_path):
    v, f = _unit_cube()
    v = v.copy(); v[0, 0] = np.nan
    p = _raw_mesh_stage(str(tmp_path / "s.usd"), verts=v, faces=f)
    issues = validate_isaac_stage(p)
    assert "non_finite_vertices" in _codes(issues)


def test_degenerate_mesh_errors(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)  # 3 verts
    f = np.array([[0, 1, 2]], dtype=np.int32)
    p = _raw_mesh_stage(str(tmp_path / "s.usd"), verts=v, faces=f)
    issues = validate_isaac_stage(p)
    assert "degenerate_mesh" in _codes(issues)


def test_z_up_metric_scene_is_clean(tmp_path):
    """A Z-up, metric, well-formed scene should be entirely clean (no ERROR/WARN)."""
    v, f = _unit_cube()
    p = _raw_mesh_stage(str(tmp_path / "s.usd"), verts=v, faces=f, up="Z",
                        approximation="convexDecomposition")
    issues = validate_isaac_stage(p)
    assert not _errors(issues)
    warns = [i for i in issues if i.severity == "WARN"]
    # only the missing-splat WARN is acceptable here (raw stage has no splat layer)
    assert {i.code for i in warns} <= {"no_splat_prim"}, \
        f"unexpected warnings: {[str(i) for i in warns]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
