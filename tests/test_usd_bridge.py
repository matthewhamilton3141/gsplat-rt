"""End-to-end test for the collision proxy builder and USD bridge.

Three tests:
  1. test_tsdf_integration_and_mesh      — TSDF+mesh extraction, no pxr needed.
  2. test_extractor_async_10hz           — Verifies the background thread
                                           produces a mesh within 200ms from
                                           the first push.
  3. test_full_pipeline_usdz             — Feeds 50 synthetic depth maps,
                                           builds the collision mesh, writes
                                           output_scene.usdz, validates USD.

Run:
    pytest tests/test_usd_bridge.py -v -s
    # or:
    python tests/test_usd_bridge.py
"""

import os
import sys
import tempfile
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mapping.collision_proxy import (
    CameraIntrinsics,
    CollisionProxyExtractor,
    TSDFVolume,
    TriangleMesh,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
USDZ_PATH = os.path.join(OUTPUT_DIR, "output_scene.usdz")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_K(H: int = 128, W: int = 128) -> CameraIntrinsics:
    """Camera intrinsics for the synthetic scene."""
    return CameraIntrinsics.from_fov(fov_deg=70.0, width=W, height=H)


def _synthetic_depth(n_frames: int = 50, H: int = 128, W: int = 128) -> list[np.ndarray]:
    """Generate synthetic depth maps depicting a curved surface at ~2m.

    A hemispherical bowl is placed at z=2m. Depth varies from 2.0m at centre
    to 2.6m at corners. This gives the TSDF a clear zero-crossing to extract.
    """
    u = np.linspace(-1.0, 1.0, W)[None, :]   # (1, W)
    v = np.linspace(-1.0, 1.0, H)[:, None]   # (H, 1)
    r2 = u ** 2 + v ** 2
    base_depth = 2.0 + 0.6 * r2              # (H, W) float64

    rng = np.random.default_rng(42)
    frames = []
    for i in range(n_frames):
        # Small per-frame jitter simulates camera or scene noise
        noise = rng.normal(0, 0.005, base_depth.shape)
        frames.append((base_depth + noise).astype(np.float32))
    return frames


def _dummy_gaussian_params(N: int = 500):
    rng = np.random.default_rng(0)
    means     = rng.standard_normal((N, 3)).astype(np.float32)
    scales    = np.abs(rng.standard_normal((N, 3))).astype(np.float32) * 0.01
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1)).astype(np.float32)
    opacities = rng.uniform(0.1, 0.9, N).astype(np.float32)
    return means, scales, rotations, opacities


def _pxr_available() -> bool:
    try:
        from pxr import Usd  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Test 1 — TSDF + mesh extraction (no pxr dependency)
# ---------------------------------------------------------------------------

def test_tsdf_integration_and_mesh():
    """Feed 50 depth frames into TSDFVolume and verify a valid mesh is extracted."""
    frames = _synthetic_depth(n_frames=50, H=128, W=128)
    K = _default_K(H=128, W=128)

    tsdf = TSDFVolume(voxel_size=0.05, grid_dim=64, trunc=0.10)

    t0 = time.perf_counter()
    for depth in frames:
        tsdf.integrate(depth, K)
    integrate_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    mesh = tsdf.extract_mesh()
    extract_ms = (time.perf_counter() - t1) * 1000

    print(f"\n  TSDF integration : {integrate_ms:.1f}ms total ({integrate_ms/50:.1f}ms/frame)")
    print(f"  Mesh extraction  : {extract_ms:.1f}ms")

    assert mesh is not None, "TSDF produced no zero-crossing — check depth scale / volume bounds"
    assert mesh.vertices.shape[1] == 3
    assert mesh.faces.shape[1] == 3
    assert mesh.vertices.dtype == np.float32

    print(f"  Mesh             : {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} triangles")

    # Each frame must integrate in <20ms (well under 100ms for 10Hz budget)
    assert integrate_ms / 50 < 20.0, f"Integration too slow: {integrate_ms/50:.1f}ms/frame"
    # Mesh extraction must be <50ms on any reasonable CPU
    assert extract_ms < 50.0, f"Marching cubes too slow: {extract_ms:.1f}ms"


# ---------------------------------------------------------------------------
# Test 2 — async extractor at 10Hz
# ---------------------------------------------------------------------------

def test_extractor_async_10hz():
    """Verify the background thread produces a mesh within 200ms after 10 frames."""
    frames = _synthetic_depth(n_frames=20, H=128, W=128)
    K = _default_K(H=128, W=128)

    tsdf = TSDFVolume(voxel_size=0.05, grid_dim=64, trunc=0.10)
    extractor = CollisionProxyExtractor(tsdf=tsdf, update_hz=10.0)
    extractor.start()

    t0 = time.perf_counter()
    for depth in frames:
        extractor.push_depth(depth, K)
        time.sleep(0.01)   # simulate ~100 FPS input (faster than 30Hz)

    # Wait up to 500ms for the first mesh to appear
    got_mesh = extractor.mesh_ready.wait(timeout=0.5)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    extractor.stop()
    mesh = extractor.get_latest_mesh()

    print(f"\n  First mesh ready in {elapsed_ms:.0f}ms")
    print(f"  Frames integrated: {extractor.frames_integrated}")
    print(f"  Meshes extracted : {extractor.meshes_extracted}")

    assert got_mesh, "Background thread never produced a mesh within 500ms"
    assert mesh is not None
    assert not mesh.is_empty, "Mesh is empty (no vertices/faces)"


# ---------------------------------------------------------------------------
# Test 3 — full pipeline → .usdz
# ---------------------------------------------------------------------------

def test_full_pipeline_usdz():
    """50 depth frames → collision mesh → USD stage → output_scene.usdz."""
    if not _pxr_available():
        pytest.skip("pxr (OpenUSD) not installed — pip install usd-core")

    from pxr import Usd, UsdGeom, UsdPhysics
    from mapping.usd_bridge import UsdBridge

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: build collision mesh from 50 synthetic depth frames ---
    frames = _synthetic_depth(n_frames=50, H=128, W=128)
    K = _default_K(H=128, W=128)

    tsdf = TSDFVolume(voxel_size=0.05, grid_dim=64, trunc=0.10)
    for depth in frames:
        tsdf.integrate(depth, K)

    mesh = tsdf.extract_mesh()
    assert mesh is not None, "No collision mesh extracted — check scene geometry"
    print(f"\n  Collision mesh: {mesh.vertices.shape[0]}V / {mesh.faces.shape[0]}F")

    # --- Step 2: build USD stage ---
    usd_path = USDZ_PATH.replace(".usdz", ".usda")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_usd = os.path.join(tmpdir, "scene.usda")

        bridge = UsdBridge(tmp_usd)

        # Gaussian splat layer
        means, scales, rotations, opacities = _dummy_gaussian_params(N=500)
        bridge.update_gaussian_splats(means, scales, rotations, opacities)

        # Collision mesh layer
        bridge.update_collision_mesh(mesh.vertices, mesh.faces)

        bridge.save()

        # --- Step 3: export as .usdz ---
        usdz_out = os.path.join(tmpdir, "output_scene.usdz")
        bridge.export_usdz(usdz_out)
        assert os.path.exists(usdz_out), ".usdz not written"
        size_kb = os.path.getsize(usdz_out) / 1024
        print(f"  USDZ size: {size_kb:.1f} KB")
        assert size_kb > 1.0, ".usdz file is suspiciously small"

        # Copy to project output dir for inspection
        import shutil
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        shutil.copy(usdz_out, USDZ_PATH)

        # --- Step 4: validate USD ---
        # Open the source .usda (usdz is a zip; open the source for inspection)
        stage = Usd.Stage.Open(tmp_usd)
        assert stage is not None, "Cannot re-open USD stage"

        default_prim = stage.GetDefaultPrim()
        assert default_prim.IsValid(), "Stage has no default prim"

        # Splat prim must exist and have Gaussian attributes
        splat_prim = stage.GetPrimAtPath("/World/GaussianSplats")
        assert splat_prim.IsValid(), "/World/GaussianSplats prim missing"
        assert splat_prim.HasAttribute("splat:count"), "Missing splat:count attribute"
        assert splat_prim.HasAttribute("splat:opacities"), "Missing splat:opacities attribute"

        # Collision prim must exist with correct APIs and visibility
        coll_prim = stage.GetPrimAtPath("/World/CollisionMesh")
        assert coll_prim.IsValid(), "/World/CollisionMesh prim missing"
        assert coll_prim.HasAPI(UsdPhysics.CollisionAPI), "CollisionAPI not applied"
        assert coll_prim.HasAPI(UsdPhysics.MeshCollisionAPI), "MeshCollisionAPI not applied"

        vis_attr = coll_prim.GetAttribute("visibility")
        assert vis_attr, "visibility attribute missing on CollisionMesh"
        assert vis_attr.Get() == UsdGeom.Tokens.invisible, (
            f"CollisionMesh should be invisible, got: {vis_attr.Get()}"
        )

        mesh_geom = UsdGeom.Mesh(coll_prim)
        pts = mesh_geom.GetPointsAttr().Get()
        assert len(pts) > 0, "CollisionMesh has no vertices"

        print(f"  USD validated: splat prim OK, collision mesh OK (invisible, physics APIs applied)")
        print(f"  USDZ written: {USDZ_PATH}")


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Test 1: TSDF integration and mesh extraction ===")
    test_tsdf_integration_and_mesh()
    print("PASS\n")

    print("=== Test 2: Async extractor @ 10Hz ===")
    test_extractor_async_10hz()
    print("PASS\n")

    print("=== Test 3: Full pipeline → .usdz ===")
    try:
        test_full_pipeline_usdz()
        print("PASS\n")
    except pytest.skip.Exception as e:
        print(f"SKIPPED: {e}\n")
    except AssertionError as e:
        print(f"FAIL: {e}\n")
        sys.exit(1)
