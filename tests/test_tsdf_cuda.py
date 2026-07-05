"""Correctness for the custom CUDA TSDF integrate kernel.

Because CI (and the dev Mac) has no CUDA device, the kernel's *arithmetic* is
verified on CPU through `integrate_reference` — a line-for-line transcription of
`tsdf_integrate_kernel`. If that matches the production vectorised numpy
`TSDFVolume.integrate` bit-for-bit (up to float rounding), the CUDA kernel,
which runs the identical math, is validated too.

  test_reference_matches_vectorized_single / _multi
      Oracle == production numpy path, one and several frames (running average).
  test_cuda_matches_reference
      GPU-gated: the compiled kernel matches the reference. Skips without CUDA.

Run:
    pytest tests/test_tsdf_cuda.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mapping.collision_proxy import CameraIntrinsics, TSDFVolume
from mapping import tsdf_cuda


def _synthetic_depth(K: CameraIntrinsics, rng) -> np.ndarray:
    """A tilted plane ~1.5 m out plus noise — gives the volume real surfaces."""
    yy, xx = np.mgrid[0:K.height, 0:K.width].astype(np.float32)
    depth = 1.5 + 0.3 * (xx / K.width) - 0.2 * (yy / K.height)
    depth += rng.normal(0, 0.01, depth.shape).astype(np.float32)
    return depth.astype(np.float32)


def _pose(rot_deg: float, tvec) -> np.ndarray:
    """Camera-to-world pose with a yaw so the R columns are all exercised."""
    a = np.radians(rot_deg)
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    T[:3, 3] = np.asarray(tvec, dtype=np.float32)
    return T


def _run_reference(vol: TSDFVolume, depth, K, pose):
    """Integrate one frame via the kernel-transcription oracle (fresh buffers)."""
    tsdf = vol._tsdf.ravel().copy()
    weight = vol._weight.ravel().copy()
    tsdf_cuda.integrate_reference(
        tsdf, weight, depth,
        pose[:3, :3], pose[:3, 3],
        vol.grid_dim, vol.voxel_size, vol.origin, K, vol.trunc)
    return tsdf, weight


def test_reference_matches_vectorized_single():
    K = CameraIntrinsics.from_fov(70.0, 96, 96)
    rng = np.random.default_rng(0)
    depth = _synthetic_depth(K, rng)
    pose = _pose(15.0, [0.1, -0.05, 0.0])

    ref = TSDFVolume(voxel_size=0.05, grid_dim=24)
    ref_tsdf, ref_weight = _run_reference(ref, depth, K, pose)

    prod = TSDFVolume(voxel_size=0.05, grid_dim=24)
    prod.integrate(depth, K, pose)

    # Some voxel must actually have been observed, else the test is vacuous.
    assert ref_weight.sum() > 0
    np.testing.assert_allclose(ref_tsdf, prod._tsdf.ravel(), rtol=0, atol=1e-5)
    np.testing.assert_array_equal(ref_weight, prod._weight.ravel())


def test_reference_matches_vectorized_multi():
    K = CameraIntrinsics.from_fov(70.0, 96, 96)
    rng = np.random.default_rng(1)
    poses = [_pose(0.0, [0, 0, 0]), _pose(10.0, [0.2, 0, 0.1]),
             _pose(-8.0, [-0.1, 0.05, 0.0])]

    ref = TSDFVolume(voxel_size=0.05, grid_dim=24)
    prod = TSDFVolume(voxel_size=0.05, grid_dim=24)
    ref_tsdf = ref._tsdf.ravel().copy()
    ref_weight = ref._weight.ravel().copy()

    for pose in poses:
        depth = _synthetic_depth(K, rng)
        tsdf_cuda.integrate_reference(
            ref_tsdf, ref_weight, depth, pose[:3, :3], pose[:3, 3],
            ref.grid_dim, ref.voxel_size, ref.origin, K, ref.trunc)
        prod.integrate(depth, K, pose)

    assert ref_weight.max() >= 2                       # accumulated over frames
    np.testing.assert_allclose(ref_tsdf, prod._tsdf.ravel(), rtol=0, atol=1e-5)
    np.testing.assert_array_equal(ref_weight, prod._weight.ravel())


@pytest.mark.skipif(not tsdf_cuda.available(), reason="CUDA extension unavailable")
def test_cuda_matches_reference():
    import torch

    K = CameraIntrinsics.from_fov(70.0, 96, 96)
    rng = np.random.default_rng(2)
    depth = _synthetic_depth(K, rng)
    pose = _pose(12.0, [0.15, -0.05, 0.05])

    vol = TSDFVolume(voxel_size=0.05, grid_dim=64)
    ref_tsdf, ref_weight = _run_reference(vol, depth, K, pose)

    dev = torch.device("cuda")
    tsdf_t = torch.from_numpy(vol._tsdf.ravel().copy()).to(dev)
    weight_t = torch.from_numpy(vol._weight.ravel().copy()).to(dev)
    depth_t = torch.from_numpy(depth).to(dev)
    R_t = torch.from_numpy(np.ascontiguousarray(pose[:3, :3])).to(dev)
    t_t = torch.from_numpy(np.ascontiguousarray(pose[:3, 3])).to(dev)

    tsdf_cuda.integrate_cuda(
        tsdf_t, weight_t, depth_t, R_t, t_t,
        vol.grid_dim, vol.voxel_size, vol.origin, K, vol.trunc)

    np.testing.assert_allclose(tsdf_t.cpu().numpy(), ref_tsdf, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(weight_t.cpu().numpy(), ref_weight)
