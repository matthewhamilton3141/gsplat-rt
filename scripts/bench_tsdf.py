"""Benchmark the TSDF integrate stage: numpy baseline vs custom CUDA kernel.

The vectorised-numpy `TSDFVolume.integrate` is the one pipeline stage still over
its per-call budget (~13 ms for a 64^3 grid). This script times it and, when the
`gaussian_kernels` extension is built on a CUDA box, times the kernel against the
same frames and reports the speed-up.

Usage:
    python scripts/bench_tsdf.py --grid 64 --frames 50
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mapping.collision_proxy import CameraIntrinsics, TSDFVolume
from mapping import tsdf_cuda


def _synthetic_depth(K, rng):
    yy, xx = np.mgrid[0:K.height, 0:K.width].astype(np.float32)
    depth = 1.5 + 0.3 * (xx / K.width) - 0.2 * (yy / K.height)
    depth += rng.normal(0, 0.01, depth.shape).astype(np.float32)
    return depth.astype(np.float32)


def _pose(i):
    a = np.radians(0.5 * i)
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    T[:3, 3] = np.array([0.01 * i, 0.0, 0.0], dtype=np.float32)
    return T


def bench_numpy(grid, depths, poses, K):
    vol = TSDFVolume(voxel_size=0.05, grid_dim=grid)
    times = []
    for depth, pose in zip(depths, poses):
        t0 = time.perf_counter()
        vol.integrate(depth, K, pose)
        times.append((time.perf_counter() - t0) * 1e3)
    return np.array(times)


def bench_cuda(grid, depths, poses, K):
    import torch
    dev = torch.device("cuda")
    N = grid
    vol = TSDFVolume(voxel_size=0.05, grid_dim=grid)
    tsdf_t = torch.from_numpy(vol._tsdf.ravel().copy()).to(dev)
    weight_t = torch.from_numpy(vol._weight.ravel().copy()).to(dev)
    times = []
    for depth, pose in zip(depths, poses):
        depth_t = torch.from_numpy(depth).to(dev)
        R_t = torch.from_numpy(np.ascontiguousarray(pose[:3, :3])).to(dev)
        t_t = torch.from_numpy(np.ascontiguousarray(pose[:3, 3])).to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tsdf_cuda.integrate_cuda(tsdf_t, weight_t, depth_t, R_t, t_t,
                                 N, vol.voxel_size, vol.origin, K, vol.trunc)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return np.array(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--res", type=int, default=518)
    args = ap.parse_args()

    K = CameraIntrinsics.from_fov(70.0, args.res, args.res)
    rng = np.random.default_rng(0)
    depths = [_synthetic_depth(K, rng) for _ in range(args.frames)]
    poses = [_pose(i) for i in range(args.frames)]

    print(f"TSDF integrate — grid={args.grid}^3, {args.frames} frames, {args.res}px depth\n")

    np_t = bench_numpy(args.grid, depths, poses, K)
    # Drop the first (warm-up / allocation) sample from the summary.
    warm = np_t[1:] if len(np_t) > 1 else np_t
    print(f"numpy   : {warm.mean():6.2f} ms/frame  (median {np.median(warm):.2f}, "
          f"min {warm.min():.2f})")

    if tsdf_cuda.available():
        cu_t = bench_cuda(args.grid, depths, poses, K)
        cw = cu_t[1:] if len(cu_t) > 1 else cu_t
        print(f"cuda    : {cw.mean():6.2f} ms/frame  (median {np.median(cw):.2f}, "
              f"min {cw.min():.2f})")
        print(f"speed-up: {warm.mean() / cw.mean():.1f}x")
        budget = "PASS" if cw.mean() < 33.3 else "OVER"
        print(f"\n30 FPS budget (33.3 ms): cuda {budget}")
    else:
        print("cuda    : extension unavailable — build with "
              "`python setup.py build_ext --inplace` on a CUDA box")


if __name__ == "__main__":
    main()
