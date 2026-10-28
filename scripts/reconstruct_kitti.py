#!/usr/bin/env python3
"""Fuse a recorded KITTI drive into a coloured world-space cloud — the AV-scale
counterpart of the TUM desk preview (``reconstruct_tum.py --ply``).

Camera-only and GPU-free: each frame's metric depth comes from SGBM on the
rectified colour stereo pair (no lidar), sky + depth-edge pixels are dropped,
the rest is back-projected and placed by the OXTS ground-truth pose, and the
fused cloud is voxel-downsampled. Writes:

    <out>.ply         x y z + rgb (renders with view_scene.py / render_turntable.py)
    <out>.traj.npy    (N,4,4) camera-to-world poses, for ``render_drive.py``

    python scripts/reconstruct_kitti.py --drive 0009 --out output/kitti_0009.ply

Expects KITTI raw at ``data/kitti_raw/<date>/<date>_drive_<id>_sync`` (or
``--base-dir``); needs ``pykitti`` (pip install pykitti).
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from slam.kitti_raw import KittiRawDrive, edge_mask, sky_mask, stereo_depth   # noqa: E402
from reconstruct_tum import write_ply_xyzrgb                                   # noqa: E402


def backproject_masked(depth, rgb, K, pose, pixel_stride):
    """Valid depth pixels → (world pts float32 (M,3), RGB uint8 (M,3))."""
    vs, us = np.mgrid[0:depth.shape[0]:pixel_stride, 0:depth.shape[1]:pixel_stride]
    z = depth[vs, us].ravel()
    us, vs = us.ravel(), vs.ravel()
    m = z > 0
    z, us, vs = z[m], us[m], vs[m]
    pts = np.stack([(us - K.cx) * z / K.fx, (vs - K.cy) * z / K.fy, z], axis=-1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):   # macOS BLAS FPE
        pts = pts @ pose[:3, :3].T + pose[:3, 3]
    return pts.astype(np.float32), rgb[vs, us]


def voxel_downsample(pts, cols, voxel):
    """Keep one point per ``voxel``-metre cell (the first seen)."""
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()
    return pts[idx], cols[idx]


def fuse_drive(drive, frame_stride=3, pixel_stride=2, max_depth=25.0, voxel=0.05,
               log=print):
    K, poses = drive.intrinsics, drive.gt_poses
    P, C = [], []
    t0 = time.perf_counter()
    for i in range(0, len(drive), frame_stride):
        left, right = drive.rgb_pair(i)
        gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        depth = stereo_depth(gl, gr, K.fx, drive.baseline, max_depth=max_depth)
        depth[sky_mask(left, horizon_row=K.cy) | edge_mask(depth)] = 0.0
        p, c = backproject_masked(depth, left, K, poses[i], pixel_stride)
        P.append(p)
        C.append(c)
    dt = time.perf_counter() - t0
    pts, cols = np.concatenate(P), np.concatenate(C)
    n_raw = len(pts)
    pts, cols = voxel_downsample(pts, cols, voxel)
    log(f"fused {len(P)} frames in {dt:.1f}s ({1e3 * dt / len(P):.0f} ms/frame CPU): "
        f"{n_raw:,} → {len(pts):,} pts after {voxel * 100:.0f} cm voxel")
    return pts, cols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default="2011_09_26")
    ap.add_argument("--drive", default="0009")
    ap.add_argument("--base-dir", default=None, help="KITTI raw root (default data/kitti_raw)")
    ap.add_argument("--out", default="output/kitti_0009.ply")
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--pixel-stride", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=25.0, help="stereo range cap (m)")
    ap.add_argument("--voxel", type=float, default=0.05, help="downsample cell (m)")
    args = ap.parse_args()

    drive = KittiRawDrive(args.date, args.drive, args.base_dir)
    print(f"KITTI {args.date} drive {args.drive}: {len(drive)} frames, "
          f"{drive.path_length:.0f} m, baseline {drive.baseline:.3f} m")
    pts, cols = fuse_drive(drive, args.frame_stride, args.pixel_stride,
                           args.max_depth, args.voxel)
    print(f"world extent (m): {np.round(np.ptp(pts, axis=0), 1)}")

    write_ply_xyzrgb(args.out, pts, cols)
    traj = os.path.splitext(args.out)[0] + ".traj.npy"
    np.save(traj, drive.gt_poses.astype(np.float32))
    print(f"wrote {args.out}  +  {traj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
