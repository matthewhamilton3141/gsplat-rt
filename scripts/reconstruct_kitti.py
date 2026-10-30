#!/usr/bin/env python3
"""Fuse a recorded KITTI drive into a coloured world-space cloud — the AV-scale
counterpart of the TUM desk preview (``reconstruct_tum.py --ply``).

Each frame's metric depth comes from one of two sources:

    --depth stereo  (default, CPU)  SGBM on the rectified colour stereo pair.
    --depth trt     (GPU box)       the pipeline's own TensorRT Depth Anything V2
                                    engine on the *left image only*, made metric by
                                    the pipeline's DepthScaleAligner against sparse
                                    ORB points triangulated between frame i and
                                    i+gap with the ground-truth relative pose.

Either way, sky + depth-edge pixels are dropped, the rest is back-projected and
placed by the OXTS ground-truth pose, and the fused cloud is voxel-downsampled.
Writes:

    <out>.ply         x y z + rgb (renders with view_scene.py / render_turntable.py)
    <out>.traj.npy    (N,4,4) camera-to-world poses, for ``render_drive.py``

    python scripts/reconstruct_kitti.py --drive 0009 --out output/kitti_0009.ply
    python scripts/reconstruct_kitti.py --depth trt --engine models/depth_engine_fp16.engine \
        --out output/kitti_0009_trt.ply

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

from slam.kitti_raw import (KittiRawDrive, edge_mask, invert_se3, sky_mask,   # noqa: E402
                            stereo_depth)
from depth.metric_scale import DepthScaleAligner, triangulate_two_view       # noqa: E402
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


def k_matrix(K):
    return np.array([[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]])


def triangulated_anchor(gray_a, gray_b, K, T_a, T_b, max_depth=40.0, max_reproj=2.0,
                        n_features=2000):
    """Sparse metric depth at frame-A pixels: ORB matches A↔B, triangulated with
    the known (metric) relative pose. Returns ``(uv (M,2), z (M,))``; M may be 0.

    Matches are cross-checked, then kept only if in front of both cameras, closer
    than ``max_depth`` and reprojecting into B within ``max_reproj`` px — forward
    motion triangulates poorly near the epipole, and this drops those points.
    """
    orb = cv2.ORB_create(n_features)
    ka, da = orb.detectAndCompute(gray_a, None)
    kb, db = orb.detectAndCompute(gray_b, None)
    if da is None or db is None:
        return np.zeros((0, 2)), np.zeros(0)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if not matches:
        return np.zeros((0, 2)), np.zeros(0)
    uv_a = np.float64([ka[m.queryIdx].pt for m in matches])
    uv_b = np.float64([kb[m.trainIdx].pt for m in matches])

    T_ab = invert_se3(T_b) @ T_a                          # X_b = T_ab · X_a
    R, t = T_ab[:3, :3], T_ab[:3, 3]
    Km = k_matrix(K)
    pts, valid = triangulate_two_view(uv_a, uv_b, Km, R, t)
    with np.errstate(all="ignore"):                       # macOS BLAS FPE; z=0 rows
        proj = (pts @ R.T + t) @ Km.T
        err = np.linalg.norm(proj[:, :2] / proj[:, 2:3] - uv_b, axis=1)
    valid &= (pts[:, 2] < max_depth) & (err < max_reproj)
    return uv_a[valid], pts[valid, 2]


class StereoDepthSource:
    """SGBM stereo depth (CPU)."""

    name = "stereo"

    def __init__(self, drive, max_depth):
        self.drive, self.max_depth = drive, max_depth

    def __call__(self, i, left, right):
        K = self.drive.intrinsics
        gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        return stereo_depth(gl, gr, K.fx, self.drive.baseline, max_depth=self.max_depth)


class MonoDepthSource:
    """Monocular relative depth (``infer(bgr) -> rel map``, larger = nearer for
    Depth Anything's disparity-like output) made metric per frame.

    ``infer`` is the TensorRT ``DepthEstimator.infer`` on the box; tests inject a
    stand-in. Only the left image is used — the right one is ignored.
    """

    name = "trt"

    def __init__(self, drive, infer, max_depth, gap=3, min_depth=2.0):
        self.drive, self.infer = drive, infer
        self.max_depth, self.min_depth, self.gap = max_depth, min_depth, gap
        self.aligner = DepthScaleAligner(space="disparity", smoothing=0.7, min_points=30)
        self.infer_ms: list[float] = []
        self.n_anchor: list[int] = []

    def __call__(self, i, left, right):
        K, poses = self.drive.intrinsics, self.drive.gt_poses
        t0 = time.perf_counter()
        rel = self.infer(cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        self.infer_ms.append(1e3 * (time.perf_counter() - t0))
        rel = cv2.resize(rel.astype(np.float32), (left.shape[1], left.shape[0]),
                         interpolation=cv2.INTER_LINEAR)

        j = min(i + self.gap, len(poses) - 1)
        if j == i:                                        # last frame: look back instead
            j = max(i - self.gap, 0)
        uv, z = triangulated_anchor(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY),
                                    cv2.cvtColor(self.drive.rgb_pair(j)[0], cv2.COLOR_RGB2GRAY),
                                    K, poses[i], poses[j])
        self.n_anchor.append(len(z))
        if len(z):
            cols = np.clip(np.rint(uv[:, 0]).astype(int), 0, rel.shape[1] - 1)
            rows = np.clip(np.rint(uv[:, 1]).astype(int), 0, rel.shape[0] - 1)
            self.aligner.fit(rel[rows, cols], z)
        depth = self.aligner.transform(rel)
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = 0.0
        return depth


def fuse_drive(drive, frame_stride=3, pixel_stride=2, max_depth=25.0, voxel=0.05,
               depth_source=None, log=print):
    K, poses = drive.intrinsics, drive.gt_poses
    source = depth_source or StereoDepthSource(drive, max_depth)
    P, C = [], []
    t0 = time.perf_counter()
    for i in range(0, len(drive), frame_stride):
        left, right = drive.rgb_pair(i)
        depth = source(i, left, right)
        depth[sky_mask(left, horizon_row=K.cy) | edge_mask(depth)] = 0.0
        p, c = backproject_masked(depth, left, K, poses[i], pixel_stride)
        P.append(p)
        C.append(c)
    dt = time.perf_counter() - t0
    pts, cols = np.concatenate(P), np.concatenate(C)
    n_raw = len(pts)
    pts, cols = voxel_downsample(pts, cols, voxel)
    log(f"fused {len(P)} frames ({source.name} depth) in {dt:.1f}s "
        f"({1e3 * dt / len(P):.0f} ms/frame end-to-end): "
        f"{n_raw:,} → {len(pts):,} pts after {voxel * 100:.0f} cm voxel")
    return pts, cols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default="2011_09_26")
    ap.add_argument("--drive", default="0009")
    ap.add_argument("--base-dir", default=None, help="KITTI raw root (default data/kitti_raw)")
    ap.add_argument("--out", default="output/kitti_0009.ply")
    ap.add_argument("--depth", choices=("stereo", "trt"), default="stereo")
    ap.add_argument("--engine", default="models/depth_engine_fp16.engine",
                    help="TensorRT depth engine for --depth trt")
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--pixel-stride", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=25.0, help="depth range cap (m)")
    ap.add_argument("--voxel", type=float, default=0.05, help="downsample cell (m)")
    args = ap.parse_args()

    drive = KittiRawDrive(args.date, args.drive, args.base_dir)
    print(f"KITTI {args.date} drive {args.drive}: {len(drive)} frames, "
          f"{drive.path_length:.0f} m, baseline {drive.baseline:.3f} m")

    source = None
    if args.depth == "trt":
        from depth.depth_estimator import DepthEstimator   # box-only: TensorRT + CUDA
        est = DepthEstimator(args.engine)
        source = MonoDepthSource(drive, est.infer, args.max_depth)
    pts, cols = fuse_drive(drive, args.frame_stride, args.pixel_stride,
                           args.max_depth, args.voxel, depth_source=source)
    if source is not None:
        ms = np.array(source.infer_ms[3:] or source.infer_ms)   # skip warm-up frames
        print(f"TensorRT depth: median {np.median(ms):.2f} ms/frame, p95 "
              f"{np.percentile(ms, 95):.2f} ms ({os.path.basename(args.engine)}); "
              f"anchor pts/frame median {int(np.median(source.n_anchor))}, "
              f"aligner fits {source.aligner.n_fits} / coasts {source.aligner.n_coasts}")
    print(f"world extent (m): {np.round(np.ptp(pts, axis=0), 1)}")

    write_ply_xyzrgb(args.out, pts, cols)
    traj = os.path.splitext(args.out)[0] + ".traj.npy"
    np.save(traj, drive.gt_poses.astype(np.float32))
    print(f"wrote {args.out}  +  {traj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
