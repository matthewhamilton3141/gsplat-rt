"""Run the RGB-D visual-odometry front-end on a TUM sequence and score it.

Tracks the sequence with RGBDOdometry, computes Absolute Trajectory Error
(ATE-RMSE) against ground truth after rigid alignment, and renders the
estimated vs ground-truth trajectory top-down.

Usage:
    python scripts/eval_odometry.py --seq data/tum/rgbd_dataset_freiburg1_desk \
        --max-frames 200 --out output/odometry_traj.png
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.tum_dataset import TUMDataset
from slam.rgbd_odometry import RGBDOdometry, ate_rmse, align_umeyama


def render_trajectories(est, gt, size=560, pad=0.12):
    """Top-down est-vs-gt trajectory overlay (est rigidly aligned to gt)."""
    ep, gp = est[:, :3, 3], gt[:, :3, 3]
    R, t, s = align_umeyama(ep, gp)
    ep = (s * (R @ ep.T)).T + t

    both = np.vstack([ep, gp])
    spread = both.max(0) - both.min(0)
    up = int(np.argmin(spread))
    h = [a for a in range(3) if a != up]
    lo, hi = both[:, h].min(0), both[:, h].max(0)
    span = (hi - lo).max() * (1 + pad) or 1.0
    ctr = (hi + lo) / 2

    def to_px(P):
        n = (P[:, h] - ctr) / span + 0.5
        ij = np.clip((n * (size - 1)).astype(int), 0, size - 1)
        ij[:, 1] = size - 1 - ij[:, 1]
        return ij

    img = np.zeros((size, size, 3), np.uint8)
    for name, traj, color in [("ground truth", gp, (80, 220, 80)),
                              ("estimated", ep, (80, 160, 255))]:
        pts = to_px(traj)
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, tuple(a), tuple(b), color, 1, cv2.LINE_AA)
    cv2.putText(img, "green = ground truth   blue = estimated (aligned)",
                (10, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="data/tum/rgbd_dataset_freiburg1_desk")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--out", default="output/odometry_traj.png")
    ap.add_argument("--frontend", choices=["orb", "superpoint"], default="orb",
                    help="feature front-end: orb (CPU baseline) or superpoint "
                         "(SuperPoint+LightGlue ONNX)")
    ap.add_argument("--sp-onnx", default="weights/sp_lg_tum.onnx",
                    help="fused SuperPoint+LightGlue ONNX (for --frontend superpoint)")
    ap.add_argument("--provider", choices=["cuda", "tensorrt", "cpu"], default="cuda",
                    help="onnxruntime execution provider (for --frontend superpoint); "
                         "tensorrt builds/caches an FP16 TRT engine from the ONNX")
    args = ap.parse_args()

    ds = TUMDataset(args.seq)
    frames = ds.frames[:args.max_frames] if args.max_frames > 0 else ds.frames

    if args.frontend == "superpoint":
        from slam.superpoint_lightglue import SuperPointLightGlueFrontend, ort_providers
        K = ds.intrinsics
        fe = SuperPointLightGlueFrontend(args.sp_onnx, height=K.height, width=K.width,
                                         providers=ort_providers(args.provider, args.sp_onnx))
        print(f"Front-end     : SuperPoint+LightGlue ONNX  (providers={fe.providers})")
        odom = RGBDOdometry(ds.intrinsics, frontend=fe)
    else:
        print("Front-end     : ORB (CPU baseline)")
        odom = RGBDOdometry(ds.intrinsics)
    gt_poses, est_poses = [], []
    ok_count = 0
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        res = odom.track(f.load_rgb(), f.load_depth(),
                         init_pose=f.pose if i == 0 else None)
        est_poses.append(res.pose)
        gt_poses.append(f.pose)
        ok_count += int(res.ok)
    dt = time.perf_counter() - t0

    est_poses = np.stack(est_poses)
    gt_poses = np.stack(gt_poses)
    rmse, err = ate_rmse(est_poses, gt_poses)

    print(f"Sequence      : {os.path.basename(args.seq)}")
    print(f"Frames tracked: {len(frames)}  ({ok_count} PnP-ok, "
          f"{len(frames) - ok_count} coasted)")
    print(f"Speed         : {len(frames) / dt:.1f} fps  ({dt / len(frames) * 1e3:.1f} ms/frame)")
    print(f"ATE-RMSE      : {rmse * 100:.1f} cm   (median {np.median(err) * 100:.1f} cm, "
          f"max {err.max() * 100:.1f} cm)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, render_trajectories(est_poses, gt_poses))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
