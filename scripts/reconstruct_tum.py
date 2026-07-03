"""Pose-aware reconstruction proof for M6.

Fuses a TUM RGB-D sequence's real metric depth into a single world-space point
cloud two ways and renders both top-down:

    left  — identity pose   (every frame dropped at the origin: today's behaviour)
    right — ground-truth pose (each frame placed by its mocap trajectory)

If pose-aware mapping works, the right panel resolves into a coherent desk
scene while the left collapses into an overlapping frustum blob. This validates
the mapping side with *known* poses before the visual-odometry front-end has to
*estimate* them.

Usage:
    python scripts/reconstruct_tum.py \
        --seq data/tum/rgbd_dataset_freiburg1_desk \
        --out output/tum_recon.png --frame-stride 3 --pixel-stride 4
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.tum_dataset import TUMDataset


def backproject(depth, K, pose, pixel_stride):
    """Depth map -> (M,3) world points. pose=None keeps them in camera frame."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H:pixel_stride, 0:W:pixel_stride]
    z = depth[vs, us].ravel()
    us, vs = us.ravel(), vs.ravel()
    valid = z > 0.1
    z, us, vs = z[valid], us[valid], vs[valid]
    x = (us - K.cx) * z / K.fx
    y = (vs - K.cy) * z / K.fy
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)   # camera frame
    if pose is not None:
        # errstate suppresses spurious float32 BLAS subnormal warnings on macOS
        # (same guard as TSDFVolume.integrate).
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            pts = pts @ pose[:3, :3].T + pose[:3, 3]        # -> world
    return pts


def fuse(ds, use_pose, frame_stride, pixel_stride):
    clouds = []
    for f in list(ds)[::frame_stride]:
        clouds.append(backproject(f.load_depth(), ds.intrinsics,
                                  f.pose if use_pose else None, pixel_stride))
    return np.concatenate(clouds, axis=0)


def topdown_image(pts, size=480, pad=0.1):
    """Orthographic top-down density render, coloured by height.

    The vertical axis is taken as the point cloud's thinnest spatial spread
    (a level-held camera moves mostly horizontally), and dropped for the plan.
    """
    spread = pts.max(0) - pts.min(0)
    up = int(np.argmin(spread))
    horiz = [a for a in range(3) if a != up]

    p2 = pts[:, horiz]
    lo, hi = p2.min(0), p2.max(0)
    span = (hi - lo).max() * (1 + pad) or 1.0
    ctr = (hi + lo) / 2
    norm = (p2 - ctr) / span + 0.5                          # -> [0,1]
    ij = np.clip((norm * (size - 1)).astype(int), 0, size - 1)

    height = pts[:, up]
    hn = (height - height.min()) / (np.ptp(height) or 1.0)
    img = np.zeros((size, size, 3), np.uint8)
    colors = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, :]
    # y-axis flips so "up" in image = larger horiz[1]
    img[size - 1 - ij[:, 1], ij[:, 0]] = colors
    return img


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="data/tum/rgbd_dataset_freiburg1_desk")
    ap.add_argument("--out", default="output/tum_recon.png")
    ap.add_argument("--frame-stride", type=int, default=3)
    ap.add_argument("--pixel-stride", type=int, default=4)
    args = ap.parse_args()

    ds = TUMDataset(args.seq)
    n_used = len(list(ds)[::args.frame_stride])
    print(f"Loaded {len(ds)} associated frames from {os.path.basename(args.seq)}; "
          f"fusing {n_used} (stride {args.frame_stride}).")

    ident = fuse(ds, False, args.frame_stride, args.pixel_stride)
    gt = fuse(ds, True, args.frame_stride, args.pixel_stride)
    print(f"identity cloud: {len(ident):,} pts   gt cloud: {len(gt):,} pts")
    print(f"gt world extent (m): {np.round(gt.max(0) - gt.min(0), 2)}")

    size = 480
    left = label(topdown_image(ident, size), "identity pose (current)")
    right = label(topdown_image(gt, size), "ground-truth pose (M6)")
    combo = np.hstack([left, np.full((size, 4, 3), 60, np.uint8), right])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, combo)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
