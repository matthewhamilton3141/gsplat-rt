#!/usr/bin/env python3
"""Render a chase-camera flythrough of a reconstructed drive (.ply + .traj.npy).

The AV-scale sibling of ``render_turntable.py``: a turntable orbit suits a desk,
but a 300 m street is long and thin, so the camera instead follows the recorded
ego trajectory — trailing and above the car, looking down the road ahead — then
cranes up into a bird's-eye reveal of the whole drive. Frames are rasterised by
the same depth-sorted numpy point splatter and piped to ffmpeg.

    python scripts/render_drive.py --ply output/kitti_0009.ply \
        --out docs/kitti_drive.mp4 --gif docs/kitti_drive.gif --still docs/kitti_drive.png

Runs anywhere (numpy + ffmpeg on PATH; no torch/CUDA).
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from viz.scene_source import read_ply              # noqa: E402
from render_turntable import _render_frame          # noqa: E402

# KITTI poses live in the first camera's frame: y points down.
WORLD_UP = np.array([0.0, -1.0, 0.0])


def resample_by_arclength(xyz, n):
    """``n`` points evenly spaced in distance along a polyline → (n,3)."""
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, xyz[:, k]) for k in range(3)], axis=-1), s[-1]


def chase_path(xyz, n, back=11.0, height=6.0, look_ahead=22.0, smooth=15.0):
    """Chase-camera ``(eye, target)`` per frame along a trajectory.

    ``eye`` trails ``back`` m behind the car and ``height`` m above it along the
    *smoothed* heading (raw per-frame heading jitters); ``target`` is the point
    ``look_ahead`` m further down the route, so the camera leans into turns.
    All distances are arc-length, so the clip moves at constant ground speed.
    """
    pts, length = resample_by_arclength(xyz, n)
    step = length / max(n - 1, 1)
    w = max(1, int(round(smooth / step)))
    pad = np.pad(pts, ((w, w), (0, 0)), mode="edge")
    kern = np.ones(2 * w + 1) / (2 * w + 1)
    sm = np.stack([np.convolve(pad[:, k], kern, mode="valid") for k in range(3)], axis=-1)
    fwd = np.gradient(sm, axis=0)
    with np.errstate(all="ignore"):                       # macOS BLAS emits spurious FPE
        fwd -= np.outer(fwd @ WORLD_UP, WORLD_UP)        # keep heading level
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9
    eye = pts - back * fwd + height * WORLD_UP
    ahead = np.minimum(np.arange(n) + int(round(look_ahead / step)), n - 1)
    return eye, pts[ahead]


def crane_path(start_eye, start_target, centre, span, n, elevation=55.0, dist=0.85):
    """Ease from the last chase pose up to a 3/4 oblique view of the whole drive.

    The end pose sits ``dist·span`` from ``centre`` at ``elevation`` degrees,
    on the far side of the drive from the car so the reveal looks back over it.
    Stopping short of straight-down keeps the look-at basis well-conditioned
    (at 90° the view direction is parallel to up and the frame rolls).
    """
    t = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))[:, None]   # ease in-out
    el = np.radians(elevation)
    hor = _level(centre - start_eye)
    end_eye = centre + dist * span * (np.sin(el) * WORLD_UP - np.cos(el) * hor)
    return (1 - t) * start_eye + t * end_eye, (1 - t) * start_target + t * centre


def _level(v):
    v = v - (v @ WORLD_UP) * WORLD_UP
    return v / (np.linalg.norm(v) + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description="Chase-cam flythrough of a reconstructed drive")
    ap.add_argument("--ply", default="output/kitti_0009.ply")
    ap.add_argument("--traj", default=None, help="poses .npy (default: <ply>.traj.npy)")
    ap.add_argument("--out", default="docs/kitti_drive.mp4")
    ap.add_argument("--gif", default=None, help="also write a GIF (every 3rd frame, half-res)")
    ap.add_argument("--still", default=None, help="also write a hero PNG (chase frame)")
    ap.add_argument("--still-at", type=float, default=0.2, help="fraction along the drive")
    ap.add_argument("--still-only", action="store_true", help="write the still, skip the video")
    ap.add_argument("--seconds", type=float, default=12.0, help="chase section length")
    ap.add_argument("--crane-seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--view-radius", type=float, default=70.0,
                    help="only splat points this close to the camera (m) in the chase")
    args = ap.parse_args()

    snap = read_ply(args.ply)
    pts, cols = snap.means.astype(np.float64), snap.colors
    traj = np.load(args.traj or os.path.splitext(args.ply)[0] + ".traj.npy")
    xyz = traj[:, :3, 3].astype(np.float64)
    print(f"{len(pts):,} points, {len(xyz)} poses", flush=True)

    n_chase = max(2, int(args.seconds * args.fps))
    n_crane = int(args.crane_seconds * args.fps)
    eye, tgt = chase_path(xyz, n_chase)
    # Frame the reveal on the *route*, not the cloud's bbox: a turning drive is
    # L-shaped and its bbox centre sits in empty space off the road.
    centre, span = xyz.mean(axis=0), float(np.ptp(xyz, axis=0).max())
    if n_crane:
        ce, ct = crane_path(eye[-1], tgt[-1], centre, span, n_crane)
        eye, tgt = np.vstack([eye, ce]), np.vstack([tgt, ct])
    bg = np.array([15, 17, 22], np.uint8)
    # Wide shots see the whole drive; a subsample keeps them fast and the splats fine.
    sub = np.random.default_rng(0).permutation(len(pts))[:1_500_000]
    pts_wide, cols_wide = pts[sub], cols[sub]

    def render(i, W, H):
        if i < n_chase:                                   # cull for speed in the chase
            near = np.linalg.norm(pts - eye[i], axis=1) < args.view_radius
            p, c, r = pts[near], cols[near], 60.0
        else:
            t = (i - n_chase) / max(n_crane - 1, 1)
            p, c, r = pts_wide, cols_wide, 60.0 + (0.4 * span - 60.0) * t
        return _render_frame(p, c, eye[i], tgt[i], WORLD_UP, W, H, args.fov, r, bg)

    if args.still:
        import cv2
        k = int(args.still_at * (n_chase - 1))
        img = render(k, args.width * 2, args.height * 2)
        os.makedirs(os.path.dirname(args.still) or ".", exist_ok=True)
        cv2.imwrite(args.still, img[:, :, ::-1])
        print(f"still → {args.still}", flush=True)
        if args.still_only:
            return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = len(eye)
    print(f"rendering {n} frames → {args.out}", flush=True)
    for i in range(n):
        ff.stdin.write(render(i, args.width, args.height).tobytes())
        if (i + 1) % args.fps == 0:
            print(f"  {i + 1}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    print(f"done: {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)", flush=True)

    if args.gif:                                          # palette GIF from the MP4
        vf = (f"fps={args.fps // 3},scale={args.width // 2}:-1:flags=lanczos,"
              "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer")
        subprocess.run(["ffmpeg", "-y", "-i", args.out, "-vf", vf, "-loop", "0", args.gif],
                       check=True, stderr=subprocess.DEVNULL)
        print(f"gif → {args.gif} ({os.path.getsize(args.gif) / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
