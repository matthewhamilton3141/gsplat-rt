#!/usr/bin/env python3
"""Render an orbiting-camera MP4 of a static reconstruction (.ply).

A GPU-free "turntable": loads a finalize-stage / point-cloud .ply the same way the
viewer does (:func:`read_ply` + auto-upright), then spins a perspective camera
around the scene and rasterises each frame with a depth-sorted point splatter
(pure numpy). Frames are piped straight to ffmpeg → H.264 MP4.

    python scripts/render_turntable.py --ply output/tum_recon_light.ply \
        --out docs/reconstruction_turntable.mp4 --seconds 6

Runs anywhere (numpy + ffmpeg on PATH; no torch/CUDA/cv2).
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from viz.scene_source import read_ply                 # noqa: E402
from mapping.visualization import orient_upright      # noqa: E402


def _look_at(cam, target, up):
    """World→camera basis (camera looks down +Z, x right, y up). Returns 3x3."""
    fwd = target - cam
    fwd /= np.linalg.norm(fwd) + 1e-12
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, fwd)
    return np.stack([right, true_up, fwd], axis=0)     # rows = cam axes


def _render_frame(pts, cols, cam, target, up, W, H, fov_deg, radius, bg):
    """Depth-sorted point splat of the cloud from one camera pose → (H,W,3) uint8."""
    R = _look_at(cam, target, up)
    with np.errstate(all="ignore"):                     # BLAS emits spurious FPE
        cc = (pts - cam) @ R.T                          # camera coords
    z = cc[:, 2]
    front = z > 1e-3
    cc, c = cc[front], cols[front]
    z = cc[:, 2]

    f = 0.5 * W / np.tan(np.radians(fov_deg) * 0.5)
    u = f * cc[:, 0] / z + W * 0.5
    v = H * 0.5 - f * cc[:, 1] / z                      # flip y for image space

    # Point size shrinks with depth; a splat of ~this many px across.
    px = np.clip((f * (radius * 0.004) / z), 1.0, 4.0)

    img = np.empty((H, W, 3), np.uint8)
    img[:] = bg
    zbuf = np.full((H, W), np.inf)

    order = np.argsort(-z)                              # far → near (painter's)
    u, v, z, c, px = u[order], v[order], z[order], c[order], px[order]
    ui, vi, ri = u.astype(np.int32), v.astype(np.int32), px.astype(np.int32)
    rgb = np.clip(c * 255.0, 0, 255).astype(np.uint8)

    for dy in range(-2, 3):                             # square splat kernel
        for dx in range(-2, 3):
            m = (np.abs(dx) <= ri) & (np.abs(dy) <= ri)
            xx, yy = ui[m] + dx, vi[m] + dy
            inb = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            xx, yy, zz, cc2 = xx[inb], yy[inb], z[m][inb], rgb[m][inb]
            # near-wins z-test (points already far→near, so last write wins anyway,
            # but the z-test keeps a nearer splat from an earlier point on top).
            closer = zz < zbuf[yy, xx]
            xx, yy, cc2 = xx[closer], yy[closer], cc2[closer]
            zbuf[yy, xx] = zz[closer]
            img[yy, xx] = cc2
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="Render an orbiting MP4 of a .ply")
    ap.add_argument("--ply", default="output/tum_recon_light.ply")
    ap.add_argument("--out", default="docs/reconstruction_turntable.mp4")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=55.0)
    ap.add_argument("--elevation", type=float, default=30.0, help="deg above horizon")
    ap.add_argument("--distance", type=float, default=2.4, help="orbit radius, ×scene radius")
    ap.add_argument("--max-points", type=int, default=250_000)
    ap.add_argument("--turns", type=float, default=1.0, help="full revolutions over the clip")
    args = ap.parse_args()

    print(f"loading {args.ply} …", flush=True)
    snap = read_ply(args.ply)
    pts, cols = snap.means, snap.colors
    print(f"  {len(pts):,} points", flush=True)

    finite = np.isfinite(pts).all(axis=1)               # drop stray inf/nan splats
    pts, cols = pts[finite], cols[finite]

    # Reject far pose-drift outliers so the framing hugs the real scene (and the
    # camera maths can't overflow on a runaway splat 10^30 away).
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    keep = d < 3.0 * np.percentile(d, 95)
    pts, cols = pts[keep], cols[keep]
    print(f"  {len(pts):,} after outlier clip", flush=True)

    pts, _ = orient_upright(pts)                        # stand it up (+Z up)
    pts = pts.astype(np.float64)

    if len(pts) > args.max_points:
        idx = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    # Robust centre + radius (ignore stray outliers).
    centre = np.median(pts, axis=0)
    radius = float(np.percentile(np.linalg.norm(pts - centre, axis=1), 90)) or 1.0
    target = centre
    up = np.array([0.0, 0.0, 1.0])
    el = np.radians(args.elevation)
    dist = args.distance * radius
    bg = np.array([15, 17, 22], np.uint8)              # near-black slate

    n_frames = max(1, int(args.seconds * args.fps))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    print(f"rendering {n_frames} frames → {args.out}", flush=True)
    for i in range(n_frames):
        az = 2.0 * np.pi * args.turns * (i / n_frames)
        cam = centre + dist * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        img = _render_frame(pts, cols, cam, target, up,
                            args.width, args.height, args.fov, radius, bg)
        ff.stdin.write(img.tobytes())
        if (i + 1) % args.fps == 0:
            print(f"  {i + 1}/{n_frames}", flush=True)

    ff.stdin.close()
    ff.wait()
    sz = os.path.getsize(args.out) / 1e6 if os.path.exists(args.out) else 0.0
    print(f"done: {args.out} ({sz:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
