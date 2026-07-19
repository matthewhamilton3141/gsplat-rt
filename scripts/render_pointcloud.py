#!/usr/bin/env python3
"""Render the pipeline's RGB reconstruction point cloud to stills + an orbit GIF (CPU, no GPU).

The exported `.usdz` carries only the (colorless) collision-proxy mesh, so a sim render of it
looks like a gray sheet. The *reconstruction itself* is a 2.1M-point RGB cloud (`tum_recon_light.
ply`) — this renders that in colour with a plain NumPy painter's-algorithm rasteriser (perspective
+ z-order), so the actual reconstructed scene is visible without Open3D/Isaac/a GPU.

    python scripts/render_pointcloud.py --ply output/tum_recon_light.ply --out docs/reconstruction
"""

import argparse
import os

import numpy as np


def load_ply_xyzrgb(path):
    with open(path, "rb") as f:
        while f.readline().strip() != b"end_header":
            pass
        raw = f.read()
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    a = np.frombuffer(raw, dtype=dt)
    P = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
    C = np.stack([a["r"], a["g"], a["b"]], 1).astype(np.float64)
    fin = np.isfinite(P).all(1)
    P, C = P[fin], C[fin]
    lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)   # drop outliers
    keep = ((P >= lo) & (P <= hi)).all(1)
    return P[keep] - (lo + hi) / 2, C[keep], float(np.linalg.norm(hi - lo))


def render(P, C, scale, az, el, W=1500, H=950, fov=55, dist_mult=0.82, bright=1.35, gamma=0.85):
    ar, er = np.radians(az), np.radians(el)
    d = np.array([np.cos(er) * np.sin(ar), np.cos(er) * np.cos(ar), np.sin(er)])   # Z-up orbit
    campos = d * scale * dist_mult
    f = -campos / np.linalg.norm(campos)
    wup = np.array([0, 0, 1.0])
    if abs(f @ wup) > 0.95:
        wup = np.array([0, 1.0, 0])
    right = np.cross(wup, f); right /= np.linalg.norm(right)
    up = np.cross(f, right)
    Q = P - campos
    Z = Q @ f
    m = Z > 1e-3
    fpx = (0.5 * W) / np.tan(np.radians(fov) / 2)
    u = Q[m] @ right * fpx / Z[m] + W / 2
    v = -(Q[m] @ up) * fpx / Z[m] + H / 2
    z = Z[m]; col = C[m]
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, col = u[ok].astype(int), v[ok].astype(int), z[ok], col[ok]
    o = np.argsort(-z)                                    # far first; near overwrites
    u, v, col = u[o], v[o], col[o]
    col = np.clip((np.clip(col / 255.0, 0, 1) ** gamma) * bright * 255.0, 0, 255).astype(np.uint8)
    img = np.full((H, W, 3), 12, np.uint8)
    for dx in (0, 1):                                    # 2x2 splat for point density
        for dy in (0, 1):
            img[np.clip(v + dy, 0, H - 1), np.clip(u + dx, 0, W - 1)] = col
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply", default="output/tum_recon_light.ply")
    ap.add_argument("--out", default="docs/reconstruction")
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--elev", type=float, default=22.0)
    args = ap.parse_args()
    from PIL import Image

    P, C, scale = load_ply_xyzrgb(args.ply)
    print(f"loaded {len(P)} pts, scale {scale:.2f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    frames = []
    for i in range(args.frames):
        az = 360.0 * i / args.frames
        im = Image.fromarray(render(P, C, scale, az, args.elev))
        frames.append(im)
        if i == 0 or az in (90.0, 180.0, 270.0):
            im.save(f"{args.out}_az{int(az)}.png")
    frames[0].save(f"{args.out}.gif", save_all=True, append_images=frames[1:],
                   duration=90, loop=0, optimize=True)
    print(f"wrote {args.out}.gif ({len(frames)} frames) + hero stills")


if __name__ == "__main__":
    main()
