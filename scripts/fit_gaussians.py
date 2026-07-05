"""Demo/benchmark for the M5 Gaussian-splat optimiser.

Builds a small ground-truth Gaussian scene, renders it from several views,
perturbs a copy, and optimises the copy back to the target views. Saves a
before/after/target strip and prints the PSNR curve + per-iteration latency.

Usage:
    python scripts/fit_gaussians.py --iters 300 --views 4 \
        --out output/gaussian_fit.png
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gaussian.gaussian_model import GaussianModel, _logit
from gaussian.optimizer import LearningRates, fit, psnr
from gaussian.rasterizer import Camera, rasterize


def truth_scene() -> GaussianModel:
    means = np.array([[-0.12, 0.06, 0.02], [0.14, -0.09, 0.2],
                      [0.03, 0.18, -0.15], [-0.05, -0.15, 0.05]])
    log_scales = np.log(np.array([[0.12, 0.09, 0.10], [0.10, 0.13, 0.08],
                                  [0.11, 0.10, 0.12], [0.09, 0.09, 0.14]]))
    quats = np.array([[1.0, 0.1, -0.05, 0.02], [0.95, -0.15, 0.1, 0.2],
                      [1.0, 0.0, 0.15, -0.05], [0.9, 0.2, 0.0, 0.1]])
    opacities = _logit(np.array([0.7, 0.65, 0.75, 0.7]))
    colors = _logit(np.array([[0.85, 0.2, 0.25], [0.2, 0.75, 0.35],
                              [0.4, 0.45, 0.9], [0.9, 0.8, 0.2]]))
    return GaussianModel(means, log_scales, quats, opacities, colors)


def make_views(model, n, res=64):
    eyes = [(0, 0, -3), (0.7, 0.2, -2.9), (-0.6, -0.3, -2.9), (0.1, 0.6, -2.9),
            (-0.2, -0.5, -3.0), (0.5, -0.4, -2.9)]
    views = []
    for i in range(n):
        cam = Camera.look_at(eye=eyes[i % len(eyes)], target=(0, 0, 0),
                             fx=1.4 * res, fy=1.4 * res, width=res, height=res)
        views.append((cam, rasterize(model, cam)[0]))
    return views


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--out", default="output/gaussian_fit.png")
    args = ap.parse_args()

    rng = np.random.default_rng(3)
    truth = truth_scene()
    views = make_views(truth, args.views, args.res)

    init = truth_scene()
    init.means += rng.normal(0, 0.04, init.means.shape)
    init.colors += rng.normal(0, 0.6, init.colors.shape)
    init.opacities += rng.normal(0, 0.3, init.opacities.shape)
    init.log_scales += rng.normal(0, 0.15, init.log_scales.shape)

    cam0, tgt0 = views[0]
    before = rasterize(init, cam0)[0]
    start = np.mean([psnr(rasterize(init, c)[0], t) for c, t in views])

    t0 = time.time()
    res = fit(init, views, iters=args.iters, lr=LearningRates(), log_every=max(1, args.iters // 10))
    dt = time.time() - t0
    after = rasterize(init, cam0)[0]

    print(f"\nPSNR {start:.2f} dB -> {res.psnrs[-1]:.2f} dB   "
          f"L1 {res.losses[0]:.4f} -> {res.losses[-1]:.5f}")
    print(f"{args.iters} iters x {args.views} views ({args.res}x{args.res}) "
          f"in {dt:.2f}s = {dt / args.iters * 1000:.2f} ms/iter")

    try:
        import cv2
        strip = np.concatenate([before, after, tgt0], axis=1)
        strip = (np.clip(strip, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, strip)
        print(f"wrote {args.out}  (before | after | target)")
    except Exception as e:  # cv2 optional
        print(f"[skip image write: {e}]")


if __name__ == "__main__":
    main()
