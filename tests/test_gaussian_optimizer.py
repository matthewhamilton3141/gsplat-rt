"""Overfit test for the M5 Gaussian optimiser: it must actually learn."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gaussian.gaussian_model import GaussianModel, _logit
from gaussian.optimizer import LearningRates, fit, psnr
from gaussian.rasterizer import Camera, rasterize


def _truth():
    means = np.array([[-0.12, 0.06, 0.02], [0.14, -0.09, 0.2], [0.03, 0.18, -0.15]])
    log_scales = np.log(np.array([[0.12, 0.09, 0.10],
                                  [0.10, 0.13, 0.08],
                                  [0.11, 0.10, 0.12]]))
    quats = np.array([[1.0, 0.1, -0.05, 0.02],
                      [0.95, -0.15, 0.1, 0.2],
                      [1.0, 0.0, 0.15, -0.05]])
    opacities = _logit(np.array([0.7, 0.65, 0.75]))
    colors = _logit(np.array([[0.85, 0.2, 0.25],
                              [0.2, 0.75, 0.35],
                              [0.4, 0.45, 0.9]]))
    return GaussianModel(means, log_scales, quats, opacities, colors)


def _views(model, n=3):
    eyes = [(0, 0, -3), (0.6, 0.2, -2.9), (-0.5, -0.3, -2.9)]
    views = []
    for i in range(n):
        cam = Camera.look_at(eye=eyes[i], target=(0, 0, 0),
                             fx=90.0, fy=90.0, width=48, height=48)
        img, _ = rasterize(model, cam)
        views.append((cam, img))
    return views


def test_overfit_multiview_improves_psnr():
    rng = np.random.default_rng(3)
    truth = _truth()
    views = _views(truth)

    # Start from the truth perturbed by noise; optimiser must recover the views.
    init = _truth()
    init.means += rng.normal(0, 0.03, init.means.shape)
    init.colors += rng.normal(0, 0.5, init.colors.shape)
    init.opacities += rng.normal(0, 0.3, init.opacities.shape)
    init.log_scales += rng.normal(0, 0.15, init.log_scales.shape)

    start_psnr = np.mean([psnr(rasterize(init, c)[0], t) for c, t in views])
    res = fit(init, views, iters=250, lr=LearningRates())
    end_psnr = res.psnrs[-1]

    # Loss decreases monotonically-ish and PSNR climbs substantially.
    assert res.losses[-1] < res.losses[0] * 0.5, (res.losses[0], res.losses[-1])
    assert end_psnr > start_psnr + 8.0, (start_psnr, end_psnr)
    assert end_psnr > 28.0, end_psnr


def test_ssim_weighted_fit_improves_structural_similarity():
    """The 3DGS loss ((1−λ)L1 + λ(1−SSIM)) must train — loss drops, and both PSNR
    and SSIM to the targets improve."""
    from gaussian.ssim import ssim

    rng = np.random.default_rng(11)
    truth = _truth()
    views = _views(truth)

    init = _truth()
    init.means += rng.normal(0, 0.03, init.means.shape)
    init.colors += rng.normal(0, 0.5, init.colors.shape)
    init.opacities += rng.normal(0, 0.3, init.opacities.shape)
    init.log_scales += rng.normal(0, 0.15, init.log_scales.shape)

    start_ssim = np.mean([ssim(rasterize(init, c)[0], t) for c, t in views])
    res = fit(init, views, iters=250, lr=LearningRates(), ssim_weight=0.2)
    end_ssim = np.mean([ssim(rasterize(init, c)[0], t) for c, t in views])

    assert res.losses[-1] < res.losses[0] * 0.6, (res.losses[0], res.losses[-1])
    assert res.psnrs[-1] > 25.0, res.psnrs[-1]
    assert end_ssim > start_ssim + 0.05, (start_ssim, end_ssim)


if __name__ == "__main__":
    import time
    t = time.time()
    sys.exit(pytest.main([__file__, "-v", "-s"]))
