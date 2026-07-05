"""Finite-difference gradient check for the M5 analytic backward pass.

Proves ``rasterize_backward`` computes correct gradients for every raw
parameter (means, log_scales, quats, opacities, colors) by comparing against
central finite differences of a squared-error loss.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gaussian.gaussian_model import GaussianModel, _logit
from gaussian.rasterizer import Camera, rasterize, rasterize_backward


def _scene():
    rng = np.random.default_rng(0)
    # Off-grid centres on purpose: a splat landing exactly on an integer pixel
    # boundary makes the tiny-eps finite difference (not the analytic gradient)
    # discontinuous, so keep every projected centre off the pixel lattice.
    means = np.array([[-0.15, 0.05, 0.03], [0.15, -0.1, 0.25], [0.04, 0.2, -0.2]])
    log_scales = np.log(np.array([[0.12, 0.08, 0.10],
                                  [0.09, 0.13, 0.07],
                                  [0.11, 0.10, 0.12]]))
    quats = np.array([[1.0, 0.15, -0.1, 0.05],
                      [0.9, -0.2, 0.1, 0.3],
                      [1.0, 0.0, 0.2, -0.1]])
    opacities = _logit(np.array([0.6, 0.55, 0.65]))
    colors = _logit(np.array([[0.8, 0.2, 0.3],
                              [0.2, 0.7, 0.4],
                              [0.5, 0.5, 0.9]]))
    model = GaussianModel(means, log_scales, quats, opacities, colors)
    cam = Camera.look_at(eye=(0, 0, -3), target=(0, 0, 0),
                         fx=90.0, fy=90.0, width=48, height=48)
    target = rng.uniform(0.0, 1.0, size=(48, 48, 3))
    return model, cam, target


def _loss_and_grad_image(model, cam, target):
    img, cache = rasterize(model, cam)
    grad_img = img - target                 # dL/dimage for L = 0.5||img-target||^2
    loss = 0.5 * float((grad_img ** 2).sum())
    return loss, grad_img, cache


def _numeric_grad(model, cam, target, field, eps=1e-6):
    base = getattr(model, field).copy()
    flat = base.ravel()
    num = np.zeros_like(flat)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        setattr(model, field, base.reshape(getattr(model, field).shape))
        lp, _, _ = _loss_and_grad_image(model, cam, target)
        flat[i] = orig - eps
        lm, _, _ = _loss_and_grad_image(model, cam, target)
        num[i] = (lp - lm) / (2 * eps)
        flat[i] = orig
    setattr(model, field, base)
    return num.reshape(base.shape)


@pytest.mark.parametrize("field", ["colors", "opacities", "means",
                                   "log_scales", "quats"])
def test_analytic_matches_numeric(field):
    model, cam, target = _scene()
    _, grad_img, cache = _loss_and_grad_image(model, cam, target)
    analytic = rasterize_backward(grad_img, cache)[field]
    numeric = _numeric_grad(model, cam, target, field)
    denom = np.maximum(np.abs(numeric).max(), 1e-6)
    rel = np.abs(analytic - numeric).max() / denom
    assert rel < 1e-4, (
        f"{field}: max rel err {rel:.2e}\nanalytic={analytic}\nnumeric={numeric}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
