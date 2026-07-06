"""Tests for windowed SSIM + the analytic D-SSIM gradient (src/gaussian/ssim.py).

Pure numpy. The centrepiece is a finite-difference check of the analytic gradient
— the same verification pattern the rasterizer's analytic gradients use.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gaussian.ssim import (  # noqa: E402
    _filter,
    dssim_loss_and_grad,
    gaussian_window,
    ssim,
)


def test_gaussian_window_normalised_and_symmetric():
    k = gaussian_window(11, 1.5)
    assert k.shape == (11,)
    assert k.sum() == pytest.approx(1.0)
    assert np.allclose(k, k[::-1])            # symmetric


def test_filter_is_self_adjoint():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((17, 19, 3))
    b = rng.standard_normal((17, 19, 3))
    k = gaussian_window()
    lhs = float(np.sum(_filter(a, k) * b))
    rhs = float(np.sum(a * _filter(b, k)))
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


def test_ssim_identity_is_one():
    rng = np.random.default_rng(1)
    x = rng.random((24, 24, 3))
    assert ssim(x, x) == pytest.approx(1.0, abs=1e-9)


def test_ssim_symmetric():
    rng = np.random.default_rng(2)
    x = rng.random((20, 28, 3))
    y = rng.random((20, 28, 3))
    assert ssim(x, y) == pytest.approx(ssim(y, x), abs=1e-12)


def test_ssim_decreases_with_corruption():
    rng = np.random.default_rng(3)
    x = rng.random((32, 32, 3))
    noisy = np.clip(x + 0.3 * rng.standard_normal(x.shape), 0, 1)
    assert ssim(x, noisy) < ssim(x, x)
    assert -1.0 <= ssim(x, noisy) <= 1.0


def test_dssim_zero_at_identity():
    rng = np.random.default_rng(4)
    x = rng.random((20, 20, 3))
    loss, grad = dssim_loss_and_grad(x, x)
    assert loss == pytest.approx(0.0, abs=1e-9)
    assert np.max(np.abs(grad)) < 1e-9        # gradient vanishes at a perfect match


def test_dssim_grad_shape_matches_input():
    rng = np.random.default_rng(5)
    x = rng.random((16, 18, 3))
    y = rng.random((16, 18, 3))
    _, grad = dssim_loss_and_grad(x, y)
    assert grad.shape == x.shape
    # grayscale (H,W) input → (H,W) gradient
    xg, yg = x[..., 0], y[..., 0]
    _, gg = dssim_loss_and_grad(xg, yg)
    assert gg.shape == xg.shape


def test_dssim_gradient_matches_finite_differences():
    rng = np.random.default_rng(6)
    x = rng.random((14, 14, 3))
    y = rng.random((14, 14, 3))
    _, grad = dssim_loss_and_grad(x, y)

    eps = 1e-6
    # Central differences at a random subset of coordinates.
    coords = [tuple(c) for c in rng.integers(
        [0, 0, 0], [14, 14, 3], size=(40, 3))]
    max_err = 0.0
    for c in coords:
        xp = x.copy(); xp[c] += eps
        xm = x.copy(); xm[c] -= eps
        lp, _ = dssim_loss_and_grad(xp, y)
        lm, _ = dssim_loss_and_grad(xm, y)
        fd = (lp - lm) / (2 * eps)
        max_err = max(max_err, abs(fd - grad[c]))
    assert max_err < 1e-5, f"analytic vs FD gradient mismatch: {max_err:.2e}"


def test_dssim_grad_sign_reduces_loss():
    # A gradient step should reduce the D-SSIM loss.
    rng = np.random.default_rng(7)
    x = rng.random((24, 24, 3))
    y = rng.random((24, 24, 3))
    loss0, grad = dssim_loss_and_grad(x, y)
    x_step = x - 1e-2 * grad / (np.linalg.norm(grad) + 1e-12)
    loss1, _ = dssim_loss_and_grad(x_step, y)
    assert loss1 < loss0
