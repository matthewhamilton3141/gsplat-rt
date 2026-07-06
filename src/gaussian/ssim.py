"""Windowed SSIM + differentiable D-SSIM loss (numpy).

The 3D Gaussian Splatting loss (Kerbl et al. 2023) is

    L = (1 - λ)·L1 + λ·(1 - SSIM),      λ = 0.2

but the M5 optimiser shipped with the L1 term only — the docstring in
``optimizer.py`` flagged the missing "windowed SSIM". This module supplies it:
mean structural similarity over an 11×11 Gaussian window (σ = 1.5, the standard
Wang et al. 2004 parameters) and, crucially, the analytic gradient
``d(1 − SSIM)/d(image)`` so it plugs straight into the existing analytic-gradient
optimiser rather than needing autodiff.

Filtering choice — self-adjoint by construction
-----------------------------------------------
The local means/variances are computed by convolving with a **separable,
symmetric, zero-padded** kernel. Zero-padded convolution with a symmetric kernel
is a symmetric linear operator (``W = Wᵀ``), so the same ``_filter`` serves as
its own adjoint in the backward pass — which keeps the gradient exact (it matches
finite differences to ~1e-6, verified in the tests) instead of only approximate
at the image borders. The mild border darkening from zero padding is immaterial
for a 0.2-weighted training loss.

Everything is pure numpy — runs and unit-tests on the dev machine; the CUDA/torch
port reuses the identical loss.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Wang et al. 2004 stabilisers for images in [0, 1].
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def gaussian_window(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Normalised 1-D Gaussian kernel (the separable SSIM window)."""
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    k = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _filter(img: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """Separable, symmetric, zero-padded 'same' convolution over (H, W, C).

    Self-adjoint: ``<_filter(a), b> == <a, _filter(b)>`` (used in the backward
    pass, where the adjoint of the mean/variance filter is the filter itself).
    """
    r = len(k1d) // 2
    h, w = img.shape[:2]

    pad_h = np.pad(img, ((r, r), (0, 0), (0, 0)))          # zeros, along H
    tmp = np.zeros_like(img)
    for t, kt in enumerate(k1d):
        tmp += kt * pad_h[t:t + h]

    pad_w = np.pad(tmp, ((0, 0), (r, r), (0, 0)))          # zeros, along W
    out = np.zeros_like(img)
    for t, kt in enumerate(k1d):
        out += kt * pad_w[:, t:t + w]
    return out


def _as_hwc(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    return a[:, :, None] if a.ndim == 2 else a


def _ssim_fields(x: np.ndarray, y: np.ndarray, k1d: np.ndarray):
    """Shared forward quantities for both SSIM value and its gradient."""
    mu_x = _filter(x, k1d)
    mu_y = _filter(y, k1d)
    vx = _filter(x * x, k1d)               # E[x²] (windowed)
    cxy = _filter(x * y, k1d)              # E[xy] (windowed)
    var_x = vx - mu_x * mu_x
    var_y = _filter(y * y, k1d) - mu_y * mu_y
    cov = cxy - mu_x * mu_y

    a1 = 2.0 * mu_x * mu_y + _C1
    a2 = 2.0 * cov + _C2
    b1 = mu_x * mu_x + mu_y * mu_y + _C1
    b2 = var_x + var_y + _C2
    s_map = (a1 * a2) / (b1 * b2)
    return dict(mu_x=mu_x, mu_y=mu_y, a1=a1, a2=a2, b1=b1, b2=b2, s_map=s_map)


def ssim(img: np.ndarray, target: np.ndarray, k1d: np.ndarray | None = None) -> float:
    """Mean SSIM over an 11×11 Gaussian window. Inputs (H,W) or (H,W,C) in [0,1]."""
    x, y = _as_hwc(img), _as_hwc(target)
    if k1d is None:
        k1d = gaussian_window()
    return float(np.mean(_ssim_fields(x, y, k1d)["s_map"]))


def dssim_loss_and_grad(
    img: np.ndarray, target: np.ndarray, k1d: np.ndarray | None = None,
) -> Tuple[float, np.ndarray]:
    """D-SSIM loss ``1 − mean(SSIM)`` and its gradient w.r.t. ``img``.

    Returns ``(loss, grad)`` with ``grad`` shaped like ``img``. The gradient is
    analytic (see module docstring) and matches finite differences to ~1e-6.
    """
    x2d = np.asarray(img, dtype=np.float64)
    x, y = _as_hwc(img), _as_hwc(target)
    if k1d is None:
        k1d = gaussian_window()

    f = _ssim_fields(x, y, k1d)
    mu_x, mu_y = f["mu_x"], f["mu_y"]
    a1, a2, b1, b2 = f["a1"], f["a2"], f["b1"], f["b2"]
    loss = 1.0 - float(np.mean(f["s_map"]))

    # Per-pixel partials of S w.r.t. the x-dependent windowed fields
    # μx = W x,  vx = W(x²),  cxy = W(xy):
    #   ∂S/∂cxy = 2·a1 / (b1·b2)
    #   ∂S/∂vx  = −a1·a2 / (b1·b2²)
    #   ∂S/∂μx  = 2μy(a2 − a1)/(b1·b2) − 2μx·a1·a2·(b2 − b1)/(b1·b2)²
    b1b2 = b1 * b2
    ds_dcxy = 2.0 * a1 / b1b2
    ds_dvx = -a1 * a2 / (b1b2 * b2)
    ds_dmux = (2.0 * mu_y * (a2 - a1) / b1b2
               - 2.0 * mu_x * a1 * a2 * (b2 - b1) / (b1b2 * b1b2))

    # Backprop through the (self-adjoint) filter and the x², xy nonlinearities.
    n = x.size
    dS_dx = (_filter(ds_dmux, k1d)
             + 2.0 * x * _filter(ds_dvx, k1d)
             + y * _filter(ds_dcxy, k1d))
    grad = -(dS_dx / n)                       # d(1 − mean S)/dx
    grad = grad.reshape(x2d.shape)            # match caller's (H,W) or (H,W,C)
    return loss, grad
