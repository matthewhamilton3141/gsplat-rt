"""Gaussian-splat optimiser (M5): fit a GaussianModel to posed target views.

Numpy Adam over the five raw parameter groups, driving the analytic gradients
from ``rasterizer.rasterize_backward``. This is the CPU reference optimiser —
correctness and the training loop shape first; the CUDA/torch port to the A10G
reuses the exact same loss and update rule.

Loss is the 3DGS photometric loss (Kerbl et al. 2023):
``(1 − λ)·L1 + λ·(1 − SSIM)`` with ``λ = ssim_weight`` (0 → pure L1, the original
behaviour). The windowed SSIM + analytic D-SSIM gradient live in ``ssim.py``.
PSNR is reported for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .gaussian_model import GaussianModel
from .rasterizer import Camera, rasterize, rasterize_backward
from .ssim import dssim_loss_and_grad

_FIELDS = ("means", "log_scales", "quats", "opacities", "colors")


@dataclass
class LearningRates:
    means: float = 0.01
    log_scales: float = 0.01
    quats: float = 0.01
    opacities: float = 0.05
    colors: float = 0.05


@dataclass
class _AdamState:
    lr: LearningRates
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)
    t: int = 0

    def step(self, model: GaussianModel, grads: dict) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t
        for name in _FIELDS:
            g = grads[name]
            if name not in self.m:
                self.m[name] = np.zeros_like(g)
                self.v[name] = np.zeros_like(g)
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * (g * g)
            mhat = self.m[name] / bc1
            vhat = self.v[name] / bc2
            lr = getattr(self.lr, name)
            param = getattr(model, name)
            param -= lr * mhat / (np.sqrt(vhat) + self.eps)


def psnr(img: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((img - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def _l1_loss_and_grad(img: np.ndarray, target: np.ndarray):
    diff = img - target
    n = diff.size
    loss = float(np.abs(diff).sum()) / n
    grad_img = np.sign(diff) / n          # dL/d(image)
    return loss, grad_img


def _photometric_loss_and_grad(img: np.ndarray, target: np.ndarray,
                               ssim_weight: float):
    """3DGS loss ``(1−λ)·L1 + λ·(1−SSIM)`` and its gradient w.r.t. the image."""
    l1, g_l1 = _l1_loss_and_grad(img, target)
    if ssim_weight <= 0.0:
        return l1, g_l1
    ds, g_ds = dssim_loss_and_grad(img, target)
    loss = (1.0 - ssim_weight) * l1 + ssim_weight * ds
    grad = (1.0 - ssim_weight) * g_l1 + ssim_weight * g_ds
    return loss, grad


@dataclass
class FitResult:
    losses: List[float]
    psnrs: List[float]


def fit(
    model: GaussianModel,
    views: List[Tuple[Camera, np.ndarray]],
    iters: int = 200,
    lr: LearningRates | None = None,
    log_every: int = 0,
    ssim_weight: float = 0.0,
) -> FitResult:
    """Optimise ``model`` in place to reproduce ``views`` = [(camera, image)].

    ``ssim_weight`` (λ) blends the D-SSIM term into the photometric loss; 0 keeps
    the original pure-L1 behaviour, 0.2 matches the 3DGS paper. Returns the
    per-iteration photometric loss and mean PSNR across the views.
    """
    opt = _AdamState(lr or LearningRates())
    losses: List[float] = []
    psnrs: List[float] = []
    for it in range(iters):
        # Accumulate gradients over all views (mini-batch = every view).
        grad_acc = {f: np.zeros_like(getattr(model, f)) for f in _FIELDS}
        loss_sum = 0.0
        psnr_sum = 0.0
        for cam, target in views:
            img, cache = rasterize(model, cam)
            loss, grad_img = _photometric_loss_and_grad(img, target, ssim_weight)
            grads = rasterize_backward(grad_img, cache)
            for f in _FIELDS:
                grad_acc[f] += grads[f]
            loss_sum += loss
            psnr_sum += psnr(img, target)
        n = len(views)
        for f in _FIELDS:
            grad_acc[f] /= n
        opt.step(model, grad_acc)
        losses.append(loss_sum / n)
        psnrs.append(psnr_sum / n)
        if log_every and (it % log_every == 0 or it == iters - 1):
            print(f"[fit] it={it:4d}  L1={losses[-1]:.5f}  PSNR={psnrs[-1]:.2f} dB")
    return FitResult(losses, psnrs)
