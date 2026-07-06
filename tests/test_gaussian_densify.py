"""Tests for Adaptive Density Control (src/gaussian/densify.py).

Mechanics (clone / split / prune + Adam-state bookkeeping) are checked
deterministically with hand-crafted gradients; a final integration test runs the
full fit loop with densification and shows the Gaussian count grow while the loss
falls. Pure numpy.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gaussian.densify import DensifyConfig, DensificationController  # noqa: E402
from gaussian.gaussian_model import GaussianModel, _logit  # noqa: E402
from gaussian.optimizer import _AdamState, LearningRates, fit, psnr  # noqa: E402
from gaussian.rasterizer import Camera, rasterize  # noqa: E402


def _model(scales, opacities=None, n=None):
    scales = np.asarray(scales, dtype=float)
    n = scales.shape[0]
    means = np.zeros((n, 3))
    means[:, 0] = np.arange(n)                       # distinct positions
    log_scales = np.log(scales)
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    op = np.full(n, _logit(0.5)) if opacities is None else _logit(np.asarray(opacities))
    colors = np.zeros((n, 3))
    return GaussianModel(means, log_scales, quats, op, colors)


def _primed_opt(model):
    """An Adam state with moments initialised (so rebuild has something to move)."""
    opt = _AdamState(LearningRates())
    from gaussian.optimizer import _FIELDS
    opt.step(model, {f: np.zeros_like(getattr(model, f)) for f in _FIELDS})
    # Tag means-moment rows with identifiable values to check preservation.
    opt.m["means"] = np.arange(model.num_gaussians * 3).reshape(-1, 3).astype(float)
    opt.v["means"] = opt.m["means"].copy()
    return opt


def _fire(controller, model, opt, viewpos_norm, means_grad=None):
    """track() one iteration then step() at an interval boundary (it=0, interval=1)."""
    n = model.num_gaussians
    viewpos = np.zeros((n, 2))
    viewpos[:, 0] = viewpos_norm
    visible = np.ones(n)
    mg = np.zeros((n, 3)) if means_grad is None else means_grad
    controller.track(mg, viewpos, visible)
    return controller.step(model, opt, it=0)


def test_clone_small_high_gradient_gaussian():
    # Two small Gaussians; only #0 has a high positional gradient → clone it.
    model = _model([[0.02, 0.02, 0.02], [0.02, 0.02, 0.02]])
    opt = _primed_opt(model)
    cfg = DensifyConfig(grad_threshold=0.1, scale_split_threshold=0.05,
                        densify_interval=1)
    ctl = DensificationController(cfg)
    stats = _fire(ctl, model, opt, viewpos_norm=[1.0, 0.0])
    assert stats["n_clone"] == 1
    assert stats["n_split"] == 0
    assert model.num_gaussians == 3                  # 2 survivors + 1 clone
    # Adam moments resized and survivor #0/#1 preserved, clone row zeroed.
    assert opt.m["means"].shape == (3, 3)
    assert np.allclose(opt.m["means"][0], [0, 1, 2])
    assert np.allclose(opt.m["means"][1], [3, 4, 5])
    assert np.allclose(opt.m["means"][2], 0.0)       # fresh clone → zero moment


def test_split_large_high_gradient_gaussian():
    # One large Gaussian, high gradient → split into split_n children, parent gone.
    model = _model([[0.3, 0.3, 0.3]])
    opt = _primed_opt(model)
    cfg = DensifyConfig(grad_threshold=0.1, scale_split_threshold=0.05,
                        densify_interval=1, split_n=2, split_scale_div=1.6)
    ctl = DensificationController(cfg)
    stats = _fire(ctl, model, opt, viewpos_norm=[1.0])
    assert stats["n_split"] == 1
    assert stats["n_clone"] == 0
    assert model.num_gaussians == 2                  # 0 survivors + 2 children
    # Children are smaller than the parent by the division factor.
    assert np.allclose(model.scales, 0.3 / 1.6, rtol=1e-6)
    assert opt.m["means"].shape == (2, 3)
    assert np.allclose(opt.m["means"], 0.0)          # both children are fresh


def test_prune_transparent_gaussian():
    # #1 is nearly transparent → pruned; low gradient so nothing densifies.
    model = _model([[0.02, 0.02, 0.02], [0.02, 0.02, 0.02]],
                   opacities=[0.6, 1e-4])
    opt = _primed_opt(model)
    cfg = DensifyConfig(grad_threshold=10.0, min_opacity=0.005, densify_interval=1)
    ctl = DensificationController(cfg)
    stats = _fire(ctl, model, opt, viewpos_norm=[0.0, 0.0])
    assert stats["n_pruned"] == 1
    assert model.num_gaussians == 1
    assert np.allclose(opt.m["means"][0], [0, 1, 2])  # survivor #0 preserved


def test_below_threshold_does_nothing():
    model = _model([[0.02, 0.02, 0.02], [0.02, 0.02, 0.02]])
    opt = _primed_opt(model)
    cfg = DensifyConfig(grad_threshold=100.0, densify_interval=1)
    ctl = DensificationController(cfg)
    stats = _fire(ctl, model, opt, viewpos_norm=[1.0, 1.0])
    assert stats["n_clone"] == 0 and stats["n_split"] == 0 and stats["n_pruned"] == 0
    assert model.num_gaussians == 2


def test_max_gaussians_cap_suppresses_densify():
    model = _model([[0.02, 0.02, 0.02], [0.02, 0.02, 0.02]])
    opt = _primed_opt(model)
    cfg = DensifyConfig(grad_threshold=0.1, densify_interval=1, max_gaussians=2)
    ctl = DensificationController(cfg)
    stats = _fire(ctl, model, opt, viewpos_norm=[1.0, 1.0])
    assert stats["n_clone"] == 0 and stats["n_split"] == 0   # at cap → prune only
    assert model.num_gaussians == 2


def test_scheduling_only_fires_on_interval():
    model = _model([[0.02, 0.02, 0.02]])
    opt = _primed_opt(model)
    ctl = DensificationController(DensifyConfig(densify_interval=10))
    ctl.track(np.zeros((1, 3)), np.ones((1, 2)), np.ones(1))
    assert ctl.step(model, opt, it=3) is None         # not a boundary
    assert ctl.step(model, opt, it=8) is None
    assert ctl.step(model, opt, it=9) is not None      # (9+1) % 10 == 0


# ---------------------------------------------------------------------------
# End-to-end: densification grows the model during a real fit
# ---------------------------------------------------------------------------

def _truth():
    means = np.array([[-0.12, 0.06, 0.02], [0.14, -0.09, 0.2], [0.03, 0.18, -0.15]])
    log_scales = np.log(np.array([[0.12, 0.09, 0.10],
                                  [0.10, 0.13, 0.08],
                                  [0.11, 0.10, 0.12]]))
    quats = np.array([[1.0, 0.1, -0.05, 0.02], [0.95, -0.15, 0.1, 0.2],
                      [1.0, 0.0, 0.15, -0.05]])
    opacities = _logit(np.array([0.7, 0.65, 0.75]))
    colors = _logit(np.array([[0.85, 0.2, 0.25], [0.2, 0.75, 0.35],
                              [0.4, 0.45, 0.9]]))
    return GaussianModel(means, log_scales, quats, opacities, colors)


def _views(model):
    eyes = [(0, 0, -3), (0.6, 0.2, -2.9), (-0.5, -0.3, -2.9)]
    views = []
    for e in eyes:
        cam = Camera.look_at(eye=e, target=(0, 0, 0), fx=90.0, fy=90.0,
                             width=48, height=48)
        views.append((cam, rasterize(model, cam)[0]))
    return views


def test_densification_grows_model_and_reduces_loss():
    views = _views(_truth())
    # Start from a single Gaussian — far too few to reconstruct three targets.
    init = GaussianModel(
        means=np.array([[0.0, 0.0, 0.0]]),
        log_scales=np.log(np.array([[0.15, 0.15, 0.15]])),
        quats=np.array([[1.0, 0.0, 0.0, 0.0]]),
        opacities=_logit(np.array([0.5])),
        colors=_logit(np.array([[0.5, 0.5, 0.5]])),
    )
    cfg = DensifyConfig(grad_threshold=1e-8, scale_split_threshold=0.1,
                        densify_interval=15, stop_iter=70, min_opacity=0.005,
                        max_gaussians=80, seed=0)
    ctl = DensificationController(cfg)

    res = fit(init, views, iters=90, lr=LearningRates(), ssim_weight=0.0,
              densifier=ctl)

    # The fit completing without a shape mismatch proves the model + Adam state
    # stayed in lock-step through every clone/split/prune.
    assert init.num_gaussians > 1, "densification did not add Gaussians"
    assert init.num_gaussians <= cfg.max_gaussians
    assert res.losses[-1] < res.losses[0], (res.losses[0], res.losses[-1])
    assert np.isfinite(res.losses[-1])


def test_finalize_gaussians_threads_densify_config():
    from gaussian.finalize import finalize_gaussians

    views = _views(_truth())
    points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    cfg = DensifyConfig(grad_threshold=1e-8, densify_interval=10, stop_iter=30,
                        max_gaussians=60)
    model, result = finalize_gaussians(points, views, iters=40,
                                       densify_config=cfg)
    assert model.num_gaussians >= 2
    assert result.losses[-1] < result.losses[0]
