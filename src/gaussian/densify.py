"""Adaptive Density Control (ADC) for the M5 Gaussian optimiser.

The signature mechanism of 3D Gaussian Splatting (Kerbl et al. 2023, §5): the
initial point cloud is too sparse to reconstruct fine detail, so during
optimisation Gaussians in under-/over-reconstructed regions are **densified** and
transparent ones are **pruned**. The trigger is the average magnitude of the
view-space (projected-pixel) position gradient — large where the render can't
place enough splats:

- **clone**  — a *small* Gaussian with a large positional gradient is duplicated
  and nudged along the loss-descent direction (grow into empty space);
- **split**  — a *large* Gaussian with a large positional gradient is replaced by
  ``split_n`` smaller children (÷``split_scale_div``) sampled from its own 3-D
  distribution (add detail where one splat is too coarse);
- **prune**  — Gaussians whose opacity falls below ``min_opacity`` (and,
  optionally, world-space giants) are removed.

The controller consumes the ``viewpos``/``visible`` signals now returned by
``rasterize_backward`` and mutates both the :class:`GaussianModel` and the
optimiser's Adam moments in lock-step (persisting Gaussians keep their momentum;
fresh children start at zero). Pure numpy — same CPU reference / A10G-port story
as the rest of M5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .gaussian_model import GaussianModel, quat_to_rotmat, sigmoid

_FIELDS = ("means", "log_scales", "quats", "opacities", "colors")


@dataclass
class DensifyConfig:
    grad_threshold: float = 0.0002
    """Average view-space position-gradient above which a Gaussian densifies."""

    scale_split_threshold: float = 0.05
    """World-space max-axis scale separating clone (≤) from split (>)."""

    min_opacity: float = 0.005
    """Prune Gaussians whose alpha = sigmoid(opacity) falls below this."""

    max_world_scale: Optional[float] = None
    """Also prune Gaussians whose world max-axis scale exceeds this (or None)."""

    split_n: int = 2
    """Children produced per split."""

    split_scale_div: float = 1.6
    """Child scale = parent scale / this (the paper's φ = 1.6)."""

    clone_offset: float = 1.0
    """Clone displacement as a fraction of the parent scale, along −∇means."""

    densify_interval: int = 50
    """Run densify/prune every this many iterations."""

    start_iter: int = 0
    """Do not densify before this iteration (warm-up)."""

    stop_iter: Optional[int] = None
    """Stop densifying after this iteration (None → never stop)."""

    max_gaussians: int = 100_000
    """Hard cap; at/above it, clone/split are suppressed (prune still runs)."""

    seed: int = 0


class DensificationController:
    """Accumulates the densify signal and applies ADC at intervals.

    Usage is driven by ``optimizer.fit(..., densifier=controller)``: each
    iteration it :meth:`track`\\ s the averaged gradients, then :meth:`step`
    performs clone/split/prune on interval boundaries.
    """

    def __init__(self, cfg: Optional[DensifyConfig] = None):
        self.cfg = cfg or DensifyConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.pos_accum: Optional[np.ndarray] = None   # Σ‖view-pos grad‖   (N,)
        self.denom: Optional[np.ndarray] = None       # #iters visible     (N,)
        self.move_accum: Optional[np.ndarray] = None  # Σ ∇means           (N,3)
        self.last_stats: dict = {}

    # -- accumulation --------------------------------------------------------

    def reset_accumulators(self, n: int) -> None:
        self.pos_accum = np.zeros(n)
        self.denom = np.zeros(n)
        self.move_accum = np.zeros((n, 3))

    def _ensure(self, n: int) -> None:
        if self.pos_accum is None or self.pos_accum.shape[0] != n:
            self.reset_accumulators(n)

    def track(self, means_grad: np.ndarray, viewpos_grad: np.ndarray,
              visible: np.ndarray) -> None:
        """Fold one iteration's gradients into the running densify signal."""
        n = viewpos_grad.shape[0]
        self._ensure(n)
        self.pos_accum += np.linalg.norm(viewpos_grad, axis=1)
        self.denom += (visible > 0)
        self.move_accum += means_grad

    def avg_grad(self) -> np.ndarray:
        """Average view-space position-gradient magnitude per Gaussian."""
        return self.pos_accum / np.maximum(self.denom, 1.0)

    # -- scheduling ----------------------------------------------------------

    def should_densify(self, it: int) -> bool:
        c = self.cfg
        if (it + 1) % c.densify_interval != 0:
            return False
        if it < c.start_iter:
            return False
        if c.stop_iter is not None and it > c.stop_iter:
            return False
        return self.denom is not None and self.denom.sum() > 0

    def step(self, model: GaussianModel, opt, it: int) -> Optional[dict]:
        """Densify + prune if this iteration is a boundary. Returns stats or None."""
        if not self.should_densify(it):
            return None
        stats = self._densify_and_prune(model, opt)
        self.reset_accumulators(model.num_gaussians)
        self.last_stats = stats
        return stats

    # -- the actual ADC operation -------------------------------------------

    def _densify_and_prune(self, model: GaussianModel, opt) -> dict:
        c = self.cfg
        n0 = model.num_gaussians
        avg = self.avg_grad()
        max_scale = model.scales.max(axis=1)

        sel = avg > c.grad_threshold
        if n0 >= c.max_gaussians:
            sel = np.zeros(n0, dtype=bool)             # at cap: prune only
        split_sel = sel & (max_scale > c.scale_split_threshold)
        clone_sel = sel & ~split_sel

        # Loss-descent direction for clone displacement (−∇means).
        move = -self.move_accum
        move /= (np.linalg.norm(move, axis=1, keepdims=True) + 1e-12)

        survivors = np.where(~split_sel)[0]            # split parents are removed
        clone_idx = np.where(clone_sel)[0]
        split_idx = np.where(split_sel)[0]

        blocks_params = {f: [getattr(model, f)[survivors]] for f in _FIELDS}
        src_blocks = [survivors.astype(np.int64)]

        # --- clones: copy + nudge along the descent direction ---------------
        if clone_idx.size:
            off = (c.clone_offset * model.scales[clone_idx].mean(axis=1, keepdims=True)
                   * move[clone_idx])
            jitter = 0.1 * model.scales[clone_idx].min(axis=1, keepdims=True) * \
                self._rng.standard_normal((clone_idx.size, 3))
            blocks_params["means"].append(model.means[clone_idx] + off + jitter)
            for f in ("log_scales", "quats", "opacities", "colors"):
                blocks_params[f].append(getattr(model, f)[clone_idx].copy())
            src_blocks.append(np.full(clone_idx.size, -1, dtype=np.int64))

        # --- splits: sample children from the parent's 3-D distribution -----
        if split_idx.size:
            k = c.split_n
            parents = np.repeat(split_idx, k)
            R = quat_to_rotmat(model.quats[parents])              # (P·k, 3, 3)
            s = model.scales[parents]                             # (P·k, 3)
            noise = self._rng.standard_normal((parents.size, 3))
            local = noise * s                                     # ~ N(0, diag(s²))
            world = np.einsum("nij,nj->ni", R, local)
            child_means = model.means[parents] + world
            child_logscales = (model.log_scales[parents]
                               - np.log(c.split_scale_div))
            blocks_params["means"].append(child_means)
            blocks_params["log_scales"].append(child_logscales)
            for f in ("quats", "opacities", "colors"):
                blocks_params[f].append(getattr(model, f)[parents].copy())
            src_blocks.append(np.full(parents.size, -1, dtype=np.int64))

        params = {f: np.concatenate(blocks_params[f], axis=0) for f in _FIELDS}
        src_index = np.concatenate(src_blocks, axis=0)

        # --- prune: transparent (and optionally world-space giant) ----------
        alpha = sigmoid(params["opacities"])
        keep = alpha >= c.min_opacity
        if c.max_world_scale is not None:
            keep &= np.exp(params["log_scales"]).max(axis=1) <= c.max_world_scale
        n_pruned = int((~keep).sum())
        for f in _FIELDS:
            params[f] = params[f][keep]
        src_index = src_index[keep]

        # --- commit to model + Adam state -----------------------------------
        n_final = params["means"].shape[0]
        model.means = np.ascontiguousarray(params["means"])
        model.log_scales = np.ascontiguousarray(params["log_scales"])
        model.quats = np.ascontiguousarray(params["quats"])
        model.opacities = np.ascontiguousarray(params["opacities"])
        model.colors = np.ascontiguousarray(params["colors"])
        opt.rebuild(src_index, n_final)

        return dict(n_before=n0, n_after=n_final, n_clone=int(clone_idx.size),
                    n_split=int(split_idx.size), n_pruned=n_pruned)
