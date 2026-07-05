"""Optimizable 3D Gaussian model (M5).

A scene is a set of N anisotropic 3D Gaussians, each parameterised by

    means      (N, 3)  world-space centre  µ
    log_scales (N, 3)  log of the per-axis stddev; scale = exp(log_scale)
    quats      (N, 4)  rotation as a (w, x, y, z) quaternion (un-normalised)
    opacities  (N,)    logit of alpha; alpha = sigmoid(opacity)
    colors     (N, 3)  view-independent RGB, stored as logits; rgb = sigmoid(color)

The *raw* parameters are stored unconstrained so an unconstrained optimiser
(Adam) can move them freely; the activations below map them into the valid
ranges the rasteriser expects. This mirrors the reference 3DGS parameterisation
(Kerbl et al. 2023) minus spherical harmonics — flat RGB keeps the first CPU
build tractable; SH bands are a later upgrade.

Pure numpy on purpose: this box has no torch (see project notes), and the whole
pipeline must stay runnable / testable GPU-free. The analytic gradients live in
``rasterizer.py``; this module only holds state + activations and their
derivatives (needed to backprop from raw params to activated values).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable logistic sigmoid.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def quat_to_rotmat(quats: np.ndarray) -> np.ndarray:
    """(N, 4) (w, x, y, z) quaternions -> (N, 3, 3) rotation matrices.

    Quaternions are normalised internally, so callers may keep them
    un-normalised in the raw parameter vector.
    """
    q = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@dataclass
class GaussianModel:
    """Container of raw (unconstrained) Gaussian parameters."""

    means: np.ndarray       # (N, 3)
    log_scales: np.ndarray  # (N, 3)
    quats: np.ndarray       # (N, 4)
    opacities: np.ndarray   # (N,)
    colors: np.ndarray      # (N, 3)

    def __post_init__(self) -> None:
        self.means = np.ascontiguousarray(self.means, dtype=np.float64)
        self.log_scales = np.ascontiguousarray(self.log_scales, dtype=np.float64)
        self.quats = np.ascontiguousarray(self.quats, dtype=np.float64)
        self.opacities = np.ascontiguousarray(self.opacities, dtype=np.float64)
        self.colors = np.ascontiguousarray(self.colors, dtype=np.float64)
        n = self.means.shape[0]
        assert self.log_scales.shape == (n, 3), self.log_scales.shape
        assert self.quats.shape == (n, 4), self.quats.shape
        assert self.opacities.shape == (n,), self.opacities.shape
        assert self.colors.shape == (n, 3), self.colors.shape

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    # --- activations (raw params -> values the rasteriser consumes) ---------
    @property
    def scales(self) -> np.ndarray:
        return np.exp(self.log_scales)

    @property
    def alphas(self) -> np.ndarray:
        return sigmoid(self.opacities)

    @property
    def rgb(self) -> np.ndarray:
        return sigmoid(self.colors)

    @property
    def rotmats(self) -> np.ndarray:
        return quat_to_rotmat(self.quats)

    def covariance3d(self) -> np.ndarray:
        """World-space 3x3 covariance Σ = R S Sᵀ Rᵀ for each Gaussian."""
        R = self.rotmats                      # (N, 3, 3)
        s = self.scales                       # (N, 3)
        M = R * s[:, None, :]                  # R @ diag(s), broadcast columns
        return M @ np.transpose(M, (0, 2, 1))  # (N, 3, 3)

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        init_scale: float = 0.02,
        init_opacity: float = 0.1,
    ) -> "GaussianModel":
        """Initialise isotropic Gaussians at a point cloud (pipeline output).

        points  : (N, 3) world-space centres.
        colors  : (N, 3) RGB in [0, 1]; defaults to mid-grey.
        """
        points = np.asarray(points, dtype=np.float64)
        n = points.shape[0]
        log_scales = np.full((n, 3), np.log(init_scale), dtype=np.float64)
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        opacities = np.full((n,), _logit(init_opacity), dtype=np.float64)
        if colors is None:
            rgb = np.full((n, 3), 0.5, dtype=np.float64)
        else:
            rgb = np.clip(np.asarray(colors, dtype=np.float64), 1e-4, 1 - 1e-4)
        return cls(points, log_scales, quats, opacities, _logit(rgb))


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))
