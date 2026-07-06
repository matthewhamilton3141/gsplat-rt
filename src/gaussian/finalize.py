"""Offline 'finalize' stage (M5 → pipeline): turn a captured point cloud +
keyframe views into an optimized Gaussian scene.

The live pipeline accumulates Gaussian *centres* (a point cloud) in the world
frame and stashes a few RGB keyframes with their camera poses. This module is
the bridge: it seeds Gaussians at those points and runs the differentiable
optimiser (`optimizer.fit`) against the keyframes, recovering per-Gaussian
colour/opacity/shape that the real-time hot path never had budget to solve.

Pure and pipeline-free so it unit-tests in isolation; `PipelineManager` only
assembles the inputs and calls `finalize_gaussians`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .gaussian_model import GaussianModel
from .optimizer import FitResult, LearningRates, fit
from .rasterizer import Camera

# Spherical-harmonics DC normalisation constant (INRIA 3DGS convention).
_SH_C0 = 0.28209479177387814


def pose_to_camera(pose_cw: Optional[np.ndarray], fx: float, fy: float,
                   width: int, height: int, near: float = 0.05) -> Camera:
    """Convert a camera-to-world 4x4 pose into a rasteriser Camera (world->cam).

    ``pose_cw`` maps camera points to world (world = R_cw @ cam + t_cw); the
    rasteriser wants the inverse (cam = R @ world + t). ``None`` → identity,
    matching the pipeline's fixed-camera default where points stay camera-frame.
    """
    if pose_cw is None:
        R = np.eye(3)
        t = np.zeros(3)
    else:
        R_cw = np.asarray(pose_cw[:3, :3], dtype=np.float64)
        t_cw = np.asarray(pose_cw[:3, 3], dtype=np.float64)
        R = R_cw.T
        t = -R_cw.T @ t_cw
    return Camera(R, t, fx, fy, width / 2.0, height / 2.0, width, height, near)


def finalize_gaussians(
    points: np.ndarray,
    views: List[Tuple[Camera, np.ndarray]],
    max_points: int = 2000,
    iters: int = 150,
    lr: LearningRates | None = None,
    init_scale: float = 0.05,
    init_opacity: float = 0.1,
    seed: int = 0,
    ssim_weight: float = 0.0,
    densify_config=None,
) -> Tuple[GaussianModel, FitResult]:
    """Seed Gaussians at ``points`` and optimise them to reproduce ``views``.

    Points are randomly subsampled to ``max_points`` to keep the CPU fit
    tractable. Returns the optimized model and the fit history.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[sel]
    model = GaussianModel.from_points(pts, init_scale=init_scale,
                                      init_opacity=init_opacity)
    densifier = None
    if densify_config is not None:
        from .densify import DensificationController
        densifier = DensificationController(densify_config)
    result = fit(model, views, iters=iters, lr=lr, ssim_weight=ssim_weight,
                 densifier=densifier)
    return model, result


def write_ply(model: GaussianModel, path: str) -> None:
    """Write the optimized Gaussians as a 3DGS .ply (INRIA field layout).

    Fields: x y z, f_dc_0..2 (SH DC colour), opacity (logit), scale_0..2 (log),
    rot_0..3 (quaternion). Readable by standard 3D Gaussian Splatting viewers.
    """
    n = model.num_gaussians
    xyz = model.means.astype(np.float32)
    # RGB -> SH DC term: rgb = SH_C0 * f_dc + 0.5.
    f_dc = ((model.rgb - 0.5) / _SH_C0).astype(np.float32)
    opacity = model.opacities.astype(np.float32).reshape(n, 1)   # logit (raw)
    scale = model.log_scales.astype(np.float32)                  # log (raw)
    quat = (model.quats / (np.linalg.norm(model.quats, axis=1, keepdims=True) + 1e-12)
            ).astype(np.float32)

    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {p}" for p in props]
    header.append("end_header")

    data = np.concatenate([xyz, f_dc, opacity, scale, quat], axis=1).astype("<f4")
    with open(path, "wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(data.tobytes())


def sh_dc_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """RGB in [0,1] -> degree-0 SH coefficients (for USD sh_coeffs export)."""
    return ((np.asarray(rgb, dtype=np.float32) - 0.5) / _SH_C0)
