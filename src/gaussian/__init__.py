"""M5 — differentiable 3D Gaussian Splatting optimiser (pure-numpy reference).

Public API:
  GaussianModel            optimizable Gaussian parameters + activations
  Camera, rasterize        forward EWA splatting render
  rasterize_backward       analytic gradients (finite-difference verified)
  fit, LearningRates, psnr Adam optimisation of a model to posed views
"""

from .gaussian_model import GaussianModel, quat_to_rotmat, sigmoid
from .rasterizer import Camera, rasterize, rasterize_backward
from .optimizer import FitResult, LearningRates, fit, psnr

__all__ = [
    "GaussianModel", "quat_to_rotmat", "sigmoid",
    "Camera", "rasterize", "rasterize_backward",
    "FitResult", "LearningRates", "fit", "psnr",
]
