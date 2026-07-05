"""Forward-render tests for the M5 EWA splatting rasteriser."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gaussian.gaussian_model import GaussianModel, _logit
from gaussian.rasterizer import Camera, rasterize


def _one_gaussian(center=(0, 0, 0), scale=0.15, opacity=0.99, color=(1, 1, 1)):
    return GaussianModel(
        means=np.array([center], dtype=np.float64),
        log_scales=np.full((1, 3), np.log(scale)),
        quats=np.array([[1.0, 0.0, 0.0, 0.0]]),
        opacities=np.array([_logit(opacity)]).reshape(1),
        colors=_logit(np.array([color], dtype=np.float64)),
    )


def _cam(W=64, H=64):
    return Camera.look_at(eye=(0, 0, -3), target=(0, 0, 0),
                          fx=80.0, fy=80.0, width=W, height=H)


def test_single_gaussian_peaks_at_centre():
    img, _ = rasterize(_one_gaussian(), _cam())
    lum = img.mean(axis=2)
    py, px = np.unravel_index(np.argmax(lum), lum.shape)
    # Brightest pixel sits at the image centre (±2 px).
    assert abs(px - 32) <= 2 and abs(py - 32) <= 2
    assert lum.max() > 0.5


def test_render_is_symmetric():
    img, _ = rasterize(_one_gaussian(), _cam())
    lum = img.mean(axis=2)
    # Centre projects to exactly pixel (32, 32); the isotropic splat is
    # point-symmetric about it, so pixels equidistant along each axis match.
    cy = cx = 32
    for d in range(1, 10):
        assert lum[cy, cx - d] == pytest.approx(lum[cy, cx + d], abs=1e-9)
        assert lum[cy - d, cx] == pytest.approx(lum[cy + d, cx], abs=1e-9)


def test_front_gaussian_occludes_back():
    # Red splat in front (z=-0.3), blue behind (z=+0.3), both on the axis.
    red = _one_gaussian(center=(0, 0, -0.3), color=(1, 0, 0))
    blue = _one_gaussian(center=(0, 0, 0.3), color=(0, 0, 1))
    model = GaussianModel(
        means=np.vstack([red.means, blue.means]),
        log_scales=np.vstack([red.log_scales, blue.log_scales]),
        quats=np.vstack([red.quats, blue.quats]),
        opacities=np.concatenate([red.opacities, blue.opacities]),
        colors=np.vstack([red.colors, blue.colors]),
    )
    img, _ = rasterize(model, _cam())
    r, g, b = img[32, 32]
    assert r > b, f"front red should dominate back blue, got r={r:.3f} b={b:.3f}"


def test_all_behind_camera_returns_background():
    model = _one_gaussian(center=(0, 0, -10))  # behind eye at z=-3
    img, _ = rasterize(model, _cam(), bg=0.25)
    assert np.allclose(img, 0.25)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
