"""Tests for the offline finalize bridge (point cloud + keyframes -> splats)."""

import os
import struct
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gaussian.gaussian_model import GaussianModel, _logit
from gaussian.finalize import (finalize_gaussians, pose_to_camera, write_ply,
                               sh_dc_from_rgb)
from gaussian.rasterizer import Camera, rasterize
from gaussian.optimizer import psnr


def _truth():
    means = np.array([[-0.12, 0.06, 0.02], [0.14, -0.09, 0.2], [0.03, 0.18, -0.15]])
    log_scales = np.log(np.full((3, 3), 0.11))
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    opacities = _logit(np.array([0.7, 0.65, 0.75]))
    colors = _logit(np.array([[0.85, 0.2, 0.25], [0.2, 0.75, 0.35], [0.4, 0.45, 0.9]]))
    return GaussianModel(means, log_scales, quats, opacities, colors)


def _views(model, n=3, res=48):
    eyes = [(0, 0, -3), (0.6, 0.2, -2.9), (-0.5, -0.3, -2.9)]
    out = []
    for i in range(n):
        cam = Camera.look_at(eye=eyes[i], target=(0, 0, 0),
                             fx=1.4 * res, fy=1.4 * res, width=res, height=res)
        out.append((cam, rasterize(model, cam)[0]))
    return out


def test_finalize_improves_psnr():
    truth = _truth()
    views = _views(truth)
    # Seed from the true centres (grey/low-opacity); optimiser recovers appearance.
    model, res = finalize_gaussians(truth.means, views, iters=200, max_points=2000)
    start = res.psnrs[0]
    assert res.psnrs[-1] > start + 5.0, (start, res.psnrs[-1])
    assert res.psnrs[-1] > 25.0


def test_finalize_subsamples_points():
    truth = _truth()
    views = _views(truth)
    pts = np.repeat(truth.means, 2000, axis=0)      # 6000 points
    model, _ = finalize_gaussians(pts, views, iters=1, max_points=500)
    assert model.num_gaussians == 500


def test_pose_to_camera_roundtrip():
    # A look_at camera defines world->cam (R, t); build the inverse camera->world
    # pose and check pose_to_camera reconstructs the original R, t.
    ref = Camera.look_at(eye=(0.4, -0.2, -2.8), target=(0, 0, 0),
                         fx=100, fy=100, width=64, height=64)
    R_cw = ref.R.T
    t_cw = -ref.R.T @ ref.t
    pose = np.eye(4)
    pose[:3, :3] = R_cw
    pose[:3, 3] = t_cw
    cam = pose_to_camera(pose, ref.fx, ref.fy, ref.width, ref.height)
    assert np.allclose(cam.R, ref.R, atol=1e-9)
    assert np.allclose(cam.t, ref.t, atol=1e-9)


def test_pose_to_camera_none_is_identity():
    cam = pose_to_camera(None, 50, 50, 32, 32)
    assert np.allclose(cam.R, np.eye(3)) and np.allclose(cam.t, 0)


def test_write_ply_roundtrip():
    truth = _truth()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scene.ply")
        write_ply(truth, path)
        with open(path, "rb") as fh:
            raw = fh.read()
        header_end = raw.index(b"end_header\n") + len(b"end_header\n")
        header = raw[:header_end].decode("ascii")
        assert "element vertex 3" in header
        assert header.count("property float") == 14
        body = np.frombuffer(raw[header_end:], dtype="<f4").reshape(3, 14)
        # First three columns are xyz — must match the model means.
        assert np.allclose(body[:, :3], truth.means.astype(np.float32), atol=1e-6)
        # DC colour column round-trips back to rgb via SH_C0.
        rgb_back = 0.28209479177387814 * body[:, 3:6] + 0.5
        assert np.allclose(rgb_back, truth.rgb, atol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
