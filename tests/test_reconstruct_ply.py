"""Colored-PLY export from reconstruct_tum.py (scripts/reconstruct_tum.py).

Verifies the back-projection + PLY writer round-trips through the viewer's reader
(`viz.scene_source.read_ply`) — point count, geometry, and crucially RGB channel
order (load_rgb is BGR, so a swap would tint the whole scene). No TUM data / GPU.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from mapping.collision_proxy import CameraIntrinsics          # noqa: E402
from viz.scene_source import read_ply                          # noqa: E402
from reconstruct_tum import backproject_colored, write_ply_xyzrgb  # noqa: E402


def _K():
    return CameraIntrinsics(fx=500.0, fy=500.0, cx=8.0, cy=6.0, width=16, height=12)


def test_backproject_colored_shapes_and_validity():
    K = _K()
    depth = np.full((12, 16), 2.0, np.float32)
    depth[0, 0] = 0.0                                          # one invalid pixel
    rgb = np.zeros((12, 16, 3), np.uint8)                      # BGR
    pts, colors = backproject_colored(depth, rgb, K, None, pixel_stride=1)
    assert pts.shape[0] == colors.shape[0] == 12 * 16 - 1      # invalid pixel dropped
    assert pts.dtype == np.float32 and colors.dtype == np.uint8
    assert np.isfinite(pts).all() and (pts[:, 2] > 0).all()


def test_colors_are_rgb_not_bgr():
    """load_rgb is BGR; the export must emit RGB. A pure-red desk must read red."""
    K = _K()
    depth = np.full((12, 16), 1.5, np.float32)
    rgb = np.zeros((12, 16, 3), np.uint8)
    rgb[..., 2] = 255                                          # BGR red channel
    _, colors = backproject_colored(depth, rgb, K, None, pixel_stride=1)
    assert (colors[:, 0] == 255).all()                        # R
    assert (colors[:, 1] == 0).all() and (colors[:, 2] == 0).all()


def test_write_ply_roundtrips_through_read_ply():
    K = _K()
    depth = np.full((12, 16), 2.0, np.float32)
    rgb = np.dstack([                                          # distinct per-channel
        np.full((12, 16), 10, np.uint8),                      # B
        np.full((12, 16), 128, np.uint8),                     # G
        np.full((12, 16), 240, np.uint8),                     # R
    ])
    pose = np.eye(4)
    pts, colors = backproject_colored(depth, rgb, K, pose, pixel_stride=2)

    with tempfile.TemporaryDirectory() as d:
        path = write_ply_xyzrgb(os.path.join(d, "cloud.ply"), pts, colors)
        snap = read_ply(path)

    assert snap.count == pts.shape[0]
    assert np.allclose(snap.means, pts, atol=1e-4)            # geometry preserved
    # colours come back as [0,1] RGB: R≈240/255, G≈128/255, B≈10/255
    assert np.allclose(snap.colors.mean(axis=0),
                       [240 / 255, 128 / 255, 10 / 255], atol=1e-2)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
