"""Geometry check for RGBDOdometry's pairwise (learned-matcher) branch.

A front-end exposing ``match_pair`` (e.g. SuperPoint+LightGlue) hands the
odometer corresponding pixels for a frame pair directly, instead of the ORB
detect+descriptor-match path. This test drives that branch with a *fake*
matcher whose correspondences come from a known camera translation, and checks
the recovered camera-to-world pose matches — no ONNX / GPU needed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mapping.collision_proxy import CameraIntrinsics  # noqa: E402
from slam.rgbd_odometry import RGBDOdometry  # noqa: E402


class _FakePairwiseFrontend:
    """Returns a fixed matched-pixel pair; presence of match_pair selects the
    pairwise branch in RGBDOdometry."""

    def __init__(self, uv0, uv1):
        self._uv0, self._uv1 = uv0, uv1

    def match_pair(self, rgb0, rgb1):
        return self._uv0, self._uv1


def _project(pts_cam, K):
    z = pts_cam[:, 2]
    u = K.fx * pts_cam[:, 0] / z + K.cx
    v = K.fy * pts_cam[:, 1] / z + K.cy
    return np.stack([u, v], axis=-1)


def test_pairwise_branch_recovers_known_translation():
    K = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                         width=640, height=480)
    rng = np.random.default_rng(0)

    # 3-D points in the previous camera frame.
    N = 200
    P0 = np.stack([
        rng.uniform(-1.0, 1.0, N),
        rng.uniform(-0.8, 0.8, N),
        rng.uniform(1.5, 3.0, N),
    ], axis=-1)

    # Camera moves +0.10 m along world +x → extrinsic (cam0->cam1) translation
    # is -0.10 x, so points shift in the new camera frame by that amount.
    tvec = np.array([-0.10, 0.0, 0.0])
    P1 = P0 + tvec

    uv0 = _project(P0, K)
    uv1 = _project(P1, K)

    # Keep correspondences that land inside the image in both views.
    inb = (
        (uv0[:, 0] >= 0) & (uv0[:, 0] < 640) & (uv0[:, 1] >= 0) & (uv0[:, 1] < 480)
        & (uv1[:, 0] >= 0) & (uv1[:, 0] < 640) & (uv1[:, 1] >= 0) & (uv1[:, 1] < 480)
    )
    uv0, uv1, P0 = uv0[inb], uv1[inb], P0[inb]

    # Depth map for the previous frame: z at each uv0 pixel.
    depth = np.zeros((480, 640), np.float32)
    for (u, v), z in zip(uv0, P0[:, 2]):
        depth[int(round(v)), int(round(u))] = z

    odo = RGBDOdometry(K, frontend=_FakePairwiseFrontend(uv0, uv1), min_matches=8)

    rgb = np.zeros((480, 640, 3), np.uint8)      # ignored by the fake matcher
    r0 = odo.track(rgb, depth)                   # seed frame → identity
    assert np.allclose(r0.pose, np.eye(4))

    r1 = odo.track(rgb, depth)                   # motion estimated here
    assert r1.ok
    assert r1.n_inliers >= 20
    # Recovered camera-to-world translation should be +0.10 m in x.
    assert np.allclose(r1.pose[:3, 3], [0.10, 0.0, 0.0], atol=0.01)
    assert np.allclose(r1.pose[:3, :3], np.eye(3), atol=1e-2)
