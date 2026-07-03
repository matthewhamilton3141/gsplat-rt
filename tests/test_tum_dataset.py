"""Tests for the TUM RGB-D loader (M6 data foundation).

Validates timestamp association, metric depth decoding, and that ground-truth
poses are well-formed SE(3). Requires an extracted fr1/desk sequence at
data/tum/rgbd_dataset_freiburg1_desk; skips cleanly if it is absent so the
suite still runs on machines without the dataset.

Run:
    pytest tests/test_tum_dataset.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.tum_dataset import TUMDataset, quaternion_to_matrix

_SEQ = os.path.join(
    os.path.dirname(__file__), "..", "data", "tum", "rgbd_dataset_freiburg1_desk"
)
_HAVE_SEQ = os.path.isdir(_SEQ)
_skip = pytest.mark.skipif(not _HAVE_SEQ, reason="fr1/desk sequence not downloaded")


def test_quaternion_identity_is_translation_only():
    T = quaternion_to_matrix(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-6)
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_quaternion_90deg_z():
    # 90 deg about +Z: x-axis -> y-axis
    T = quaternion_to_matrix(0, 0, 0, 0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4))
    assert np.allclose(T[:3, :3] @ np.array([1, 0, 0.0]), [0, 1, 0], atol=1e-6)


@_skip
def test_dataset_associates_most_frames():
    ds = TUMDataset(_SEQ)
    # fr1/desk has ~595 depth frames; nearest-timestamp pairing should keep most.
    assert len(ds) > 500, f"only associated {len(ds)} frames"


@_skip
def test_timestamps_monotonic():
    ds = TUMDataset(_SEQ)
    ts = [f.timestamp for f in ds]
    assert all(b > a for a, b in zip(ts, ts[1:])), "timestamps not strictly increasing"


@_skip
def test_depth_is_metric_and_plausible():
    ds = TUMDataset(_SEQ)
    depth = ds[0].load_depth()
    assert depth.dtype == np.float32
    assert depth.shape == (480, 640)
    valid = depth[depth > 0]
    assert valid.size > 0, "depth frame has no valid pixels"
    # A desk scene: nothing below ~0.3 m or beyond the Kinect's ~10 m range.
    assert 0.3 < valid.mean() < 10.0
    assert valid.max() < 12.0


@_skip
def test_rgb_shape():
    ds = TUMDataset(_SEQ)
    img = ds[0].load_rgb()
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8


@_skip
def test_poses_are_valid_se3():
    ds = TUMDataset(_SEQ)
    poses = ds.poses()
    assert poses.shape == (len(ds), 4, 4)
    for T in poses[::50]:
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-4), "rotation not orthonormal"
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-4), "rotation det != 1"
        assert np.allclose(T[3], [0, 0, 0, 1])


@_skip
def test_intrinsics_are_freiburg1():
    ds = TUMDataset(_SEQ)
    k = ds.intrinsics
    assert abs(k.fx - 517.306408) < 1e-3
    assert abs(k.cx - 318.643040) < 1e-3
    assert (k.width, k.height) == (640, 480)
