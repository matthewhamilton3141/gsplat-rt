"""Two-view geometry for the monocular scale anchor (src/slam/monocular_scale.py).

Tests the geometry core with synthetic correspondences (needs cv2, no GPU). The
ORB front-end is exercised implicitly by the pipeline; here we verify the pose
recovery + triangulation math that turns matches into a metric-consistent
reference.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

cv2 = pytest.importorskip("cv2")

from slam.monocular_scale import estimate_relative_pose  # noqa: E402
from depth.metric_scale import triangulate_two_view      # noqa: E402


def _K():
    return np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _project(K, R, t, X):
    Xc = (R @ X.T).T + t
    uvw = (K @ Xc.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def _scene(rng, n=200):
    X = np.column_stack([
        rng.uniform(-3, 3, n),
        rng.uniform(-2, 2, n),
        rng.uniform(3, 9, n),
    ])
    ang = 0.08
    R = np.array([[np.cos(ang), 0, np.sin(ang)],
                  [0, 1, 0],
                  [-np.sin(ang), 0, np.cos(ang)]])
    t = np.array([0.7, 0.05, 0.1])
    return X, R, t


def test_estimate_relative_pose_recovers_rotation_and_direction():
    rng = np.random.default_rng(0)
    K = _K()
    X, R, t = _scene(rng)
    uv_a = _project(K, np.eye(3), np.zeros(3), X)
    uv_b = _project(K, R, t, X)

    out = estimate_relative_pose(uv_a, uv_b, K)
    assert out is not None
    R_est, t_unit, inliers = out

    assert np.allclose(R_est, R, atol=1e-2)
    # Translation recovered up to scale + sign; compare unit directions.
    t_dir = t / np.linalg.norm(t)
    cos = abs(float(t_unit @ t_dir))
    assert cos > 0.99
    assert inliers.sum() >= 100


def test_pose_plus_triangulation_recovers_depth_up_to_baseline():
    rng = np.random.default_rng(1)
    K = _K()
    X, R, t = _scene(rng)
    uv_a = _project(K, np.eye(3), np.zeros(3), X)
    uv_b = _project(K, R, t, X)

    R_est, t_unit, inliers = estimate_relative_pose(uv_a, uv_b, K)
    # Scale the recovered unit translation back to the TRUE baseline, then the
    # triangulated depths must match ground truth.
    sign = np.sign(t_unit @ (t / np.linalg.norm(t)))
    t_metric = sign * np.linalg.norm(t) * t_unit
    pts, valid = triangulate_two_view(uv_a[inliers], uv_b[inliers], K, R_est, t_metric)
    z_true = X[inliers, 2]
    assert np.allclose(pts[valid, 2], z_true[valid], rtol=1e-2, atol=1e-2)


def test_estimate_relative_pose_rejects_too_few_points():
    K = _K()
    pts = np.random.default_rng(2).uniform(0, 640, size=(4, 2))
    assert estimate_relative_pose(pts, pts, K) is None
