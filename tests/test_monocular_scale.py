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

from slam.monocular_scale import (  # noqa: E402
    MonocularScaleReference,
    estimate_relative_pose,
)
from depth.metric_scale import triangulate_two_view      # noqa: E402

try:
    from mapping.collision_proxy import CameraIntrinsics
except ImportError:
    from src.mapping.collision_proxy import CameraIntrinsics


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


def _proj_cam(K, Xcam):
    uvw = (K @ Xcam.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def test_monocular_reference_global_scale_consistency():
    """Cross-frame propagation: a camera moving with VARYING per-step baselines
    still yields depths that are the true depth times ONE global constant at
    every frame — i.e. no scale drift. The _geometry_step core is driven with
    synthetic correspondences (identity indices link landmarks across frames)."""
    rng = np.random.default_rng(40)
    K = _K()                                            # fx=fy=600, c=(320,240)
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0,
                            width=640, height=480)
    ref = MonocularScaleReference(intr, min_matches=6, min_shared=6, anchor=1.0)

    n = 90
    Xw = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-2, 2, n),
                          rng.uniform(4, 10, n)])       # world points, R=I cameras
    # Camera centres with DIFFERENT step sizes each pair (varying baseline).
    centres = [np.array([0.0, 0.0, 0.0]),
               np.array([0.5, 0.0, 0.2]),               # baseline 0.539
               np.array([1.7, 0.0, -0.1]),              # baseline 1.237
               np.array([2.0, 0.0, 0.3])]               # baseline 0.500
    ids = np.arange(n)
    depth_map = np.ones((480, 640))

    baseline_01 = float(np.linalg.norm(centres[1] - centres[0]))
    expected_const = 1.0 / baseline_01                  # global gauge from anchor=1

    for i in range(len(centres) - 1):
        Xa = Xw - centres[i]                            # cam-A coords (R=I)
        Xb = Xw - centres[i + 1]
        uv_a = _proj_cam(K, Xa)
        uv_b = _proj_cam(K, Xb)
        out = ref._geometry_step(uv_a, uv_b, ids, ids, depth_map)
        assert out is not None
        _, metric_cur = out
        assert len(metric_cur) == n                     # clean data → all inliers
        z_true_cur = Xb[:, 2]
        ratio = metric_cur / z_true_cur
        # Internally consistent across landmarks *and* equal to the global const
        # every frame despite the baseline changing — that's the drift-free claim.
        assert np.allclose(ratio, expected_const, rtol=1e-3), (
            f"frame {i + 1}: ratio spread "
            f"{ratio.min():.4f}..{ratio.max():.4f}, expected {expected_const:.4f}")

    assert ref.baseline is not None


def test_geometry_step_equal_length_under_cheirality_drop(monkeypatch):
    """Regression: pred_values and metric_cur must stay equal-length even when
    triangulation drops some inliers as cheirality failures.

    The bug: pred_values was sampled at ALL inlier pixels (uv_b) while metric_cur
    was the valid-only subset, so the two arrays diverged whenever a point failed
    the in-front-of-both-cameras check. DepthScaleAligner.fit then raised
    'pred_values and ref_depth must have the same length' and the live run
    silently skipped the scale update (the 'scale fit failed on frame N' errors).

    Cheirality drops are hard to force through real geometry (RANSAC + robust
    triangulation reject the very points that would flip), so we drive the exact
    condition directly: a pose with all matches as inliers, and a triangulation
    whose `valid` mask drops a few. Only the fixed indexing keeps the outputs
    aligned.
    """
    from slam import monocular_scale as m

    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0,
                            width=640, height=480)
    ref = MonocularScaleReference(intr, min_matches=6, min_shared=0, anchor=1.0)

    n = 20
    rng = np.random.default_rng(3)
    uv = rng.uniform(20, 600, (n, 2)).astype(np.float64)
    ids = np.arange(n)

    # Pose stub: identity rotation, unit +X baseline, every match an inlier.
    monkeypatch.setattr(m, "estimate_relative_pose",
                        lambda a, b, K: (np.eye(3), np.array([1.0, 0.0, 0.0]),
                                         np.ones(n, dtype=bool)))

    # Triangulation stub: valid depths for all, but the cheirality mask drops the
    # last 5 — so pts_a[valid] (and metric_cur) are shorter than the inlier set.
    def fake_triangulate(uv_a, uv_b, K, R, t):
        pts = np.column_stack([uv_a, np.full(len(uv_a), 2.0)])   # z = 2 m
        valid = np.ones(len(uv_a), dtype=bool)
        valid[-5:] = False
        return pts, valid
    monkeypatch.setattr(m, "triangulate_two_view", fake_triangulate)

    depth_map = np.ones((480, 640))
    pred_values, metric_cur = ref._geometry_step(uv, uv, ids, ids, depth_map)
    assert len(pred_values) == len(metric_cur) == n - 5   # the invariant that broke


def test_monocular_reference_first_frame_returns_none():
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0,
                            width=640, height=480)
    ref = MonocularScaleReference(intr)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert ref(frame, np.ones((480, 640), dtype=np.float32)) is None
