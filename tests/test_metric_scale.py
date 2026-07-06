"""Tests for the relative→metric depth aligner (src/depth/metric_scale.py).

Pure numpy — runs on the dev machine (no cv2/torch/CUDA). Covers the numerical
kernel (closed-form + robust), the disparity/depth space handling, and the
stateful per-stream estimator (smoothing, coasting, clamping).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from depth.metric_scale import (  # noqa: E402
    DepthScaleAligner,
    ScaleShift,
    align_scale_shift,
    align_scale_shift_robust,
    sample_dense_reference,
    triangulate_two_view,
    triangulated_scale_reference,
)


# ---------------------------------------------------------------------------
# Closed-form kernel
# ---------------------------------------------------------------------------

def test_align_recovers_exact_affine():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0.1, 5.0, size=200)
    s_true, t_true = 2.3, -0.7
    target = s_true * pred + t_true
    s, t = align_scale_shift(pred, target)
    assert s == pytest.approx(s_true, rel=1e-9)
    assert t == pytest.approx(t_true, abs=1e-9)


def test_align_scale_only_ignores_shift_term():
    rng = np.random.default_rng(1)
    pred = rng.uniform(0.1, 5.0, size=200)
    target = 1.7 * pred                       # passes through origin
    s, t = align_scale_shift(pred, target, fit_shift=False)
    assert t == 0.0
    assert s == pytest.approx(1.7, rel=1e-9)


def test_align_weighted_least_squares():
    # Two clusters; heavy weight on the second should pull the fit to it.
    pred = np.array([0.0, 1.0, 2.0, 3.0])
    target = np.array([0.0, 1.0, 2.0, 9.0])   # last point is an "outlier"
    w = np.array([1.0, 1.0, 1.0, 1000.0])
    s, t = align_scale_shift(pred, target, weights=w)
    # Fit should nearly pass through the heavily-weighted last point.
    assert (s * 3.0 + t) == pytest.approx(9.0, abs=0.05)


def test_align_degenerate_constant_pred_returns_shift_match():
    pred = np.full(50, 2.0)
    target = np.full(50, 5.0)
    s, t = align_scale_shift(pred, target)
    # Scale unidentifiable → identity scale, shift matches the means (2 -> 5).
    assert s == 1.0
    assert (s * 2.0 + t) == pytest.approx(5.0, abs=1e-9)


def test_align_zero_weight_is_identity():
    pred = np.arange(10.0)
    target = 3.0 * pred + 1.0
    s, t = align_scale_shift(pred, target, weights=np.zeros(10))
    assert (s, t) == (1.0, 0.0)


def test_align_shape_mismatch_raises():
    with pytest.raises(ValueError):
        align_scale_shift(np.arange(5.0), np.arange(4.0))


# ---------------------------------------------------------------------------
# Robust kernel
# ---------------------------------------------------------------------------

def test_robust_downweights_gross_outliers():
    rng = np.random.default_rng(2)
    pred = rng.uniform(0.5, 5.0, size=300)
    s_true, t_true = 1.5, 0.3
    target = s_true * pred + t_true
    # Corrupt 15% of the targets with large blunders.
    n_bad = 45
    bad = rng.choice(300, size=n_bad, replace=False)
    target[bad] += rng.uniform(-20, 20, size=n_bad)

    s_plain, t_plain = align_scale_shift(pred, target)
    s_rob, t_rob = align_scale_shift_robust(pred, target)

    # Robust fit is much closer to truth than the plain least-squares fit.
    assert abs(s_rob - s_true) < abs(s_plain - s_true)
    assert s_rob == pytest.approx(s_true, abs=0.15)
    assert t_rob == pytest.approx(t_true, abs=0.3)


def test_robust_matches_plain_when_clean():
    rng = np.random.default_rng(3)
    pred = rng.uniform(0.5, 5.0, size=100)
    target = 0.8 * pred - 0.2
    s, t = align_scale_shift_robust(pred, target)
    assert s == pytest.approx(0.8, abs=1e-6)
    assert t == pytest.approx(-0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# Stateful aligner — metric recovery
# ---------------------------------------------------------------------------

def _make_disparity_scene(rng, H=48, W=64, alpha=3.0, beta=0.5):
    """A metric depth map + an affine-invariant disparity prediction of it.

    Returns (z_true, pred) where pred = alpha*(1/z) + beta with unknown alpha,
    beta — exactly the affine-invariance Depth Anything exhibits in disparity.
    """
    z_true = rng.uniform(0.6, 6.0, size=(H, W))
    pred = alpha * (1.0 / z_true) + beta
    return z_true, pred


def test_disparity_aligner_recovers_metric_depth():
    rng = np.random.default_rng(4)
    z_true, pred = _make_disparity_scene(rng)

    # Sparse metric reference at a random subset of pixels.
    H, W = z_true.shape
    idx = rng.choice(H * W, size=150, replace=False)
    pv = pred.ravel()[idx]
    rv = z_true.ravel()[idx]

    aligner = DepthScaleAligner(space="disparity", smoothing=0.0)
    metric = aligner(pred, pv, rv)

    assert aligner.initialised
    # The whole dense map should now be metric to numerical precision.
    assert np.allclose(metric, z_true, rtol=1e-4, atol=1e-4)
    assert aligner.last_residual < 1e-6


def test_depth_space_aligner_recovers_metric_depth():
    rng = np.random.default_rng(5)
    z_true = rng.uniform(0.6, 6.0, size=(40, 40))
    pred = 4.0 * z_true - 1.2                  # depth-like affine prediction
    pv, rv = sample_dense_reference(pred, z_true, max_points=300)

    aligner = DepthScaleAligner(space="depth", smoothing=0.0)
    metric = aligner(pred, pv, rv)
    assert np.allclose(metric, z_true, rtol=1e-5, atol=1e-5)


def test_scale_recovery_invariant_to_prediction_gauge():
    """However the model rescales its raw output, we recover the same metric."""
    rng = np.random.default_rng(6)
    z_true, _ = _make_disparity_scene(rng, alpha=1.0, beta=0.0)
    disp = 1.0 / z_true

    H, W = z_true.shape
    idx = rng.choice(H * W, size=200, replace=False)
    rv = z_true.ravel()[idx]

    outputs = []
    for alpha, beta in [(1.0, 0.0), (7.3, 2.1), (0.05, -0.01)]:
        pred = alpha * disp + beta
        aligner = DepthScaleAligner(space="disparity", smoothing=0.0)
        metric = aligner(pred, pred.ravel()[idx], rv)
        outputs.append(metric)
    for m in outputs[1:]:
        assert np.allclose(m, outputs[0], rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Stateful aligner — temporal behaviour
# ---------------------------------------------------------------------------

def test_transform_before_fit_is_identity_in_fit_space():
    aligner = DepthScaleAligner(space="depth")
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = aligner.transform(pred)
    assert not aligner.initialised
    assert np.allclose(out, pred)             # s=1,t=0 in depth space


def test_first_fit_initialises_without_smoothing_toward_identity():
    rng = np.random.default_rng(7)
    pred = rng.uniform(0.5, 5.0, size=100)
    target = 3.0 * pred                        # depth space, s=3
    aligner = DepthScaleAligner(space="depth", smoothing=0.9)
    aligner.fit(pred, target)
    # Despite heavy smoothing, the *first* fit lands on the true scale, not 0.9
    # of the way from identity to it.
    assert aligner.params.scale == pytest.approx(3.0, rel=1e-6)


def test_smoothing_blends_successive_fits():
    aligner = DepthScaleAligner(space="depth", smoothing=0.75, min_points=2)
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    aligner.fit(pred, 2.0 * pred)             # scale 2 (initialises)
    aligner.fit(pred, 6.0 * pred)             # scale 6 (blended)
    # 0.75*2 + 0.25*6 = 3.0
    assert aligner.params.scale == pytest.approx(3.0, rel=1e-6)


def test_coasts_on_insufficient_reference():
    aligner = DepthScaleAligner(space="depth", min_points=20, smoothing=0.0)
    pred = np.arange(1.0, 41.0)
    good = aligner.fit(pred, 2.5 * pred)
    assert good is not None
    before = aligner.params.as_tuple()

    # Too few points → refused; params must not move.
    refused = aligner.fit(np.array([1.0, 2.0]), np.array([2.5, 5.0]))
    assert refused is None
    assert aligner.params.as_tuple() == before
    assert aligner.n_coasts == 1
    # Transform still applies the last good scale.
    assert np.allclose(aligner.transform(pred), 2.5 * pred)


def test_reset_clears_state():
    aligner = DepthScaleAligner(space="depth", min_points=2)
    aligner.fit(np.arange(1.0, 11.0), 2.0 * np.arange(1.0, 11.0))
    assert aligner.initialised
    aligner.reset()
    assert not aligner.initialised
    assert aligner.n_fits == 0
    assert np.isnan(aligner.last_residual)


# ---------------------------------------------------------------------------
# Transform guards (clamp / invalid)
# ---------------------------------------------------------------------------

def test_transform_zeros_out_nonpositive_disparity():
    # After a fit, some pixels' disparity is <= 0 → metric depth undefined → 0.
    aligner = DepthScaleAligner(space="disparity", clamp=(0.05, 100.0),
                                min_points=2, smoothing=0.0)
    # Fit identity in disparity space: target 1/z, pred already = 1/z.
    z = np.array([1.0, 2.0, 4.0, 8.0])
    aligner.fit(1.0 / z, z)
    pred = np.array([[1.0, 0.0], [-0.5, 0.25]])   # disparities incl. 0 and negative
    out = aligner.transform(pred)
    assert out[0, 1] == 0.0                    # disparity 0 → 0
    assert out[1, 0] == 0.0                    # negative disparity → 0
    assert out[0, 0] == pytest.approx(1.0, rel=1e-6)
    assert out[1, 1] == pytest.approx(4.0, rel=1e-6)


def test_transform_clamps_metric_range():
    aligner = DepthScaleAligner(space="depth", clamp=(1.0, 5.0),
                                min_points=2, smoothing=0.0)
    aligner.fit(np.array([1.0, 2.0]), np.array([1.0, 2.0]))   # identity
    out = aligner.transform(np.array([0.1, 3.0, 50.0]))
    assert out[0] == pytest.approx(1.0)       # clamped up
    assert out[1] == pytest.approx(3.0)       # untouched
    assert out[2] == pytest.approx(5.0)       # clamped down


def test_transform_returns_float32():
    aligner = DepthScaleAligner(space="depth")
    out = aligner.transform(np.ones((4, 4), dtype=np.float64))
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Two-view triangulation (monocular sparse anchor)
# ---------------------------------------------------------------------------

def _project(K, R, t, X):
    """Project world/cam-A 3-D points into a camera (R,t)."""
    Xc = (R @ X.T).T + t
    uvw = (K @ Xc.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def _test_intrinsics():
    return np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def test_triangulation_recovers_3d_points():
    rng = np.random.default_rng(10)
    K = _test_intrinsics()
    # Points in front of camera A (z in [2,6]).
    X = np.column_stack([
        rng.uniform(-2, 2, 60),
        rng.uniform(-2, 2, 60),
        rng.uniform(2, 6, 60),
    ])
    # Camera B translated sideways (a real baseline) with a small rotation.
    ang = 0.05
    R = np.array([[np.cos(ang), 0, np.sin(ang)],
                  [0, 1, 0],
                  [-np.sin(ang), 0, np.cos(ang)]])
    t = np.array([0.4, 0.0, 0.0])

    uv_a = _project(K, np.eye(3), np.zeros(3), X)
    uv_b = _project(K, R, t, X)

    pts, valid = triangulate_two_view(uv_a, uv_b, K, R, t)
    assert valid.all()
    assert np.allclose(pts, X, atol=1e-6)


def test_triangulation_cheirality_rejects_behind_camera():
    K = _test_intrinsics()
    # One point behind camera A (negative z) — project math still gives pixels,
    # but the cheirality check must flag it invalid.
    X = np.array([[0.0, 0.0, 3.0], [0.5, 0.5, -3.0]])
    R = np.eye(3)
    t = np.array([0.3, 0.0, 0.0])
    uv_a = _project(K, np.eye(3), np.zeros(3), X)
    uv_b = _project(K, R, t, X)
    _, valid = triangulate_two_view(uv_a, uv_b, K, R, t)
    assert valid[0]
    assert not valid[1]


def test_triangulation_scale_matches_baseline_gauge():
    """Depths scale linearly with the baseline: unit-baseline VO → arbitrary but
    self-consistent scale; the aligner then locks it to metric."""
    rng = np.random.default_rng(11)
    K = _test_intrinsics()
    X = np.column_stack([rng.uniform(-1, 1, 40), rng.uniform(-1, 1, 40),
                         rng.uniform(2, 5, 40)])
    R = np.eye(3)
    uv_a = _project(K, np.eye(3), np.zeros(3), X)

    # True baseline vs a half-length (unit-gauge) baseline in the same direction.
    t_true = np.array([0.5, 0.0, 0.0])
    uv_b = _project(K, R, t_true, X)
    pts_true, _ = triangulate_two_view(uv_a, uv_b, K, R, t_true)
    pts_half, _ = triangulate_two_view(uv_a, uv_b, K, R, 0.5 * t_true)
    # Halving the assumed baseline halves every triangulated depth.
    assert np.allclose(pts_half[:, 2], 0.5 * pts_true[:, 2], rtol=1e-6)


def test_triangulated_scale_reference_feeds_aligner_to_metric():
    """End-to-end: VO matches (unit-gauge baseline) + affine-invariant prediction
    → aligner recovers true metric depth at the triangulated pixels."""
    rng = np.random.default_rng(12)
    K = _test_intrinsics()
    H, W = 480, 640
    X = np.column_stack([rng.uniform(-2, 2, 120), rng.uniform(-1.5, 1.5, 120),
                         rng.uniform(2, 6, 120)])
    R = np.eye(3)
    t = np.array([0.6, 0.0, 0.0])                 # true metric baseline
    uv_a = _project(K, np.eye(3), np.zeros(3), X)
    uv_b = _project(K, R, t, X)

    # Affine-invariant disparity prediction over the whole image; we only read it
    # at the matched pixels via the reference builder.
    pred_map = np.zeros((H, W))
    cols = np.clip(np.rint(uv_a[:, 0]).astype(int), 0, W - 1)
    rows = np.clip(np.rint(uv_a[:, 1]).astype(int), 0, H - 1)
    pred_map[rows, cols] = 4.0 * (1.0 / X[:, 2]) + 0.3     # α·disp + β

    ref = triangulated_scale_reference(uv_a, uv_b, pred_map, K, R, t)
    assert ref is not None
    pred_values, ref_depth = ref
    assert np.allclose(np.sort(ref_depth), np.sort(X[:, 2]), atol=1e-4)

    aligner = DepthScaleAligner(space="disparity", smoothing=0.0, min_points=10)
    aligner.fit(pred_values, ref_depth)
    # Aligner recovers the true metric depth from the affine prediction.
    metric = aligner.transform(pred_values)
    assert np.allclose(np.sort(metric), np.sort(ref_depth.astype(np.float32)),
                       rtol=1e-3, atol=1e-3)


def test_triangulated_reference_none_when_all_behind():
    K = _test_intrinsics()
    X = np.array([[0.0, 0.0, -3.0], [0.2, 0.1, -4.0]])
    R = np.eye(3)
    t = np.array([0.3, 0.0, 0.0])
    uv_a = _project(K, np.eye(3), np.zeros(3), X)
    uv_b = _project(K, R, t, X)
    assert triangulated_scale_reference(uv_a, uv_b, np.zeros((480, 640)),
                                        K, R, t) is None


# ---------------------------------------------------------------------------
# Dense-reference helper
# ---------------------------------------------------------------------------

def test_sample_dense_reference_masks_invalid_and_subsamples():
    rng = np.random.default_rng(8)
    pred = rng.uniform(0.1, 5.0, size=(50, 50))
    ref = rng.uniform(0.5, 6.0, size=(50, 50))
    ref[ref < 1.0] = 0.0                       # mark some invalid (no return)
    pv, rv = sample_dense_reference(pred, ref, max_points=200, rng=rng)
    assert pv.shape == rv.shape
    assert pv.size <= 200
    assert np.all(rv > 0.0)                     # invalid pixels excluded


def test_sample_dense_reference_shape_mismatch_raises():
    with pytest.raises(ValueError):
        sample_dense_reference(np.zeros((4, 4)), np.zeros((4, 5)))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_invalid_space_raises():
    with pytest.raises(ValueError):
        DepthScaleAligner(space="inverse")


def test_invalid_smoothing_raises():
    with pytest.raises(ValueError):
        DepthScaleAligner(smoothing=1.0)


def test_scaleshift_as_tuple():
    assert ScaleShift(2.0, -1.0).as_tuple() == (2.0, -1.0)
