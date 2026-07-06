"""Turn Depth Anything's *relative* depth into *metric* depth.

Why this stage exists
---------------------
Depth Anything V2 (the relative model we run) is **affine-invariant**: its output
is only correct up to an unknown scale and shift. Left alone, a monocular stream
cannot produce a metric point cloud or a metric TSDF — the map's absolute size is
undefined and, worse, the per-frame scale drifts, so successive frames fuse
inconsistently. The rest of the pipeline (`_backproject_gaussians`, the TSDF
push, `RGBDOdometry`) all assume the depth value *is* metric ``z`` in metres, so
without an alignment step the live monocular path is geometrically meaningless.

This module recovers the missing degrees of freedom the standard way (the
MiDaS/DPT evaluation protocol): fit a linear map from the prediction to a sparse
**metric reference** and apply it to the whole frame.

    disparity space :   1/z  ≈  s · d + t          (Depth Anything is disparity-like)
    depth space     :    z   ≈  s · d + t          (already depth-like predictions)

``s`` (scale) and ``t`` (shift) are solved per frame by weighted least squares
with an optional robust IRLS (Huber) reweighting so a handful of bad reference
points can't wreck the fit. An exponential moving average over ``(s, t)`` keeps
the scale steady frame-to-frame, which is what lets the aligned depth fuse into a
single coherent volume.

Reference sources (wired in the pipeline layer, not here):
- **Live monocular:** sparse 3-D points triangulated by the VO front-end give a
  metric-*consistent* depth at their pixels; aligning the dense map to them locks
  the frame to the VO scale. A one-time global anchor fixes absolute metres.
- **RGB-D / TUM:** the sensor depth map is the reference directly — this is also
  how we *measure* how well alignment recovers true metric depth.

Design constraint: pure numpy, and no cv2/torch import at module load, so this
imports and unit-tests on the numpy-only dev machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Core least-squares fit (the tested numerical kernel)
# ---------------------------------------------------------------------------

def align_scale_shift(
    pred: np.ndarray,
    target: np.ndarray,
    weights: Optional[np.ndarray] = None,
    fit_shift: bool = True,
) -> Tuple[float, float]:
    """Weighted least-squares fit of ``target ≈ s·pred + t``.

    Closed form (no iteration). Solves

        min_{s,t}  Σ_i w_i (s·pred_i + t − target_i)²

    Args:
        pred:      (N,) predictions (relative depth or disparity).
        target:    (N,) metric reference in the *same space* as ``pred``.
        weights:   (N,) non-negative weights, or None for uniform.
        fit_shift: when False, forces ``t = 0`` (pure scale — a 1-parameter fit
                   appropriate when the prediction is known to pass through the
                   origin, e.g. metric-disparity with no offset).

    Returns:
        ``(s, t)`` floats. On a degenerate system (no spread in ``pred`` or zero
        total weight) returns ``(1.0, 0.0)`` — the identity, so callers degrade
        to "pass the prediction through unchanged" rather than divide by zero.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    if weights is None:
        w = np.ones_like(pred)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != pred.shape:
            raise ValueError("weights shape must match pred")
        w = np.clip(w, 0.0, None)

    sw = w.sum()
    if sw <= _EPS:
        return 1.0, 0.0

    if not fit_shift:
        denom = float((w * pred * pred).sum())
        if denom <= _EPS:
            return 1.0, 0.0
        s = float((w * pred * target).sum()) / denom
        return s, 0.0

    # Normal equations for [s, t] with design matrix [pred, 1]:
    #   [Σw p²  Σw p] [s]   [Σw p y]
    #   [Σw p   Σw ] [t] = [Σw y ]
    swp = float((w * pred).sum())
    swpp = float((w * pred * pred).sum())
    swy = float((w * target).sum())
    swpy = float((w * pred * target).sum())

    det = swpp * sw - swp * swp
    if abs(det) <= _EPS:
        # No spread in pred (all reference pixels share a prediction value): scale
        # is unidentifiable. Fall back to a shift-only match of the means.
        return 1.0, float((swy - swp) / sw)

    s = (swpy * sw - swp * swy) / det
    t = (swpp * swy - swp * swpy) / det
    return float(s), float(t)


def align_scale_shift_robust(
    pred: np.ndarray,
    target: np.ndarray,
    weights: Optional[np.ndarray] = None,
    fit_shift: bool = True,
    iters: int = 5,
    huber_delta: float = 1.345,
) -> Tuple[float, float]:
    """Robust (IRLS / Huber) version of :func:`align_scale_shift`.

    Starts from the plain weighted fit, then reweights points by a Huber factor
    on their residual (normalised by a robust MAD scale) for a few iterations.
    Down-weights outliers — reference points on depth discontinuities, moving
    objects, or triangulation blunders — without discarding them hard.

    ``huber_delta`` is in units of robust standard deviations (1.345 gives ~95%
    efficiency at the Gaussian). Falls back gracefully to the plain fit when the
    residuals collapse (perfect fit) or too few points remain.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    base_w = (np.ones_like(pred) if weights is None
              else np.clip(np.asarray(weights, np.float64).ravel(), 0.0, None))

    s, t = align_scale_shift(pred, target, base_w, fit_shift)
    for _ in range(max(0, iters - 1)):
        resid = s * pred + t - target
        # Robust scale via MAD (median absolute deviation → std estimate).
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad
        if scale <= _EPS:
            break  # residuals already ~0; the plain fit is optimal
        u = np.abs(resid) / (huber_delta * scale)
        huber = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, _EPS))
        w = base_w * huber
        s_new, t_new = align_scale_shift(pred, target, w, fit_shift)
        if abs(s_new - s) <= 1e-9 and abs(t_new - t) <= 1e-9:
            s, t = s_new, t_new
            break
        s, t = s_new, t_new
    return s, t


# ---------------------------------------------------------------------------
# Space conversions (predictions may be disparity- or depth-like)
# ---------------------------------------------------------------------------

def _to_target_space(metric_depth: np.ndarray, space: str) -> np.ndarray:
    """Metric depth (metres) → the space the fit is performed in."""
    if space == "depth":
        return metric_depth
    if space == "disparity":
        return 1.0 / np.maximum(metric_depth, _EPS)
    raise ValueError(f"space must be 'depth' or 'disparity', got {space!r}")


def _from_fit_space(values: np.ndarray, space: str) -> np.ndarray:
    """Fitted values (s·pred + t) → metric depth (metres).

    In disparity space a non-positive disparity means infinite/undefined depth
    (a pixel whose fitted disparity crossed zero); we emit ``+inf`` there so the
    caller's finite-and-positive filter drops it as "no return" rather than
    clamping it to a spurious far surface that would corrupt the TSDF.
    """
    if space == "depth":
        return values
    with np.errstate(divide="ignore"):
        return np.where(values > _EPS, 1.0 / values, np.inf)


# ---------------------------------------------------------------------------
# Stateful per-stream estimator
# ---------------------------------------------------------------------------

@dataclass
class ScaleShift:
    """The affine parameters mapping a prediction into fit space."""
    scale: float = 1.0
    shift: float = 0.0

    def as_tuple(self) -> Tuple[float, float]:
        return self.scale, self.shift


class DepthScaleAligner:
    """Per-stream relative→metric depth aligner with temporal smoothing.

    Feed each frame's relative prediction plus a sparse (or dense) metric
    reference; the aligner solves ``(s, t)``, smooths it against the running
    estimate, and applies it to the whole map. When a frame has no usable
    reference it *coasts* on the last good parameters, so the metric scale
    persists through reference dropouts instead of snapping back to identity.

    Args:
        space:       'disparity' (Depth Anything default — output is inverse
                     depth) or 'depth' (already depth-like predictions).
        fit_shift:   include the shift term ``t`` (recommended for affine-
                     invariant models; set False for pure-scale references).
        robust:      use Huber IRLS reweighting (recommended for VO points).
        smoothing:   EMA weight on the *previous* parameters in [0, 1). 0 uses
                     each frame's raw fit; 0.8 is heavily damped. The first fit
                     initialises the state directly (no smoothing toward identity).
        min_points:  below this many valid reference points the fit is refused
                     and the frame coasts on the current parameters.
        clamp:       (min_m, max_m) metric-depth clamp applied on transform, or
                     None. Keeps a bad disparity crossing zero from producing
                     absurd or negative depths downstream.
    """

    def __init__(
        self,
        space: str = "disparity",
        fit_shift: bool = True,
        robust: bool = True,
        smoothing: float = 0.7,
        min_points: int = 20,
        clamp: Optional[Tuple[float, float]] = (0.05, 100.0),
    ):
        if space not in ("disparity", "depth"):
            raise ValueError(f"space must be 'depth' or 'disparity', got {space!r}")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0, 1)")
        self.space = space
        self.fit_shift = fit_shift
        self.robust = robust
        self.smoothing = smoothing
        self.min_points = min_points
        self.clamp = clamp

        self.params: Optional[ScaleShift] = None   # None until the first good fit
        self.n_fits: int = 0
        self.n_coasts: int = 0
        self.last_residual: float = float("nan")    # RMS residual (fit space)

    # -- state ---------------------------------------------------------------

    @property
    def initialised(self) -> bool:
        return self.params is not None

    def reset(self) -> None:
        self.params = None
        self.n_fits = 0
        self.n_coasts = 0
        self.last_residual = float("nan")

    # -- fitting -------------------------------------------------------------

    def fit(
        self,
        pred_values: np.ndarray,
        ref_depth: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Optional[ScaleShift]:
        """Update the running ``(s, t)`` from a batch of paired samples.

        Args:
            pred_values: (N,) raw predictions at the reference pixels (relative
                         depth / disparity, whatever the model emits).
            ref_depth:   (N,) metric reference depth in metres at those pixels.
            weights:     (N,) optional per-point confidence.

        Returns:
            The smoothed :class:`ScaleShift` now in effect, or None if the frame
            was refused (too few valid points) — in which case the aligner
            coasts on its current parameters.
        """
        pred_values = np.asarray(pred_values, dtype=np.float64).ravel()
        ref_depth = np.asarray(ref_depth, dtype=np.float64).ravel()
        if pred_values.shape != ref_depth.shape:
            raise ValueError("pred_values and ref_depth must have the same length")

        valid = np.isfinite(pred_values) & np.isfinite(ref_depth) & (ref_depth > _EPS)
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64).ravel()
            valid &= np.isfinite(weights) & (weights > 0.0)

        if int(valid.sum()) < self.min_points:
            self.n_coasts += 1
            return None

        p = pred_values[valid]
        y = _to_target_space(ref_depth[valid], self.space)
        w = None if weights is None else weights[valid]

        if self.robust:
            s, t = align_scale_shift_robust(p, y, w, self.fit_shift)
        else:
            s, t = align_scale_shift(p, y, w, self.fit_shift)

        resid = s * p + t - y
        self.last_residual = float(np.sqrt(np.mean(resid ** 2)))

        # Temporal EMA. First fit initialises directly so we don't smooth toward
        # the identity (which would bias the early frames small/large).
        if self.params is None or self.smoothing <= 0.0:
            self.params = ScaleShift(s, t)
        else:
            a = self.smoothing
            self.params = ScaleShift(
                a * self.params.scale + (1.0 - a) * s,
                a * self.params.shift + (1.0 - a) * t,
            )
        self.n_fits += 1
        return self.params

    # -- applying ------------------------------------------------------------

    def transform(self, pred: np.ndarray) -> np.ndarray:
        """Apply the current ``(s, t)`` to a relative map → metric depth (m).

        Before any successful fit this is the identity in *depth* space and a
        plain inversion in *disparity* space (``s=1, t=0``), so the pipeline
        still runs — just not yet metric. Non-positive/invalid metric depths are
        set to 0, which the downstream ``z > 0.1`` masks treat as "no return".
        """
        pred = np.asarray(pred, dtype=np.float64)
        s, t = (self.params.as_tuple() if self.params is not None else (1.0, 0.0))
        metric = _from_fit_space(s * pred + t, self.space)

        # Kill NaNs/infs and non-positive depths (e.g. disparity that crossed 0).
        metric = np.where(np.isfinite(metric) & (metric > 0.0), metric, 0.0)
        if self.clamp is not None:
            lo, hi = self.clamp
            nonzero = metric > 0.0
            metric = np.where(nonzero, np.clip(metric, lo, hi), 0.0)
        return metric.astype(np.float32)

    def __call__(
        self,
        pred_map: np.ndarray,
        pred_values: np.ndarray,
        ref_depth: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit on the sparse reference, then transform the full map. Convenience
        wrapper around :meth:`fit` + :meth:`transform`."""
        self.fit(pred_values, ref_depth, weights)
        return self.transform(pred_map)


# ---------------------------------------------------------------------------
# Sparse reference from two-view triangulation (live monocular anchor)
# ---------------------------------------------------------------------------

def triangulate_two_view(
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear (DLT) triangulation of matched pixels across two views.

    The monocular scale anchor: a VO front-end supplies feature correspondences
    and the relative pose ``(R, t)`` mapping camera-A coordinates into camera-B
    (``X_b = R·X_a + t``). Triangulating the matches gives a 3-D point per match
    in camera A's frame; its ``z`` is a **metric-consistent** depth at that pixel
    — consistent up to the single global gauge of ``t`` (unit-baseline VO ⇒ an
    arbitrary but self-consistent scale; a metric baseline ⇒ true metres).

    Args:
        uv_a: (N,2) pixel coords in image A.
        uv_b: (N,2) matched pixel coords in image B.
        K:    (3,3) camera intrinsics (shared by both views).
        R:    (3,3) rotation of the A→B relative pose.
        t:    (3,) translation of the A→B relative pose.

    Returns:
        ``(pts_a, valid)`` — ``pts_a`` is (N,3) in camera A's frame; ``valid`` is
        an (N,) bool mask that is True only where the point is in front of *both*
        cameras (positive depth — the cheirality check). Downstream should keep
        only ``pts_a[valid]``.
    """
    uv_a = np.asarray(uv_a, dtype=np.float64).reshape(-1, 2)
    uv_b = np.asarray(uv_b, dtype=np.float64).reshape(-1, 2)
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    if uv_a.shape != uv_b.shape:
        raise ValueError("uv_a and uv_b must have the same shape")

    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])       # A: K[I|0]
    P1 = K @ np.hstack([R, t.reshape(3, 1)])                # B: K[R|t]

    n = uv_a.shape[0]
    pts = np.zeros((n, 3), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        ua, va = uv_a[i]
        ub, vb = uv_b[i]
        A = np.stack([
            ua * P0[2] - P0[0],
            va * P0[2] - P0[1],
            ub * P1[2] - P1[0],
            vb * P1[2] - P1[1],
        ])
        # Homogeneous solution = right singular vector of smallest singular value.
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) <= _EPS:
            continue                                        # point at infinity
        Xa = X[:3] / X[3]                                   # camera-A coords
        za = Xa[2]
        Xb = R @ Xa + t
        if za > _EPS and Xb[2] > _EPS:                      # in front of both
            pts[i] = Xa
            valid[i] = True
    return pts, valid


def triangulated_scale_reference(
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    pred_map: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build ``(pred_values, ref_depth)`` for the aligner from VO matches.

    Triangulates the correspondences, reads the *predicted* value at each valid
    frame-A pixel, and pairs it with the triangulated depth. Returns None if no
    correspondence survives the cheirality check. ``pred_map`` is indexed at the
    integer-rounded frame-A pixels.
    """
    pts_a, valid = triangulate_two_view(uv_a, uv_b, K, R, t)
    if not np.any(valid):
        return None
    uv = np.asarray(uv_a, dtype=np.float64).reshape(-1, 2)[valid]
    ref_depth = pts_a[valid, 2]
    h, w = pred_map.shape[:2]
    cols = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
    rows = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
    pred_values = np.asarray(pred_map)[rows, cols]
    return pred_values, ref_depth


# ---------------------------------------------------------------------------
# Cross-frame scale propagation (globally-consistent monocular scale)
# ---------------------------------------------------------------------------

def estimate_relative_scale(
    numer: np.ndarray,
    denom: np.ndarray,
    robust: bool = True,
    reject_sigma: float = 3.0,
) -> Optional[float]:
    """Robust ``median(numer / denom)`` over paired positive samples.

    The workhorse of scale propagation: ``numer`` are a shared landmark's depths
    in the running global gauge, ``denom`` the same landmark's depths in the new
    pair's unit gauge, so their ratio is the baseline that maps new depths into
    the global gauge. Uses a median (breakdown 50%) and one MAD-based outlier
    rejection pass. Returns None if nothing valid remains.
    """
    numer = np.asarray(numer, dtype=np.float64).ravel()
    denom = np.asarray(denom, dtype=np.float64).ravel()
    m = np.isfinite(numer) & np.isfinite(denom) & (numer > _EPS) & (denom > _EPS)
    if not np.any(m):
        return None
    r = numer[m] / denom[m]
    if not robust or r.size < 3:
        return float(np.median(r))
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    if mad > _EPS:
        keep = np.abs(r - med) <= reject_sigma * 1.4826 * mad
        if np.any(keep):
            r = r[keep]
    return float(np.median(r))


class ScalePropagator:
    """Chains per-pair unit-gauge baselines into one globally-consistent scale.

    Monocular two-view geometry gives each frame pair a relative pose with a
    *unit* translation, so each pair's triangulated depths live in their own
    baseline gauge — this is the source of monocular scale drift. Given the
    depths of landmarks shared with the previous pair (already in the global
    gauge), this recovers the factor ``baseline`` that maps the new pair's
    unit-gauge depths into the same global gauge:

        baseline = median( shared_prev_global_depth / shared_new_local_depth )
        metric_depth = baseline · local_depth

    The first pair has no shared history, so it defines the gauge from ``anchor``
    (1.0 → consistent-but-arbitrary absolute scale, the honest monocular limit;
    set it to a known metric baseline to pin true metres). When too few
    landmarks are shared (fast motion, a cut) it *coasts* on the last baseline
    rather than snapping the scale.
    """

    def __init__(self, anchor: float = 1.0, min_shared: int = 6):
        if anchor <= 0.0:
            raise ValueError("anchor must be positive")
        self.anchor = anchor
        self.min_shared = min_shared
        self.baseline: Optional[float] = None   # None until the first pair
        self.n_updates: int = 0
        self.n_coasts: int = 0

    @property
    def initialised(self) -> bool:
        return self.baseline is not None

    def reset(self) -> None:
        self.baseline = None
        self.n_updates = 0
        self.n_coasts = 0

    def update(self, prev_global: np.ndarray, new_local: np.ndarray) -> float:
        """Fold one pair's shared landmarks into the running baseline.

        Args:
            prev_global: (M,) depths of the shared landmarks in the global gauge
                         (from the previous pair).
            new_local:   (M,) the same landmarks' depths in this pair's unit gauge.

        Returns:
            The baseline now in effect (``metric = baseline · local``).
        """
        prev_global = np.asarray(prev_global, dtype=np.float64).ravel()
        new_local = np.asarray(new_local, dtype=np.float64).ravel()
        valid = (np.isfinite(prev_global) & np.isfinite(new_local)
                 & (prev_global > _EPS) & (new_local > _EPS))
        n = int(valid.sum())

        if n < self.min_shared:
            if self.baseline is None:
                self.baseline = self.anchor          # first pair: define the gauge
            else:
                self.n_coasts += 1                    # dropout: hold last scale
            return self.baseline

        b = estimate_relative_scale(prev_global[valid], new_local[valid])
        if b is None or not np.isfinite(b) or b <= 0.0:
            if self.baseline is None:
                self.baseline = self.anchor
            else:
                self.n_coasts += 1
            return self.baseline

        self.baseline = b
        self.n_updates += 1
        return self.baseline


# ---------------------------------------------------------------------------
# Dense-reference convenience (RGB-D / TUM validation path)
# ---------------------------------------------------------------------------

def sample_dense_reference(
    pred_map: np.ndarray,
    ref_map: np.ndarray,
    max_points: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair a prediction map with a co-registered dense metric depth map.

    Returns ``(pred_values, ref_depth)`` sampled at pixels where the reference
    is valid (``> 0``). Used for the RGB-D/TUM path where a true metric depth
    image is available — both for aligning and for measuring how metric the
    aligned prediction becomes. Sub-samples to ``max_points`` for a fast fit.
    """
    pred_map = np.asarray(pred_map, dtype=np.float64)
    ref_map = np.asarray(ref_map, dtype=np.float64)
    if pred_map.shape != ref_map.shape:
        raise ValueError(f"map shape mismatch: {pred_map.shape} vs {ref_map.shape}")

    valid = np.isfinite(pred_map) & np.isfinite(ref_map) & (ref_map > 0.0)
    pv = pred_map[valid]
    rv = ref_map[valid]
    if max_points and pv.size > max_points:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(pv.size, size=max_points, replace=False)
        pv, rv = pv[idx], rv[idx]
    return pv, rv
