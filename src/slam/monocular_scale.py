"""Live monocular metric-scale anchor for the pipeline.

Depth Anything gives *relative* depth; to make the monocular path metric we need
a per-frame metric reference to align it to. On a pure monocular stream that
reference comes from **two-view geometry**: match features to the previous
frame, recover the relative pose from the essential matrix, and triangulate the
matches. Each triangulated point's depth is a metric-*consistent* sample at its
pixel, which the :class:`DepthScaleAligner` uses to lock the dense map's scale.

Scale gauge — read this before trusting the numbers
---------------------------------------------------
``cv2.recoverPose`` returns a **unit** translation, so a single frame-pair's
triangulated depths are only defined up to that pair's baseline — and the
baseline changes every frame as the camera speeds up and slows down, which is
exactly monocular scale drift. This class defeats the drift with **cross-frame
scale propagation** (:class:`~depth.metric_scale.ScalePropagator`): landmarks
shared with the previous pair are already in a running global gauge, so their
depth ratio pins the new pair's baseline into that same gauge. The per-frame
scale is therefore globally consistent, not just locally valid.

That still leaves **one** free number — the absolute size of the whole
reconstruction — which is genuinely unobservable from a single moving camera.
Pin it either way:

- **Known baseline:** pass ``anchor`` = the real first-pair camera translation
  (metres, from an IMU / wheel odometry / a rig) → depths come out in metres.
- **Arbitrary gauge:** leave ``anchor=1.0`` → a consistent but arbitrary absolute
  scale; multiply by one external metric cue (RGB-D calibration frame, known
  object size, camera height + ground plane) to make it metric.

The geometry (:func:`estimate_relative_pose`) and the propagation
(:meth:`MonocularScaleReference._geometry_step`) are separated from the ORB
front-end so both can be unit-tested with synthetic correspondences.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from mapping.collision_proxy import CameraIntrinsics
    from depth.metric_scale import ScalePropagator, triangulate_two_view
except ImportError:  # pragma: no cover - import-path shim
    from src.mapping.collision_proxy import CameraIntrinsics
    from src.depth.metric_scale import ScalePropagator, triangulate_two_view


def _k_matrix(K: CameraIntrinsics) -> np.ndarray:
    return np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1]], dtype=np.float64)


def estimate_relative_pose(
    uv_a: np.ndarray,
    uv_b: np.ndarray,
    K: np.ndarray,
    ransac_thresh_px: float = 1.0,
    prob: float = 0.999,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Essential-matrix relative pose ``A→B`` from correspondences.

    Args:
        uv_a, uv_b: (N,2) matched pixel coords in image A and image B.
        K:          (3,3) intrinsics.

    Returns:
        ``(R, t_unit, inlier_mask)`` where ``X_b = R·X_a + t_unit`` and
        ``t_unit`` is a **unit** translation (scale gauge — see module docstring),
        or None if the essential matrix / pose could not be recovered (too few
        points, degenerate/planar motion, no cheirality-consistent solution).
    """
    uv_a = np.asarray(uv_a, dtype=np.float64).reshape(-1, 2)
    uv_b = np.asarray(uv_b, dtype=np.float64).reshape(-1, 2)
    if uv_a.shape[0] < 5 or uv_a.shape != uv_b.shape:
        return None

    E, mask = cv2.findEssentialMat(
        uv_a, uv_b, K, method=cv2.RANSAC, prob=prob, threshold=ransac_thresh_px)
    if E is None or E.shape != (3, 3):
        return None

    n_in, R, t, mask_pose = cv2.recoverPose(E, uv_a, uv_b, K, mask=mask)
    if n_in < 5:
        return None
    inliers = (mask_pose.ravel() > 0)
    return R, t.reshape(3), inliers


class MonocularScaleReference:
    """Callable scale-reference for the pipeline's monocular path.

    Mirrors the ``scale_reference(frame_bgr, rel_depth) -> (pred_values,
    ref_depth)`` contract PipelineManager expects. Holds the previous frame's
    ORB features *and* the global-gauge depth of every landmark it triangulated;
    each call matches to the current frame, recovers the relative pose,
    triangulates the inliers, propagates scale through the landmarks shared with
    the previous pair, and returns predicted-value / metric-depth pairs at the
    current frame's pixels. Returns None on the first frame and whenever geometry
    is degenerate — the aligner then coasts on its current scale.

    The RGB frame is resized to the depth map's resolution so pixels and
    intrinsics agree (same convention as OdometryPoseProvider).
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        n_features: int = 1500,
        ratio: float = 0.75,
        min_matches: int = 12,
        anchor: float = 1.0,
        min_shared: int = 6,
    ):
        self.K = intrinsics
        self._Kmat = _k_matrix(intrinsics)
        self.ratio = ratio
        self.min_matches = min_matches
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._propagator = ScalePropagator(anchor=anchor, min_shared=min_shared)

        self._prev: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (xy, des)
        # Global-gauge depth of each previous-frame landmark, keyed by that
        # frame's keypoint index. Rebuilt every step; the bridge between pairs.
        self._landmarks: dict = {}

    # -- observability -------------------------------------------------------

    @property
    def baseline(self) -> Optional[float]:
        """Current propagated baseline (global gauge), or None before init."""
        return self._propagator.baseline

    def _features(self, gray: np.ndarray):
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            return np.empty((0, 2), np.float32), None
        return np.array([k.pt for k in kp], dtype=np.float32), des

    def _match(self, des_a, des_b) -> np.ndarray:
        if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
            return np.empty((0, 2), np.int32)
        knn = self._matcher.knnMatch(des_a, des_b, k=2)
        good = [[m.queryIdx, m.trainIdx]
                for m, n in knn if m.distance < self.ratio * n.distance]
        return np.array(good, dtype=np.int32) if good else np.empty((0, 2), np.int32)

    def _geometry_step(
        self,
        uv_prev: np.ndarray,
        uv_cur: np.ndarray,
        prev_ids: np.ndarray,
        cur_ids: np.ndarray,
        rel_depth: np.ndarray,
    ):
        """Pose + triangulate + propagate for one matched pair (testable core).

        ``uv_prev``/``uv_cur`` are matched pixels in the previous/current frame;
        ``prev_ids``/``cur_ids`` are their keypoint indices in each frame (the
        identity that links landmarks across pairs). Triangulates in the
        *previous* frame (A) so a landmark's depth there can be compared with the
        depth the previous pair stored for it; also carries each point forward to
        the current frame (B) to build this frame's reference and next step's
        landmark map. Returns ``(pred_values, metric_depth)`` at current-frame
        pixels, or None if the step is degenerate (the landmark map is cleared so
        a broken step can't create phantom correspondences).
        """
        pose = estimate_relative_pose(uv_prev, uv_cur, self._Kmat)
        if pose is None:
            self._landmarks = {}
            return None
        R, t_unit, inliers = pose
        if int(inliers.sum()) < self.min_matches:
            self._landmarks = {}
            return None

        uv_a = uv_prev[inliers]
        uv_b = uv_cur[inliers]
        pts_a, valid = triangulate_two_view(uv_a, uv_b, self._Kmat, R, t_unit)
        if not np.any(valid):
            self._landmarks = {}
            return None

        pts_a = pts_a[valid]
        z_prev = pts_a[:, 2]                                  # unit-gauge depth at A
        z_cur = (pts_a @ R.T + t_unit)[:, 2]                 # unit-gauge depth at B
        ids_prev = np.asarray(prev_ids)[inliers][valid]
        ids_cur = np.asarray(cur_ids)[inliers][valid]

        # Landmarks shared with the previous pair fix the global baseline.
        shared_prev_global, shared_new_local = [], []
        for pid, zp in zip(ids_prev, z_prev):
            g = self._landmarks.get(int(pid))
            if g is not None:
                shared_prev_global.append(g)
                shared_new_local.append(zp)
        baseline = self._propagator.update(
            np.asarray(shared_prev_global), np.asarray(shared_new_local))

        metric_cur = baseline * z_cur                        # global-gauge metres
        # Carry landmarks forward, keyed by current-frame keypoint index.
        self._landmarks = {int(cid): float(md)
                           for cid, md in zip(ids_cur, metric_cur)}

        # Sample the predicted value at the SAME cheirality-valid current-frame
        # pixels that produced metric_cur. uv_b spans all inliers; the triangulation
        # keeps only the `valid` subset, so pred_values must be filtered by `valid`
        # too or the aligner gets mismatched-length arrays (a silent scale-fit skip).
        uv_b_valid = uv_b[valid]
        h, w = rel_depth.shape[:2]
        cols = np.clip(np.rint(uv_b_valid[:, 0]).astype(int), 0, w - 1)
        rows = np.clip(np.rint(uv_b_valid[:, 1]).astype(int), 0, h - 1)
        pred_values = np.asarray(rel_depth)[rows, cols]
        return pred_values, metric_cur

    def __call__(self, frame_bgr: np.ndarray, rel_depth: np.ndarray):
        h, w = rel_depth.shape[:2]
        if frame_bgr.shape[:2] != (h, w):
            frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        xy, des = self._features(gray)

        if self._prev is None:
            self._prev = (xy, des)
            return None

        prev_xy, prev_des = self._prev
        self._prev = (xy, des)
        matches = self._match(prev_des, des)
        if len(matches) < self.min_matches:
            self._landmarks = {}
            return None

        return self._geometry_step(
            prev_xy[matches[:, 0]], xy[matches[:, 1]],
            matches[:, 0], matches[:, 1], rel_depth)
