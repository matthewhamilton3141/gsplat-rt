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
triangulated depths are only defined up to that pair's baseline. Two honest ways
to make them absolute:

- **Known baseline:** pass ``baseline_m`` = the real inter-frame camera
  translation (from an IMU, wheel odometry, or a constant-velocity rig). Depths
  come out in metres directly.
- **Metric anchor frame:** leave ``baseline_m=1.0`` (unit gauge) and fix the one
  global factor once from an external metric cue (an RGB-D calibration frame, a
  known object size, camera-height + ground plane). The aligner's temporal EMA
  then keeps subsequent frames consistent.

This module provides the machinery and the geometry; choosing the global gauge is
a deployment decision, made explicit rather than hidden. The geometry core
(:func:`estimate_relative_pose`) is separated from the ORB front-end so it can be
unit-tested with synthetic correspondences.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from mapping.collision_proxy import CameraIntrinsics
    from depth.metric_scale import triangulated_scale_reference
except ImportError:  # pragma: no cover - import-path shim
    from src.mapping.collision_proxy import CameraIntrinsics
    from src.depth.metric_scale import triangulated_scale_reference


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
    ORB features; each call matches to the current frame, recovers the relative
    pose, triangulates the inliers, and returns predicted-value / metric-depth
    pairs. Returns None on the first frame and whenever geometry is degenerate —
    the aligner then coasts on its current scale.

    The RGB frame is resized to the depth map's resolution so pixels and
    intrinsics agree (same convention as OdometryPoseProvider).
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        n_features: int = 1500,
        ratio: float = 0.75,
        min_matches: int = 12,
        baseline_m: float = 1.0,
    ):
        self.K = intrinsics
        self._Kmat = _k_matrix(intrinsics)
        self.ratio = ratio
        self.min_matches = min_matches
        self.baseline_m = baseline_m
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (xy, des)

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
            return None

        uv_a = prev_xy[matches[:, 0]]      # previous frame == triangulation frame A
        uv_b = xy[matches[:, 1]]
        pose = estimate_relative_pose(uv_a, uv_b, self._Kmat)
        if pose is None:
            return None
        R, t_unit, inliers = pose
        if int(inliers.sum()) < self.min_matches:
            return None

        # Triangulate against the *previous* frame — but the aligner is fitting
        # THIS frame's rel_depth. We instead triangulate in the current frame's
        # coordinates by swapping the roles (B is the current frame): use the
        # inverse relative pose so depths land at current-frame pixels.
        R_ba = R.T
        t_ba = -R.T @ (self.baseline_m * t_unit)
        ref = triangulated_scale_reference(
            uv_b[inliers], uv_a[inliers], rel_depth, self._Kmat, R_ba, t_ba)
        return ref
