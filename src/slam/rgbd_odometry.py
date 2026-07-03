"""Frame-to-frame RGB-D visual odometry — the M6 SLAM front-end (CPU baseline).

Pipeline per frame pair (i -> i+1):
    1. ORB keypoints + descriptors on both grayscale images.
    2. Ratio-test descriptor matching.
    3. Back-project frame-i matches to 3-D using frame-i's metric depth.
    4. solvePnPRansac(3-D_i, 2-D_{i+1}) -> relative camera motion.
    5. Compose onto the running camera-to-world pose.

This is the provider-agnostic baseline: pure OpenCV + numpy, no GPU. It defines
the pose-estimation *interface* and the ATE evaluation harness. The A10G upgrade
swaps step 1-2 for a SuperPoint + SuperGlue learned front-end (needs torch/TRT)
without touching the geometry or the mapping wiring.

Pose convention: poses are 4x4 camera-to-world SE(3). solvePnP returns the
extrinsic mapping cam_i coords -> cam_{i+1} coords (T_rel), so the next
camera-to-world pose is  P_{i+1} = P_i @ inv(T_rel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from mapping.collision_proxy import CameraIntrinsics
except ImportError:
    from src.mapping.collision_proxy import CameraIntrinsics


def _k_matrix(K: CameraIntrinsics) -> np.ndarray:
    return np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1]], dtype=np.float64)


def _invert_se3(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


@dataclass
class TrackResult:
    pose: np.ndarray            # (4,4) camera-to-world
    n_matches: int              # descriptor matches this step
    n_inliers: int              # PnP RANSAC inliers (0 if PnP skipped/failed)
    ok: bool                    # True if a pose was estimated (not a fallback)


class RGBDOdometry:
    """Stateful frame-to-frame RGB-D visual odometer.

    Call :meth:`track` with each (rgb, depth) pair in order. The first call
    seeds the reference frame and returns the initial pose (identity unless
    ``init_pose`` is given). Each later call estimates motion from the previous
    frame and returns the updated camera-to-world pose.

    On a degenerate step (too few matches or a failed PnP) the last relative
    motion is re-applied (constant-velocity fallback) so tracking never stalls;
    ``TrackResult.ok`` is False for those steps.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        n_features: int = 1500,
        ratio: float = 0.75,
        min_matches: int = 12,
        ransac_reproj_px: float = 3.0,
    ):
        self.K = intrinsics
        self._Kmat = _k_matrix(intrinsics)
        self.ratio = ratio
        self.min_matches = min_matches
        self.ransac_reproj_px = ransac_reproj_px

        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._pose = np.eye(4, dtype=np.float64)
        self._last_rel = np.eye(4, dtype=np.float64)
        self._prev: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # kp_xy, des, depth
        self.trajectory: List[np.ndarray] = []

    # -- helpers -------------------------------------------------------------

    def _features(self, rgb: np.ndarray):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            return np.empty((0, 2), np.float32), None
        xy = np.array([k.pt for k in kp], dtype=np.float32)
        return xy, des

    def _match(self, des_a, des_b) -> np.ndarray:
        """Ratio-tested matches as an (M,2) array of [idx_a, idx_b]."""
        if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
            return np.empty((0, 2), np.int32)
        knn = self._matcher.knnMatch(des_a, des_b, k=2)
        good = [[m.queryIdx, m.trainIdx] for m, n in knn if m.distance < self.ratio * n.distance]
        return np.array(good, dtype=np.int32) if good else np.empty((0, 2), np.int32)

    def _backproject(self, xy: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pixels + depth -> (3-D camera points, valid mask)."""
        u, v = xy[:, 0], xy[:, 1]
        ui = np.clip(np.rint(v).astype(int), 0, depth.shape[0] - 1)
        uj = np.clip(np.rint(u).astype(int), 0, depth.shape[1] - 1)
        z = depth[ui, uj]
        valid = z > 0.1
        x = (u - self.K.cx) * z / self.K.fx
        y = (v - self.K.cy) * z / self.K.fy
        return np.stack([x, y, z], axis=-1).astype(np.float64), valid

    # -- main API ------------------------------------------------------------

    def track(self, rgb: np.ndarray, depth: np.ndarray,
              init_pose: Optional[np.ndarray] = None) -> TrackResult:
        xy, des = self._features(rgb)

        if self._prev is None:
            if init_pose is not None:
                self._pose = init_pose.astype(np.float64).copy()
            self._prev = (xy, des, depth)
            self.trajectory.append(self._pose.copy())
            return TrackResult(self._pose.copy(), 0, 0, True)

        prev_xy, prev_des, prev_depth = self._prev
        matches = self._match(prev_des, des)

        ok = False
        n_inliers = 0
        if len(matches) >= self.min_matches:
            pts3d_prev, valid = self._backproject(prev_xy[matches[:, 0]], prev_depth)
            obj = pts3d_prev[valid]
            img = xy[matches[:, 1]][valid].astype(np.float64)
            if len(obj) >= self.min_matches:
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj, img, self._Kmat, None,
                    reprojectionError=self.ransac_reproj_px,
                    iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if retval and inliers is not None and len(inliers) >= 6:
                    R, _ = cv2.Rodrigues(rvec)
                    T_rel = np.eye(4)                 # cam_prev -> cam_cur extrinsic
                    T_rel[:3, :3] = R
                    T_rel[:3, 3] = tvec.ravel()
                    self._last_rel = _invert_se3(T_rel)   # cam_cur <- cam_prev, as c2w step
                    n_inliers = len(inliers)
                    ok = True

        # Compose (or coast on last relative motion if this step failed)
        self._pose = self._pose @ self._last_rel
        self._prev = (xy, des, depth)
        self.trajectory.append(self._pose.copy())
        return TrackResult(self._pose.copy(), len(matches), n_inliers, ok)


class OdometryPoseProvider:
    """Adapts RGBDOdometry to the PipelineManager pose-provider contract.

    Callable as ``provider(frame_bgr, depth) -> (4,4) camera-to-world``. The RGB
    frame is resized to the depth map's resolution so pixels and intrinsics
    agree. Intended for metric, scale-consistent depth (RGB-D sensor / TUM);
    on monocular relative depth the estimated scale drifts frame to frame.
    """

    def __init__(self, intrinsics: CameraIntrinsics, **kwargs):
        self._odom = RGBDOdometry(intrinsics, **kwargs)
        self._hw = (intrinsics.height, intrinsics.width)

    def __call__(self, frame_bgr: np.ndarray, depth: np.ndarray) -> np.ndarray:
        if frame_bgr.shape[:2] != depth.shape[:2]:
            frame_bgr = cv2.resize(frame_bgr, (depth.shape[1], depth.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        return self._odom.track(frame_bgr, depth).pose


# ---------------------------------------------------------------------------
# Trajectory evaluation — Absolute Trajectory Error (TUM standard)
# ---------------------------------------------------------------------------

def align_umeyama(src: np.ndarray, dst: np.ndarray,
                  with_scale: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rigid (optionally similarity) alignment mapping src -> dst.

    Returns (R, t, s) minimising sum || s*R*src_i + t - dst_i ||^2 (Kabsch/Umeyama).
    RGB-D depth is metric, so scale is fixed to 1 by default.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    H = sc.T @ dc / len(src)
    U, D, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    S = np.diag([1.0, 1.0, d])
    R = Vt.T @ S @ U.T
    s = (D * np.array([1, 1, d])).sum() / (sc ** 2).sum() * len(src) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def ate_rmse(est_poses: np.ndarray, gt_poses: np.ndarray,
             with_scale: bool = False) -> Tuple[float, np.ndarray]:
    """Absolute Trajectory Error (RMSE, metres) after best rigid alignment.

    est_poses, gt_poses : (N,4,4) camera-to-world, frame-associated.
    Returns (rmse, per_frame_errors).
    """
    est = np.asarray(est_poses)[:, :3, 3]
    gt = np.asarray(gt_poses)[:, :3, 3]
    R, t, s = align_umeyama(est, gt, with_scale)
    aligned = (s * (R @ est.T)).T + t
    err = np.linalg.norm(aligned - gt, axis=1)
    return float(np.sqrt((err ** 2).mean())), err
