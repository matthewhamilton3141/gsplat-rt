"""Frame-to-frame RGB-D visual odometry — the M6 SLAM front-end (CPU baseline).

Pipeline per frame pair (i -> i+1):
    1. ORB keypoints + descriptors on both grayscale images.
    2. Ratio-test descriptor matching.
    3. Back-project frame-i matches to 3-D using frame-i's metric depth.
    4. solvePnPRansac(3-D_i, 2-D_{i+1}) -> relative camera motion.
    5. Compose onto the running camera-to-world pose.

This is the provider-agnostic baseline: pure OpenCV + numpy, no GPU. It defines
the pose-estimation *interface* and the ATE evaluation harness. Step 1-2 (detect
+ match) are factored behind a pluggable ``Frontend``; the default is ORB, and
the A10G upgrade injects a SuperPoint + LightGlue learned front-end (torch/TRT)
with the same contract, leaving the geometry and mapping wiring untouched.

Pose convention: poses are 4x4 camera-to-world SE(3). solvePnP returns the
extrinsic mapping cam_i coords -> cam_{i+1} coords (T_rel), so the next
camera-to-world pose is  P_{i+1} = P_i @ inv(T_rel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

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


def _rotation_angle(R: np.ndarray) -> float:
    """Geodesic rotation angle (radians) of a 3x3 rotation matrix."""
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


@dataclass
class TrackResult:
    pose: np.ndarray            # (4,4) camera-to-world
    n_matches: int              # descriptor matches this step
    n_inliers: int              # PnP RANSAC inliers (0 if PnP skipped/failed)
    ok: bool                    # True if a pose was estimated (not a fallback)


@dataclass
class Keyframe:
    """One keyframe: a re-usable tracking anchor (Stage 1 of loop closure)."""
    id: int
    pose: np.ndarray                        # (4,4) camera-to-world
    depth: np.ndarray                       # metric depth, for back-projection
    xy: Optional[np.ndarray] = None         # keypoints (detect/match front-ends)
    des: Optional[object] = None            # descriptors (detect/match front-ends)
    rgb: Optional[np.ndarray] = None        # raw frame (pairwise match_pair front-ends)


class KeyframeDB:
    """Ordered store of keyframes. Stage 1 only appends + reads the latest; loop
    closure (Stage 2/3) will add retrieval + a pose graph over these nodes."""

    def __init__(self):
        self.keyframes: List[Keyframe] = []

    def add(self, kf: Keyframe) -> None:
        self.keyframes.append(kf)

    @property
    def current(self) -> Optional[Keyframe]:
        return self.keyframes[-1] if self.keyframes else None

    def __len__(self) -> int:
        return len(self.keyframes)


class Frontend(Protocol):
    """Detect/describe + match contract consumed by :class:`RGBDOdometry`.

    ``detect(rgb) -> (keypoints_xy (N,2) float32, descriptors)``
    ``match(kp0, desc0, kp1, desc1) -> (M,2) int32 array of [idx0, idx1]``

    Keypoint arrays are passed to ``match`` for both frames so a position-aware
    matcher (e.g. LightGlue) can use them; descriptor-NN matchers ignore them.
    """

    def detect(self, rgb: np.ndarray) -> Tuple[np.ndarray, object]: ...

    def match(self, kp0: np.ndarray, desc0, kp1: np.ndarray, desc1) -> np.ndarray: ...


class ORBFrontend:
    """CPU baseline front-end: ORB detect/describe + ratio-tested BF matching.

    Reproduces the original in-line ORB path exactly, so the ATE baseline is
    unchanged. The keypoint arrays handed to :meth:`match` are ignored (Hamming
    descriptor NN is position-free) — they exist only to satisfy the contract.
    """

    def __init__(self, n_features: int = 1500, ratio: float = 0.75):
        self.ratio = ratio
        self._orb = cv2.ORB_create(nfeatures=n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, rgb: np.ndarray) -> Tuple[np.ndarray, object]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            return np.empty((0, 2), np.float32), None
        xy = np.array([k.pt for k in kp], dtype=np.float32)
        return xy, des

    def match(self, kp0, desc0, kp1, desc1) -> np.ndarray:
        """Ratio-tested matches as an (M,2) array of [idx0, idx1]."""
        if desc0 is None or desc1 is None or len(desc0) < 2 or len(desc1) < 2:
            return np.empty((0, 2), np.int32)
        knn = self._matcher.knnMatch(desc0, desc1, k=2)
        good = [[m.queryIdx, m.trainIdx] for m, n in knn
                if m.distance < self.ratio * n.distance]
        return np.array(good, dtype=np.int32) if good else np.empty((0, 2), np.int32)


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
        frontend: Optional[Frontend] = None,
        keyframe: bool = False,
        kf_trans_thresh: float = 0.10,
        kf_rot_thresh_deg: float = 10.0,
        kf_min_inlier_ratio: float = 0.5,
    ):
        self.K = intrinsics
        self._Kmat = _k_matrix(intrinsics)
        self.min_matches = min_matches
        self.ransac_reproj_px = ransac_reproj_px

        # Stage-1 keyframing (opt-in; off = byte-identical frame-to-frame baseline).
        # Track each frame against the current keyframe rather than the immediately
        # previous frame: less short-term drift, and the fused SuperPoint ONNX only
        # re-extracts the keyframe once. A new keyframe is inserted when the camera
        # has moved past a translation/rotation threshold or tracking weakens.
        self.keyframe = keyframe
        self.kf_trans_thresh = kf_trans_thresh
        self.kf_rot_thresh = np.deg2rad(kf_rot_thresh_deg)
        self.kf_min_inlier_ratio = kf_min_inlier_ratio
        self.keyframes = KeyframeDB()

        # Pluggable detect/describe + match front-end. Default = ORB (CPU
        # baseline); the A10G upgrade injects a SuperPoint + LightGlue front-end
        # with the same contract, leaving all geometry/eval below untouched.
        self._frontend = frontend if frontend is not None else ORBFrontend(n_features, ratio)

        self._pose = np.eye(4, dtype=np.float64)
        self._last_rel = np.eye(4, dtype=np.float64)
        self._prev: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None  # kp_xy, des, depth
        self.trajectory: List[np.ndarray] = []

    # -- helpers -------------------------------------------------------------

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
        """Estimate the next camera-to-world pose from an (rgb, depth) pair.

        Dispatches to the pairwise branch when the front-end exposes
        ``match_pair`` (learned matchers like LightGlue jointly match an image
        pair, returning corresponding pixels directly), otherwise the detect +
        descriptor-match branch (the ORB baseline).
        """
        if self.keyframe:
            return self._track_keyframe(rgb, depth, init_pose)
        if hasattr(self._frontend, "match_pair"):
            return self._track_pairwise(rgb, depth, init_pose)
        return self._track_detect_match(rgb, depth, init_pose)

    def _track_detect_match(self, rgb: np.ndarray, depth: np.ndarray,
                            init_pose: Optional[np.ndarray] = None) -> TrackResult:
        xy, des = self._frontend.detect(rgb)

        if self._prev is None:
            if init_pose is not None:
                self._pose = init_pose.astype(np.float64).copy()
            self._prev = (xy, des, depth)
            self.trajectory.append(self._pose.copy())
            return TrackResult(self._pose.copy(), 0, 0, True)

        prev_xy, prev_des, prev_depth = self._prev
        matches = self._frontend.match(prev_xy, prev_des, xy, des)

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

    def _track_pairwise(self, rgb: np.ndarray, depth: np.ndarray,
                        init_pose: Optional[np.ndarray] = None) -> TrackResult:
        """Learned-matcher branch: the front-end matches (prev_rgb, rgb)
        directly to corresponding pixels (uv0, uv1); uv0 back-projects through
        the previous depth into the same RANSAC-PnP as the ORB path."""
        if self._prev is None:
            if init_pose is not None:
                self._pose = init_pose.astype(np.float64).copy()
            self._prev = (rgb, depth)     # pairwise caches the raw frame, not features
            self.trajectory.append(self._pose.copy())
            return TrackResult(self._pose.copy(), 0, 0, True)

        prev_rgb, prev_depth = self._prev
        uv0, uv1 = self._frontend.match_pair(prev_rgb, rgb)
        uv0 = np.asarray(uv0, dtype=np.float32).reshape(-1, 2)
        uv1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
        n_matches = len(uv0)

        ok = False
        n_inliers = 0
        if n_matches >= self.min_matches:
            pts3d_prev, valid = self._backproject(uv0, prev_depth)
            obj = pts3d_prev[valid]
            img = uv1[valid]
            if len(obj) >= self.min_matches:
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj, img, self._Kmat, None,
                    reprojectionError=self.ransac_reproj_px,
                    iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if retval and inliers is not None and len(inliers) >= 6:
                    R, _ = cv2.Rodrigues(rvec)
                    T_rel = np.eye(4)
                    T_rel[:3, :3] = R
                    T_rel[:3, 3] = tvec.ravel()
                    self._last_rel = _invert_se3(T_rel)
                    n_inliers = len(inliers)
                    ok = True

        self._pose = self._pose @ self._last_rel
        self._prev = (rgb, depth)
        self.trajectory.append(self._pose.copy())
        return TrackResult(self._pose.copy(), n_matches, n_inliers, ok)

    # -- Stage-1 keyframe tracking -------------------------------------------

    def _pnp_relative(self, obj: np.ndarray, img: np.ndarray):
        """RANSAC-PnP of 3-D reference points against current-frame pixels.

        Returns ``(rel, n_inliers, ok)`` where ``rel`` is the cam_cur <- cam_ref
        camera-to-world *step* (so ``cur_pose = ref_pose @ rel``), or None on fail.
        """
        if len(obj) < self.min_matches:
            return None, 0, False
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, self._Kmat, None,
            reprojectionError=self.ransac_reproj_px,
            iterationsCount=100, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if retval and inliers is not None and len(inliers) >= 6:
            R, _ = cv2.Rodrigues(rvec)
            T_rel = np.eye(4)                     # cam_ref -> cam_cur extrinsic
            T_rel[:3, :3] = R
            T_rel[:3, 3] = tvec.ravel()
            return _invert_se3(T_rel), len(inliers), True
        return None, 0, False

    def _insert_keyframe(self, pose, xy, des, depth, rgb, pairwise) -> None:
        self.keyframes.add(Keyframe(
            id=len(self.keyframes), pose=pose.copy(), depth=depth,
            xy=None if pairwise else xy, des=None if pairwise else des,
            rgb=rgb.copy() if pairwise else None,
        ))

    def _need_keyframe(self, kf_pose, pose, inlier_ratio) -> bool:
        rel = _invert_se3(kf_pose) @ pose
        trans = float(np.linalg.norm(rel[:3, 3]))
        rot = _rotation_angle(rel[:3, :3])
        return (trans > self.kf_trans_thresh or rot > self.kf_rot_thresh
                or inlier_ratio < self.kf_min_inlier_ratio)

    def _track_keyframe(self, rgb: np.ndarray, depth: np.ndarray,
                        init_pose: Optional[np.ndarray] = None) -> TrackResult:
        """Track against the current keyframe; promote frames to keyframes on demand.

        Works for both front-end kinds via the same seam as ``track``: pairwise
        (``match_pair``) keyframes cache the raw frame; detect/match keyframes cache
        keypoints + descriptors.
        """
        pairwise = hasattr(self._frontend, "match_pair")
        cur_xy, cur_des = (None, None) if pairwise else self._frontend.detect(rgb)

        kf = self.keyframes.current
        if kf is None:                            # first frame → keyframe 0
            pose = (init_pose.astype(np.float64).copy()
                    if init_pose is not None else self._pose)
            self._pose = pose
            self._insert_keyframe(pose, cur_xy, cur_des, depth, rgb, pairwise)
            self.trajectory.append(pose.copy())
            return TrackResult(pose.copy(), 0, 0, True)

        # Correspondences current-frame ↔ current keyframe.
        rel, n_matches, n_inliers, ok = None, 0, 0, False
        if pairwise:
            uv0, uv1 = self._frontend.match_pair(kf.rgb, rgb)
            n_matches = len(uv0)
            if n_matches >= self.min_matches:
                obj3d, valid = self._backproject(uv0, kf.depth)
                rel, n_inliers, ok = self._pnp_relative(
                    obj3d[valid], np.asarray(uv1)[valid].astype(np.float64))
        else:
            matches = self._frontend.match(kf.xy, kf.des, cur_xy, cur_des)
            n_matches = len(matches)
            if n_matches >= self.min_matches:
                obj3d, valid = self._backproject(kf.xy[matches[:, 0]], kf.depth)
                rel, n_inliers, ok = self._pnp_relative(
                    obj3d[valid], cur_xy[matches[:, 1]][valid].astype(np.float64))

        prev_pose = self._pose
        if ok:
            new_pose = kf.pose @ rel
            self._last_rel = _invert_se3(prev_pose) @ new_pose   # step, for coasting
        else:
            new_pose = prev_pose @ self._last_rel                # constant-velocity coast
        self._pose = new_pose
        self.trajectory.append(new_pose.copy())

        # A well-tracked frame that has moved far enough (or whose match is
        # weakening) becomes the next anchor. Don't anchor on a failed step.
        inlier_ratio = n_inliers / max(n_matches, 1)
        if ok and self._need_keyframe(kf.pose, new_pose, inlier_ratio):
            self._insert_keyframe(new_pose, cur_xy, cur_des, depth, rgb, pairwise)

        return TrackResult(new_pose.copy(), n_matches, n_inliers, ok)


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
