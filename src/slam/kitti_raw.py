"""Loader + stereo depth for KITTI raw drives (Geiger et al., IJRR 2013).

A KITTI raw drive is a directory of the form::

    <base>/2011_09_26/
        calib_cam_to_cam.txt  calib_imu_to_velo.txt  calib_velo_to_cam.txt
        2011_09_26_drive_0009_sync/
            image_02/data/*.png   rectified left colour camera
            image_03/data/*.png   rectified right colour camera
            oxts/data/*.txt       GPS/IMU packet per frame (ground-truth pose)

This is the AV-scale counterpart to :mod:`slam.tum_dataset`: it supplies colour
frames, a *camera-only* metric depth map (OpenCV SGBM on the rectified colour
stereo pair — no lidar, no GPU) and the OXTS ground-truth camera trajectory, so a
recorded drive can be fused into a world-space cloud exactly the way the TUM desk
preview is.

Parsing calibration + OXTS is delegated to **pykitti** (MIT,
https://github.com/utiasSTARS/pykitti) rather than hand-rolled; it is imported
lazily so the rest of the package never needs it. KITTI raw data is CC BY-NC-SA 3.0
and is not redistributed here (``data/`` is gitignored).

Poses are camera-2-to-world 4x4 SE(3), first frame at the identity, so the world
frame is the first camera's: x right, y **down**, z forward.
"""

from __future__ import annotations

import os
from functools import cached_property
from typing import Optional

import numpy as np

try:                                              # package-relative (src on path)
    from mapping.collision_proxy import CameraIntrinsics
except ImportError:                               # fallback for `import src.slam...`
    from src.mapping.collision_proxy import CameraIntrinsics

DEFAULT_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kitti_raw")


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of one (4,4) or a stack (N,4,4) of rigid transforms."""
    R, t = T[..., :3, :3], T[..., :3, 3:]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ t
    out[..., 3, 3] = 1.0
    return out


def stereo_depth(left_gray: np.ndarray, right_gray: np.ndarray, fx: float, baseline: float,
                 num_disparities: int = 128, block_size: int = 5,
                 min_depth: float = 2.0, max_depth: float = 25.0) -> np.ndarray:
    """Metric depth (m) from a rectified grayscale pair via SGBM; 0 marks invalid.

    ``max_depth`` is deliberately short (25 m): at KITTI's fx≈721 px, b≈0.54 m one
    1/16-px disparity step at 25 m is ~5 cm, but at 80 m it is metres — far stereo
    points smear into streaks that ruin a fused cloud.
    """
    import cv2

    b = block_size
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disparities, blockSize=b,
        P1=8 * b * b, P2=32 * b * b,             # OpenCV's recommended smoothness terms
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        mode=cv2.StereoSGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0   # int16 Q4 -> px
    depth = np.zeros_like(disp)
    valid = disp > 0.0
    depth[valid] = fx * baseline / disp[valid]
    depth[(depth < min_depth) | (depth > max_depth)] = 0.0
    return depth


def sky_mask(rgb: np.ndarray, horizon_row: float) -> np.ndarray:
    """True where a pixel above ``horizon_row`` looks like clear sky.

    Sky is textureless, so SGBM "matches" it at arbitrary short range and the
    fused cloud grows light-blue blobs floating over the road. Blue-dominant,
    bright pixels above the horizon are dropped before back-projection.
    """
    rgb = rgb.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    skyish = (b > r + 15) & (b >= g) & (b > 140)
    rows = np.arange(rgb.shape[0])[:, None] < horizon_row
    return skyish & rows


def edge_mask(depth: np.ndarray, rel_jump: float = 0.08) -> np.ndarray:
    """True at depth discontinuities ("flying pixels" between fore/background).

    A pixel is flagged when the depth range over its 3x3 neighbourhood exceeds
    ``rel_jump`` of its own depth — SGBM interpolates across those edges, leaving
    a sheet of points smeared between the object and whatever is behind it.
    """
    import cv2

    k = np.ones((3, 3), np.uint8)
    d = np.where(depth > 0, depth, np.float32(0)).astype(np.float32)
    big = np.where(depth > 0, depth, np.float32(1e6)).astype(np.float32)
    rng = cv2.dilate(d, k) - cv2.erode(big, k)
    return (depth > 0) & (rng > rel_jump * depth)


class KittiRawDrive:
    """Colour stereo + ground-truth pose view over one KITTI raw ``*_sync`` drive."""

    def __init__(self, date: str = "2011_09_26", drive: str = "0009",
                 base_dir: Optional[str] = None):
        import pykitti                                # lazy: optional dependency

        self.base_dir = base_dir or DEFAULT_BASE_DIR
        if not os.path.isdir(os.path.join(self.base_dir, date)):
            raise FileNotFoundError(f"no KITTI raw data at {os.path.join(self.base_dir, date)}")
        self.date, self.drive = date, drive
        self._data = pykitti.raw(self.base_dir, date, drive)

    def __len__(self) -> int:
        return len(self._data.timestamps)

    @cached_property
    def intrinsics(self) -> CameraIntrinsics:
        P = self._data.calib.P_rect_20
        w, h = self._data.get_cam2(0).size
        return CameraIntrinsics(width=w, height=h, fx=float(P[0, 0]), fy=float(P[1, 1]),
                                cx=float(P[0, 2]), cy=float(P[1, 2]))

    @cached_property
    def baseline(self) -> float:
        """Rectified cam2→cam3 baseline (m), from the ``-fx·b`` term of each ``P_rect``."""
        P2, P3 = self._data.calib.P_rect_20, self._data.calib.P_rect_30
        return float(abs(P3[0, 3] - P2[0, 3]) / P2[0, 0])

    def rgb_pair(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Rectified (left, right) colour frames as RGB uint8 arrays."""
        left, right = self._data.get_rgb(i)
        return np.asarray(left), np.asarray(right)

    @cached_property
    def gt_poses(self) -> np.ndarray:
        """OXTS ground-truth cam2-to-world poses (N,4,4), first frame at the identity."""
        T_imu_cam2 = invert_se3(np.asarray(self._data.calib.T_cam2_imu, dtype=np.float64))
        poses = np.stack([np.asarray(o.T_w_imu, dtype=np.float64) @ T_imu_cam2
                          for o in self._data.oxts])
        return invert_se3(poses[0]) @ poses

    @property
    def path_length(self) -> float:
        xyz = self.gt_poses[:, :3, 3]
        return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
