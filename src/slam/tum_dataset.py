"""Loader for the TUM RGB-D SLAM benchmark (Sturm et al., IROS 2012).

A TUM sequence is a directory of the form::

    rgbd_dataset_freiburg1_desk/
        rgb/            <timestamp>.png   8-bit BGR, 640x480
        depth/          <timestamp>.png   16-bit, depth_m = pixel / 5000
        rgb.txt         timestamp -> rgb filename
        depth.txt       timestamp -> depth filename
        groundtruth.txt timestamp tx ty tz qx qy qz qw   (motion-capture pose)

The three streams are logged on independent clocks, so this loader associates
them by nearest timestamp (the same greedy scheme as TUM's own associate.py):
each RGB frame is paired with the closest depth frame and the closest
ground-truth pose, dropping frames with no match inside `max_diff` seconds.

This is the M6 data foundation: it supplies *real* metric depth and a
*ground-truth* camera trajectory, so the mapping path can be validated with
known poses before the visual-odometry front-end (rgbd_odometry.py) estimates
its own. Poses are camera-to-world 4x4 SE(3) matrices.

Intrinsics are the published freiburg1 pinhole model. Lens distortion (the
fr1 d-coefficients) is not undistorted here; note it as a refinement for the
tracker rather than the loader.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Reuse the pipeline's pinhole model so intrinsics flow straight into the TSDF.
try:                                              # package-relative (src on path)
    from mapping.collision_proxy import CameraIntrinsics
except ImportError:                               # fallback for `import src.slam...`
    from src.mapping.collision_proxy import CameraIntrinsics


# Published intrinsics per TUM camera. See:
# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
_INTRINSICS_BY_CAM = {
    "freiburg1": dict(fx=517.306408, fy=516.469215, cx=318.643040, cy=255.313989),
    "freiburg2": dict(fx=520.908620, fy=521.007327, cx=325.141442, cy=249.701764),
    "freiburg3": dict(fx=535.4,      fy=539.2,      cx=320.1,      cy=247.6),
}
_DEFAULT_CAM = "freiburg1"

# TUM depth PNGs are stored as uint16 where 5000 counts == 1 metre.
DEPTH_SCALE = 5000.0


def quaternion_to_matrix(tx: float, ty: float, tz: float,
                         qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """(translation, unit quaternion xyzw) -> 4x4 camera-to-world SE(3)."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 0.0 if n == 0.0 else 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s

    T = np.eye(4, dtype=np.float32)
    T[0, 0] = 1.0 - (yy + zz); T[0, 1] = xy - wz;         T[0, 2] = xz + wy
    T[1, 0] = xy + wz;         T[1, 1] = 1.0 - (xx + zz); T[1, 2] = yz - wx
    T[2, 0] = xz - wy;         T[2, 1] = yz + wx;         T[2, 2] = 1.0 - (xx + yy)
    T[0, 3] = tx; T[1, 3] = ty; T[2, 3] = tz
    return T


def _read_tum_txt(path: str) -> List[List[float]]:
    """Read a whitespace TUM index/trajectory file, skipping '#' comments."""
    rows: List[List[float]] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows


def _nearest(sorted_ts: List[float], query: float) -> int:
    """Index into sorted_ts of the entry closest to query (assumes non-empty)."""
    i = bisect.bisect_left(sorted_ts, query)
    if i == 0:
        return 0
    if i >= len(sorted_ts):
        return len(sorted_ts) - 1
    before, after = sorted_ts[i - 1], sorted_ts[i]
    return i if (after - query) < (query - before) else i - 1


@dataclass
class TUMFrame:
    """One associated (rgb, depth, pose) sample. Pixels are loaded on demand."""
    timestamp: float
    rgb_path: str
    depth_path: str
    pose: np.ndarray            # (4,4) float32 camera-to-world

    def load_rgb(self) -> np.ndarray:
        """8-bit BGR image (H, W, 3) as OpenCV returns it."""
        import cv2
        img = cv2.imread(self.rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.rgb_path)
        return img

    def load_depth(self) -> np.ndarray:
        """Metric depth (H, W) float32 in metres; 0.0 marks invalid pixels."""
        import cv2
        raw = cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(self.depth_path)
        return raw.astype(np.float32) / DEPTH_SCALE


class TUMDataset:
    """Associated view over one extracted TUM RGB-D sequence.

    Parameters
    ----------
    root : str
        Path to the extracted ``rgbd_dataset_freiburgX_*`` directory.
    max_diff : float
        Max timestamp gap (s) tolerated when pairing rgb<->depth<->pose.
        Frames without a match within this window are dropped.

    Frames are exposed via ``len()``, indexing, and iteration; each is a
    :class:`TUMFrame`. Camera model is on ``.intrinsics``.
    """

    def __init__(self, root: str, max_diff: float = 0.02):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"TUM sequence directory not found: {root}")
        self.root = root
        self.max_diff = float(max_diff)
        self.intrinsics = self._intrinsics_for(root)
        self.frames: List[TUMFrame] = self._associate()
        if not self.frames:
            raise RuntimeError(
                f"No rgb/depth/pose triples matched within {max_diff}s in {root}"
            )

    # -- setup ---------------------------------------------------------------

    @staticmethod
    def _intrinsics_for(root: str) -> CameraIntrinsics:
        name = os.path.basename(os.path.normpath(root))
        cam = next((c for c in _INTRINSICS_BY_CAM if c in name), _DEFAULT_CAM)
        k = _INTRINSICS_BY_CAM[cam]
        return CameraIntrinsics(width=640, height=480, **k)

    def _associate(self) -> List[TUMFrame]:
        rgb = [(float(r[0]), r[1]) for r in _read_tum_txt(os.path.join(self.root, "rgb.txt"))]
        depth = [(float(r[0]), r[1]) for r in _read_tum_txt(os.path.join(self.root, "depth.txt"))]
        gt_rows = _read_tum_txt(os.path.join(self.root, "groundtruth.txt"))

        depth_ts = [t for t, _ in depth]
        gt_ts = [float(r[0]) for r in gt_rows]

        frames: List[TUMFrame] = []
        for t_rgb, rgb_file in rgb:
            di = _nearest(depth_ts, t_rgb)
            gi = _nearest(gt_ts, t_rgb)
            if abs(depth_ts[di] - t_rgb) > self.max_diff:
                continue
            if abs(gt_ts[gi] - t_rgb) > self.max_diff:
                continue
            row = gt_rows[gi]
            pose = quaternion_to_matrix(*(float(x) for x in row[1:8]))
            frames.append(TUMFrame(
                timestamp=t_rgb,
                rgb_path=os.path.join(self.root, rgb_file),
                depth_path=os.path.join(self.root, depth[di][1]),
                pose=pose,
            ))
        return frames

    # -- access --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> TUMFrame:
        return self.frames[i]

    def __iter__(self):
        return iter(self.frames)

    def poses(self) -> np.ndarray:
        """Ground-truth trajectory as (N, 4, 4) camera-to-world matrices."""
        return np.stack([f.pose for f in self.frames], axis=0)
