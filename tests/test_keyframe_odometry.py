"""Stage-1 keyframing for the SLAM front-end (src/slam/rgbd_odometry.py).

Frame-to-keyframe tracking + keyframe insertion, exercised deterministically on a
synthetic scene (no GPU, no real features): a fake detect/match front-end returns
exact projections keyed by frame index, pairing keypoints by a point-ID descriptor.
Verifies the KeyframeDB, insertion triggers, trajectory accuracy, and that keyframe
mode is opt-in (off ⇒ no keyframes, the untouched frame-to-frame baseline).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mapping.collision_proxy import CameraIntrinsics                      # noqa: E402
from slam.rgbd_odometry import KeyframeDB, Keyframe, RGBDOdometry          # noqa: E402


def _K():
    return CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def _project(Pc, K):
    z = Pc[:, 2]
    return np.stack([K.fx * Pc[:, 0] / z + K.cx, K.fy * Pc[:, 1] / z + K.cy], axis=-1)


def _make_scene(n_frames=12, step=0.03, seed=0):
    """A camera translating +x through a fixed world cloud (identity rotation).

    Returns per-frame (uv float[M,2], ids int[M], depth map, c2w pose), keeping
    only points visible in *every* frame so correspondences are complete.
    """
    K = _K()
    rng = np.random.default_rng(seed)
    N = 300
    Pw = np.stack([rng.uniform(-1.0, 1.0, N), rng.uniform(-0.8, 0.8, N),
                   rng.uniform(1.5, 3.0, N)], axis=-1)
    centers = np.array([[i * step, 0.0, 0.0] for i in range(n_frames)])

    # Visibility ∩ over all frames (in front + inside the image).
    vis = np.ones(N, bool)
    for c in centers:
        Pc = Pw - c                                 # identity rotation
        uv = _project(Pc, K)
        vis &= (Pc[:, 2] > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < 640) \
            & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
    Pw = Pw[vis]
    ids = np.arange(Pw.shape[0])

    frames = []
    for i, c in enumerate(centers):
        Pc = Pw - c
        uv = _project(Pc, K)
        depth = np.zeros((480, 640), np.float32)
        for (u, v), z in zip(uv, Pc[:, 2]):
            depth[int(round(v)), int(round(u))] = z
        pose = np.eye(4)
        pose[:3, 3] = c                             # c2w: camera center = c
        frames.append((uv, ids, depth, pose))
    return frames


class _FakeDetectMatch:
    """detect(rgb)->(xy, ids); match pairs keypoints by shared point ID. The frame
    index is stashed in rgb[0,0,0] so detect can return that frame's projections."""

    def __init__(self, frames):
        self._frames = frames

    def detect(self, rgb):
        i = int(rgb[0, 0, 0])
        uv, ids, _, _ = self._frames[i]
        return uv, ids

    def match(self, xy0, des0, xy1, des1):
        lookup = {int(pid): j for j, pid in enumerate(des1)}
        return np.array([[j0, lookup[int(pid)]] for j0, pid in enumerate(des0)
                         if int(pid) in lookup], dtype=np.int32)


def _rgb_for(i):
    img = np.zeros((480, 640, 3), np.uint8)
    img[0, 0, 0] = i
    return img


# ---------------------------------------------------------------------------
# KeyframeDB
# ---------------------------------------------------------------------------

def test_keyframe_db_basics():
    db = KeyframeDB()
    assert len(db) == 0 and db.current is None
    kf = Keyframe(id=0, pose=np.eye(4), depth=np.zeros((4, 4), np.float32))
    db.add(kf)
    assert len(db) == 1 and db.current is kf


# ---------------------------------------------------------------------------
# Keyframe tracking
# ---------------------------------------------------------------------------

def test_first_frame_becomes_keyframe():
    frames = _make_scene()
    odo = RGBDOdometry(_K(), frontend=_FakeDetectMatch(frames), keyframe=True)
    odo.track(_rgb_for(0), frames[0][2])
    assert len(odo.keyframes) == 1
    assert np.allclose(odo.keyframes.current.pose, np.eye(4))


def test_keyframes_inserted_as_camera_moves_and_trajectory_tracks():
    frames = _make_scene(n_frames=12, step=0.03)      # 0.33 m total, thresh 0.10 m
    odo = RGBDOdometry(_K(), frontend=_FakeDetectMatch(frames),
                       keyframe=True, kf_trans_thresh=0.10)
    for i, (_, _, depth, _) in enumerate(frames):
        odo.track(_rgb_for(i), depth)

    # Multiple keyframes were spawned as the baseline grew past 0.10 m.
    assert len(odo.keyframes) >= 2
    # Recovered trajectory matches ground truth (exact synthetic correspondences).
    for est, (_, _, _, gt) in zip(odo.trajectory, frames):
        assert np.linalg.norm(est[:3, 3] - gt[:3, 3]) < 5e-3


def test_keyframe_off_creates_no_keyframes():
    """Opt-in: the default frame-to-frame path never touches the KeyframeDB."""
    frames = _make_scene(n_frames=5)
    odo = RGBDOdometry(_K(), frontend=_FakeDetectMatch(frames))   # keyframe=False
    for i, (_, _, depth, _) in enumerate(frames):
        odo.track(_rgb_for(i), depth)
    assert len(odo.keyframes) == 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
