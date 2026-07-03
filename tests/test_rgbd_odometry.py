"""Tests for the RGB-D visual-odometry front-end (M6 tracker).

Two layers:
  1. Pure geometry — Umeyama alignment and ATE, on synthetic data (no dataset).
  2. End-to-end — track a fr1/desk segment and assert the ATE baseline holds.
     Skips if the sequence isn't downloaded.

Run:
    pytest tests/test_rgbd_odometry.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.rgbd_odometry import RGBDOdometry, align_umeyama, ate_rmse
from slam.tum_dataset import TUMDataset

_SEQ = os.path.join(
    os.path.dirname(__file__), "..", "data", "tum", "rgbd_dataset_freiburg1_desk"
)
_skip = pytest.mark.skipif(not os.path.isdir(_SEQ), reason="fr1/desk not downloaded")


# -- geometry (no dataset) ---------------------------------------------------

def test_align_umeyama_recovers_known_transform():
    rng = np.random.default_rng(0)
    src = rng.standard_normal((50, 3))
    theta = 0.6
    R_true = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
    t_true = np.array([1.0, -2.0, 0.5])
    dst = (R_true @ src.T).T + t_true

    R, t, s = align_umeyama(src, dst)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-6)


def test_ate_is_zero_under_rigid_transform():
    # A trajectory that differs from GT by only a rigid transform has ~0 ATE.
    rng = np.random.default_rng(1)
    gt = np.tile(np.eye(4), (30, 1, 1))
    gt[:, :3, 3] = np.cumsum(rng.standard_normal((30, 3)) * 0.1, axis=0)

    theta = 0.4
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    est = gt.copy()
    est[:, :3, 3] = (R @ gt[:, :3, 3].T).T + np.array([5.0, 1.0, -3.0])

    rmse, _ = ate_rmse(est, gt)
    assert rmse < 1e-6


# -- end to end (needs dataset) ----------------------------------------------

@_skip
def test_odometry_tracks_fr1_desk():
    ds = TUMDataset(_SEQ)
    frames = ds.frames[:120]
    odom = RGBDOdometry(ds.intrinsics)

    est, gt, ok = [], [], 0
    for i, f in enumerate(frames):
        r = odom.track(f.load_rgb(), f.load_depth(),
                       init_pose=f.pose if i == 0 else None)
        est.append(r.pose); gt.append(f.pose); ok += int(r.ok)

    rmse, _ = ate_rmse(np.stack(est), np.stack(gt))
    # Most frames should track via PnP, not the constant-velocity fallback.
    assert ok >= int(0.9 * len(frames)), f"only {ok}/{len(frames)} PnP-tracked"
    # Baseline frame-to-frame ORB+PnP on this segment sits well under 15 cm.
    assert rmse < 0.15, f"ATE-RMSE {rmse*100:.1f} cm exceeds baseline budget"
