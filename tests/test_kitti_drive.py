"""Tests for the KITTI drive preview path (loader filters + fusion + chase camera).

The stereo / sky / edge filters, fusion helpers and camera paths are tested on
synthetic inputs so they run anywhere. The end-to-end row needs a KITTI raw
drive at data/kitti_raw/2011_09_26/..._0009_sync and pykitti; it skips cleanly
without them.

Run:
    pytest tests/test_kitti_drive.py -v
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from slam.kitti_raw import edge_mask, invert_se3, sky_mask, stereo_depth   # noqa: E402
from reconstruct_kitti import backproject_masked, voxel_downsample           # noqa: E402
from render_drive import WORLD_UP, chase_path, crane_path, resample_by_arclength  # noqa: E402
from mapping.collision_proxy import CameraIntrinsics                         # noqa: E402

_DRIVE = os.path.join(_ROOT, "data", "kitti_raw", "2011_09_26", "2011_09_26_drive_0009_sync")
try:
    import pykitti  # noqa: F401
    _HAVE_PYKITTI = True
except ImportError:
    _HAVE_PYKITTI = False
_skip = pytest.mark.skipif(not (os.path.isdir(_DRIVE) and _HAVE_PYKITTI),
                           reason="KITTI raw drive 0009 / pykitti not available")


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_invert_se3_single_and_stacked():
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = _rot_y(0.7), [1.0, -2.0, 3.0]
    assert np.allclose(invert_se3(T) @ T, np.eye(4), atol=1e-12)
    stack = np.stack([T, np.eye(4)])
    assert np.allclose(invert_se3(stack) @ stack, np.eye(4), atol=1e-12)


def test_stereo_depth_recovers_known_shift():
    # Fronto-parallel textured plane shifted by a known disparity d → depth fx·b/d.
    rng = np.random.default_rng(0)
    tex = (rng.random((120, 400)) * 255).astype(np.uint8)
    d = 16
    left, right = tex[:, :-d], tex[:, d:]                # right[x - d] == left[x]
    fx, b = 700.0, 0.5
    depth = stereo_depth(left, right, fx, b, num_disparities=64,
                         min_depth=1.0, max_depth=100.0)
    valid = depth[:, 80:-10][depth[:, 80:-10] > 0]       # skip the no-match left band
    assert valid.size > 0.8 * depth[:, 80:-10].size
    assert np.median(valid) == pytest.approx(fx * b / d, rel=0.02)


def test_stereo_depth_range_cap_marks_invalid():
    rng = np.random.default_rng(1)
    tex = (rng.random((80, 300)) * 255).astype(np.uint8)
    left, right = tex[:, :-4], tex[:, 4:]                # d=4 → 87.5 m, beyond cap
    depth = stereo_depth(left, right, 700.0, 0.5, num_disparities=32, max_depth=25.0)
    assert not (depth > 25.0).any()
    uncapped = stereo_depth(left, right, 700.0, 0.5, num_disparities=32, max_depth=100.0)
    assert (uncapped > 25.0).mean() > 0.5                # the cap is what removed them


def test_sky_mask_only_above_horizon_and_blue():
    img = np.zeros((10, 4, 3), np.uint8)
    img[:] = (120, 170, 230)                             # clear-sky blue everywhere
    img[:, 3] = (120, 120, 120)                          # a grey column (building)
    m = sky_mask(img, horizon_row=5)
    assert m[:5, :3].all()                               # sky above horizon
    assert not m[5:].any()                               # same colour below → kept
    assert not m[:, 3].any()                             # grey never sky


def test_edge_mask_flags_depth_discontinuity_not_flat():
    depth = np.full((20, 20), 10.0, np.float32)
    depth[:, 10:] = 20.0                                 # 10 m → 20 m step
    m = edge_mask(depth)
    assert m[:, 9:11].all()                              # both sides of the step
    assert not m[:, :8].any() and not m[:, 12:].any()    # flat regions untouched


def test_edge_mask_ignores_invalid_neighbours():
    depth = np.full((10, 10), 10.0, np.float32)
    depth[:, 5:] = 0.0                                   # invalid, not a jump
    assert not edge_mask(depth).any()


def test_backproject_masked_places_points_by_pose():
    K = CameraIntrinsics(fx=100.0, fy=100.0, cx=2.0, cy=2.0, width=5, height=5)
    depth = np.zeros((5, 5), np.float32)
    depth[2, 2] = 4.0                                    # principal point, 4 m ahead
    rgb = np.zeros((5, 5, 3), np.uint8)
    rgb[2, 2] = (9, 8, 7)
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 0.0, 10.0]
    pts, cols = backproject_masked(depth, rgb, K, pose, pixel_stride=1)
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0], [1.0, 0.0, 14.0])
    assert cols[0].tolist() == [9, 8, 7]


def test_voxel_downsample_one_point_per_cell():
    pts = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.30, 0, 0]], np.float32)
    cols = np.arange(9, dtype=np.uint8).reshape(3, 3)
    p, c = voxel_downsample(pts, cols, voxel=0.1)
    assert len(p) == 2
    assert c[0].tolist() == [0, 1, 2]                    # first point in the cell kept


def test_resample_by_arclength_is_uniform():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 9]], float)   # 1 m then 9 m
    pts, length = resample_by_arclength(xyz, 11)
    assert length == pytest.approx(10.0)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.allclose(steps, 1.0, atol=1e-9)


def test_chase_camera_trails_above_and_looks_ahead():
    # Straight road along +z (KITTI forward); world up is -y.
    xyz = np.stack([np.zeros(200), np.zeros(200), np.linspace(0, 200, 200)], axis=-1)
    eye, tgt = chase_path(xyz, 100, back=10.0, height=5.0, look_ahead=20.0)
    car, _ = resample_by_arclength(xyz, 100)
    mid = 50
    assert eye[mid] @ WORLD_UP == pytest.approx(5.0, abs=1e-6)       # above the road
    assert car[mid, 2] - eye[mid, 2] == pytest.approx(10.0, abs=0.5)  # behind the car
    assert tgt[mid, 2] - car[mid, 2] == pytest.approx(20.0, abs=2.5)  # looks ahead


def test_crane_starts_at_chase_and_ends_elevated():
    start_eye, start_tgt = np.array([0.0, -5.0, -10.0]), np.zeros(3)
    centre = np.array([0.0, 0.0, 50.0])
    eye, tgt = crane_path(start_eye, start_tgt, centre, span=100.0, n=30, elevation=55.0)
    assert np.allclose(eye[0], start_eye) and np.allclose(tgt[0], start_tgt)
    assert np.allclose(tgt[-1], centre)
    v = eye[-1] - centre
    el = np.degrees(np.arcsin((v @ WORLD_UP) / np.linalg.norm(v)))
    assert el == pytest.approx(55.0, abs=1e-6)          # oblique, never straight down


@_skip
def test_real_drive_fuses_coherent_street():
    from slam.kitti_raw import KittiRawDrive
    from reconstruct_kitti import fuse_drive

    drive = KittiRawDrive("2011_09_26", "0009")
    assert len(drive) == 447
    assert 0.5 < drive.baseline < 0.6                    # KITTI colour pair ≈ 0.54 m
    # KITTI's calibration rotations are only orthonormal to ~1e-7, so pose 0 is
    # identity to that precision, not exactly.
    assert np.allclose(drive.gt_poses[0], np.eye(4), atol=1e-5)
    assert drive.path_length == pytest.approx(333.0, abs=5.0)

    # Fuse a short stretch: every point must land near the driven route.
    class _First(KittiRawDrive):
        def __len__(self):
            return 30

    short = _First("2011_09_26", "0009")
    pts, cols = fuse_drive(short, frame_stride=10, pixel_stride=8, log=lambda *_: None)
    assert len(pts) > 5_000 and pts.dtype == np.float32 and cols.shape == pts.shape
    route = drive.gt_poses[:30, :3, 3]
    d = np.min(np.linalg.norm(pts[:, None, :] - route[None, ::5], axis=-1), axis=1)
    assert np.percentile(d, 99) < 30.0                   # within the 25 m stereo cap


@_skip
def test_triangulated_anchor_agrees_with_stereo_on_real_frames():
    import cv2
    from slam.kitti_raw import KittiRawDrive
    from reconstruct_kitti import triangulated_anchor

    drive = KittiRawDrive("2011_09_26", "0009")
    K, P = drive.intrinsics, drive.gt_poses
    for i in (30, 300):
        left, right = drive.rgb_pair(i)
        gl, gr = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (left, right))
        gb = cv2.cvtColor(drive.rgb_pair(i + 3)[0], cv2.COLOR_RGB2GRAY)
        uv, z = triangulated_anchor(gl, gb, K, P[i], P[i + 3])
        assert len(z) > 100
        s = stereo_depth(gl, gr, K.fx, drive.baseline, max_depth=40.0)
        s = s[np.rint(uv[:, 1]).astype(int), np.rint(uv[:, 0]).astype(int)]
        m = s > 0
        # Two independent metric sources (GT-pose triangulation vs stereo baseline).
        assert np.median(np.abs(z[m] - s[m]) / s[m]) < 0.08


@_skip
def test_mono_source_recovers_metric_depth_from_affine_relative_map():
    """A stand-in "mono model" whose output is an unknown affine of inverse depth
    (what Depth Anything emits) must come back metric after alignment."""
    import cv2
    from slam.kitti_raw import KittiRawDrive
    from reconstruct_kitti import MonoDepthSource

    drive = KittiRawDrive("2011_09_26", "0009")
    K = drive.intrinsics

    def gt_like(i):
        gl, gr = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in drive.rgb_pair(i))
        return stereo_depth(gl, gr, K.fx, drive.baseline, max_depth=40.0)

    def fake_infer(bgr):
        z = gt_like(fake_infer.i)
        disp = np.where(z > 0, 1.0 / np.maximum(z, 1e-3), 0).astype(np.float32)
        holes = (z == 0).astype(np.uint8)
        disp = cv2.inpaint(disp, holes, 3, cv2.INPAINT_NS)   # dense, like a real model
        return cv2.resize(7.3 * disp + 0.02, (518, 518))    # unknown scale + shift

    src = MonoDepthSource(drive, fake_infer, max_depth=25.0)
    errs = []
    for i in range(0, 60, 6):
        fake_infer.i = i
        left, right = drive.rgb_pair(i)
        d = src(i, left, right)
        z = gt_like(i)
        m = (d > 0) & (z > 0) & (z < 25.0)
        errs.append(np.median(np.abs(d[m] - z[m]) / z[m]))
    assert src.aligner.n_fits == len(errs)
    assert np.median(errs) < 0.10
