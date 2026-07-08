"""Pipeline integration for the metric-scale stage (src/pipeline_manager.py).

Uses the mock depth estimator (no GPU) driven by a real temp video clip — the
same robust pattern as test_finalize_pipeline — so it runs on the dev machine.
Verifies:
  - disabled by default → depth consumers see raw depth (no behaviour change);
  - enabled with a reference → depth is rescaled to metric *before* the pose
    provider / TSDF / Gaussian back-projection consume it.

We tap the depth the consumers actually receive via a recording pose provider,
which the coordinator calls with the (already-aligned) depth map.
"""

import os
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_manager import PipelineConfig, PipelineManager  # noqa: E402


def _make_video(path, n_frames=120, fps=120.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    yy, xx = np.mgrid[0:480, 0:640]
    base = np.stack([
        (xx / 640.0 * 255).astype(np.uint8),
        (yy / 480.0 * 255).astype(np.uint8),
        np.full_like(xx, 128, dtype=np.uint8),
    ], axis=-1)
    for i in range(n_frames):
        frame = base.copy()
        x0 = int(20 + i) % 560
        frame[200:280, x0:x0 + 80] = (255, 255, 255)
        writer.write(frame)
    writer.release()


class _RecordingPose:
    """Pose provider that records the depth maps it is handed (thread-safe)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.depths = []

    def __call__(self, frame, depth):
        with self._lock:
            self.depths.append(np.array(depth, copy=True))
        return np.eye(4, dtype=np.float64)

    def latest(self):
        with self._lock:
            return None if not self.depths else self.depths[-1]


def _base_cfg(tmp, **kw):
    base = dict(
        output_dir=tmp, usd_stem="mscale",
        # Force the mock depth estimator regardless of environment: on a GPU box
        # a real engine at the default path would otherwise be auto-selected and
        # break the exact-scale asserts below (this file is mock-by-design).
        engine_path="/nonexistent/force-mock.engine",
        depth_input_h=64, depth_input_w=64, gaussian_sample_step=8,
        write_previews=False, usd_update_interval_s=60.0,
        usd_update_frame_count=10_000, tsdf_grid_dim=32, tsdf_voxel_size=0.10,
    )
    base.update(kw)
    return PipelineConfig(**base)


def _run(cfg, **mgr_kw):
    rec = _RecordingPose()
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "clip.mp4")
        _make_video(video)
        cfg.video_source = video
        cfg.output_dir = tmp
        mgr = PipelineManager(cfg, pose_provider=rec, **mgr_kw)
        with mgr:
            t0 = time.time()
            while mgr.frames_processed < 5 and time.time() - t0 < 3.0:
                time.sleep(0.02)
            stats = mgr.stats()
        assert mgr.frames_processed > 0, "no frames processed"
        return mgr, rec, stats


def test_metric_scale_disabled_by_default_passes_raw_depth():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _base_cfg(tmp)
    assert cfg.metric_scale_enabled is False
    mgr, rec, _ = _run(cfg)
    assert mgr._aligner is None
    got = rec.latest()
    assert got is not None
    # Mock depth is a ~2 m bowl; unscaled it stays ~2 m.
    assert 1.5 < float(np.median(got)) < 3.5


def test_metric_scale_rescales_depth_before_consumers():
    # Force a known scale: reference pairs each frame's own predicted values with
    # 3x those values → in depth space the fit locks scale to 3.0.
    def scale_reference(frame, depth):
        d = depth.ravel()
        idx = np.linspace(0, d.size - 1, 200).astype(int)
        pv = d[idx]
        return pv, 3.0 * pv

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _base_cfg(tmp, metric_scale_enabled=True,
                        metric_scale_space="depth", metric_scale_smoothing=0.0,
                        metric_scale_min_points=10, metric_scale_clamp=None)
        mgr, rec, stats = _run(cfg, scale_reference=scale_reference)

        assert mgr._aligner is not None
        got = rec.latest()
        assert got is not None
        # Depth the pose provider received is 3x the raw mock bowl (~2.4 m → ~7.2 m).
        assert 6.5 < float(np.median(got)) < 8.0
        assert stats.get("metric_scale") == pytest.approx(3.0, rel=1e-3)


def test_metric_scale_enabled_without_reference_is_identity():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _base_cfg(tmp, metric_scale_enabled=True,
                        metric_scale_space="depth", metric_scale_clamp=None)
    mgr, rec, _ = _run(cfg)                     # no scale_reference, monocular off
    got = rec.latest()
    assert got is not None
    # No reference → aligner is identity in depth space → raw ~2 m bowl.
    assert 1.5 < float(np.median(got)) < 3.5


def test_monocular_auto_wiring_builds_reference_and_runs():
    # metric_scale_monocular=True + no injected reference → the pipeline builds a
    # MonocularScaleReference from its intrinsics and runs the mono path end to
    # end. (The synthetic clip has no real 3-D parallax, so the aligner mostly
    # coasts; the point here is the wiring + that it runs without crashing.)
    from slam.monocular_scale import MonocularScaleReference

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _base_cfg(tmp, metric_scale_enabled=True,
                        metric_scale_monocular=True, metric_scale_space="depth")
    mgr, rec, _ = _run(cfg)
    assert isinstance(mgr._scale_reference, MonocularScaleReference)
    assert mgr._aligner is not None
    assert mgr.frames_processed > 0
    assert rec.latest() is not None            # depth reached the consumers
