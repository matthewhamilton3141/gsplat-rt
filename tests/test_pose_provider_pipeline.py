"""Config-driven pose-provider wiring in PipelineManager (M6 live integration).

Verifies the auto-build path that turns cfg.pose_tracking into a live pose
provider, without a GPU/ONNX:
  - 'orb'        → builds an OdometryPoseProvider and drives the pipeline;
  - 'superpoint' with a missing ONNX / no onnxruntime → caught, coasts at
    identity (never crashes the run);
  - a learned pairwise front-end (fake match_pair) threads through the
    coordinator when injected.

All runs force the mock depth estimator (nonexistent engine path) so they are
deterministic on any machine.
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_manager import PipelineConfig, PipelineManager  # noqa: E402
from slam.rgbd_odometry import OdometryPoseProvider  # noqa: E402

_FORCE_MOCK = "/nonexistent/force-mock.engine"


def _make_video(path, n_frames=120, fps=120.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    for i in range(n_frames):
        frame = base.copy()
        frame[100:180, (20 + i) % 560:(20 + i) % 560 + 60] = 255   # a moving patch
        writer.write(frame)
    writer.release()


def _cfg(tmp, **kw):
    base = dict(video_source=None, output_dir=tmp, usd_stem="pose",
                engine_path=_FORCE_MOCK, usd_update_interval_s=60.0,
                usd_update_frame_count=10_000, tsdf_grid_dim=32,
                tsdf_voxel_size=0.10, write_previews=False)
    base.update(kw)
    return PipelineConfig(**base)


def test_orb_pose_provider_autowired():
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "clip.mp4")
        _make_video(video)
        pm = PipelineManager(_cfg(tmp, video_source=video, pose_tracking="orb"))
        with pm:
            time.sleep(1.5)
            stats = pm.stats()
        # Config auto-built the provider, and it drove the pipeline.
        assert isinstance(pm._pose_provider, OdometryPoseProvider)
        assert stats["frames"] > 0
        assert stats["gaussians"] > 0


def test_superpoint_missing_onnx_coasts():
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "clip.mp4")
        _make_video(video)
        cfg = _cfg(tmp, video_source=video, pose_tracking="superpoint",
                   pose_onnx_path="/nonexistent/sp_lg.onnx")
        pm = PipelineManager(cfg)
        with pm:                                  # must not raise
            time.sleep(1.0)
            stats = pm.stats()
        # Build failed gracefully → no provider, pipeline still fused (identity).
        assert pm._pose_provider is None
        assert stats["frames"] > 0


class _FakePairwiseFrontend:
    """Learned-style front-end: presence of match_pair selects the pairwise
    branch; returns arbitrary correspondences and records calls."""

    def __init__(self):
        self.calls = 0

    def match_pair(self, rgb0, rgb1):
        self.calls += 1
        rng = np.random.default_rng(self.calls)
        uv0 = rng.uniform(10, 500, (30, 2)).astype(np.float32)
        uv1 = uv0 + rng.uniform(-2, 2, (30, 2)).astype(np.float32)
        return uv0, uv1


def test_learned_pairwise_frontend_threads_through_pipeline():
    from mapping.collision_proxy import CameraIntrinsics
    cfg0 = PipelineConfig()
    K = CameraIntrinsics(fx=500.0, fy=500.0,
                         cx=cfg0.depth_input_w / 2, cy=cfg0.depth_input_h / 2,
                         width=cfg0.depth_input_w, height=cfg0.depth_input_h)
    fe = _FakePairwiseFrontend()
    provider = OdometryPoseProvider(K, frontend=fe)

    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "clip.mp4")
        _make_video(video)
        pm = PipelineManager(_cfg(tmp, video_source=video), pose_provider=provider)
        with pm:
            time.sleep(1.5)
            stats = pm.stats()
        assert fe.calls > 0, "learned match_pair front-end was never called"
        assert stats["frames"] > 0
        assert stats["gaussians"] > 0
