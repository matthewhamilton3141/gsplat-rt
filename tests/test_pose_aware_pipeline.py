"""Tests for M6 step 4: pose-aware fusion in the live pipeline.

  test_backproject_camera_vs_world
      The Gaussian back-projection places points in the camera frame by
      default and in the world frame when a pose is supplied.

  test_pipeline_runs_with_pose_provider
      A full pipeline run with a pose provider stays crash-free, invokes the
      provider, and still accumulates splats — proving the pose threads through
      the coordinator without breaking the existing contract.

Run:
    pytest tests/test_pose_aware_pipeline.py -v
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_manager import PipelineConfig, PipelineManager


def _make_video(path: str, n_frames: int = 120, fps: float = 120.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    rng = np.random.default_rng(7)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
    writer.release()


def test_backproject_camera_vs_world():
    pm = PipelineManager(PipelineConfig())          # not started; direct method test
    depth = np.full((pm._config.depth_input_h, pm._config.depth_input_w), 2.0, np.float32)

    pm._gaussian_positions.clear()
    pm._backproject_gaussians(depth, pose=None)
    cam = np.array(pm._gaussian_positions)
    # Camera-frame points sit near the optical axis: |x|, |y| are a few metres.
    assert abs(cam[:, 0].mean()) < 1.0

    shift = np.eye(4, dtype=np.float32)
    shift[:3, 3] = [10.0, 0.0, 0.0]                 # translate camera +10 m in world X
    pm._gaussian_positions.clear()
    pm._backproject_gaussians(depth, pose=shift)
    world = np.array(pm._gaussian_positions)
    # Same geometry, now offset by the pose translation.
    assert world[:, 0].min() > 5.0
    assert np.allclose(world[:, 0] - cam[:, 0], 10.0, atol=1e-4)


def test_pipeline_runs_with_pose_provider():
    calls = {"n": 0}

    def provider(frame, depth):
        calls["n"] += 1
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = 0.001 * calls["n"]                # gently translating camera
        return T

    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "clip.mp4")
        _make_video(video, n_frames=120)
        cfg = PipelineConfig(
            video_source=video, output_dir=tmp, usd_stem="pose_scene",
            usd_update_interval_s=60.0, usd_update_frame_count=40,
            tsdf_grid_dim=32, tsdf_voxel_size=0.10, write_previews=False,
        )
        pm = PipelineManager(cfg, pose_provider=provider)
        with pm:
            time.sleep(1.5)
            stats = pm.stats()

        assert calls["n"] > 0, "pose provider was never called"
        assert stats["frames"] > 0
        assert stats["gaussians"] > 0
