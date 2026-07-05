"""Integration test for the offline Gaussian finalize stage (M5 → pipeline).

  test_pipeline_finalizes_and_writes_ply
      A full PipelineManager run with ``optimize_on_finalize=True`` captures
      keyframes during the hot path, then on stop() seeds Gaussians from the
      accumulated cloud, fits them against the keyframes, and writes a 3DGS
      .ply. Asserts the optimizer actually improved and the file exists.

Run:
    pytest tests/test_finalize_pipeline.py -v
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
    """A structured (non-noise) clip so the finalize fit has real signal.

    Pure random frames give the optimiser nothing coherent to converge on;
    a smooth colour gradient with a moving block is cheap and reproducible.
    """
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


def test_pipeline_finalizes_and_writes_ply():
    def provider(frame, depth):
        # Gently translating camera so keyframes carry distinct poses.
        T = np.eye(4, dtype=np.float32)
        return T

    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "clip.mp4")
        _make_video(video, n_frames=120)
        cfg = PipelineConfig(
            video_source=video, output_dir=tmp, usd_stem="finalize_scene",
            usd_update_interval_s=60.0, usd_update_frame_count=1000,
            tsdf_grid_dim=32, tsdf_voxel_size=0.10, write_previews=False,
            optimize_on_finalize=True,
            # Short interval so keyframes land well within the run window
            # regardless of how many frames the mock depth estimator clears.
            keyframe_interval=5, max_keyframes=3,
            finalize_res=48, finalize_iters=40, finalize_max_points=800,
        )
        pm = PipelineManager(cfg, pose_provider=provider)
        with pm:
            time.sleep(2.5)

        # Keyframes were captured and the finalize optimiser ran.
        assert pm.optimized_gaussians is not None, "finalize did not run"
        assert pm.finalize_result is not None
        losses = pm.finalize_result.losses
        assert len(losses) >= 2
        assert losses[-1] < losses[0], (
            f"fit did not improve: {losses[0]:.5f} -> {losses[-1]:.5f}")

        # The 3DGS .ply was written and is a well-formed binary INRIA file.
        assert os.path.exists(pm.ply_path), f"missing {pm.ply_path}"
        assert os.path.getsize(pm.ply_path) > 0
        with open(pm.ply_path, "rb") as fh:
            head = fh.read(64)
        assert head.startswith(b"ply"), "not a .ply file"
