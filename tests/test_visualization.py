"""Tests for the 2-D visual artifacts: occupancy map + splat preview.

Covers three layers:
  1. TSDFVolume.occupancy_grid_2d — shape / dtype / state semantics.
  2. save_occupancy_png / save_splat_preview — valid, correctly-sized PNGs,
     plus the empty-input contract.
  3. Pipeline integration — a live run drops both PNGs next to the .usdz.

No GPU or pxr required (the pipeline falls back to the mock depth estimator).

Run:
    pytest tests/test_visualization.py -v
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mapping.collision_proxy import CameraIntrinsics, TSDFVolume
from mapping.visualization import save_occupancy_png, save_splat_preview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_tsdf(grid_dim: int = 32) -> TSDFVolume:
    """Integrate one wall-like depth map so the volume has all three states."""
    tsdf = TSDFVolume(voxel_size=0.05, grid_dim=grid_dim)
    K = CameraIntrinsics.from_fov(70.0, 128, 128)
    depth = np.full((128, 128), 1.5, dtype=np.float32)   # flat wall at 1.5 m
    tsdf.integrate(depth, K, None)
    return tsdf


# ---------------------------------------------------------------------------
# occupancy_grid_2d
# ---------------------------------------------------------------------------

def test_occupancy_grid_shape_and_dtype():
    tsdf = _synthetic_tsdf(grid_dim=32)
    grid = tsdf.occupancy_grid_2d()
    assert grid.shape == (32, 32)
    assert grid.dtype == np.int8
    assert set(np.unique(grid)).issubset({-1, 0, 1})


def test_occupancy_states_present():
    """A partially-observed volume must show unknown + at least one seen state."""
    tsdf = _synthetic_tsdf(grid_dim=32)
    grid = tsdf.occupancy_grid_2d()
    assert np.any(grid == -1), "expected some unobserved (unknown) columns"
    assert np.any(grid >= 0), "expected some observed columns"


def test_occupancy_empty_volume_all_unknown():
    tsdf = TSDFVolume(voxel_size=0.05, grid_dim=16)
    grid = tsdf.occupancy_grid_2d()
    assert np.all(grid == -1)


# ---------------------------------------------------------------------------
# save_occupancy_png
# ---------------------------------------------------------------------------

def test_save_occupancy_png_dimensions():
    tsdf = _synthetic_tsdf(grid_dim=32)
    grid = tsdf.occupancy_grid_2d()
    with tempfile.TemporaryDirectory() as d:
        path = save_occupancy_png(grid, os.path.join(d, "occ.png"), cell_px=8)
        assert os.path.exists(path)
        img = cv2.imread(path)
        assert img is not None
        assert img.shape == (32 * 8, 32 * 8, 3)


# ---------------------------------------------------------------------------
# save_splat_preview
# ---------------------------------------------------------------------------

def test_save_splat_preview_writes_png():
    rng = np.random.default_rng(0)
    # Points in front of the camera (z > 0), spread across the frustum
    pts = np.column_stack([
        rng.uniform(-1, 1, 500),
        rng.uniform(-1, 1, 500),
        rng.uniform(1.0, 3.0, 500),
    ]).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = save_splat_preview(
            pts, fx=259.0, fy=259.0, cx=259.0, cy=259.0,
            width=518, height=518, path=os.path.join(d, "prev.png"),
        )
        assert path is not None
        img = cv2.imread(path)
        assert img.shape == (518, 518, 3)
        # Something was actually drawn (not a flat background)
        assert img.std() > 1.0


def test_save_splat_preview_empty_returns_none():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "prev.png")
        assert save_splat_preview(
            np.empty((0, 3), np.float32), 259, 259, 259, 259, 518, 518, out
        ) is None
        assert not os.path.exists(out)


def test_save_splat_preview_all_behind_camera_returns_none():
    pts = np.array([[0.0, 0.0, -1.0], [0.1, 0.1, -2.0]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "prev.png")
        assert save_splat_preview(pts, 259, 259, 259, 259, 518, 518, out) is None
        assert not os.path.exists(out)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def test_pipeline_writes_preview_pngs():
    """A short live run (mock depth) must emit both preview PNGs."""
    from pipeline_manager import PipelineConfig, PipelineManager

    with tempfile.TemporaryDirectory() as d:
        video_path = os.path.join(d, "clip.mp4")
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480)
        )
        rng = np.random.default_rng(1)
        for _ in range(120):
            writer.write(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
        writer.release()

        cfg = PipelineConfig(
            video_source=video_path,
            output_dir=os.path.join(d, "out"),
            usd_update_interval_s=0.2,
        )
        manager = PipelineManager(cfg)
        manager.start()
        time.sleep(1.0)          # let a few export ticks fire
        manager.stop(flush_usd=True)

        assert os.path.exists(manager.occupancy_png_path), "occupancy PNG missing"
        assert os.path.exists(manager.preview_png_path), "splat preview PNG missing"
        assert cv2.imread(manager.occupancy_png_path) is not None
        assert cv2.imread(manager.preview_png_path) is not None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
