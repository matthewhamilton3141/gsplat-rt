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
from mapping.visualization import (
    estimate_up,
    occupancy_to_ascii,
    save_occupancy_png,
    save_points_preview,
    save_splat_preview,
)


def _rot(ax, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if ax == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if ax == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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


def test_save_occupancy_png_crop_shrinks_to_observed():
    """Cropping trims the mostly-unknown volume down to the observed patch."""
    grid = np.full((40, 40), -1, dtype=np.int8)
    grid[18:22, 18:22] = 1                          # small observed block
    with tempfile.TemporaryDirectory() as d:
        full = cv2.imread(save_occupancy_png(
            grid, os.path.join(d, "full.png"), cell_px=4, crop=False))
        crop = cv2.imread(save_occupancy_png(
            grid, os.path.join(d, "crop.png"), cell_px=4, crop=True, crop_margin=2))
        assert full.shape == (40 * 4, 40 * 4, 3)
        # 4-wide block + 2 margin each side = 8 cells
        assert crop.shape == (8 * 4, 8 * 4, 3)


# ---------------------------------------------------------------------------
# save_points_preview / save_splat_preview (auto-framed)
# ---------------------------------------------------------------------------

def _spread_cloud(n: int = 500, seed: int = 0, offset=(0.0, 0.0, 2.0)) -> np.ndarray:
    """A cloud sitting anywhere in space — auto-framing must still frame it."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(-1, 1, n) + offset[0],
        rng.uniform(-1, 1, n) + offset[1],
        rng.uniform(-0.5, 0.5, n) + offset[2],
    ]).astype(np.float32)


def test_save_points_preview_writes_png():
    with tempfile.TemporaryDirectory() as d:
        path = save_points_preview(_spread_cloud(), os.path.join(d, "pts.png"),
                                   width=518, height=518)
        assert path is not None
        img = cv2.imread(path)
        assert img.shape == (518, 518, 3)
        assert img.std() > 1.0                       # something was drawn


def test_save_splat_preview_writes_png():
    with tempfile.TemporaryDirectory() as d:
        path = save_splat_preview(_spread_cloud(), os.path.join(d, "prev.png"),
                                  width=518, height=518)
        assert path is not None
        img = cv2.imread(path)
        assert img.shape == (518, 518, 3)
        assert img.std() > 1.0


def test_auto_frame_handles_world_space_offset():
    """A cloud far from the origin still fills the frame (the pose-tracking fix):
    projecting through an origin camera would push it off-screen, auto-framing
    recentres on it. Verify the drawn content isn't stuck in one corner."""
    with tempfile.TemporaryDirectory() as d:
        path = save_points_preview(
            _spread_cloud(offset=(30.0, -20.0, 50.0)),
            os.path.join(d, "far.png"), width=256, height=256)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert img is not None
        drawn = np.argwhere(img > 30)                # non-background pixels
        assert drawn.shape[0] > 0
        cy, cx = drawn.mean(axis=0)
        # Centroid of drawn content lands near the middle, not jammed in a corner.
        assert 64 < cx < 192 and 64 < cy < 192


def test_splat_preview_uses_supplied_colors():
    """Per-splat colours propagate: an all-red cloud renders red, not the ramp."""
    pts = _spread_cloud(n=300)
    colors = np.tile([1.0, 0.0, 0.0], (pts.shape[0], 1)).astype(np.float32)  # RGB red
    with tempfile.TemporaryDirectory() as d:
        path = save_splat_preview(pts, os.path.join(d, "red.png"),
                                  width=256, height=256, colors=colors)
        img = cv2.imread(path)                       # BGR
        drawn = img[img.sum(axis=2) > 30]            # non-background pixels
        assert drawn.shape[0] > 0
        # Red channel (BGR index 2) dominates.
        assert drawn[:, 2].mean() > drawn[:, 0].mean() + 40
        assert drawn[:, 2].mean() > drawn[:, 1].mean() + 40


def test_prep_points_rejects_outliers():
    """The MAD gate drops far-flung specks but keeps the whole main body."""
    from mapping.visualization import _prep_points

    rng = np.random.default_rng(2)
    body = rng.normal(0, 0.5, (5000, 3)).astype(np.float32) + [10, -5, 20]
    stray = rng.uniform(0, 1, (50, 3)).astype(np.float32) + [40, 40, 60]  # far cluster
    cloud = np.vstack([body, stray])
    kept, _ = _prep_points(cloud, None, max_points=0, reject_outliers=True)
    assert body.shape[0] <= kept.shape[0] < cloud.shape[0]     # stray gone, body intact
    # every kept point is near the body centroid, none out at the stray cluster
    assert np.linalg.norm(kept - np.median(body, axis=0), axis=1).max() < 20.0


def test_outliers_do_not_pull_the_frame():
    """A stray cluster must not drag the auto-frame off the main body: the drawn
    content stays centred once outliers are rejected."""
    rng = np.random.default_rng(5)
    body = rng.normal(0, 0.4, (4000, 3)).astype(np.float32) + [10, -5, 20]
    stray = rng.uniform(0, 0.5, (300, 3)).astype(np.float32) + [25, 10, 22]
    cloud = np.vstack([body, stray])
    with tempfile.TemporaryDirectory() as d:
        path = save_points_preview(cloud, os.path.join(d, "c.png"),
                                   width=256, height=256)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        drawn = np.argwhere(img > 30)
        cy, cx = drawn.mean(axis=0)
        assert 80 < cx < 176 and 80 < cy < 176      # centred, not shoved aside


def test_estimate_up_recovers_plane_normal():
    """A tilted planar 'desk' — estimate_up (points-only) recovers its normal
    regardless of the world tilt, which is what makes the preview render level."""
    rng = np.random.default_rng(0)
    n = 8000
    plane = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-0.7, 0.7, n),
                             rng.normal(0, 0.02, n)])            # normal ~ +Z
    for rx, ry, rz in [(35, 20, 10), (-50, 70, -25)]:
        R = _rot("x", rx) @ _rot("y", ry) @ _rot("z", rz)
        P = plane @ R.T + [30, -20, 50]                         # tilt + offset far
        up = estimate_up(P)
        true_up = R @ np.array([0, 0, 1.0])
        true_up /= np.linalg.norm(true_up)
        misalign = np.degrees(np.arccos(np.clip(abs(up @ true_up), -1, 1)))
        assert misalign < 5.0, f"tilt {(rx, ry, rz)}: {misalign:.1f} deg off"


def test_estimate_up_uses_cam_up_hint():
    """With a cam_up reference and a weakly-planar cloud, the estimate stays
    aligned to the hint rather than latching onto a spurious plane."""
    rng = np.random.default_rng(1)
    blob = rng.normal(0, 1.0, (4000, 3))                        # no dominant plane
    hint = np.array([0.0, 0.0, 1.0])
    up = estimate_up(blob, cam_up=hint)
    assert abs(up @ hint) > np.cos(np.deg2rad(45))              # broadly follows hint


def test_upright_preview_consistent_across_world_tilt():
    """The same desk at two different world tilts must render to a similar image
    once uprighted — the whole point of the auto-orientation."""
    rng = np.random.default_rng(3)
    n = 12000
    plane = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-0.7, 0.7, n),
                             rng.normal(0, 0.02, n)])
    imgs = []
    with tempfile.TemporaryDirectory() as d:
        for i, (rx, ry, rz) in enumerate([(30, 15, 0), (-40, 55, 20)]):
            R = _rot("x", rx) @ _rot("y", ry) @ _rot("z", rz)
            P = (plane @ R.T + [10, -5, 25]).astype(np.float32)
            p = save_points_preview(P, os.path.join(d, f"{i}.png"), 128, 128)
            imgs.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 30)
        # Both drawn footprints occupy a similar region (IoU high) — same framing.
        inter = np.logical_and(imgs[0], imgs[1]).sum()
        union = np.logical_or(imgs[0], imgs[1]).sum()
        assert union > 0 and inter / union > 0.6


def test_previews_empty_returns_none():
    with tempfile.TemporaryDirectory() as d:
        empty = np.empty((0, 3), np.float32)
        assert save_points_preview(empty, os.path.join(d, "p.png")) is None
        assert save_splat_preview(empty, os.path.join(d, "s.png")) is None
        assert not os.path.exists(os.path.join(d, "p.png"))
        assert not os.path.exists(os.path.join(d, "s.png"))


# ---------------------------------------------------------------------------
# occupancy_to_ascii
# ---------------------------------------------------------------------------

def test_ascii_map_downsamples_to_max_cols():
    tsdf = _synthetic_tsdf(grid_dim=64)
    grid = tsdf.occupancy_grid_2d()
    art = occupancy_to_ascii(grid, max_cols=20, color=False)
    lines = art.split("\n")
    # Every row fits the requested width, and there are rows to show
    assert lines and all(len(ln) <= 20 for ln in lines)


def test_ascii_map_plain_charset():
    tsdf = _synthetic_tsdf(grid_dim=32)
    art = occupancy_to_ascii(tsdf.occupancy_grid_2d(), color=False)
    assert "\033[" not in art                       # no ANSI escapes
    assert set(art) <= {"#", ".", " ", "\n"}


def test_ascii_map_color_has_ansi():
    tsdf = _synthetic_tsdf(grid_dim=32)
    art = occupancy_to_ascii(tsdf.occupancy_grid_2d(), color=True)
    assert "\033[" in art                            # colored output


def test_ascii_map_occupied_survives_downsample():
    """A single occupied cell must not vanish when the grid is reduced."""
    grid = np.full((64, 64), -1, dtype=np.int8)
    grid[10, 10] = 1
    art = occupancy_to_ascii(grid, max_cols=16, color=False)
    assert "#" in art


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

        # Live dashboard snapshot must reflect a running pipeline
        s = manager.stats()
        assert s["frames"] > 0
        # "mock" on a CPU box, "tensorrt" when a real engine is present (GPU box).
        assert s["depth_backend"] in ("mock", "tensorrt")
        assert set(s) >= {"frames", "exports", "gaussians", "depth_ms", "depth_backend"}

        manager.stop(flush_usd=True)

        assert os.path.exists(manager.occupancy_png_path), "occupancy PNG missing"
        assert os.path.exists(manager.preview_png_path), "splat preview PNG missing"
        assert os.path.exists(manager.points_png_path), "points preview PNG missing"
        assert cv2.imread(manager.occupancy_png_path) is not None
        assert cv2.imread(manager.preview_png_path) is not None
        assert cv2.imread(manager.points_png_path) is not None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
