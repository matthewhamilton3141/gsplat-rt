"""Tests for the viewer's scene sources + .ply reader (src/viz/scene_source.py).

Pure numpy — no browser, no GPU. Exercises the data path the web viewer serves.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from viz.scene_source import (  # noqa: E402
    PipelineSceneSource,
    PlySceneSource,
    SceneSnapshot,
    SyntheticSceneSource,
    height_colormap,
    read_ply,
)
from gaussian.gaussian_model import GaussianModel  # noqa: E402
from gaussian.finalize import write_ply  # noqa: E402


def test_height_colormap_range_and_shape():
    c = height_colormap(np.linspace(0, 5, 50))
    assert c.shape == (50, 3)
    assert c.min() >= 0.0 and c.max() <= 1.0
    # Low vs high map to distinct hues (blue-ish vs red-ish).
    assert not np.allclose(c[0], c[-1])


def test_height_colormap_empty():
    assert height_colormap(np.array([])).shape == (0, 3)


def test_snapshot_bbox_and_decimate():
    means = np.random.default_rng(0).uniform(-2, 2, (100, 3))
    snap = SceneSnapshot(means, np.ones((100, 3)) * 0.5,
                         np.full(100, 0.05), np.full(100, 0.9))
    lo, hi = snap.bbox()
    assert len(lo) == 3 and all(h >= l for l, h in zip(lo, hi))
    small = snap.decimated(20)
    assert small.count == 20
    assert snap.decimated(1000).count == 100        # no-op when already small


def test_synthetic_source_snapshot():
    src = SyntheticSceneSource(n=500)
    snap = src.snapshot()
    assert snap.count == 500
    assert snap.colors.shape == (500, 3)
    assert snap.scales.shape == (500,)
    assert snap.stats["source"] == "synthetic"
    assert src.snapshot().stats["tick"] == 2         # ticks advance


def test_ply_roundtrip_recovers_scene():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, (200, 3))
    rgb = rng.uniform(0, 1, (200, 3))
    model = GaussianModel.from_points(pts, colors=rgb, init_scale=0.04,
                                      init_opacity=0.3)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "scene.ply")
        write_ply(model, path)
        snap = read_ply(path)

    assert snap.count == 200
    assert np.allclose(snap.means, pts, atol=1e-4)
    assert np.allclose(snap.colors, rgb, atol=1e-3)      # SH DC round-trip
    assert np.allclose(snap.scales, 0.04, atol=1e-4)     # exp(log_scale)
    assert np.allclose(snap.opacities, 0.3, atol=1e-3)   # sigmoid(logit)


def test_ply_source_reads_file():
    model = GaussianModel.from_points(np.zeros((3, 3)))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.ply")
        write_ply(model, path)
        snap = PlySceneSource(path).snapshot()
    assert snap.count == 3
    assert snap.stats["source"] == "ply"


class _FakeManager:
    """Duck-typed stand-in for PipelineManager (no threads/GPU)."""

    def __init__(self, pts, occ=None, stats=None, optimized=None):
        self._pts = pts
        self._occ = occ
        self._stats = stats or {"frames": 7, "depth_backend": "mock"}
        self.optimized_gaussians = optimized

    def latest_gaussians(self):
        return self._pts

    def latest_occupancy(self):
        return self._occ

    def stats(self):
        return dict(self._stats)


def test_pipeline_source_raw_cloud_height_coloured():
    pts = np.random.default_rng(2).uniform(-1, 1, (40, 3))
    occ = np.random.default_rng(3).integers(-1, 2, (16, 16))
    src = PipelineSceneSource(_FakeManager(pts, occ=occ))
    snap = src.snapshot()
    assert snap.count == 40
    assert snap.colors.shape == (40, 3)                  # height colormap filled
    assert snap.occupancy.shape == (16, 16)
    assert snap.stats["count"] == 40
    assert snap.stats["depth_backend"] == "mock"


def test_pipeline_source_prefers_optimized_gaussians():
    model = GaussianModel.from_points(
        np.random.default_rng(4).uniform(-1, 1, (25, 3)),
        colors=np.full((25, 3), 0.7))
    src = PipelineSceneSource(_FakeManager(None, optimized=model))
    snap = src.snapshot()
    assert snap.count == 25
    assert np.allclose(snap.colors, 0.7, atol=1e-6)      # per-splat colour used


def test_pipeline_source_empty_cloud():
    snap = PipelineSceneSource(_FakeManager(None)).snapshot()
    assert snap.count == 0
    assert snap.means.shape == (0, 3)


class _FakeManagerWithColors(_FakeManager):
    def __init__(self, pts, cols, **kw):
        super().__init__(pts, **kw)
        self._cols = cols

    def latest_gaussian_colors(self):
        return self._cols


def test_pipeline_source_uses_sampled_colors():
    pts = np.random.default_rng(5).uniform(-1, 1, (30, 3))
    cols = np.random.default_rng(6).uniform(0, 1, (30, 3))
    snap = PipelineSceneSource(_FakeManagerWithColors(pts, cols)).snapshot()
    assert snap.count == 30
    assert np.allclose(snap.colors, cols, atol=1e-9)      # real colours, not height ramp


def test_pipeline_source_truncates_color_length_mismatch():
    # Writer appended points after colours were snapshotted → longer pts.
    pts = np.random.default_rng(7).uniform(-1, 1, (30, 3))
    cols = np.random.default_rng(8).uniform(0, 1, (25, 3))
    snap = PipelineSceneSource(_FakeManagerWithColors(pts, cols)).snapshot()
    assert snap.count == 25                                # truncated to common length
    assert np.allclose(snap.colors, cols[:25], atol=1e-9)
