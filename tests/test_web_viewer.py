"""Tests for the stdlib web viewer server (src/viz/web_viewer.py).

Starts the real ThreadingHTTPServer on an ephemeral port and hits the endpoints
with urllib — no browser needed, verifies the JSON scene feed the frontend
consumes. Pure stdlib + numpy.
"""

import json
import os
import sys
import urllib.request

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from viz.scene_source import SceneSnapshot, SyntheticSceneSource  # noqa: E402
from viz.web_viewer import WebViewer  # noqa: E402


def _get(url, decode_json=True):
    with urllib.request.urlopen(url, timeout=5) as r:
        body = r.read()
        ctype = r.headers.get("Content-Type", "")
    return json.loads(body) if decode_json else (body.decode(), ctype)


class _StaticSource:
    """Source with a fixed snapshot incl. occupancy (synthetic has none)."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self._snap = SceneSnapshot(
            means=rng.uniform(-1, 1, (30, 3)),
            colors=rng.uniform(0, 1, (30, 3)),
            scales=np.full(30, 0.05), opacities=np.full(30, 0.8),
            occupancy=rng.integers(-1, 2, (8, 10)),
            stats={"source": "test", "frames": 3})

    def snapshot(self):
        return self._snap


@pytest.fixture
def viewer():
    v = WebViewer(SyntheticSceneSource(n=200), port=0).start()
    try:
        yield v
    finally:
        v.stop()


def test_serves_index_and_js(viewer):
    html, ctype = _get(viewer.url, decode_json=False)
    assert "text/html" in ctype
    assert "gsplat-rt" in html and "viewer.js" in html
    js, jctype = _get(viewer.url + "viewer.js", decode_json=False)
    assert "javascript" in jctype
    assert "THREE" in js and "api/scene" in js


def test_scene_endpoint_shape(viewer):
    scn = _get(viewer.url + "api/scene")
    n = scn["count"]
    assert n == 200
    assert len(scn["means"]) == 3 * n
    assert len(scn["colors"]) == 3 * n
    assert len(scn["scales"]) == n
    assert len(scn["opacities"]) == n
    assert "min" in scn["bbox"] and "max" in scn["bbox"]


def test_scene_decimation():
    v = WebViewer(SyntheticSceneSource(n=5000), port=0, max_points=1000).start()
    try:
        scn = _get(v.url + "api/scene")
        assert scn["count"] == 1000
        assert len(scn["means"]) == 3000
    finally:
        v.stop()


def test_scene_endpoint_includes_anisotropy(viewer):
    # SyntheticSceneSource is anisotropic → scales3 + quats present.
    scn = _get(viewer.url + "api/scene")
    n = scn["count"]
    assert len(scn["scales3"]) == 3 * n
    assert len(scn["quats"]) == 4 * n


def test_scene_endpoint_omits_anisotropy_for_point_cloud():
    # A snapshot with no scales3/quats (raw cloud) → keys absent.
    v = WebViewer(_StaticSource(), port=0).start()
    try:
        scn = _get(v.url + "api/scene")
        assert "scales3" not in scn and "quats" not in scn
    finally:
        v.stop()


def test_stats_endpoint(viewer):
    stats = _get(viewer.url + "api/stats")
    assert stats["source"] == "synthetic"
    assert "count" in stats


def test_occupancy_endpoint_present_and_absent():
    # Synthetic source → no occupancy → {}
    v = WebViewer(SyntheticSceneSource(n=50), port=0).start()
    try:
        assert _get(v.url + "api/occupancy") == {}
    finally:
        v.stop()
    # Static source with a grid → {w, h, data}
    v2 = WebViewer(_StaticSource(), port=0).start()
    try:
        occ = _get(v2.url + "api/occupancy")
        assert occ["w"] == 8 and occ["h"] == 10
        assert len(occ["data"]) == 80
        assert set(occ["data"]) <= {-1, 0, 1}
    finally:
        v2.stop()


def test_unknown_path_404(viewer):
    with pytest.raises(urllib.error.HTTPError) as ei:
        _get(viewer.url + "nope")
    assert ei.value.code == 404


def test_url_and_port_resolved(viewer):
    assert viewer.port > 0
    assert viewer.url.startswith("http://127.0.0.1:")
