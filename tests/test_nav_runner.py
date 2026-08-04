"""Tests for the browser nav runner (src/viz/nav_runner.py) and its web-viewer endpoints.

CPU/NumPy + OpenCV + stdlib only — no torch, no GPU, no browser. Verifies the JSON the frontend
polls (static scene geometry + dynamic car snapshot), that the shielded DWA car actually drives
the reconstructed scene to the goal, and that the /api/nav[_scene] endpoints serve it (and return
empty when no runner is attached).

Run:
    pytest tests/test_nav_runner.py -v
"""

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from viz.nav_runner import NavRunner, make_nav_runner  # noqa: E402
from viz.scene_source import SyntheticSceneSource  # noqa: E402
from viz.web_viewer import WebViewer  # noqa: E402


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def test_scene_json_wellformed():
    nav = NavRunner()
    sc = nav.scene_json()
    assert len(sc["bounds"]) == 4 and sc["bounds"][2] > sc["bounds"][0]
    assert sc["cell_size"] > 0
    assert len(sc["cells"]) > 0 and all(len(c) == 2 for c in sc["cells"])
    assert len(sc["goal"]) == 2
    assert sc["car"]["radius"] > 0 and sc["car"]["length"] > sc["car"]["width"]


def test_snapshot_shape():
    nav = NavRunner()
    st = nav.snapshot()
    assert len(st["car"]) == 2
    assert isinstance(st["heading"], float)
    assert len(st["trail"]) >= 1
    assert len(st["lidar"]) == nav.cfg.n_lidar_beams
    assert st["outcome"] == "run" and st["step"] == 0


def test_step_advances_car_and_trail():
    nav = NavRunner()
    x0 = nav.env.robot_xy.copy()
    n0 = len(nav.snapshot()["trail"])
    for _ in range(10):
        nav.step_once()
    st = nav.snapshot()
    assert st["step"] == 10
    assert len(st["trail"]) == n0 + 10
    assert nav.env.robot_xy[0] > x0[0]                 # moved down the road (+x)


def test_runner_reaches_goal():
    # Drive the default scene to termination with step_once (no thread/sleep) — the shielded DWA
    # car should reach the goal, never collide.
    nav = NavRunner()
    outcome = "run"
    for _ in range(nav.cfg.task.max_steps):
        running = nav.step_once()
        if not running:
            outcome = nav.snapshot()["outcome"]
            break
    assert outcome == "reached", f"expected the car to reach the goal, got {outcome}"


def test_thread_start_stop():
    nav = NavRunner().start()
    try:
        import time
        time.sleep(0.3)                                # let a few real-time steps run
        assert nav.snapshot()["step"] >= 1
    finally:
        nav.stop()
    assert nav._thread is None


def test_make_nav_runner_requires_start_goal_for_occupancy(tmp_path):
    import numpy as np
    p = os.path.join(tmp_path, "occ.npy")
    np.save(p, np.zeros((10, 10), np.int8))
    try:
        make_nav_runner(occupancy=p, resolution=0.2)   # missing start/goal
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError when start/goal are missing")


def test_web_viewer_serves_nav_endpoints():
    nav = NavRunner()
    v = WebViewer(SyntheticSceneSource(n=50), port=0, nav=nav).start()
    try:
        sc = _get(v.url + "api/nav_scene")
        assert sc["bounds"] and sc["cells"] and sc["car"]["radius"] > 0
        st = _get(v.url + "api/nav")
        assert len(st["car"]) == 2 and "outcome" in st
    finally:
        v.stop()


def test_web_viewer_nav_absent_returns_empty():
    v = WebViewer(SyntheticSceneSource(n=50), port=0).start()
    try:
        assert _get(v.url + "api/nav_scene") == {}
        assert _get(v.url + "api/nav") == {}
    finally:
        v.stop()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
