"""Headless checks for the viser 3-D viewer (src/viz/viser_viewer.py).

viser is an optional dependency; these skip cleanly when it is absent. We never
open a browser — we build the server on an ephemeral port, confirm it stands up
with the scene populated, then stop it. The uprighting math is exercised without
viser too.

Run: pytest tests/test_viser_viewer.py -v
"""

import os
import socket
import sys

import numpy as np
import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mapping.visualization import orient_upright        # noqa: E402
from viz.scene_source import SyntheticSceneSource        # noqa: E402


def _rot_x(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ---------------------------------------------------------------------------
# orient_upright (no viser needed)
# ---------------------------------------------------------------------------

def test_orient_upright_levels_a_tilted_plane():
    rng = np.random.default_rng(0)
    plane = np.column_stack([rng.uniform(-1, 1, 6000), rng.uniform(-1, 1, 6000),
                             rng.normal(0, 0.02, 6000)])          # normal ~ +Z
    tilted = plane @ _rot_x(50).T + [5, -3, 8]
    out, R = orient_upright(tilted)
    assert out.shape == tilted.shape and out.dtype == np.float32
    # After uprighting the plane normal should sit along world +Z: the residual
    # spread along Z is tiny compared to the in-plane extent.
    c = np.median(out, axis=0)
    Q = out - c
    z_spread = Q[:, 2].std()
    xy_spread = np.linalg.norm(Q[:, :2].std(axis=0))
    assert z_spread < 0.15 * xy_spread


def test_orient_upright_empty():
    out, R = orient_upright(np.empty((0, 3), np.float32))
    assert out.shape == (0, 3)
    assert np.allclose(R, np.eye(3))


# ---------------------------------------------------------------------------
# viser server (skips without viser)
# ---------------------------------------------------------------------------

def test_serve_snapshot_builds_and_stops():
    pytest.importorskip("viser")
    from viz.viser_viewer import serve_snapshot

    port = _free_port()
    snap = SyntheticSceneSource(n=2000, shape="sphere").snapshot()
    server = serve_snapshot(snap, port=port, max_points=1000, block=False)
    try:
        assert server is not None
        assert server.get_port() == port                  # actually bound the port
    finally:
        server.stop()


def test_serve_snapshot_with_camera_poses():
    pytest.importorskip("viser")
    from viz.viser_viewer import serve_snapshot

    snap = SyntheticSceneSource(n=1500, shape="plane").snapshot()
    poses = [np.eye(4) for _ in range(3)]
    for i, p in enumerate(poses):
        p[:3, 3] = [i * 0.2, 0.0, 0.0]
    port = _free_port()
    server = serve_snapshot(snap, camera_poses=poses, port=port,
                            max_points=1000, block=False)
    try:
        assert server.get_port() == port
    finally:
        server.stop()
