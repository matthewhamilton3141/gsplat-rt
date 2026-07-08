"""Live integration: PipelineManager → PipelineSceneSource → WebViewer.

Runs a short real pipeline (mock depth, temp video — GPU-free) with the viewer
attached and confirms the browser-facing scene feed reflects the running scene.
The money path, end to end. Needs cv2 (present on the dev box).
"""

import json
import os
import sys
import tempfile
import time
import urllib.request

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

cv2 = pytest.importorskip("cv2")

from pipeline_manager import PipelineConfig, PipelineManager  # noqa: E402
from viz import PipelineSceneSource, WebViewer  # noqa: E402


def _make_video(path, n=120, fps=120.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    yy, xx = np.mgrid[0:480, 0:640]
    base = np.stack([(xx / 640 * 255).astype(np.uint8),
                     (yy / 480 * 255).astype(np.uint8),
                     np.full_like(xx, 128, np.uint8)], axis=-1)
    for i in range(n):
        f = base.copy()
        f[200:280, (20 + i) % 560:(20 + i) % 560 + 80] = (255, 255, 255)
        w.write(f)
    w.release()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def test_live_pipeline_feeds_viewer():
    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "clip.mp4")
        _make_video(video)
        cfg = PipelineConfig(video_source=video, output_dir=tmp,
                             write_previews=False, usd_update_interval_s=60.0,
                             tsdf_grid_dim=32, tsdf_voxel_size=0.1)
        mgr = PipelineManager(cfg).start()
        viewer = WebViewer(PipelineSceneSource(mgr), port=0).start()
        try:
            # Let the coordinator process frames and accumulate Gaussians.
            t0 = time.time()
            while mgr.frames_processed < 5 and time.time() - t0 < 4.0:
                time.sleep(0.05)

            scn = _get(viewer.url + "api/scene")
            stats = _get(viewer.url + "api/stats")
        finally:
            viewer.stop()
            mgr.stop(flush_usd=False)

        assert mgr.frames_processed > 0
        assert scn["count"] > 0                      # splats reached the browser feed
        assert len(scn["means"]) == 3 * scn["count"]
        assert stats["depth_backend"] == "mock"      # GPU-free run
        assert stats["frames"] > 0
