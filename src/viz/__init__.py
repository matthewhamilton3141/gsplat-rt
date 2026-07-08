"""Real-time web visualiser for the Gaussian SLAM pipeline.

A dependency-light browser view of the live scene: the Python side is pure
stdlib (``http.server``) + numpy, and the browser side pulls Three.js from a CDN.
No new Python packages, nothing on the pipeline's hot path.

  scene_source  — turn a live pipeline / a .ply file / a synthetic scene into a
                  serialisable SceneSnapshot
  web_viewer    — a threaded HTTP server that serves the SPA + JSON scene feed
"""

from .scene_source import (  # noqa: F401
    PipelineSceneSource,
    PlySceneSource,
    SceneSnapshot,
    SyntheticSceneSource,
    read_ply,
)
from .web_viewer import WebViewer  # noqa: F401
