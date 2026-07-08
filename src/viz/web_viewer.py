"""A threaded, stdlib-only HTTP server that streams the live scene to a browser.

No framework, no new dependencies: ``http.server.ThreadingHTTPServer`` serves a
single-page app (``static/index.html`` + ``static/viewer.js``, which pull Three.js
from a CDN) plus three JSON endpoints the page polls:

    GET /               → the SPA
    GET /viewer.js      → the renderer
    GET /api/scene      → decimated splats {means, colors, scales, opacities, bbox}
    GET /api/occupancy  → top-down grid {w, h, data} (or {} when absent)
    GET /api/stats      → the pipeline's live stats dict

The server only ever *reads* the scene source, off the pipeline's hot path, so
attaching a viewer never perturbs throughput. Bind ``port=0`` for an ephemeral
port (tests read :attr:`port` back).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

logger = logging.getLogger(__name__)


class _DualStackServer(ThreadingHTTPServer):
    """IPv6 server that also accepts IPv4 connections.

    macOS resolves ``localhost`` to IPv6 ``::1`` first, so an IPv4-only bind on
    ``127.0.0.1`` leaves ``http://localhost:PORT`` unreachable. Binding IPv6
    ``::`` with ``IPV6_V6ONLY`` off accepts both ``localhost``/``::1`` and
    ``127.0.0.1`` (as an IPv4-mapped address).
    """

    address_family = socket.AF_INET6
    daemon_threads = True

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            pass
        super().server_bind()

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_CONTENT_TYPES = {".html": "text/html; charset=utf-8",
                  ".js": "application/javascript; charset=utf-8",
                  ".css": "text/css; charset=utf-8"}


def _scene_json(source, max_points: int) -> dict:
    snap = source.snapshot().decimated(max_points)
    lo, hi = snap.bbox()
    out = {
        "count": snap.count,
        "means": np.round(snap.means, 4).ravel().tolist(),
        "colors": np.round(snap.colors, 3).ravel().tolist(),
        "scales": np.round(snap.scales, 4).tolist(),
        "opacities": np.round(snap.opacities, 3).tolist(),
        "bbox": {"min": lo, "max": hi},
        "stats": snap.stats,
    }
    # Per-splat anisotropy (3-axis scale + orientation) for the ellipse renderer;
    # absent for a raw point cloud → the viewer draws round discs.
    if snap.anisotropic:
        out["scales3"] = np.round(snap.scales3, 4).ravel().tolist()
        out["quats"] = np.round(snap.quats, 5).ravel().tolist()
    return out


def _occupancy_json(source) -> dict:
    snap = source.snapshot()
    if snap.occupancy is None:
        return {}
    g = np.asarray(snap.occupancy)
    return {"w": int(g.shape[0]), "h": int(g.shape[1]),
            "data": g.astype(int).ravel().tolist()}


def _stats_json(source) -> dict:
    return dict(source.snapshot().stats)


class _Handler(BaseHTTPRequestHandler):
    server_version = "gsplatViewer/1.0"

    # Route table populated from the server instance's source.
    def do_GET(self):  # noqa: N802 (stdlib signature)
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                return self._send_static("index.html")
            if path == "/viewer.js":
                return self._send_static("viewer.js")
            if path == "/api/scene":
                return self._send_json(_scene_json(self.server.source,
                                                   self.server.max_points))
            if path == "/api/occupancy":
                return self._send_json(_occupancy_json(self.server.source))
            if path == "/api/stats":
                return self._send_json(_stats_json(self.server.source))
            self.send_error(404, "Not found")
        except BrokenPipeError:
            pass                                    # client navigated away mid-send
        except Exception:                           # never let one request kill the server
            logger.exception("viewer request failed: %s", path)
            try:
                self.send_error(500, "Viewer error")
            except Exception:
                pass

    # -- helpers -------------------------------------------------------------

    def _send_static(self, name: str):
        fpath = os.path.join(_STATIC_DIR, name)
        with open(fpath, "rb") as fh:
            body = fh.read()
        ctype = _CONTENT_TYPES.get(os.path.splitext(name)[1], "application/octet-stream")
        self._respond(200, body, ctype)

    def _send_json(self, obj: dict):
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self._respond(200, body, "application/json")

    def _respond(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):              # silence per-request stderr spam
        logger.debug("viewer %s", fmt % args)


class WebViewer:
    """Serves a scene source over HTTP in a background thread.

    ``source`` is anything with ``.snapshot() -> SceneSnapshot`` (see
    scene_source). Use as a context manager or call :meth:`start` / :meth:`stop`.
    """

    def __init__(self, source, host: str = "127.0.0.1", port: int = 8000,
                 max_points: int = 20000):
        self.source = source
        self.host = host
        self._requested_port = port
        self.max_points = max_points
        self._httpd = None
        self._thread = None

    @property
    def port(self) -> int:
        """The bound port (meaningful after start(); resolves an ephemeral 0)."""
        return self._httpd.server_address[1] if self._httpd else self._requested_port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> "WebViewer":
        if self._httpd is not None:
            raise RuntimeError("WebViewer already started")
        # For loopback-ish hosts, bind dual-stack so both localhost (IPv6 ::1) and
        # 127.0.0.1 (IPv4) reach the server; fall back to a plain IPv4 bind if the
        # box has no IPv6.
        httpd = None
        if self.host in ("127.0.0.1", "localhost", "::1", "::", ""):
            try:
                httpd = _DualStackServer(("::", self._requested_port), _Handler)
            except OSError:
                httpd = None
        if httpd is None:
            httpd = ThreadingHTTPServer((self.host, self._requested_port), _Handler)
        httpd.source = self.source              # handler reads these off the server
        httpd.max_points = self.max_points
        httpd.daemon_threads = True
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever,
                                        name="WebViewer", daemon=True)
        self._thread.start()
        logger.info("WebViewer serving at %s", self.url)
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            self._httpd = None
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
