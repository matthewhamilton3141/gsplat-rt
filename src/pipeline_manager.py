"""Central orchestrator for the Real-Time Gaussian SLAM pipeline.

Thread topology
---------------
  [VideoCapture thread]  ──queue──►  [Coordinator thread]
                                           │
                                           ├─ depth infer (sync, TRT buffers reused)
                                           │
                                           ├─ push_depth ──►  [TSDFWorker thread]
                                           │
                                           └─ periodic USD export (sync in coordinator)

Lock-free hot path
------------------
- VideoStream and CollisionProxyExtractor use queue.Queue for inter-thread
  handoff — no user-level mutex on the critical path.
- Gaussian positions accumulate in a collections.deque whose .extend() is
  atomic under the CPython GIL; no explicit lock needed.
- USD stage access is single-threaded (coordinator only), so UsdBridge needs
  no synchronisation.

Exception isolation
-------------------
Every thread target is wrapped in _coordinator_main() / the subsystem classes'
own try/except wrappers. Exceptions are captured in self._thread_errors and
self._stop_event is set so the coordinator exits its loop cleanly. Calling
stop() re-raises the first captured exception in the caller's thread.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tuneable parameters for one pipeline run."""

    video_source: Union[int, str] = 0
    """Webcam index or path to a video file."""

    engine_path: str = "models/depth_engine.engine"
    """TensorRT engine produced by compile_trt.py."""

    output_dir: str = "output"
    """Directory for .usda / .usdz scene files."""

    usd_stem: str = "live_scene"
    """Base filename stem — produces <stem>.usda and <stem>.usdz."""

    usd_update_interval_s: float = 3.0
    """Trigger a USD export this often (seconds)."""

    usd_update_frame_count: int = 100
    """Also trigger a USD export after this many frames, whichever comes first."""

    write_previews: bool = True
    """Write 2-D preview PNGs (occupancy map + splat render) on each export."""

    max_gaussians_export: int = 5_000
    """Maximum Gaussian splat centres kept in the ring buffer."""

    tsdf_voxel_size: float = 0.05
    """TSDF voxel edge length in metres."""

    tsdf_grid_dim: int = 64
    """Voxels along each TSDF axis (cube)."""

    depth_input_h: int = 518
    depth_input_w: int = 518
    """Expected spatial size of the depth maps from the TRT engine."""

    gaussian_sample_step: int = 16
    """Pixel stride used when sub-sampling depth for Gaussian positions."""

    camera_fov_deg: float = 70.0
    """Symmetric horizontal FOV used for back-projection and TSDF intrinsics."""


# ---------------------------------------------------------------------------
# Mock depth estimator (CUDA / TRT absent)
# ---------------------------------------------------------------------------

class _MockDepthEstimator:
    """Returns a synthetic bowl-shaped depth map — no GPU required.

    Mimics the DepthEstimator interface (.infer / context manager) so the
    pipeline coordinator needs no branch.
    """

    def __init__(self, H: int = 518, W: int = 518):
        v, u = np.meshgrid(
            np.linspace(-1.0, 1.0, H, dtype=np.float32),
            np.linspace(-1.0, 1.0, W, dtype=np.float32),
            indexing="ij",
        )
        # Bowl at ~2m; depth increases toward image corners
        self._base: np.ndarray = 2.0 + 0.6 * (u ** 2 + v ** 2)
        self._rng = np.random.default_rng(0)

    def infer(self, frame: np.ndarray) -> np.ndarray:
        noise = self._rng.standard_normal(self._base.shape).astype(np.float32) * 0.005
        return self._base + noise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# ---------------------------------------------------------------------------
# PipelineManager
# ---------------------------------------------------------------------------

class PipelineManager:
    """Coordinates VideoStream → depth inference → TSDF → USD in real time.

    Lifecycle::

        manager = PipelineManager(config)
        manager.start()          # spawns coordinator + subsystem threads
        # ... wait / do other work ...
        manager.stop()           # signals stop, joins threads, flushes USD

    Or as a context manager::

        with PipelineManager(config) as m:
            time.sleep(10)       # let it run; stop() called on __exit__
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()
        cfg = self._config

        self._stop_event = threading.Event()
        self._started = False
        self._thread_errors: dict[str, Exception] = {}

        # Component handles — set during start()
        self._video_stream = None
        self._depth_estimator = None
        self._collision_extractor = None
        self._usd_bridge = None
        self._coordinator_thread: Optional[threading.Thread] = None

        # Public metrics (written only by coordinator thread)
        self.frames_processed: int = 0
        self.usd_exports: int = 0
        self._last_export_frame: int = 0

        # Gaussian ring buffer — deque.extend is atomic in CPython
        self._gaussian_positions: deque = deque(maxlen=cfg.max_gaussians_export)

        # Pre-allocated depth sampling grid and camera model (immutable after init)
        self._init_sampling_grid()

        # Final output paths (set during start() once output_dir is known)
        self.usd_path: str = ""
        self.usdz_path: str = ""
        self.occupancy_png_path: str = ""
        self.preview_png_path: str = ""

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_sampling_grid(self) -> None:
        """Pre-compute depth pixel indices and pinhole intrinsics.

        These arrays are written once here and treated as read-only during
        run_pipeline() — eliminating per-frame index allocation.
        """
        cfg = self._config
        H, W = cfg.depth_input_h, cfg.depth_input_w
        step = cfg.gaussian_sample_step

        v_idx, u_idx = np.meshgrid(
            np.arange(0, H, step, dtype=np.int32),
            np.arange(0, W, step, dtype=np.int32),
            indexing="ij",
        )
        self._sample_v: np.ndarray = v_idx.ravel()   # (~33×33 = 1089 for 518/16)
        self._sample_u: np.ndarray = u_idx.ravel()

        fov_rad = math.radians(cfg.camera_fov_deg)
        self._fx: float = (W / 2.0) / math.tan(fov_rad / 2.0)
        self._fy: float = self._fx
        self._cx: float = W / 2.0
        self._cy: float = H / 2.0

    def _init_depth_estimator(self):
        """Return a real TRT DepthEstimator if CUDA + engine are present, else mock."""
        engine_path = self._config.engine_path
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            import tensorrt  # noqa: F401
            if not os.path.exists(engine_path):
                raise FileNotFoundError(f"Engine not found: {engine_path}")
            from depth.depth_estimator import DepthEstimator
            est = DepthEstimator(engine_path)
            logger.info("TRT DepthEstimator loaded: %s", engine_path)
            return est
        except (ImportError, RuntimeError, FileNotFoundError) as exc:
            logger.warning("TRT depth unavailable (%s) — using _MockDepthEstimator", exc)
            return _MockDepthEstimator(
                H=self._config.depth_input_h,
                W=self._config.depth_input_w,
            )

    def _init_usd_bridge(self) -> None:
        """Create the in-memory USD stage. No-op if pxr is not installed."""
        try:
            from mapping.usd_bridge import UsdBridge
            self._usd_bridge = UsdBridge(self.usd_path)
            logger.info("UsdBridge initialised: %s", self.usd_path)
        except ImportError:
            logger.warning("pxr not installed — USD export disabled for this run")
            self._usd_bridge = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "PipelineManager":
        """Initialise all subsystems and launch the coordinator thread."""
        if self._started:
            raise RuntimeError("PipelineManager is already running — call stop() first.")

        cfg = self._config
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.usd_path  = os.path.join(cfg.output_dir, f"{cfg.usd_stem}.usda")
        self.usdz_path = os.path.join(cfg.output_dir, f"{cfg.usd_stem}.usdz")
        self.occupancy_png_path = os.path.join(cfg.output_dir, f"{cfg.usd_stem}_occupancy.png")
        self.preview_png_path   = os.path.join(cfg.output_dir, f"{cfg.usd_stem}_splat_preview.png")

        logger.info("PipelineManager starting — output: %s", cfg.output_dir)

        # ---- Video ingestion ----
        from ingestion.video_stream import VideoStream
        self._video_stream = VideoStream(source=cfg.video_source)
        self._video_stream.start()

        # ---- Depth estimator ----
        self._depth_estimator = self._init_depth_estimator()

        # ---- TSDF collision proxy ----
        from mapping.collision_proxy import (
            CameraIntrinsics, CollisionProxyExtractor, TSDFVolume,
        )
        tsdf = TSDFVolume(voxel_size=cfg.tsdf_voxel_size, grid_dim=cfg.tsdf_grid_dim)
        self._camera_k = CameraIntrinsics(
            fx=self._fx, fy=self._fy,
            cx=self._cx, cy=self._cy,
            width=cfg.depth_input_w, height=cfg.depth_input_h,
        )
        self._collision_extractor = CollisionProxyExtractor(tsdf=tsdf, update_hz=10.0)
        self._collision_extractor.start()

        # ---- USD bridge ----
        self._init_usd_bridge()

        # ---- Coordinator thread ----
        self._stop_event.clear()
        self._started = True
        self._coordinator_thread = threading.Thread(
            target=self._coordinator_main,
            daemon=True,   # dies with the process if stop() is never called
            name="PipelineCoordinator",
        )
        self._coordinator_thread.start()
        logger.info("PipelineManager running")
        return self

    def stop(self, flush_usd: bool = True, timeout_s: float = 10.0) -> None:
        """Signal shutdown, join all threads, optionally flush the final USD scene.

        Thread join order matters:
          1. Coordinator   — exits its loop on _stop_event; may be mid-export
          2. TSDF worker   — may be mid-integration; stop() waits for clean exit
          3. VideoCapture  — stop() releases the capture device
          4. Final USD     — safe now that all writers are done

        Re-raises the first exception from any crashed background thread.
        """
        if not self._started:
            return

        logger.info("PipelineManager stopping (flush_usd=%s) …", flush_usd)
        self._stop_event.set()

        if self._coordinator_thread and self._coordinator_thread.is_alive():
            self._coordinator_thread.join(timeout=timeout_s)
            if self._coordinator_thread.is_alive():
                logger.warning("Coordinator thread still alive after %.1fs — proceeding", timeout_s)

        if self._collision_extractor:
            self._collision_extractor.stop()
        if self._video_stream:
            self._video_stream.stop()

        # Final USD snapshot — coordinator is done, so USD bridge access is safe
        if flush_usd:
            try:
                self._trigger_usd_export(final=True)
            except Exception:
                logger.exception("Final USD flush failed")

        self._started = False
        logger.info(
            "PipelineManager stopped — frames=%d  usd_exports=%d",
            self.frames_processed, self.usd_exports,
        )

        # Surface any thread crash to the caller
        if self._thread_errors:
            name, exc = next(iter(self._thread_errors.items()))
            raise RuntimeError(f"Background thread '{name}' raised an exception") from exc

    def __enter__(self) -> "PipelineManager":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Core pipeline loop
    # ------------------------------------------------------------------

    def run_pipeline(self) -> None:
        """Lock-free coordinator loop — the heart of the pipeline.

        Hot-path allocations avoided:
        - Depth inference: TRT buffers pre-allocated in DepthEstimator.__init__
        - Gaussian sampling: index arrays pre-allocated in _init_sampling_grid
        - deque.extend: amortised O(1), no malloc inside CPython's deque

        The only per-frame heap activity is the small (~1 KB) masked sub-array
        from depth backprojection, unavoidable for variable-coverage scenes.
        """
        last_usd_t = time.monotonic()

        while not self._stop_event.is_set():
            # --- Frame acquisition (non-blocking; 33ms timeout = 30 Hz pull) ---
            frame = self._video_stream.get_frame(timeout=0.033)
            if frame is None:
                continue

            # --- Depth estimation (TRT device buffers reused across calls) ---
            try:
                depth = self._depth_estimator.infer(frame)
            except Exception:
                logger.exception("Depth infer failed on frame %d", self.frames_processed)
                continue

            self.frames_processed += 1

            # --- TSDF update (non-blocking push; drops oldest depth on queue overflow) ---
            self._collision_extractor.push_depth(depth, self._camera_k)

            # --- Gaussian accumulation (fast back-projection, O(sample_pixels)) ---
            self._backproject_gaussians(depth)

            # --- Periodic USD export ---
            now = time.monotonic()
            frames_since = self.frames_processed - self._last_export_frame

            if (now - last_usd_t >= self._config.usd_update_interval_s
                    or frames_since >= self._config.usd_update_frame_count):
                try:
                    self._trigger_usd_export()
                except Exception:
                    logger.exception("USD export failed at frame %d", self.frames_processed)
                last_usd_t = now
                self._last_export_frame = self.frames_processed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coordinator_main(self) -> None:
        """Thread target: run the pipeline loop and capture any crash."""
        try:
            self.run_pipeline()
        except Exception as exc:
            logger.exception("Coordinator thread crashed")
            self._thread_errors["coordinator"] = exc
            self._stop_event.set()

    def _backproject_gaussians(self, depth: np.ndarray) -> None:
        """Sub-sample depth and back-project valid pixels to world-space 3D points.

        Uses pre-allocated index arrays (_sample_v, _sample_u).  The only
        per-call allocations are the small boolean mask and the resulting
        positions array (typically < 4 KB for 1089 samples).
        """
        z = depth[self._sample_v, self._sample_u]       # (S,) float32, no copy
        valid = z > 0.1
        if not np.any(valid):
            return

        z_v = z[valid]
        x_v = (self._sample_u[valid] - self._cx) * z_v / self._fx
        y_v = (self._sample_v[valid] - self._cy) * z_v / self._fy

        pts = np.stack([x_v, y_v, z_v], axis=-1)        # (M, 3)
        # extend is amortised O(1) and GIL-atomic for CPython's deque
        self._gaussian_positions.extend(pts.tolist())

    def _write_previews(self) -> None:
        """Write the 2-D occupancy map + splat preview PNGs.

        Independent of USD (no `pxr` needed) and best-effort: any failure is
        logged but never interrupts the export or the pipeline. Each PNG is
        overwritten in place, so the files always show the latest scene.
        """
        if not self._config.write_previews:
            return
        try:
            from mapping.visualization import save_occupancy_png, save_splat_preview
        except ImportError:
            return

        try:
            grid = self._collision_extractor.get_latest_occupancy()
            if grid is not None:
                save_occupancy_png(grid, self.occupancy_png_path)
        except Exception:
            logger.exception("Occupancy PNG write failed")

        try:
            positions = list(self._gaussian_positions)
            if positions:
                save_splat_preview(
                    positions,
                    self._fx, self._fy, self._cx, self._cy,
                    self._config.depth_input_w, self._config.depth_input_h,
                    self.preview_png_path,
                )
        except Exception:
            logger.exception("Splat preview write failed")

    def _trigger_usd_export(self, final: bool = False) -> None:
        """Snapshot the current Gaussian buffer + latest collision mesh to .usdz.

        The .usdz is written to a temporary path then renamed atomically so
        readers (e.g. Isaac Sim) never observe a half-written archive.
        """
        # 2-D previews first — they don't need pxr, so they're produced even
        # when USD export is unavailable on this machine.
        self._write_previews()

        if self._usd_bridge is None:
            return

        label = "final" if final else f"#{self.usd_exports + 1}"
        logger.info("USD export %s (frame=%d) …", label, self.frames_processed)

        # ---- Gaussian splat layer ----
        positions = list(self._gaussian_positions)   # snapshot the deque
        if positions:
            n = len(positions)
            means     = np.array(positions,        dtype=np.float32)          # (N, 3)
            scales    = np.full((n, 3), 0.05,      dtype=np.float32)
            rotations = np.zeros((n, 4),            dtype=np.float32)
            rotations[:, 0] = 1.0                                              # identity quat
            opacities = np.full(n, 0.8,            dtype=np.float32)
            self._usd_bridge.update_gaussian_splats(means, scales, rotations, opacities)

        # ---- Collision mesh layer ----
        mesh = self._collision_extractor.get_latest_mesh()
        if mesh is not None and not mesh.is_empty:
            self._usd_bridge.update_collision_mesh(mesh.vertices, mesh.faces)

        # ---- Flush .usda then atomically replace .usdz ----
        self._usd_bridge.save()

        tmp_usdz = self.usdz_path + ".tmp"
        try:
            self._usd_bridge.export_usdz(tmp_usdz)
            os.replace(tmp_usdz, self.usdz_path)      # POSIX rename — atomic
            self.usd_exports += 1
            logger.info("USD export %s → %s", label, self.usdz_path)
        except Exception:
            logger.exception("export_usdz failed")
            if os.path.exists(tmp_usdz):
                try:
                    os.remove(tmp_usdz)
                except OSError:
                    pass
            raise
