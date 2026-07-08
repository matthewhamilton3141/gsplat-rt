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

    loop_source: bool = False
    """Rewind a file source when it ends instead of stopping (no-op for webcam)."""

    realtime_source: bool = False
    """Pace a file source to its frame rate instead of reading at disk speed."""

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

    # ---- Offline Gaussian finalize (M5) ----
    optimize_on_finalize: bool = False
    """Run the differentiable Gaussian optimiser once on stop(), fitting the
    accumulated points to captured keyframes. This is an *offline* refinement,
    not part of the 30 FPS hot path — CPU numpy is far too slow for per-frame
    optimisation, so it runs once at shutdown."""

    keyframe_interval: int = 15
    """Capture an RGB keyframe (for the finalize optimiser) every N frames."""

    max_keyframes: int = 6
    """Ring-buffer size of keyframes retained for the finalize fit."""

    finalize_res: int = 96
    """Square resolution the keyframes are optimised at (CPU budget)."""

    finalize_iters: int = 150
    """Optimiser iterations in the finalize stage."""

    finalize_max_points: int = 2000
    """Cap on Gaussians seeded from the accumulated cloud during finalize."""

    finalize_ssim_weight: float = 0.2
    """λ for the finalize photometric loss ``(1−λ)·L1 + λ·(1−SSIM)`` (Kerbl et al.
    use 0.2). 0 → pure L1 (the original behaviour)."""

    finalize_densify: bool = False
    """Run Adaptive Density Control during the finalize fit (clone/split under-/
    over-reconstructed Gaussians, prune transparent ones). Off by default — the
    seeded cloud is usually dense enough and densification lengthens the CPU fit;
    enable for sparse seeds / higher-fidelity offline exports."""

    # ---- Monocular metric scale (relative → metric depth) ----
    metric_scale_enabled: bool = False
    """Insert the scale/shift aligner between depth inference and the
    pose/TSDF/Gaussian consumers, turning Depth Anything's *relative* depth into
    *metric* depth. Off by default → existing RGB-D/TUM runs (already metric) are
    unchanged. Requires a ``scale_reference`` provider (see PipelineManager) to
    supply the per-frame metric anchor; without one the aligner just passes depth
    through (identity), so enabling this alone is a no-op."""

    metric_scale_space: str = "disparity"
    """'disparity' (Depth Anything's output is inverse-depth) or 'depth'."""

    metric_scale_smoothing: float = 0.7
    """EMA weight on the previous (scale, shift) for frame-to-frame stability."""

    metric_scale_robust: bool = True
    """Huber-IRLS reweighting in the fit (robust to bad reference points)."""

    metric_scale_min_points: int = 20
    """Minimum valid reference points to accept a fit; else coast on last scale."""

    metric_scale_clamp: tuple = (0.05, 100.0)
    """(min, max) metres clamp applied to aligned depth, or None to disable."""

    metric_scale_monocular: bool = False
    """Auto-build a MonocularScaleReference (two-view triangulation + cross-frame
    scale propagation) from the camera intrinsics as the aligner's reference,
    when no ``scale_reference`` is injected. This is what makes a *pure monocular*
    stream produce metric depth end-to-end. Needs cv2; falls back to identity
    (with a warning) if unavailable. Ignored when metric_scale_enabled is False
    or an explicit scale_reference was passed to PipelineManager."""

    metric_scale_anchor: float = 1.0
    """Absolute-scale gauge for the monocular reference: a known first-pair camera
    translation (metres) pins true metric scale; 1.0 gives a globally-consistent
    but arbitrary absolute scale (the honest monocular limit)."""

    # ---- Pose tracking (M6 visual odometry) ------------------------------
    pose_tracking: str = "none"
    """Auto-build a VO pose provider when none is injected to PipelineManager:
    'none' (fixed camera, identity), 'orb' (CPU baseline), or 'superpoint'
    (SuperPoint+LightGlue ONNX). Only produces a coherent world map with metric,
    scale-consistent depth (RGB-D sensor / TUM). On build failure the pipeline
    coasts at identity (a warning), so a run never crashes for lack of a tracker."""

    pose_onnx_path: str = "models/sp_lg_tum.onnx"
    """Fused SuperPoint+LightGlue ONNX, for pose_tracking='superpoint'
    (regenerate with scripts/export_sp_lg.sh)."""

    pose_onnx_hw: tuple = (480, 640)
    """(height, width) the pose ONNX was exported at (model input size)."""

    pose_backend: str = "tensorrt"
    """onnxruntime provider for pose_tracking='superpoint': 'tensorrt' (FP16 TRT
    EP + engine cache — measured 7.4 ms/frame on A10G), 'cuda', or 'cpu'."""


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

    def __init__(self, config: Optional[PipelineConfig] = None, pose_provider=None,
                 scale_reference=None):
        self._config = config or PipelineConfig()
        cfg = self._config

        # Optional per-frame pose source (M6). A callable (frame_bgr, depth) ->
        # (4,4) camera-to-world matrix, or None. When None, every frame is fused
        # at identity (the original fixed-camera behaviour). A pose provider only
        # produces a coherent world map when depth is metric and scale-consistent
        # across frames (an RGB-D sensor / TUM), which is exactly what the
        # OdometryPoseProvider in src/slam expects.
        self._pose_provider = pose_provider

        # Optional per-frame metric-scale anchor for the monocular path. A
        # callable (frame_bgr, rel_depth) -> (pred_values, ref_depth[, weights])
        # or None. Feeds the DepthScaleAligner so relative depth becomes metric
        # before pose/TSDF/Gaussians consume it. Only used when
        # config.metric_scale_enabled; see _apply_metric_scale.
        self._scale_reference = scale_reference
        self._aligner = None            # built in start() when enabled

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

        # Rolling depth-inference wall-clock times (ms) for the live dashboard.
        # deque.append is GIL-atomic; readers see a slightly stale mean, which is
        # fine for a status line. This is a coarse wall-clock figure, not the
        # CUDA-event benchmark in tests/test_depth_inference.py.
        self._depth_ms: deque = deque(maxlen=120)

        # Wall-clock timestamps of recently processed frames, for a rolling FPS
        # readout (deque.append is GIL-atomic; stats() derives fps from the span).
        self._frame_times: deque = deque(maxlen=120)

        # Gaussian ring buffer — deque.extend is atomic in CPython
        self._gaussian_positions: deque = deque(maxlen=cfg.max_gaussians_export)
        # Parallel per-point source-frame colours (viewer-only; same maxlen so it
        # stays aligned with _gaussian_positions). Empty when no frame is sampled.
        self._gaussian_colors: deque = deque(maxlen=cfg.max_gaussians_export)

        # Keyframe ring buffer for the offline finalize optimiser (M5): each is
        # (rgb float32 [finalize_res, finalize_res, 3] in [0,1], pose or None).
        self._keyframes: deque = deque(maxlen=cfg.max_keyframes)
        # Set by run_finalize(); the optimized scene the final export prefers.
        self.optimized_gaussians = None            # GaussianModel | None
        self.finalize_result = None                # FitResult | None

        # Pre-allocated depth sampling grid and camera model (immutable after init)
        self._init_sampling_grid()

        # Final output paths (set during start() once output_dir is known)
        self.usd_path: str = ""
        self.usdz_path: str = ""
        self.occupancy_png_path: str = ""
        self.preview_png_path: str = ""
        self.ply_path: str = ""

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

    def _init_aligner(self):
        """Build the DepthScaleAligner if metric scale is enabled, else None."""
        cfg = self._config
        if not cfg.metric_scale_enabled:
            return None
        from depth.metric_scale import DepthScaleAligner
        aligner = DepthScaleAligner(
            space=cfg.metric_scale_space,
            robust=cfg.metric_scale_robust,
            smoothing=cfg.metric_scale_smoothing,
            min_points=cfg.metric_scale_min_points,
            clamp=cfg.metric_scale_clamp,
        )
        logger.info("Metric-scale aligner active (space=%s)", cfg.metric_scale_space)
        return aligner

    def _maybe_build_monocular_reference(self) -> None:
        """Auto-wire a MonocularScaleReference when the monocular path is on.

        Only when the aligner is enabled, no explicit scale_reference was
        injected, and metric_scale_monocular is set. Requires cv2; on failure the
        aligner falls back to identity (a warning), so the pipeline still runs.
        """
        cfg = self._config
        if self._aligner is None or self._scale_reference is not None:
            return
        if not cfg.metric_scale_monocular:
            logger.warning(
                "metric_scale_enabled but no scale_reference and "
                "metric_scale_monocular=False — depth passes through un-scaled "
                "(aligner runs as identity).")
            return
        try:
            from slam.monocular_scale import MonocularScaleReference
            self._scale_reference = MonocularScaleReference(
                self._camera_k, anchor=cfg.metric_scale_anchor)
            logger.info("Monocular scale reference active (anchor=%.3f)",
                        cfg.metric_scale_anchor)
        except Exception:
            logger.exception(
                "Could not build MonocularScaleReference — aligner runs as "
                "identity (depth stays relative).")

    def _maybe_build_pose_provider(self) -> None:
        """Auto-wire a VO pose provider from config when none was injected.

        ``cfg.pose_tracking`` selects the front-end: 'orb' (CPU baseline) or
        'superpoint' (SuperPoint+LightGlue ONNX via onnxruntime, provider from
        ``cfg.pose_backend``). Needs metric, scale-consistent depth to yield a
        coherent world map. Any failure (missing ONNX, onnxruntime absent) is
        caught and the pipeline coasts at identity — a run never crashes here.
        """
        cfg = self._config
        if self._pose_provider is not None or cfg.pose_tracking == "none":
            return
        try:
            from slam.rgbd_odometry import OdometryPoseProvider
            if cfg.pose_tracking == "orb":
                self._pose_provider = OdometryPoseProvider(self._camera_k)
                logger.info("Pose provider: ORB visual odometry")
            elif cfg.pose_tracking == "superpoint":
                from slam.superpoint_lightglue import (
                    SuperPointLightGlueFrontend, ort_providers,
                )
                h, w = cfg.pose_onnx_hw
                fe = SuperPointLightGlueFrontend(
                    cfg.pose_onnx_path, height=h, width=w,
                    providers=ort_providers(cfg.pose_backend, cfg.pose_onnx_path))
                self._pose_provider = OdometryPoseProvider(self._camera_k, frontend=fe)
                logger.info("Pose provider: SuperPoint+LightGlue (%s, providers=%s)",
                            cfg.pose_onnx_path, fe.providers)
            else:
                logger.warning("Unknown pose_tracking=%r — fusing at identity",
                               cfg.pose_tracking)
        except Exception:
            logger.exception(
                "Could not build pose provider (pose_tracking=%s) — fusing at "
                "identity (fixed camera).", cfg.pose_tracking)

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
        self.ply_path           = os.path.join(cfg.output_dir, f"{cfg.usd_stem}.ply")

        logger.info("PipelineManager starting — output: %s", cfg.output_dir)

        # ---- Video ingestion ----
        from ingestion.video_stream import VideoStream
        self._video_stream = VideoStream(
            source=cfg.video_source,
            loop=cfg.loop_source,
            realtime=cfg.realtime_source,
        )
        self._video_stream.start()

        # ---- Depth estimator ----
        self._depth_estimator = self._init_depth_estimator()

        # ---- Metric-scale aligner (relative → metric depth) ----
        self._aligner = self._init_aligner()

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

        # ---- Monocular scale reference (needs intrinsics, so after _camera_k) ----
        self._maybe_build_monocular_reference()

        # ---- Pose provider (needs intrinsics; skipped if one was injected) ----
        self._maybe_build_pose_provider()

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

        # Offline Gaussian optimisation (M5), before the final snapshot so the
        # exported scene carries optimized splats rather than raw defaults.
        if self._config.optimize_on_finalize:
            try:
                self.run_finalize()
            except Exception:
                logger.exception("Gaussian finalize failed")

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
    # Observability
    # ------------------------------------------------------------------

    @property
    def depth_backend(self) -> str:
        """'tensorrt' if the real TRT engine is loaded, else 'mock'."""
        if isinstance(self._depth_estimator, _MockDepthEstimator):
            return "mock"
        return "tensorrt" if self._depth_estimator is not None else "unset"

    def stats(self) -> dict:
        """Non-blocking snapshot of live metrics for a status display.

        Safe to call from another thread — reads GIL-atomic counters and a
        rolling latency window written only by the coordinator.
        """
        samples = list(self._depth_ms)   # copy; deque may mutate mid-read
        depth_ms = sum(samples) / len(samples) if samples else 0.0

        # Rolling throughput from the frame-timestamp window.
        times = list(self._frame_times)
        fps = 0.0
        if len(times) >= 2:
            span = times[-1] - times[0]
            if span > 0:
                fps = (len(times) - 1) / span

        stats = {
            "frames": self.frames_processed,
            "exports": self.usd_exports,
            "gaussians": len(self._gaussian_positions),
            "depth_ms": depth_ms,
            "fps": fps,
            "depth_backend": self.depth_backend,
        }
        if self._aligner is not None and self._aligner.params is not None:
            # Current metric scale in effect (None until the first good fit).
            stats["metric_scale"] = self._aligner.params.scale
        return stats

    def latest_occupancy(self) -> Optional[np.ndarray]:
        """Most recent top-down occupancy grid, or None. For live visualization."""
        if self._collision_extractor is None:
            return None
        return self._collision_extractor.get_latest_occupancy()

    def latest_gaussians(self) -> Optional[np.ndarray]:
        """Snapshot of the accumulated Gaussian centres as an (N, 3) array, or
        None if empty. Copies the ring buffer, so it's safe to call from another
        thread (e.g. the web viewer) while the coordinator keeps appending."""
        pts = list(self._gaussian_positions)   # deque copy is GIL-atomic
        return np.asarray(pts, dtype=np.float64) if pts else None

    def latest_gaussian_colors(self) -> Optional[np.ndarray]:
        """Snapshot of the accumulated per-point source-frame colours as (N, 3)
        RGB in [0, 1], or None if none were sampled (no frame / colour disabled).
        Viewer pairs these with latest_gaussians(); callers should tolerate a
        small length mismatch from the concurrent writer."""
        cols = list(self._gaussian_colors)
        return np.asarray(cols, dtype=np.float64) if cols else None

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
                _t0 = time.perf_counter()
                depth = self._depth_estimator.infer(frame)
                self._depth_ms.append((time.perf_counter() - _t0) * 1e3)
            except Exception:
                logger.exception("Depth infer failed on frame %d", self.frames_processed)
                continue

            self.frames_processed += 1
            self._frame_times.append(time.monotonic())

            # --- Metric scale (relative → metric depth, before any consumer) ---
            if self._aligner is not None:
                depth = self._apply_metric_scale(frame, depth)

            # --- Pose estimation (M6; None → identity, fixed-camera fusion) ---
            pose = None
            if self._pose_provider is not None:
                try:
                    pose = self._pose_provider(frame, depth)
                except Exception:
                    logger.exception("Pose provider failed on frame %d — using identity",
                                     self.frames_processed)

            # --- TSDF update (non-blocking push; drops oldest depth on queue overflow) ---
            self._collision_extractor.push_depth(depth, self._camera_k, pose)

            # --- Gaussian accumulation (fast back-projection, O(sample_pixels)) ---
            self._backproject_gaussians(depth, frame, pose)

            # --- Keyframe capture for the offline finalize optimiser (M5) ---
            if self._config.optimize_on_finalize:
                self._maybe_capture_keyframe(frame, pose)

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

    def _apply_metric_scale(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Fit the aligner on this frame's reference (if any) and rescale depth.

        Best-effort: a failing or empty reference provider never crashes the
        loop — the aligner simply coasts on its current (scale, shift) and still
        returns a metric map. Before the first successful fit it is the identity,
        so the pipeline runs from frame 0 (just not yet metric).
        """
        if self._scale_reference is not None:
            try:
                ref = self._scale_reference(frame, depth)
            except Exception:
                logger.exception("scale_reference failed on frame %d",
                                 self.frames_processed)
                ref = None
            if ref is not None:
                try:
                    self._aligner.fit(*ref)
                except Exception:
                    logger.exception("scale fit failed on frame %d",
                                     self.frames_processed)
        return self._aligner.transform(depth)

    def _backproject_gaussians(self, depth: np.ndarray,
                               frame: Optional[np.ndarray] = None,
                               pose: Optional[np.ndarray] = None) -> None:
        """Sub-sample depth and back-project valid pixels to 3D points.

        Uses pre-allocated index arrays (_sample_v, _sample_u).  The only
        per-call allocations are the small boolean mask and the resulting
        positions array (typically < 4 KB for 1089 samples).

        With `pose` (a 4x4 camera-to-world matrix) the points are transformed
        into the world frame so successive frames accumulate into one coherent
        cloud; without it they stay in the camera frame (fixed-camera default).

        When `frame` (the source BGR image) is given, the source pixel's colour
        is sampled at each kept point into a parallel buffer — purely for the
        live viewer's benefit (real per-splat colour instead of a height ramp).
        It does not touch the positions, the TSDF, the finalize optimiser, or any
        exported product; a ~1k-pixel lookup, negligible on the hot path.
        """
        z = depth[self._sample_v, self._sample_u]       # (S,) float32, no copy
        valid = z > 0.1
        if not np.any(valid):
            return

        z_v = z[valid]
        x_v = (self._sample_u[valid] - self._cx) * z_v / self._fx
        y_v = (self._sample_v[valid] - self._cy) * z_v / self._fy

        pts = np.stack([x_v, y_v, z_v], axis=-1)        # (M, 3) camera frame
        if pose is not None:
            # camera → world; errstate mutes float32 BLAS subnormal warnings (macOS)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                pts = pts @ pose[:3, :3].T.astype(np.float32) + pose[:3, 3].astype(np.float32)
        # extend is amortised O(1) and GIL-atomic for CPython's deque
        self._gaussian_positions.extend(pts.tolist())

        if frame is not None:
            self._sample_gaussian_colors(frame, valid)

    def _sample_gaussian_colors(self, frame: np.ndarray, valid: np.ndarray) -> None:
        """Append the source-frame colour of each kept sample to the colour buffer.

        Kept in lockstep with `_gaussian_positions` (same valid mask, same order),
        so the viewer can pair them. Depth-grid coords (in depth_input_w×h) are
        scaled to the frame's own resolution — no resize.
        """
        try:
            fh, fw = frame.shape[:2]
            sx = fw / float(self._config.depth_input_w)
            sy = fh / float(self._config.depth_input_h)
            fu = np.clip((self._sample_u[valid] * sx).astype(np.intp), 0, fw - 1)
            fv = np.clip((self._sample_v[valid] * sy).astype(np.intp), 0, fh - 1)
            bgr = frame[fv, fu]                          # (M, 3) uint8 BGR
            rgb = bgr[:, ::-1].astype(np.float32) / 255.0
            self._gaussian_colors.extend(rgb.tolist())
        except Exception:
            logger.exception("gaussian colour sampling failed (viewer-only)")

    def _maybe_capture_keyframe(self, frame: np.ndarray,
                                pose: Optional[np.ndarray]) -> None:
        """Every keyframe_interval frames, stash a small RGB view + pose.

        Cheap (one resize + colour swap every N frames), off the critical
        latency, and bounded by the ring buffer. These are the target views the
        offline finalize optimiser fits the Gaussians against.
        """
        if self.frames_processed % self._config.keyframe_interval != 0:
            return
        try:
            import cv2
            res = self._config.finalize_res
            small = cv2.resize(frame, (res, res), interpolation=cv2.INTER_AREA)
            rgb = small[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, [0,1]
        except Exception:
            logger.exception("Keyframe capture failed on frame %d", self.frames_processed)
            return
        pose_copy = None if pose is None else np.array(pose, dtype=np.float64)
        self._keyframes.append((np.ascontiguousarray(rgb), pose_copy))

    def run_finalize(self) -> bool:
        """Offline: optimise the accumulated Gaussians against captured keyframes.

        Runs once (typically from stop()). Populates self.optimized_gaussians /
        self.finalize_result and writes a 3DGS .ply. Returns True on success.
        Never part of the real-time loop — pure numpy is far too slow per frame.
        """
        keyframes = list(self._keyframes)
        positions = list(self._gaussian_positions)
        if not keyframes or not positions:
            logger.warning("Finalize skipped — keyframes=%d points=%d",
                           len(keyframes), len(positions))
            return False

        from gaussian.finalize import finalize_gaussians, pose_to_camera, write_ply
        from gaussian.gaussian_model import GaussianModel
        from gaussian.optimizer import psnr
        from gaussian.rasterizer import rasterize

        res = self._config.finalize_res
        fov_rad = math.radians(self._config.camera_fov_deg)
        fx = (res / 2.0) / math.tan(fov_rad / 2.0)
        views = [
            (pose_to_camera(pose, fx, fx, res, res), rgb)
            for rgb, pose in keyframes
        ]
        points = np.asarray(positions, dtype=np.float64)

        seed = GaussianModel.from_points(points[:self._config.finalize_max_points])
        start = float(np.mean([psnr(rasterize(seed, cam)[0], tgt) for cam, tgt in views]))
        logger.info("Finalize: %d points, %d keyframes, %d iters @ %dpx (start PSNR %.2f dB)",
                    len(positions), len(views), self._config.finalize_iters, res, start)

        densify_config = None
        if self._config.finalize_densify:
            from gaussian.densify import DensifyConfig
            densify_config = DensifyConfig(
                densify_interval=max(1, self._config.finalize_iters // 5),
                stop_iter=int(self._config.finalize_iters * 0.8),
                max_gaussians=self._config.finalize_max_points * 4,
            )

        model, result = finalize_gaussians(
            points, views,
            max_points=self._config.finalize_max_points,
            iters=self._config.finalize_iters,
            ssim_weight=self._config.finalize_ssim_weight,
            densify_config=densify_config,
        )
        self.optimized_gaussians = model
        self.finalize_result = result
        logger.info("Finalize done: PSNR %.2f → %.2f dB, L1 %.4f → %.5f",
                    start, result.psnrs[-1], result.losses[0], result.losses[-1])

        try:
            write_ply(model, self.ply_path)
            logger.info("Wrote optimized splats → %s", self.ply_path)
        except Exception:
            logger.exception("PLY write failed")
        return True

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

    def _splat_export_arrays(self):
        """Assemble (means, scales, rotations, opacities, sh_coeffs) for USD.

        Prefers the optimized Gaussians from run_finalize() when present (real
        per-splat colour/opacity/shape); otherwise falls back to the raw
        accumulated centres with default splat attributes. Returns None if the
        scene is empty.
        """
        model = self.optimized_gaussians
        if model is not None:
            from gaussian.finalize import sh_dc_from_rgb
            means     = model.means.astype(np.float32)
            scales    = model.scales.astype(np.float32)        # linear stddev
            rotations = (model.quats /
                         (np.linalg.norm(model.quats, axis=1, keepdims=True) + 1e-12)
                         ).astype(np.float32)
            opacities = model.alphas.astype(np.float32)        # sigmoid-space
            sh        = sh_dc_from_rgb(model.rgb)               # (N, 3) DC term
            return means, scales, rotations, opacities, sh

        positions = list(self._gaussian_positions)   # snapshot the deque
        if not positions:
            return None
        n = len(positions)
        means     = np.array(positions, dtype=np.float32)
        scales    = np.full((n, 3), 0.05, dtype=np.float32)
        rotations = np.zeros((n, 4), dtype=np.float32)
        rotations[:, 0] = 1.0                                   # identity quat
        opacities = np.full(n, 0.8, dtype=np.float32)
        return means, scales, rotations, opacities, None

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
        arrays = self._splat_export_arrays()
        if arrays is not None:
            means, scales, rotations, opacities, sh = arrays
            self._usd_bridge.update_gaussian_splats(
                means, scales, rotations, opacities, sh_coeffs=sh)

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
