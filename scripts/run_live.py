"""Run the pipeline live and watch it work — the interactive entrypoint.

Unlike bench_pipeline.py (which measures and exits), this streams a live view so
you can see the pipeline running on a GPU box over SSH:

  - a status line: FPS, depth latency, splat count, USD exports, backend
  - optionally (--ascii-map) the top-down occupancy map redrawn in the terminal,
    so you watch the scene reconstruct without copying a PNG back

Logging is configured to INFO here (the library leaves it unconfigured), so the
pipeline's own "USD export …" / start-stop messages also appear.

Usage:
    # webcam
    python scripts/run_live.py --source 0 --ascii-map
    # a video file, 20 s, into a chosen output dir
    python scripts/run_live.py --source clip.mp4 --duration 20 --output-dir output/run1
    # SuperPoint+LightGlue pose tracking on a GPU box (TensorRT FP16 front-end)
    python scripts/run_live.py --source clip.mp4 --pose-tracking superpoint \
        --pose-backend tensorrt --duration 30

Stops on --duration, when a file source is exhausted, or on Ctrl-C.
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_manager import PipelineConfig, PipelineManager
from mapping.visualization import occupancy_to_ascii

_CLEAR = "\033[2J\033[H"   # clear screen + home cursor


def _parse_source(s: str):
    """Webcam index (int) if all-digits, else a file path / URL string."""
    return int(s) if s.isdigit() else s


def main() -> int:
    ap = argparse.ArgumentParser(description="Run gsplat-rt live with a terminal dashboard")
    ap.add_argument("--source", default="0", help="webcam index or video path/URL")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to run (0 = until Ctrl-C)")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--interval", type=float, default=2.0, help="USD/preview export interval (s)")
    ap.add_argument("--ascii-map", action="store_true", help="draw the occupancy map in the terminal")
    ap.add_argument("--no-color", action="store_true", help="plain ASCII (no ANSI colors)")
    ap.add_argument("--refresh", type=float, default=0.5, help="dashboard refresh period (s)")
    ap.add_argument("--loop", action="store_true", help="rewind a file source at its end (endless source)")
    ap.add_argument("--realtime", action="store_true", help="play a file at its frame rate, not disk speed")

    # --- M6 pose tracking (visual odometry front-end) ---
    ap.add_argument("--pose-tracking", choices=["none", "orb", "superpoint"], default="none",
                    help="per-frame pose source: 'none' fuses at identity (fixed camera), "
                         "'orb' = CPU baseline VO, 'superpoint' = SuperPoint+LightGlue ONNX")
    ap.add_argument("--pose-backend", choices=["tensorrt", "cuda", "cpu"], default="tensorrt",
                    help="onnxruntime provider for --pose-tracking superpoint")
    ap.add_argument("--pose-onnx", default="models/sp_lg_tum.onnx",
                    help="fused SuperPoint+LightGlue ONNX (for --pose-tracking superpoint); "
                         "run scripts/export_sp_lg.sh to produce it")

    # --- Monocular metric scale (relative Depth Anything → metric depth) ---
    ap.add_argument("--metric-scale", action="store_true",
                    help="align relative depth to a metric scale before fusion")
    ap.add_argument("--metric-scale-monocular", action="store_true",
                    help="derive the metric anchor from two-view geometry on the live "
                         "stream (implies --metric-scale)")
    ap.add_argument("--metric-scale-anchor", type=float, default=1.0,
                    help="first-pair baseline in metres pinning absolute scale "
                         "(1.0 = consistent-but-arbitrary gauge)")

    # --- Camera intrinsics (a coherent metric map needs the real camera model;
    #     without these run_live falls back to a generic FOV guess) ---
    ap.add_argument("--tum-intrinsics", action="store_true",
                    help="use TUM freiburg1 intrinsics (fx=517.31 fy=516.47 "
                         "cx=318.64 cy=255.31 @ 640x480) — for the TUM RGB clips")
    ap.add_argument("--camera-fx", type=float, help="focal-x, with --camera-fy/cx/cy + --camera-native-hw")
    ap.add_argument("--camera-fy", type=float, help="focal-y")
    ap.add_argument("--camera-cx", type=float, help="principal-point x")
    ap.add_argument("--camera-cy", type=float, help="principal-point y")
    ap.add_argument("--camera-native-hw", type=int, nargs=2, metavar=("H", "W"),
                    help="pixel resolution the --camera-* intrinsics were measured at")
    args = ap.parse_args()

    # Resolve intrinsics: --tum-intrinsics preset, or a full custom set, or None
    # (generic FOV fallback in PipelineManager).
    cam_intr = cam_hw = None
    if args.tum_intrinsics:
        cam_intr = (517.306408, 516.469215, 318.643040, 255.313989)
        cam_hw = (480, 640)
    elif any(v is not None for v in
             (args.camera_fx, args.camera_fy, args.camera_cx, args.camera_cy)):
        if (any(v is None for v in
                (args.camera_fx, args.camera_fy, args.camera_cx, args.camera_cy))
                or args.camera_native_hw is None):
            ap.error("--camera-fx/fy/cx/cy and --camera-native-hw must be given together")
        cam_intr = (args.camera_fx, args.camera_fy, args.camera_cx, args.camera_cy)
        cam_hw = tuple(args.camera_native_hw)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    metric_scale = args.metric_scale or args.metric_scale_monocular

    cfg = PipelineConfig(
        video_source=_parse_source(args.source),
        output_dir=args.output_dir,
        usd_update_interval_s=args.interval,
        loop_source=args.loop,
        realtime_source=args.realtime,
        pose_tracking=args.pose_tracking,
        pose_backend=args.pose_backend,
        pose_onnx_path=args.pose_onnx,
        metric_scale_enabled=metric_scale,
        metric_scale_monocular=args.metric_scale_monocular,
        metric_scale_anchor=args.metric_scale_anchor,
        camera_intrinsics=cam_intr,
        camera_intrinsics_hw=cam_hw,
    )

    manager = PipelineManager(cfg)
    manager.start()
    print(f"\nPipeline started — depth backend: {manager.depth_backend}")
    if args.pose_tracking != "none":
        _be = f" ({args.pose_backend})" if args.pose_tracking == "superpoint" else ""
        print(f"Pose tracking: {args.pose_tracking}{_be}"
              f"{'  metric-scale: on' if metric_scale else ''}")
    if cam_intr is not None:
        src = "TUM fr1" if args.tum_intrinsics else "custom"
        print(f"Camera intrinsics: {src}  fx={cam_intr[0]:.1f} fy={cam_intr[1]:.1f} "
              f"cx={cam_intr[2]:.1f} cy={cam_intr[3]:.1f} @ {cam_hw[1]}x{cam_hw[0]}")
    else:
        print(f"Camera intrinsics: generic FOV {cfg.camera_fov_deg:.0f}° "
              "(no --tum-intrinsics/--camera-*; map geometry approximate)")
    print(f"Outputs → {os.path.abspath(args.output_dir)}")
    print("Press Ctrl-C to stop.\n")

    t_start = time.monotonic()
    last_t, last_frames = t_start, 0
    fps = 0.0
    try:
        while True:
            time.sleep(args.refresh)
            now = time.monotonic()
            s = manager.stats()

            # FPS from the frame delta over the refresh window
            df = s["frames"] - last_frames
            dt = now - last_t
            if dt > 0 and df > 0:
                fps = df / dt
            last_t, last_frames = now, s["frames"]

            elapsed = now - t_start
            status = (
                f"[{elapsed:6.1f}s] depth={s['depth_backend']:<9} "
                f"fps={fps:5.1f}  frames={s['frames']:<6} "
                f"depth={s['depth_ms']:5.1f}ms  splats={s['gaussians']:<6} "
                f"exports={s['exports']}"
            )

            if args.ascii_map:
                grid = manager.latest_occupancy()
                sys.stdout.write(_CLEAR)
                print(status + "\n")
                if grid is not None:
                    print(occupancy_to_ascii(grid, color=not args.no_color))
                    print("\n\033[91m█\033[0m occupied   \033[90m·\033[0m free   "
                          "(blank) unknown   — X→right, depth↑" if not args.no_color
                          else "# occupied   . free   (blank) unknown")
                else:
                    print("(waiting for first occupancy grid — needs a surface in view)")
            else:
                sys.stdout.write("\r" + status)
                sys.stdout.flush()

            # Stop conditions: duration elapsed, or a file source ran dry
            if args.duration > 0 and elapsed >= args.duration:
                break
            if not args.loop and df == 0 and s["frames"] > 0 and elapsed > 2.0:
                # No new frames for a full refresh after processing started —
                # a file source has almost certainly been exhausted. (A looped
                # source never exhausts, so this check is skipped for --loop.)
                print("\nSource exhausted — stopping.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted — stopping.")
    finally:
        manager.stop(flush_usd=True)

    s = manager.stats()
    print(f"\nDone. frames={s['frames']} exports={s['exports']} "
          f"splats={s['gaussians']} depth~{s['depth_ms']:.1f}ms")
    print("Wrote:")
    for p in (manager.usdz_path, manager.occupancy_png_path, manager.preview_png_path):
        mark = "✓" if p and os.path.exists(p) else "·"
        print(f"  {mark} {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
