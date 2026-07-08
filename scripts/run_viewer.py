"""Serve a live 3-D view of the pipeline (or a .ply) in your browser.

Two modes:

    # live: run the pipeline and stream its splats + occupancy to the browser
    python scripts/run_viewer.py --source 0
    python scripts/run_viewer.py --source clip.mp4 --duration 30

    # static: just view a finalize-stage .ply (or any 3DGS .ply)
    python scripts/run_viewer.py --ply output/live_scene.ply

Then open http://localhost:8000. The viewer is decoupled from the pipeline's hot
path (it only *reads* snapshots), so attaching it never perturbs throughput.

Runs GPU-free: without CUDA/TensorRT the pipeline uses its mock depth estimator,
so you can drive the whole viewer on a laptop. --demo needs neither a GPU nor a
video source (procedural scene).
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from viz import (PipelineSceneSource, PlySceneSource, SyntheticSceneSource,
                 WebViewer)


def _parse_source(s: str):
    return int(s) if s.isdigit() else s


def main() -> int:
    ap = argparse.ArgumentParser(description="Live browser viewer for gsplat-rt")
    ap.add_argument("--source", default=None,
                    help="webcam index or video path for a live pipeline run")
    ap.add_argument("--ply", default=None, help="view a static .ply instead")
    ap.add_argument("--demo", action="store_true",
                    help="serve a procedural scene (no pipeline, no GPU)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-points", type=int, default=20000,
                    help="decimate the scene to at most this many splats")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop the live run after N seconds (0 = until Ctrl-C)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manager = None
    if args.ply:
        source = PlySceneSource(args.ply)
        label = f"ply: {args.ply}"
    elif args.demo or args.source is None:
        source = SyntheticSceneSource()
        label = "demo (procedural scene)"
    else:
        from pipeline_manager import PipelineConfig, PipelineManager
        cfg = PipelineConfig(video_source=_parse_source(args.source),
                             realtime_source=isinstance(_parse_source(args.source), str))
        manager = PipelineManager(cfg)          # built now, started after the viewer
        source = PipelineSceneSource(manager)
        label = f"live pipeline: {args.source}"

    # Start the web server first so the page is reachable immediately — before we
    # touch the camera (which on macOS may block on a permission prompt).
    viewer = WebViewer(source, host=args.host, port=args.port,
                       max_points=args.max_points).start()
    print(f"\n  gsplat-rt viewer — {label}\n  open  {viewer.url}\n  Ctrl-C to stop\n")

    if manager is not None:
        print("  starting pipeline… (grant camera access if macOS prompts)\n")
        manager.start()

    try:
        t0 = time.time()
        while True:
            time.sleep(0.5)
            if args.duration and time.time() - t0 >= args.duration:
                break
            if manager is not None and manager._thread_errors:
                print("pipeline thread error — stopping"); break
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        viewer.stop()
        if manager is not None:
            manager.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
