#!/usr/bin/env python3
"""Interactive viser 3-D viewer for a reconstruction.

Loads a scene the same way the web viewer does (a finalize `.ply`, or a synthetic
demo), auto-uprights it, and serves an orbit-able point cloud in the browser.

    # a finalize-stage / any 3DGS .ply
    python scripts/view_scene.py --ply output/live_scene.ply

    # a synthetic scene (no pipeline / GPU needed) — smoke-test the viewer
    python scripts/view_scene.py --demo

Runs anywhere (pure numpy + viser; no torch/CUDA). Ctrl-C to stop.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from viz.scene_source import PlySceneSource, SyntheticSceneSource  # noqa: E402
from viz.viser_viewer import serve_snapshot                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="viser 3-D viewer for gsplat-rt")
    ap.add_argument("--ply", default=None, help="view a static 3DGS .ply")
    ap.add_argument("--demo", action="store_true",
                    help="view a synthetic scene (no .ply needed)")
    ap.add_argument("--scene", default="sphere", choices=["sphere", "plane", "axes"],
                    help="synthetic scene shape for --demo")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-points", type=int, default=300_000,
                    help="decimate the scene to at most this many points")
    ap.add_argument("--point-size", type=float, default=None,
                    help="override the auto point size (world units)")
    args = ap.parse_args()

    if args.ply:
        if not os.path.exists(args.ply):
            print(f"error: no such .ply: {args.ply}", file=sys.stderr)
            return 2
        source = PlySceneSource(args.ply)
    elif args.demo:
        source = SyntheticSceneSource(shape=args.scene)
    else:
        ap.error("give --ply PATH or --demo")

    snap = source.snapshot()
    print(f"Loaded {snap.count} points from "
          f"{'--ply ' + args.ply if args.ply else 'synthetic ' + args.scene}")
    serve_snapshot(snap, port=args.port, max_points=args.max_points,
                   point_size=args.point_size, block=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
