#!/usr/bin/env python3
"""Drive the shielded car through a REAL NVIDIA NuRec reconstructed drive — the digital twin.

Point this at the `clipgt/` directory of an unpacked NuRec clip (its `.usdz` is a container zip:
`unzip clip.usdz -d clip/`). It builds an occupancy grid from the clip's ground-truth road
boundaries + tracked obstacles, follows the recorded ego trajectory with a pure-pursuit look-ahead
so the DWA planner tracks the actual driven route (through curves/interchanges a single far goal
can't), keeps the braking safety shield in the loop, and renders the whole run top-down.

    # unpack a clip fetched via scripts/fetch_nurec.sh, then:
    unzip -o ~/nurec/**/<clip>.usdz -x checkpoint.ckpt volume.nurec 'frames/*' -d /tmp/clip
    python scripts/nav/nurec_drive.py --clipgt /tmp/clip/clipgt --out docs/nurec_drive.mp4

All CPU (no GPU/box). Needs pyarrow for the GT (`pip install pyarrow`).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from isaac.car_sim import BicycleNavEnv, car_dwa_action, car_safety_shield  # noqa: E402
from isaac.driving_scene import driving_scene_config  # noqa: E402
from isaac.nurec_scene import nurec_gt_to_scene, path_arclength, pursuit_target  # noqa: E402
import drive_scene as DS  # noqa: E402  (reuse the top-down renderer)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clipgt", required=True, help="path to an unpacked NuRec clip's clipgt/ dir")
    ap.add_argument("--resolution", type=float, default=0.3, help="metres/cell")
    ap.add_argument("--lookahead", type=float, default=10.0, help="pure-pursuit look-ahead (m)")
    ap.add_argument("--robot-radius", type=float, default=0.9)
    ap.add_argument("--max-speed", type=float, default=6.0)
    ap.add_argument("--horizon", type=int, default=25, help="DWA rollout horizon (steps)")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ppm", type=int, default=4, help="pixels per metre")
    ap.add_argument("--out", default="docs/nurec_drive.mp4")
    ap.add_argument("--png", default=None)
    ap.add_argument("--gif", default=None, help="also write an animated GIF here")
    args = ap.parse_args()

    import cv2

    grid, start, goal, ego = nurec_gt_to_scene(args.clipgt, resolution=args.resolution)
    arc = path_arclength(ego)
    print(f"scene {grid.shape}, route {arc[-1]:.0f} m, start {np.round(start, 1)} -> goal {np.round(goal, 1)}")

    cfg = driving_scene_config(grid, start, goal, robot_radius=args.robot_radius,
                               safety_margin=0.35, max_speed=args.max_speed, max_steer=0.55,
                               wheelbase=2.8, max_ang_vel=3.0, n_lidar_beams=24, lidar_fov=np.pi,
                               lidar_range=25.0)
    cfg.task.goal_radius = 4.0
    cfg.task.max_steps = args.max_steps
    env = BicycleNavEnv(cfg)
    env.reset(seed=0)

    pad, ppm = 10, args.ppm
    bg, W, H = DS._background(grid, grid.bounds, ppm, pad, cv2)
    to_px = DS._tf(grid.bounds, ppm, pad, H)
    ego_px = to_px(ego)

    def frame(outcome=None):
        f = DS._frame(bg, env, traj, cfg, ppm, pad, H, to_px, cv2, outcome)
        cv2.polylines(f, [ego_px.reshape(-1, 1, 2)], False, (120, 180, 120), 1)  # faint ego route
        return f

    traj = [env.robot_xy.copy()]
    frames, outcome = [], "timeout"
    for t in range(args.max_steps):
        tgt = pursuit_target(ego, arc, env.robot_xy, args.lookahead)
        raw = car_dwa_action(env.robot_xy, env._heading, tgt, env._obstacles, cfg,
                             horizon=args.horizon)
        act = raw if args.no_shield else car_safety_shield(
            raw, env.robot_xy, env._heading, env._obstacles, cfg)
        _, _, term, trunc, info = env.step(act)
        traj.append(env.robot_xy.copy())
        if t % 3 == 0:
            frames.append(frame())
        if term or trunc:
            outcome = "reached" if info["reached"] else "collided" if info["collided"] else "timeout"
            break
    print(f"outcome={outcome} steps={info['step']} final={np.round(env.robot_xy, 1)}")
    for _ in range(int(args.fps)):
        frames.append(frame(outcome))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"wrote {args.out} ({len(frames)} frames)")
    if args.png:
        cv2.imwrite(args.png, frames[-1])
        print(f"       still -> {args.png}")
    if args.gif:
        import imageio
        rgb = [f[::2, ::2, ::-1] for f in frames[::3]]     # BGR->RGB, downscaled + decimated
        imageio.mimsave(args.gif, rgb, fps=max(1, args.fps // 3), loop=0)
        print(f"       gif   -> {args.gif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
