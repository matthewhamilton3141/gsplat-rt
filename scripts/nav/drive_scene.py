#!/usr/bin/env python3
"""Render a car driving a *reconstructed-style* scene under the braking shield — the digital twin.

The AV-meaningful use of reconstruction is a digital twin: rebuild a recorded scene into geometry,
then test a driving policy *inside* it, closed-loop. This animates exactly that — a kinematic
bicycle car, driven by the lidar gap-follower behind the braking safety shield, weaving the
parked-car slalom of a driving-scale occupancy scene. Every frame draws the occupancy world, the
oriented car body, its lidar fan, trail, and the goal.

Synthetic scene (default) or a real reconstructed occupancy map:
    python scripts/nav/drive_scene.py --out docs/car_digital_twin.mp4
    python scripts/nav/drive_scene.py --occupancy path/to/map.npy --resolution 0.1

Pure NumPy + OpenCV (+ imageio for the GIF). No GPU.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from isaac.car_sim import BicycleNavEnv, car_dwa_action, car_safety_shield  # noqa: E402
from isaac.driving_scene import (  # noqa: E402
    driving_scene_config, load_occupancy_grid, make_driving_scene,
)

# BGR palette (matches the occupancy-map look used elsewhere in the repo).
_FREE = (245, 245, 245)
_OCC = (95, 95, 95)
_CAR = (200, 120, 40)
_GOAL = (60, 180, 60)
_TRAIL = (170, 110, 40)
_LIDAR = (200, 205, 150)
_OUTCOME = {"reached": (60, 180, 60), "collided": (40, 40, 220), "timeout": (200, 160, 40)}


def _tf(bounds, ppm, pad, h):
    """World (x, y) -> pixel (col, row) mapper; +y is up (floor-plan)."""
    x_min, y_min, _, _ = bounds
    def to_px(pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        xs = (pts[:, 0] - x_min) * ppm + pad
        ys = h - pad - (pts[:, 1] - y_min) * ppm
        return np.stack([xs, ys], axis=1).astype(np.int32)
    return to_px


def _background(grid, bounds, ppm, pad, cv2):
    """Static canvas: occupancy grid upscaled to pixels (occupied grey, free light)."""
    x_min, y_min, x_max, y_max = bounds
    w = int((x_max - x_min) * ppm) + 2 * pad
    h = int((y_max - y_min) * ppm) + 2 * pad
    img = np.full((h, w, 3), _FREE, np.uint8)
    occ = grid.occupancy
    # Nearest-neighbour upscale of the (H,W) occupancy to the interior pixel box, then flip rows
    # (grid row = +y up in world, image row = down).
    inner_w = int((x_max - x_min) * ppm)
    inner_h = int((y_max - y_min) * ppm)
    up = cv2.resize(occ, (inner_w, inner_h), interpolation=cv2.INTER_NEAREST)
    up = np.flipud(up)
    ys, xs = np.nonzero(up)
    img[ys + pad, xs + pad] = _OCC
    return img, w, h


def _draw_car(img, xy, heading, robot_radius, ppm, to_px, cv2, color):
    """Oriented car body (a rectangle) + a heading arrow."""
    length, width = robot_radius * 2.6, robot_radius * 1.6
    c, s = np.cos(heading), np.sin(heading)
    fwd = np.array([c, s]); left = np.array([-s, c])
    corners = [xy + 0.5 * length * fwd + 0.5 * width * left,
               xy + 0.5 * length * fwd - 0.5 * width * left,
               xy - 0.5 * length * fwd - 0.5 * width * left,
               xy - 0.5 * length * fwd + 0.5 * width * left]
    cv2.fillConvexPoly(img, to_px(corners).reshape(-1, 1, 2), color)
    tip = xy + 0.7 * length * fwd
    cv2.arrowedLine(img, tuple(to_px(xy)[0]), tuple(to_px(tip)[0]), (255, 255, 255), 2,
                    tipLength=0.4)


def _frame(bg, env, traj, cfg, ppm, pad, h, to_px, cv2, outcome=None):
    img = bg.copy()
    g = to_px(env._goal)[0]
    cv2.drawMarker(img, tuple(g), _GOAL, cv2.MARKER_STAR, 26, 3)
    if len(traj) > 1:
        cv2.polylines(img, [to_px(traj).reshape(-1, 1, 2)], False, _TRAIL, 2)
    origin = env.robot_xy
    o = tuple(to_px(origin)[0])
    if cfg.n_lidar_beams:
        n = cfg.n_lidar_beams
        angles = (env._heading + np.linspace(-cfg.lidar_fov / 2, cfg.lidar_fov / 2, n)
                  if n > 1 else np.array([env._heading]))
        dists = env._lidar() * cfg.lidar_range
        for ang, d in zip(angles, dists):
            end = origin + d * np.array([np.cos(ang), np.sin(ang)])
            cv2.line(img, o, tuple(to_px(end)[0]), _LIDAR, 1)
    _draw_car(img, origin, env._heading, cfg.robot_radius, ppm, to_px, cv2,
              _OUTCOME.get(outcome, _CAR))
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--occupancy", default=None, help="real occupancy map (.npy/.png); else synthetic")
    ap.add_argument("--resolution", type=float, default=0.1, help="metres/cell for --occupancy")
    ap.add_argument("--start", type=float, nargs=3, default=None, metavar=("X", "Y", "H"))
    ap.add_argument("--goal", type=float, nargs=2, default=None, metavar=("X", "Y"))
    ap.add_argument("--length", type=float, default=20.0, help="synthetic road length (m)")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--ppm", type=int, default=34, help="pixels per metre")
    ap.add_argument("--no-shield", action="store_true", help="drive without the braking shield")
    ap.add_argument("--out", default="docs/car_digital_twin.mp4")
    ap.add_argument("--gif", default=None, help="also write an animated GIF here")
    ap.add_argument("--png", default=None, help="also write the final still here")
    args = ap.parse_args()

    import cv2

    if args.occupancy:
        grid = load_occupancy_grid(args.occupancy, args.resolution)
        if args.start is None or args.goal is None:
            raise SystemExit("--occupancy requires --start X Y H and --goal X Y")
        start, goal = tuple(args.start), tuple(args.goal)
    else:
        grid, start, goal = make_driving_scene(length=args.length)
        if args.start is not None:
            start = tuple(args.start)
        if args.goal is not None:
            goal = tuple(args.goal)

    cfg = driving_scene_config(grid, start, goal)
    cfg.task.max_steps = args.max_steps
    pad = 16
    to_px = _tf(grid.bounds, args.ppm, pad, 0)         # h filled in after background sizing
    bg, w, h = _background(grid, grid.bounds, args.ppm, pad, cv2)
    to_px = _tf(grid.bounds, args.ppm, pad, h)         # rebuild with correct height

    env = BicycleNavEnv(cfg)
    obs, _ = env.reset(seed=0)
    traj = [env.robot_xy.copy()]
    frames = [_frame(bg, env, traj, cfg, args.ppm, pad, h, to_px, cv2)]
    outcome = "timeout"
    for _ in range(args.max_steps):
        raw = car_dwa_action(env.robot_xy, env._heading, env._goal, env._obstacles, cfg)
        act = raw if args.no_shield else car_safety_shield(
            raw, env.robot_xy, env._heading, env._obstacles, cfg)
        obs, _, term, trunc, info = env.step(act)
        traj.append(env.robot_xy.copy())
        frames.append(_frame(bg, env, traj, cfg, args.ppm, pad, h, to_px, cv2))
        if term or trunc:
            outcome = "reached" if info["reached"] else "collided" if info["collided"] else "timeout"
            break
    for _ in range(int(args.fps * 1.0)):               # hold the final frame, coloured by outcome
        frames.append(_frame(bg, env, traj, cfg, args.ppm, pad, h, to_px, cv2, outcome))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"[car] outcome={outcome} steps={info['step']} -> {args.out} "
          f"({len(frames)} frames @ {args.fps}fps)")

    if args.png:
        cv2.imwrite(args.png, frames[-1])
        print(f"       still -> {args.png}")
    if args.gif:
        import imageio
        rgb = [f[:, :, ::-1] for f in frames[::2]]     # BGR->RGB, halve for size
        imageio.mimsave(args.gif, rgb, fps=args.fps // 2, loop=0)
        print(f"       gif   -> {args.gif}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
