#!/usr/bin/env python3
"""Render an animated MP4 of a nav policy *actually navigating* — the flagship, watchable.

Rolls a policy out on procedurally-generated obstacle fields and animates each control step:
the robot (disc + heading), its lidar fan, the trajectory trail, obstacles, and the goal.
The point isn't a number — it's showing the env + controller work as a system. Works with the
pure-NumPy controllers on any machine (`avoidance` / `heuristic`); `ppo` loads a trained
stable-baselines3 policy (needs sb3 + the saved `.zip`, i.e. the box).

    python scripts/nav/render_policy.py --policy avoidance --episodes 3 --out /tmp/nav.mp4
    python scripts/nav/render_policy.py --policy ppo --model-path ~/nav_ppo/ppo_nav.zip ...
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from isaac.nav_sim import (  # noqa: E402
    DiffDriveNavEnv, NavSimConfig, avoidance_action, heuristic_action,
)
from isaac.nav_task import NavTaskConfig  # noqa: E402

# BGR palette (matches the pipeline's occupancy-map look).
_FREE = (245, 245, 245)
_OCC = (40, 40, 220)
_WALL = (90, 90, 90)
_ROBOT = (200, 120, 40)
_GOAL = (60, 180, 60)
_TRAIL = (170, 110, 40)
_LIDAR = (200, 205, 150)
_OUTCOME = {"reached": (60, 180, 60), "collided": (40, 40, 220), "timeout": (200, 160, 40)}


def _to_px(pts, cfg, ppm, pad, h):
    x_min, y_min, _, _ = cfg.bounds
    xs = ((pts[:, 0] - x_min) * ppm + pad).astype(np.int32)
    ys = (h - pad - (pts[:, 1] - y_min) * ppm).astype(np.int32)     # +y is up (floor-plan)
    return np.stack([xs, ys], axis=1)


def _draw(env, traj, cfg, ppm, pad, cv2, outcome=None):
    x_min, y_min, x_max, y_max = cfg.bounds
    w = int((x_max - x_min) * ppm) + 2 * pad
    h = int((y_max - y_min) * ppm) + 2 * pad
    img = np.full((h, w, 3), _WALL, np.uint8)
    cv2.rectangle(img, (pad, pad), (w - pad, h - pad), _FREE, -1)
    for cx, cy, r in env._obstacles:
        c = _to_px(np.array([[cx, cy]]), cfg, ppm, pad, h)[0]
        cv2.circle(img, tuple(c), int(r * ppm), _OCC, -1)
    g = _to_px(np.array([env._goal]), cfg, ppm, pad, h)[0]
    cv2.drawMarker(img, tuple(g), _GOAL, cv2.MARKER_STAR, 22, 2)
    if len(traj) > 1:
        pix = _to_px(np.array(traj), cfg, ppm, pad, h)
        cv2.polylines(img, [pix.reshape(-1, 1, 2)], False, _TRAIL, 2)
    origin = env.robot_xy
    o = _to_px(np.array([origin]), cfg, ppm, pad, h)[0]
    if cfg.n_lidar_beams:
        n = cfg.n_lidar_beams
        angles = (env._heading + np.linspace(-cfg.lidar_fov / 2, cfg.lidar_fov / 2, n)
                  if n > 1 else np.array([env._heading]))
        dists = env._lidar() * cfg.lidar_range
        for ang, d in zip(angles, dists):
            end = origin + d * np.array([np.cos(ang), np.sin(ang)])
            e = _to_px(np.array([end]), cfg, ppm, pad, h)[0]
            cv2.line(img, tuple(o), tuple(e), _LIDAR, 1)
    col = _OUTCOME.get(outcome, _ROBOT)
    cv2.circle(img, tuple(o), int(cfg.robot_radius * ppm), col, -1)
    hd = origin + 0.45 * np.array([np.cos(env._heading), np.sin(env._heading)])
    e = _to_px(np.array([hd]), cfg, ppm, pad, h)[0]
    cv2.arrowedLine(img, tuple(o), tuple(e), (255, 255, 255), 2, tipLength=0.4)
    return img


def _get_policy(name, model_path):
    if name == "avoidance":
        return lambda obs, cfg: avoidance_action(obs, cfg)
    if name == "heuristic":
        return lambda obs, cfg: heuristic_action(obs, cfg)
    if name == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        return lambda obs, cfg: model.predict(obs, deterministic=True)[0]
    raise ValueError(f"unknown policy {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", choices=["avoidance", "heuristic", "ppo"], default="avoidance")
    ap.add_argument("--model-path", default=os.path.expanduser("~/nav_ppo/ppo_nav.zip"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--obstacles", type=int, default=5)
    ap.add_argument("--beams", type=int, default=16)
    ap.add_argument("--occupancy", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--ppm", type=int, default=60, help="pixels per metre")
    ap.add_argument("--out", default="nav_policy.mp4")
    args = ap.parse_args()

    import cv2

    cfg = NavSimConfig(bounds=(-5.0, -5.0, 5.0, 5.0), n_lidar_beams=args.beams,
                       occupancy_size=args.occupancy, randomize_obstacles=args.obstacles,
                       task=NavTaskConfig(max_steps=args.max_steps))
    policy = _get_policy(args.policy, args.model_path)
    pad = 20

    frames, results = [], {"reached": 0, "collided": 0, "timeout": 0}
    for ep in range(args.episodes):
        env = DiffDriveNavEnv(cfg)
        obs, _ = env.reset(seed=args.seed + ep)
        traj = [env.robot_xy.copy()]
        outcome = "timeout"
        for _ in range(args.max_steps):
            frames.append(_draw(env, traj, cfg, args.ppm, pad, cv2))
            obs, r, term, trunc, info = env.step(policy(obs, cfg))
            traj.append(env.robot_xy.copy())
            if term or trunc:
                outcome = ("reached" if info["reached"]
                           else "collided" if info["collided"] else "timeout")
                break
        results[outcome] += 1
        # hold the final frame (coloured by outcome) briefly
        for _ in range(int(args.fps * 0.7)):
            frames.append(_draw(env, traj, cfg, args.ppm, pad, cv2, outcome))

    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"[{args.policy}] {args.episodes} eps: {results}  -> wrote {args.out} "
          f"({len(frames)} frames @ {args.fps}fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
