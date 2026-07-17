#!/usr/bin/env python3
"""Roll out the pure-NumPy diff-drive nav env and render trajectories over a floor plan.

A Mac-doable smoke/demo for the Phase-4 nav flagship core (src/isaac/nav_sim.py): it runs
episodes with a chosen controller, prints success / reward / step-count stats, and writes a
top-down PNG of the trajectories over the obstacle map — mirroring the occupancy-map palette
the pipeline already emits, so the trained-agent trajectory can later overlay a real
`*_occupancy.png` (M7 viz goal: "close the loop reconstruct -> train").

No GPU / torch / gymnasium needed. Examples:
    python scripts/nav/random_rollout.py                       # heuristic, default scene
    python scripts/nav/random_rollout.py --controller random --episodes 20
    python scripts/nav/random_rollout.py --controller both --out /tmp/nav.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from isaac.nav_sim import (  # noqa: E402
    DiffDriveNavEnv, NavSimConfig, avoidance_action, heuristic_action,
)
from isaac.nav_task import ACT_DIM, NavTaskConfig  # noqa: E402

# Match src/mapping/visualization.py so this reads like the pipeline's occupancy PNGs (BGR).
_FREE_BGR = (245, 245, 245)
_OCCUPIED_BGR = (40, 40, 220)
_WALL_BGR = (90, 90, 90)
_OUTCOME_BGR = {"reached": (60, 180, 60), "collided": (40, 40, 220), "timeout": (200, 160, 40)}


def default_scene(n_lidar_beams: int = 0) -> NavSimConfig:
    """A small room with three circular obstacles between a fixed start and goal."""
    # The first obstacle sits squarely on the start→goal diagonal, so blind go-to-goal drives
    # into it while the lidar gap-follower detours around it to the goal — makes the controller
    # difference visible (blind collides ~step 34; avoidance reaches ~step 110).
    obstacles = np.array([[0.3, 0.3, 0.6],
                          [-1.4, 1.4, 0.4],
                          [1.4, -1.4, 0.4]])
    return NavSimConfig(
        task=NavTaskConfig(max_steps=600, goal_radius=0.3),
        bounds=(-3.0, -3.0, 3.0, 3.0),
        obstacles=obstacles,
        fixed_start=(-2.5, -2.5, 0.0),
        fixed_goal=(2.5, 2.5),
        n_lidar_beams=n_lidar_beams,
    )


def pick_action(controller: str, obs: np.ndarray, cfg: NavSimConfig,
                rng: np.random.Generator) -> np.ndarray:
    if controller == "random":
        return np.array([rng.uniform(0.0, cfg.max_lin_vel),
                         rng.uniform(-cfg.max_ang_vel, cfg.max_ang_vel)], np.float32)
    if controller == "avoidance":
        return avoidance_action(obs, cfg)
    return heuristic_action(obs, cfg)


def run_episode(env: DiffDriveNavEnv, controller: str, seed: int,
                rng: np.random.Generator):
    """Run one episode; return (trajectory Nx2, outcome str, total_reward, steps)."""
    obs, info = env.reset(seed=seed)
    traj = [info["robot_xy"].copy()]
    total = 0.0
    outcome = "timeout"
    while True:
        obs, r, term, trunc, info = env.step(pick_action(controller, obs, env.cfg, rng))
        traj.append(info["robot_xy"].copy())
        total += r
        if term:
            outcome = "reached" if info["reached"] else "collided"
            break
        if trunc:
            break
    return np.array(traj), outcome, total, info["step"]


def _world_to_px(pts: np.ndarray, cfg: NavSimConfig, px_per_m: int, pad: int):
    """Map world (x, y) metres to image pixels; y-up so the map reads like a floor plan."""
    x_min, y_min, x_max, y_max = cfg.bounds
    h = int((y_max - y_min) * px_per_m) + 2 * pad
    xs = ((pts[:, 0] - x_min) * px_per_m + pad).astype(np.int32)
    ys = (h - pad - (pts[:, 1] - y_min) * px_per_m).astype(np.int32)   # flip: +y is up
    return np.stack([xs, ys], axis=1)


def render(trajectories, outcomes, cfg: NavSimConfig, path: str,
           px_per_m: int = 60, pad: int = 20) -> str:
    import cv2

    x_min, y_min, x_max, y_max = cfg.bounds
    w = int((x_max - x_min) * px_per_m) + 2 * pad
    h = int((y_max - y_min) * px_per_m) + 2 * pad
    img = np.full((h, w, 3), _WALL_BGR, np.uint8)
    cv2.rectangle(img, (pad, pad), (w - pad, h - pad), _FREE_BGR, -1)

    for cx, cy, r in cfg.obstacle_array():
        c = _world_to_px(np.array([[cx, cy]]), cfg, px_per_m, pad)[0]
        cv2.circle(img, tuple(c), int(r * px_per_m), _OCCUPIED_BGR, -1)

    for traj, outcome in zip(trajectories, outcomes):
        pix = _world_to_px(traj, cfg, px_per_m, pad)
        cv2.polylines(img, [pix.reshape(-1, 1, 2)], False, _OUTCOME_BGR[outcome], 2)

    start = _world_to_px(trajectories[0][:1], cfg, px_per_m, pad)[0]
    goal = _world_to_px(np.array([cfg.fixed_goal]), cfg, px_per_m, pad)[0]
    cv2.circle(img, tuple(start), 6, (0, 0, 0), -1)
    cv2.drawMarker(img, tuple(goal), (0, 150, 0), cv2.MARKER_STAR, 18, 2)

    cv2.imwrite(path, img)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--controller",
                    choices=["heuristic", "avoidance", "random", "all"], default="avoidance")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beams", type=int, default=0,
                    help="lidar beams (obs only; auto-set to 15 for the avoidance controller)")
    ap.add_argument("--out", default="nav_rollout.png")
    args = ap.parse_args()

    controllers = (["heuristic", "avoidance", "random"] if args.controller == "all"
                   else [args.controller])
    # The gap-follower needs the lidar fan; default it on so the demo is meaningful out of the box.
    beams = args.beams
    if beams == 0 and "avoidance" in controllers:
        beams = 15
    cfg = default_scene(n_lidar_beams=beams)
    env = DiffDriveNavEnv(cfg)
    rng = np.random.default_rng(args.seed)

    trajectories, outcomes = [], []
    for controller in controllers:
        results = {"reached": 0, "collided": 0, "timeout": 0}
        rewards, steps = [], []
        for ep in range(args.episodes):
            traj, outcome, total, n = run_episode(env, controller, args.seed + ep, rng)
            results[outcome] += 1
            rewards.append(total)
            steps.append(n)
            trajectories.append(traj)
            outcomes.append(outcome)
        n = args.episodes
        print(f"[{controller:9s}] success {results['reached']}/{n} "
              f"({100 * results['reached'] / n:.0f}%)  "
              f"collided {results['collided']}  timeout {results['timeout']}  "
              f"| reward {np.mean(rewards):+.2f}±{np.std(rewards):.2f}  "
              f"steps {np.mean(steps):.0f}")

    path = render(trajectories, outcomes, cfg, args.out)
    print(f"wrote {path}  ({len(trajectories)} trajectories; "
          f"green=reached red=collided amber=timeout, black dot=start, star=goal)")


if __name__ == "__main__":
    main()
