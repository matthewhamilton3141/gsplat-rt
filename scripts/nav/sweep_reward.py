#!/usr/bin/env python3
"""Reward-shaping sweep: push the PPO nav policy's collisions toward 0 without losing its
speed win. Box-only (needs torch + sb3 + gymnasium; the env core is pure NumPy).

The trained policy (see scripts/nav/RESULTS.md) reaches ~98% of held-out random scenes ~40%
faster than the hand-written gap-follower, but at a ~2% collision rate — the speed/safety
trade. The sparse `collision_penalty` only fires *on* the hit, so it gives the policy no
gradient to stay clear. This sweep trains a small grid of reward configs — bumped collision
penalty and/or the new dense **clearance penalty** (`nav_task.clearance_penalty`, which ramps
up as the robot nears an obstacle *before* contact) — and evaluates every candidate on one
**identical** held-out set of random scenes, so the collision comparison is apples-to-apples.

Each config is trained with byte-identical PPO hyperparameters (`train_ppo.make_ppo`); the
only thing that varies is the reward. Reports reached% / collisions / mean-steps per config
+ the hand-written baseline, and flags the best "0 collisions, speed kept" pick.

    python scripts/nav/sweep_reward.py --timesteps 500000 --beams 16 --obstacles 5 \
        --eval-episodes 200 --out ~/nav_ppo_sweep

⚠ Save --out under ~/ (NOT /tmp — the box wipes /tmp on stop/start).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from isaac.nav_sim import NavSimConfig, avoidance_action   # noqa: E402
from isaac.nav_task import NavTaskConfig                    # noqa: E402
from train_ppo import _rollout, make_ppo                    # noqa: E402


# (tag, collision_penalty, clearance_margin, clearance_penalty). The first is the current
# reward (the 2%-collision baseline); the rest add sparse and/or dense safety shaping.
DEFAULT_GRID = [
    ("baseline",   5.0, 0.30, 0.0),   # current reward — reproduces the ~2% collision point
    ("cp10",      10.0, 0.30, 0.0),   # just a harder collision penalty (still sparse)
    ("clear_soft", 5.0, 0.30, 0.5),   # gentle dense keep-clear, base collision penalty
    ("clear_firm",10.0, 0.40, 1.0),   # firmer + wider margin + harder collision penalty
    ("clear_wide", 8.0, 0.50, 1.0),   # widest margin (earliest avoidance), moderate penalty
]


def make_cfg(args, collision_penalty, clearance_margin, clearance_penalty) -> NavSimConfig:
    """A sweep-point env config: fixed sensing/scene knobs, varied reward shaping."""
    return NavSimConfig(
        bounds=(-5.0, -5.0, 5.0, 5.0),
        n_lidar_beams=args.beams,
        occupancy_size=args.occupancy,
        randomize_obstacles=args.obstacles,
        task=NavTaskConfig(
            max_steps=args.max_steps,
            collision_penalty=collision_penalty,
            clearance_margin=clearance_margin,
            clearance_penalty=clearance_penalty,
        ),
    )


def train_and_eval(tag, cfg, args, device, seed0):
    """Train one PPO policy on `cfg`, save it, and evaluate on the shared held-out scenes."""
    from stable_baselines3.common.env_util import make_vec_env
    from isaac.nav_gym import NavGymEnv

    print(f"\n=== [{tag}] training {args.timesteps} steps "
          f"(collision_penalty={cfg.task.collision_penalty}, "
          f"clearance_margin={cfg.task.clearance_margin}, "
          f"clearance_penalty={cfg.task.clearance_penalty}) ===", flush=True)
    venv = make_vec_env(lambda: NavGymEnv(cfg), n_envs=args.n_envs, seed=args.seed)
    model = make_ppo(venv, args.seed, device)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    out_dir = os.path.join(os.path.expanduser(args.out), tag)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "ppo_nav")
    model.save(save_path)

    # Held-out eval on the SAME scenes for every config (seed0 fixed across the sweep) so the
    # only variable is the learned policy. The eval env's reward config is irrelevant to the
    # outcome tallies (reached/collided/steps come from the sim, not the reward), so evaluate
    # every policy on the baseline env for a clean, identical comparison.
    eval_cfg = make_cfg(args, 5.0, 0.30, 0.0)
    reached, collided, n, ms = _rollout(
        lambda: NavGymEnv(eval_cfg),
        lambda obs, env: model.predict(obs, deterministic=True)[0],
        args.eval_episodes, args.max_steps, seed0)
    print(f"    [{tag}] reached {reached}/{n} ({100*reached/n:.0f}%)  "
          f"collided {collided}  mean-steps {ms:.0f}  -> {save_path}.zip", flush=True)
    return {"tag": tag, "collision_penalty": cfg.task.collision_penalty,
            "clearance_margin": cfg.task.clearance_margin,
            "clearance_penalty": cfg.task.clearance_penalty,
            "reached": reached, "collided": collided, "n": n, "mean_steps": ms}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=500_000,
                    help="training steps per config (fewer than the flagship 1.5M; this is a "
                         "relative comparison — retrain the winner longer if desired)")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--beams", type=int, default=16)
    ap.add_argument("--occupancy", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--out", default=os.path.expanduser("~/nav_ppo_sweep"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from isaac.nav_gym import NavGymEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed0 = 1_000_000  # held-out eval seed base (disjoint from training seeds)
    print(f"reward-shaping sweep: {len(DEFAULT_GRID)} configs x {args.timesteps} steps, "
          f"device={device}, {args.eval_episodes} held-out episodes each", flush=True)

    rows = [train_and_eval(tag, make_cfg(args, cp, cm, cpen), args, device, seed0)
            for (tag, cp, cm, cpen) in DEFAULT_GRID]

    # Reference: the hand-written gap-follower on the identical held-out scenes.
    ref_cfg = make_cfg(args, 5.0, 0.30, 0.0)
    a_reached, a_collided, a_n, a_ms = _rollout(
        lambda: NavGymEnv(ref_cfg),
        lambda obs, env: avoidance_action(obs, env.sim.cfg),
        args.eval_episodes, args.max_steps, seed0)

    print(f"\n=== reward-shaping sweep results "
          f"({args.eval_episodes} held-out episodes, identical scenes) ===")
    print(f"  {'config':11s}  {'cp':>4s} {'marg':>4s} {'clr':>4s}  "
          f"{'reached':>8s}  {'collided':>8s}  {'steps':>6s}")
    for r in rows:
        print(f"  {r['tag']:11s}  {r['collision_penalty']:4.0f} {r['clearance_margin']:4.2f} "
              f"{r['clearance_penalty']:4.1f}  {r['reached']:3d}/{r['n']:<3d} "
              f"({100*r['reached']/r['n']:3.0f}%)  {r['collided']:8d}  {r['mean_steps']:6.0f}")
    print(f"  {'avoidance':11s}  {'-':>4s} {'-':>4s} {'-':>4s}  "
          f"{a_reached:3d}/{a_n:<3d} ({100*a_reached/a_n:3.0f}%)  {a_collided:8d}  {a_ms:6.0f}")

    # Recommended pick: fewest collisions, then — among the near-tie — the fastest (fewest
    # mean-steps), provided reached% stays within 3 pts of the baseline (don't trade success
    # for safety). Purely advisory; the human makes the call.
    base = rows[0]
    ok = [r for r in rows if r["reached"] >= base["reached"] - 0.03 * base["n"]]
    pick = min(ok or rows, key=lambda r: (r["collided"], r["mean_steps"]))
    print(f"\n  suggested: '{pick['tag']}' — {pick['collided']} collisions, "
          f"{100*pick['reached']/pick['n']:.0f}% reached, {pick['mean_steps']:.0f} steps "
          f"(baseline: {base['collided']} collisions, {base['mean_steps']:.0f} steps)")

    summary = os.path.join(os.path.expanduser(args.out), "sweep_results.json")
    os.makedirs(os.path.dirname(summary), exist_ok=True)
    with open(summary, "w") as f:
        json.dump({"grid": rows, "avoidance": {"reached": a_reached, "collided": a_collided,
                                               "n": a_n, "mean_steps": a_ms},
                   "suggested": pick["tag"], "args": vars(args)}, f, indent=2)
    print(f"  wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
