#!/usr/bin/env python3
"""Evaluate a trained nav policy with vs without the hard safety shield — the experiment the
reward-shaping sweep pointed to. Box-only (needs sb3 to load the policy; the shield core in
`nav_sim` and its tests run anywhere).

The reward sweep showed shaping is a convergence-speed lever, not a path to zero collisions:
at full budget the PPO policy sits at ~98% reached / ~2% collisions / 58 steps, and softer
reward can't push collisions to 0 without surrendering the speed win. `nav_sim.safety_shield`
is the frontier-breaker instead: a one-step-lookahead filter that throttles any commanded
forward speed whose predicted pose would enter the obstacle margin (and forbids forward motion
outright when boxed in), wrapping the *already-trained* policy at runtime — no retraining.

This loads a saved policy and rolls it out on one identical held-out scene set three ways —
the raw policy, the shielded policy, and the hand-written gap-follower — reporting reached /
collided / mean-steps so the shield's effect (collisions → 0? at what speed cost?) is measured,
not assumed.

    python scripts/nav/eval_shield.py --policy ~/nav_ppo/ppo_nav.zip --episodes 200
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from isaac.nav_sim import NavSimConfig, avoidance_action, safety_shield   # noqa: E402
from isaac.nav_task import NavTaskConfig                                  # noqa: E402
from train_ppo import _rollout                                           # noqa: E402


def make_cfg(args) -> NavSimConfig:
    """The flagship training/eval scene distribution (must match what the policy saw)."""
    return NavSimConfig(
        bounds=(-5.0, -5.0, 5.0, 5.0),
        n_lidar_beams=args.beams,
        occupancy_size=args.occupancy,
        randomize_obstacles=args.obstacles,
        safety_margin=args.safety_margin,
        task=NavTaskConfig(max_steps=args.max_steps),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=os.path.expanduser("~/nav_ppo/ppo_nav.zip"),
                    help="saved sb3 PPO policy (.zip)")
    ap.add_argument("--beams", type=int, default=16)
    ap.add_argument("--occupancy", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--safety-margin", type=float, default=0.05,
                    help="clearance (m) the shield keeps free ahead of the robot")
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from isaac.nav_gym import NavGymEnv

    model = PPO.load(args.policy, device="cpu")
    print(f"loaded {args.policy}; safety_margin={args.safety_margin} m, "
          f"{args.episodes} held-out episodes (identical scenes)")

    cfg = make_cfg(args)

    def env_ctor():
        return NavGymEnv(cfg)

    def raw_policy(obs, env):
        return model.predict(obs, deterministic=True)[0]

    def shielded_policy(obs, env):
        a = model.predict(obs, deterministic=True)[0]
        return safety_shield(a, env.sim.robot_xy, env.sim._heading, env.sim._obstacles,
                             env.sim.cfg)

    def avoid_policy(obs, env):
        return avoidance_action(obs, env.sim.cfg)

    seed0 = 1_000_000   # same held-out base seed as train_ppo / sweep_reward
    runs = [("PPO raw", raw_policy), ("PPO + shield", shielded_policy),
            ("avoidance", avoid_policy)]
    print(f"\n=== shield eval ({args.episodes} held-out episodes, identical scenes) ===")
    print(f"  {'policy':14s}  {'reached':>9s}  {'collided':>8s}  {'mean-steps':>10s}")
    results = {}
    for name, pol in runs:
        reached, collided, n, ms = _rollout(env_ctor, pol, args.episodes, args.max_steps, seed0)
        results[name] = (reached, collided, n, ms)
        print(f"  {name:14s}  {reached:3d}/{n:<3d} ({100*reached/n:3.0f}%)  "
              f"{collided:8d}  {ms:10.0f}")

    raw, shd = results["PPO raw"], results["PPO + shield"]
    print(f"\n  shield effect: collisions {raw[1]} -> {shd[1]} "
          f"({raw[1]-shd[1]:+d}); reached {100*raw[0]/raw[2]:.0f}% -> {100*shd[0]/shd[2]:.0f}%; "
          f"steps {raw[3]:.0f} -> {shd[3]:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
