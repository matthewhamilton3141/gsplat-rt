#!/usr/bin/env python3
"""Sim-to-sim transfer: run a policy trained in the kinematic sim inside the PyBullet rigid-body
sim, and compare. Box-only (needs sb3 to load the policy + pybullet for the physics backend).

The nav policy was trained in `DiffDriveNavEnv` — an analytic unicycle where a collision is an
exact geometric overlap and motion is exact integration. The real question for any sim-trained
policy is whether it survives contact with *physics*: mass, friction, contact-resolved collisions.
`nav_pybullet.PyBulletNavEnv` swaps in exactly that (reusing the same scene sampling, observation,
reward, and shield), so the same policy on the same held-out scenes can be scored in both worlds.

Reports reached / collided / mean-steps for the policy in the kinematic sim vs the PyBullet sim
(both through the safety shield, the deployment config). A small gap = the policy transfers.

    python scripts/nav/eval_pybullet.py --policy ~/nav_ppo_shielded/ppo_nav.zip --episodes 100
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from isaac.nav_sim import DiffDriveNavEnv, NavSimConfig, safety_shield   # noqa: E402
from isaac.nav_task import NavTaskConfig                                 # noqa: E402


def make_cfg(args) -> NavSimConfig:
    return NavSimConfig(
        bounds=(-5.0, -5.0, 5.0, 5.0),
        n_lidar_beams=args.beams,
        occupancy_size=args.occupancy,
        randomize_obstacles=args.obstacles,
        use_safety_shield=False,      # shield applied explicitly in the rollout (both envs)
        task=NavTaskConfig(max_steps=args.max_steps),
    )


def _rollout(make_env, model, n, max_steps, seed0):
    reached = collided = 0
    steps = []
    for s in range(n):
        env = make_env()
        obs, _ = env.reset(seed=seed0 + s)
        c = False
        t = 0
        for t in range(max_steps):
            a = model.predict(obs, deterministic=True)[0]
            kin = env._kin if hasattr(env, "_kin") else env      # PyBullet vs kinematic
            a = safety_shield(a, kin.robot_xy, kin._heading, kin._obstacles, env.cfg)
            obs, r, term, trunc, info = env.step(a)
            c = c or info["collided"]
            if term or trunc:
                break
        reached += int(info["reached"])
        collided += int(c)
        steps.append(t + 1)
        if hasattr(env, "close"):
            env.close()
    return reached, collided, n, float(np.mean(steps))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=os.path.expanduser("~/nav_ppo_shielded/ppo_nav.zip"))
    ap.add_argument("--beams", type=int, default=16)
    ap.add_argument("--occupancy", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from isaac.nav_pybullet import PyBulletNavEnv

    model = PPO.load(args.policy, device="cpu")
    cfg = make_cfg(args)
    print(f"loaded {args.policy}; sim-to-sim transfer, {args.episodes} held-out scenes "
          f"({args.obstacles} obstacles), both shielded")

    seed0 = 1_000_000
    kin = _rollout(lambda: DiffDriveNavEnv(cfg), model, args.episodes, args.max_steps, seed0)
    phys = _rollout(lambda: PyBulletNavEnv(cfg), model, args.episodes, args.max_steps, seed0)

    print(f"\n=== sim-to-sim transfer ({args.episodes} identical held-out scenes, shielded) ===")
    print(f"  {'world':18s}  {'reached':>9s}  {'collided':>8s}  {'mean-steps':>10s}")
    for name, (reached, collided, n, ms) in (("kinematic (train)", kin),
                                             ("PyBullet (physics)", phys)):
        print(f"  {name:18s}  {reached:3d}/{n:<3d} ({100*reached/n:3.0f}%)  "
              f"{collided:8d}  {ms:10.0f}")
    print(f"\n  transfer gap: reached {100*kin[0]/kin[2]:.0f}% -> {100*phys[0]/phys[2]:.0f}%; "
          f"collisions {kin[1]} -> {phys[1]}; steps {kin[3]:.0f} -> {phys[3]:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
