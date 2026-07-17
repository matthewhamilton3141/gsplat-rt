#!/usr/bin/env python3
"""Train a PPO policy on the nav env (stable-baselines3) and evaluate it against the
lidar gap-following baseline on fresh random scenes — the roadmap's Phase-4 learned-policy
milestone. Box-only (needs torch + sb3 + gymnasium; the env core is pure NumPy).

    python scripts/nav/train_ppo.py --timesteps 300000 --beams 16 --obstacles 5

Reports success / collision / mean-steps for the trained policy vs the hand-written
`avoidance_action` on a held-out set of randomized fields — the honest "did learning beat
the heuristic" comparison.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from isaac.nav_sim import avoidance_action  # noqa: E402
from isaac.nav_task import NavTaskConfig     # noqa: E402
from isaac.nav_sim import NavSimConfig       # noqa: E402


def make_cfg(args) -> NavSimConfig:
    return NavSimConfig(
        bounds=(-5.0, -5.0, 5.0, 5.0),
        n_lidar_beams=args.beams,
        occupancy_size=args.occupancy,
        randomize_obstacles=args.obstacles,
        task=NavTaskConfig(max_steps=args.max_steps),
    )


def _rollout(env_ctor, policy, n, max_steps, seed0):
    """Run `n` episodes on fresh envs; return (reached, collided, n, mean_steps)."""
    reached = collided = 0
    steps = []
    for s in range(n):
        env = env_ctor()
        obs, _ = env.reset(seed=seed0 + s)
        t = 0
        while t < max_steps:
            obs, r, term, trunc, info = env.step(policy(obs, env))
            t += 1
            if term or trunc:
                reached += int(info["reached"])
                collided += int(info["collided"])
                break
        steps.append(t)
    return reached, collided, n, float(np.mean(steps))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--beams", type=int, default=16)
    ap.add_argument("--occupancy", type=int, default=0, help="egocentric grid size (0=off)")
    ap.add_argument("--obstacles", type=int, default=5, help="random obstacles per episode")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--out", default="/tmp/nav_ppo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from isaac.nav_gym import NavGymEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training PPO on {args.n_envs} envs, {args.timesteps} steps, device={device}")
    print(f"obs: 7 goal + {args.beams} lidar + {args.occupancy ** 2} occupancy; "
          f"{args.obstacles} random obstacles/episode")

    venv = make_vec_env(lambda: NavGymEnv(make_cfg(args)), n_envs=args.n_envs, seed=args.seed)
    model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device=device,
                n_steps=1024, batch_size=1024, gae_lambda=0.95, gamma=0.99, ent_coef=0.005)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    os.makedirs(args.out, exist_ok=True)
    save_path = os.path.join(args.out, "ppo_nav")
    model.save(save_path)
    print(f"saved policy -> {save_path}.zip")

    # --- held-out eval: trained PPO vs the hand-written gap-follower on the same scenes ---
    def env_ctor():
        return NavGymEnv(make_cfg(args))

    def ppo_policy(obs, env):
        return model.predict(obs, deterministic=True)[0]

    def avoid_policy(obs, env):
        return avoidance_action(obs, env.sim.cfg)

    seed0 = 1_000_000
    ppo = _rollout(env_ctor, ppo_policy, args.eval_episodes, args.max_steps, seed0)
    avo = _rollout(env_ctor, avoid_policy, args.eval_episodes, args.max_steps, seed0)
    print("\n=== held-out eval ({} episodes, identical scenes) ===".format(args.eval_episodes))
    for name, (reached, collided, n, ms) in (("PPO", ppo), ("avoidance", avo)):
        print(f"  {name:9s}: reached {reached}/{n} ({100 * reached / n:.0f}%)  "
              f"collided {collided}  mean-steps {ms:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
