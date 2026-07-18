#!/usr/bin/env python3
"""Does the occupancy map help? — lidar-only vs lidar+occupancy nav policy on a denser field.
Box-only (needs torch + sb3 + gymnasium; the env core is pure NumPy).

The reconstruction pipeline already emits a top-down **occupancy map** (`*_occupancy.png`); the
nav env can feed an egocentric slice of exactly that representation as an extra observation
block. This closes the loop between the two halves of the project — reconstruction → navigation
— by asking, measured: on a *denser* obstacle field, does giving the (provably-safe, shielded)
policy that occupancy grid on top of its lidar actually improve navigation, or does the lidar
already carry enough?

Trains two policies that differ ONLY in the observation — `lidar` (7 goal + 16 lidar) vs
`lidar+occupancy` (… + an occupancy_size² egocentric grid) — both **through the safety shield**
(so collisions are 0 by construction and the comparison is purely reached% / efficiency), on
the same denser scene distribution, and evaluates both on one identical held-out set.

    python scripts/nav/compare_obs.py --timesteps 1500000 --obstacles 8 --occupancy 8 \
        --eval-episodes 200 --out ~/nav_ppo_obs

⚠ Save --out under ~/ (NOT /tmp — the box wipes /tmp on stop/start).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from isaac.nav_sim import NavSimConfig, avoidance_action, safety_shield   # noqa: E402
from isaac.nav_task import NavTaskConfig                                  # noqa: E402
from train_ppo import _rollout, make_ppo                                  # noqa: E402


def make_cfg(args, occupancy: int) -> NavSimConfig:
    """Denser-field config, shield-in-the-loop; `occupancy` (0 or N) is the only thing varied."""
    return NavSimConfig(
        bounds=(-5.0, -5.0, 5.0, 5.0),
        n_lidar_beams=args.beams,
        occupancy_size=occupancy,
        randomize_obstacles=args.obstacles,
        use_safety_shield=True,
        task=NavTaskConfig(max_steps=args.max_steps),
    )


def train_and_eval(tag, occupancy, args, device, seed0):
    from stable_baselines3.common.env_util import make_vec_env
    from isaac.nav_gym import NavGymEnv

    cfg = make_cfg(args, occupancy)
    extra = f"{occupancy*occupancy} occupancy" if occupancy else "no occupancy"
    print(f"\n=== [{tag}] training {args.timesteps} steps "
          f"(obs: 7 goal + {args.beams} lidar + {extra}; {args.obstacles} obstacles, shielded) ===",
          flush=True)
    venv = make_vec_env(lambda: NavGymEnv(cfg), n_envs=args.n_envs, seed=args.seed)
    model = make_ppo(venv, args.seed, device)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    out_dir = os.path.join(os.path.expanduser(args.out), tag)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "ppo_nav")
    model.save(save_path)

    reached, collided, n, ms = _rollout(
        lambda: NavGymEnv(cfg),
        lambda obs, env: model.predict(obs, deterministic=True)[0],
        args.eval_episodes, args.max_steps, seed0)
    print(f"    [{tag}] reached {reached}/{n} ({100*reached/n:.0f}%)  collided {collided}  "
          f"mean-steps {ms:.0f}  -> {save_path}.zip", flush=True)
    return {"tag": tag, "occupancy": occupancy, "reached": reached, "collided": collided,
            "n": n, "mean_steps": ms}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--beams", type=int, default=16)
    ap.add_argument("--occupancy", type=int, default=8, help="occupancy grid side (N -> N*N cells)")
    ap.add_argument("--obstacles", type=int, default=8, help="random obstacles/episode (denser)")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--out", default=os.path.expanduser("~/nav_ppo_obs"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from isaac.nav_gym import NavGymEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed0 = 1_000_000
    print(f"occupancy comparison: lidar vs lidar+occupancy, {args.obstacles}-obstacle field, "
          f"shielded, {args.timesteps} steps each, device={device}", flush=True)

    rows = [train_and_eval("lidar", 0, args, device, seed0),
            train_and_eval("lidar+occupancy", args.occupancy, args, device, seed0)]

    # Reference: the hand-written gap-follower on the identical denser held-out scenes.
    ref = make_cfg(args, 0)
    a = _rollout(lambda: NavGymEnv(ref), lambda obs, env: avoidance_action(obs, env.sim.cfg),
                 args.eval_episodes, args.max_steps, seed0)

    print(f"\n=== occupancy comparison ({args.obstacles}-obstacle field, "
          f"{args.eval_episodes} held-out episodes, identical scenes, all shielded) ===")
    print(f"  {'obs':16s}  {'reached':>9s}  {'collided':>8s}  {'mean-steps':>10s}")
    for r in rows:
        print(f"  {r['tag']:16s}  {r['reached']:3d}/{r['n']:<3d} ({100*r['reached']/r['n']:3.0f}%)  "
              f"{r['collided']:8d}  {r['mean_steps']:10.0f}")
    print(f"  {'avoidance':16s}  {a[0]:3d}/{a[2]:<3d} ({100*a[0]/a[2]:3.0f}%)  {a[1]:8d}  {a[3]:10.0f}")

    ld, oc = rows[0], rows[1]
    print(f"\n  occupancy effect: reached {100*ld['reached']/ld['n']:.0f}% -> "
          f"{100*oc['reached']/oc['n']:.0f}%; steps {ld['mean_steps']:.0f} -> {oc['mean_steps']:.0f}")

    summary = os.path.join(os.path.expanduser(args.out), "compare_obs_results.json")
    os.makedirs(os.path.dirname(summary), exist_ok=True)
    with open(summary, "w") as f:
        json.dump({"rows": rows, "avoidance": {"reached": a[0], "collided": a[1], "n": a[2],
                                               "mean_steps": a[3]}, "args": vars(args)}, f, indent=2)
    print(f"  wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
