# Nav flagship — learned policy vs hand-written baseline (measured)

The M7 navigation flagship: a differential-drive robot reaching a goal amid obstacles, built
backend-agnostic (pure-NumPy core in `src/isaac/nav_sim.py`, tested on a laptop) with a thin
`gymnasium` shell (`nav_gym.py`) for RL. This is the roadmap's "build the flagship first,
Isaac is a port" milestone. All numbers measured, nothing assumed.

## Setup
- **Env:** `DiffDriveNavEnv` — unicycle kinematics, circular obstacles + wall collision, **16-beam
  lidar** observation (`[nav_task goal (7)] + [lidar (16)]`), dense progress reward. **5 random
  obstacles regenerated every episode** (`randomize_obstacles`), start & goal kept clear.
- **Algo:** PPO (stable-baselines3, `MlpPolicy`), 16 parallel envs, **1.5M steps** on the A10G
  box (CPU rollouts — MLP-PPO is CPU-bound). `scripts/nav/train_ppo.py`.
- **Eval:** 200 held-out randomized scenes, *identical* scenes for both policies.

## Result (A10G, 200 held-out episodes)
| policy | reached | collided | mean steps to goal |
|---|---|---|---|
| **PPO (learned, 1.5M steps)** | **195 / 200 (98%)** | 4 | **58** |
| `avoidance_action` (hand-written gap-follower) | 193 / 200 (96%) | **0** | 99 |

Training reward plateaued at `ep_rew_mean ≈ 14`.

## Reading it honestly
- **Learning worked:** from a useless random policy (≈5% at 5k steps) to **98%** success — on par
  with the hand-tuned heuristic, on scenes neither was shown during training.
- **The learned win is efficiency:** the PPO policy reaches the goal in **~41% fewer steps** (58
  vs 99) — it learned to take more direct routes instead of the gap-follower's conservative wide
  detours.
- **The honest cost is a bit of safety:** 4 collisions (2%) vs the heuristic's 0. The classic
  speed↔safety tradeoff — PPO exploits the dense progress reward toward faster paths and clips an
  obstacle occasionally. A heavier `collision_penalty` (currently 5.0) or a safety margin in the
  reward would trade some of that speed back; a clean next experiment.

## Next
- Reward-shaping sweep (collision penalty / safety margin) to push collisions → 0 without losing
  the speed win; add the **egocentric occupancy grid** to the obs (already in the env) and measure
  whether the richer map helps in denser fields.
- Curriculum over `randomize_obstacles` count. Port the trained policy onto a PyBullet rigid-body
  backend, then the Isaac Lab adapter (`isaac_nav_env.py`, same env contract).
