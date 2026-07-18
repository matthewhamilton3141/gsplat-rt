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
  obstacle occasionally. The reward-shaping sweep below chased this.

## Reward-shaping sweep — push collisions → 0 (measured, A10G)
The sparse `collision_penalty` only fires *on* the hit, giving the policy no gradient to stay
clear. So I added an optional **dense clearance penalty** (`nav_task.clearance_penalty`) that
ramps up as the robot's clearance to the nearest obstacle/wall falls below `clearance_margin`,
peaking at contact — a gradient to avoid *before* colliding. `scripts/nav/sweep_reward.py` trains
a 5-config grid (sparse ± dense shaping) and evaluates each on one **identical** held-out scene
set. To keep the sweep cheap it trains each config at a reduced **500k** steps (⅓ of the flagship
budget), so read the sweep as a *relative* comparison, then the winner was retrained at the full
1.5M.

| config | reward shape | reached | collided | mean steps |
|---|---|---|---|---|
| baseline | cp=5, no clearance | 78% | 44 | 48 |
| cp10 | cp=10 (sparse only) | 68% | 64 | 44 |
| clear_soft | cp=5 + dense 0.5, margin 0.30 | 68% | 64 | 44 |
| **clear_firm** | cp=10 + dense 1.0, margin 0.40 | **94%** | **10** | 59 |
| clear_wide | cp=8 + dense 1.0, margin 0.50 | 88% | 14 | 78 |
| *avoidance (ref)* | hand-written gap-follower | 96% | 0 | 99 |

*(500k-step budget; the baseline is undertrained vs the 1.5M flagship — hence 78%, not 98%.)*

**Reading it honestly — two findings, both measured:**
- **At a reduced budget, the dense shaping is a large sample-efficiency win.** `clear_firm` cut
  collisions **44 → 10 (−77%)** *and* raised reached **78% → 94%** vs the same-budget baseline.
  Notably, bumping the *sparse* penalty alone (`cp10`) or a *gentle* dense penalty (`clear_soft`)
  **did not help** (both 68% / 64 collisions) — it's specifically the firm dense keep-clear
  gradient that works, and over-widening the margin (`clear_wide`) traded speed for nothing.
- **At the full 1.5M budget the effect largely washes out.** Retraining `clear_firm` at 1.5M:
  **98% reached, 3 collisions, 58 steps** — vs the un-shaped flagship's **98% / 4 / 58**. So the
  shaping barely moved final collisions (**4 → 3**); the un-shaped policy simply *converges* to the
  same speed/safety frontier given enough samples. **The dense clearance penalty is a
  convergence-speed lever, not a path to zero collisions.**

| policy (1.5M steps) | reached | collided | mean steps |
|---|---|---|---|
| flagship (no shaping) | 98% | 4 | 58 |
| **clear_firm (dense clearance)** | **98%** | **3** | **58** |

**The honest conclusion:** reward shaping alone doesn't break the speed↔safety frontier here — PPO
at 98%/58-steps sits at ~1.5–2% collisions, while the only 0-collision policy (the heuristic) pays
~70% more steps (99 vs 58). Reaching *exactly* 0 without surrendering the speed win likely needs a
**hard safety layer** (action masking / a shielded controller near obstacles) rather than a softer
reward — a bigger change than shaping. Saved policies: `~/nav_ppo_sweep/<config>/` +
`~/nav_ppo_clearfirm/` on the box; sweep summary `~/nav_ppo_sweep/sweep_results.json`.

## Next
- **Hard safety shield** (clip actions that would enter an obstacle's margin) — the frontier-
  breaking change the reward sweep showed shaping can't deliver.
- Add the **egocentric occupancy grid** to the obs (already in the env) for denser fields;
  curriculum over `randomize_obstacles`. Port onto a PyBullet rigid-body backend, then the Isaac
  Lab adapter (`isaac_nav_env.py`, same env contract).
