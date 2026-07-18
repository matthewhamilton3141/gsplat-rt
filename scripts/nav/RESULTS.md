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
~70% more steps (99 vs 58). Reaching *exactly* 0 without surrendering the speed win needs a **hard
safety layer**, not a softer reward — done next.

## Hard safety shield — the frontier-breaker (measured, A10G)
`nav_sim.safety_shield` is a **one-step-lookahead collision filter** that wraps *any* policy at
runtime (no retraining): it scales a commanded forward speed down to the largest fraction whose
*predicted* next pose keeps clearance ≥ `safety_margin`, and forbids forward motion entirely when
boxed in (rotating in place can't collide). Its lookahead reuses the exact `predict_pose` /
`clearance_at` the sim integrates with, so it cannot disagree with the dynamics. `eval_shield.py`
rolls the **already-trained flagship policy** raw vs shielded on the same 200 held-out scenes.

| policy (flagship, 1.5M) | reached | collided | mean steps |
|---|---|---|---|
| PPO raw | 98% | 4 | 58 |
| **PPO + safety shield** | **95%** | **0** | 79 |
| `avoidance` (hand-written) | 96% | 0 | 99 |

**It works — collisions 4 → 0**, the thing reward shaping couldn't do (4 → 3). Wrapping the policy
only at eval time costs a little (reached 98% → 95%, steps 58 → 79) because the policy never trained
with the shield. First validated off-box on a reckless go-to-goal controller (53 → 0 collisions on
200 random scenes) before spending any GPU time.

## Shield-in-the-loop — train *through* the shield (measured, A10G): the capstone
`train_ppo --shield` applies the shield at the RL boundary (`NavGymEnv.step`, pure core untouched)
so the policy trains **through** the filter and adapts to it — it can commit to aggressive, direct
paths knowing the shield guarantees it can't collide. Retrained at the full 1.5M budget, evaluated
through the shield (its deployment configuration):

| policy (1.5M, 200 held-out scenes) | reached | collided | mean steps |
|---|---|---|---|
| heuristic (`avoidance`) — safe but slow | 96% | 0 | 99 |
| PPO raw — fast but unsafe | 98% | 4 | 58 |
| `clear_firm` (reward shaping) — couldn't fix it | 98% | 3 | 58 |
| PPO + shield (eval-time wrap) — safe, small cost | 95% | 0 | 79 |
| **PPO shield-in-the-loop — safe *and* best** | **100%** | **0** | **56** |

**Training through the shield doesn't just recover the eval-time cost — it dominates the raw
flagship on every axis: 100% reached (vs 98%), 0 collisions (vs 4), 56 steps (vs 58).** The policy
learns to *rely* on the shield: freed from having to be cautious itself, it takes the most direct
routes and lets the filter handle the rare near-miss — provably safe (0 collisions, guaranteed by
the shield) yet faster and more reliable than the unshielded policy that occasionally crashed.

**The nav arc, end to end:** hand-written heuristic (safe, slow) → learned PPO (fast, 2% crashes) →
reward shaping (can't reach 0 — and we said so) → hard safety shield (0 crashes, provably) →
**shield-in-the-loop (0 crashes *and* the fastest, most reliable policy of them all).** Every rung
measured on the box, nothing assumed.

## Does the occupancy map help? — lidar vs lidar+occupancy on a denser field (measured, A10G)
Closing the loop between the two halves of the project: the reconstruction pipeline emits a
top-down occupancy map (`*_occupancy.png`), and the nav env can feed an egocentric 8×8 slice of
exactly that as an extra observation block. On a **denser 8-obstacle field** (vs the 5 above),
`compare_obs.py` trains two shielded policies differing *only* in the observation and evaluates
both on one identical held-out set (collisions are 0 by construction, so this isolates reached% /
efficiency):

| observation (8-obstacle field, shielded) | reached | collided | mean steps |
|---|---|---|---|
| heuristic (`avoidance`) | 92% | 0 | 126 |
| lidar (7 goal + 16 lidar) | 100% | 0 | 59 |
| **lidar + 8×8 occupancy grid** | **100%** | 0 | **56** |

**Honest finding: the occupancy grid barely helps here — it's a marginal efficiency refinement,
not a categorical win.** Both learned policies already saturate the task (100% reached, 0
collisions via the shield); adding the 64-cell occupancy map only trims mean steps 59 → 56 (~5%)
and closes the last episode (199 → 200). The 16-beam lidar already carries enough local structure
for an 8-obstacle field, so the richer map is largely redundant *at this density* — a result worth
stating plainly rather than dressing up. (Two things that *did* hold: the denser field is genuinely
harder — the hand-written heuristic drops to 92% / 126 steps — and the shielded learned policies
still crush it at 100% / 0 / ~57, ~2× faster than the heuristic.) Where occupancy would likely earn
its keep: still-denser or non-convex clutter where beams miss what a top-down map captures — the
natural follow-up.

## Make it physical — PyBullet rigid-body backend + sim-to-sim transfer (measured, A10G)
The whole nav flagship so far runs in `DiffDriveNavEnv`, an analytic unicycle: motion is exact
integration and a "collision" is a geometric overlap. The real test of a sim-trained policy is
whether it survives *physics*. `nav_pybullet.PyBulletNavEnv` swaps in a **real rigid-body cylinder
in PyBullet** — mass, friction, contact-resolved collisions — while reusing the tested core for
everything else (an internal `DiffDriveNavEnv` samples the scene so a seed yields the *identical*
field, and supplies the observation + `nav_task` reward/termination; PyBullet owns only how the
robot moves and collides). 6 conformance tests pass on the box (contract, same-seed scene match,
physics motion, contact-based collision, shielded-heuristic reach, shield-prevents-collision).

Then the payoff — `eval_pybullet.py` runs the **kinematic-trained** shield-in-the-loop policy in
*both* worlds on identical held-out scenes (both shielded):

| world (same policy, 100 held-out scenes, shielded) | reached | collided | mean steps |
|---|---|---|---|
| kinematic (where it trained) | 100% | 0 | 57 |
| **PyBullet (rigid-body physics)** | **99%** | **0** | 63 |

**The policy transfers cleanly to physics with no retraining:** reached 100% → 99% (a single
episode), **0 collisions preserved** (the shield's safety guarantee holds under real contacts, not
just analytic ones), and mean steps 57 → 63 (~10% slower — the honest cost of friction and
contact dynamics the kinematic sim idealizes away). A small, well-characterized sim-to-sim gap is
exactly what you want to see before a hardware or Isaac Lab port: the learned behavior is not an
artifact of the kinematic idealization.

## Next
- **Isaac Lab port** — `isaac_nav_env.py` already targets the same tested contract; the PyBullet
  result de-risks it (the policy survives real physics). Needs the Isaac stack (`isaac_setup.sh`,
  `phase0_smoke.py`, and the Y-up→Z-up resolution) — see the M7 groundwork notes.
- **Occupancy in harder clutter** — non-convex arrangements where lidar beams miss what a top-down
  map catches (it didn't help at 8 convex obstacles).
