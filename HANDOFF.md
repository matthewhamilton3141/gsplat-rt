# gsplat-rt — session handoff (2026-07-18)

Human-readable "pick up here." Persistent context also lives in Claude memory (`MEMORY.md`
auto-loads each session); this is the plain-English summary of where things stand.

## Current state (clean — project reads as complete)
- **`main` is clean, 0 open PRs, box STOPPED.** Everything below is merged and measured.
- Full test suite green: **229 passed, 5 skipped** (GPU/dataset rows skip off-box).
- **A10G box credits refreshed (~$9.31) 2026-07-18** for the safety-shield runs; box STOPPED
  after each (bills ~$1.21/hr running + $0.04 storage). Nothing is waiting on a GPU run.

## ⭐ Guiding preference (set by you, still governs)
**"I don't really care about the numbers, I want it to actually work and be meaningful."**
→ Favor demonstrable, honest, coherent artifacts over chasing marginal speedup decimals. Report
measured numbers only; correct claims *down* when a fresh run disagrees.

## The two flagship arcs (both complete, all merged to main)

### LingBot-Map → TensorRT study (PRs #15–#17, #20, #21, #23)
A VGGT-style streaming-reconstruction foundation model taken to TensorRT on the A10G, every
figure reproduced on the box. Writeup: `scripts/lingbot_trt/RESULTS.md` (Stages 0–7).
- Profiled first: `global_blocks` 45% + DPT/camera heads 17.5% dominate — *not* the frame blocks
  (which is why the naive Stage-4 frame-block swap only moved the whole model ~1.08×).
- `global_blocks` (stateful, complex-RoPE + growing KV cache): 1.53× per-block → **1.069×** e2e.
- **DPT head** (static): 2.93× per-head → **1.098×** e2e, parity verified (1.19%, 0 NaN).
- **Stage 7 capstone — both levers stacked: 1.187× whole-model** (7.69 → 9.13 fps, parity 3.23%,
  0 NaN). The two disjoint slices (62.7% of runtime in TRT) **compound** above either alone — the
  mirror image of the per-block dilution lesson. This was the "shelved combined number"; we ran it.

### M7 nav RL flagship (PRs #16, #18, #19, #22, #24)
Backend-agnostic diff-drive navigation (pure-NumPy core, CPU-trainable). Writeup:
`scripts/nav/RESULTS.md`.
- `DiffDriveNavEnv` core + lidar gap-following baseline + egocentric occupancy obs + per-episode
  obstacle randomization; `nav_gym.NavGymEnv` (only gymnasium dep) + `scripts/nav/train_ppo.py`.
- **Trained PPO: 98% reached on held-out scenes, ~40% faster than the heuristic** (2% collisions).
- **`render_policy.py`** → committed `docs/nav_ppo_policy.gif` (in README) + `.mp4`.
- **Reward-shaping sweep (PR #24)** — the "push collisions → 0" experiment, done and reported
  honestly: added an optional **dense clearance penalty** + `sweep_reward.py` (5-config grid).
  MEASURED: a large *sample-efficiency* win at reduced budget (clear_firm 94%/10-collisions vs
  baseline 78%/44) but it **washes out at full budget** (clear_firm 98%/3/58 vs the un-shaped
  flagship 98%/4/58 — barely 4→3). **Conclusion: reward shaping is a convergence-speed lever, not
  a path to zero collisions.** The frontier-breaker is a *hard safety shield*, not softer reward.
- **Hard safety shield (PR #25) + shield-in-the-loop (PR #26) — the nav capstone.**
  `nav_sim.safety_shield` = a one-step-lookahead collision filter over *any* policy (its lookahead
  reuses the exact `predict_pose`/`clearance_at` the sim integrates with). Wrapping the flagship at
  eval time took collisions **4 → 0** (95%/0/79) — what shaping couldn't. Then training PPO
  *through* the shield (`train_ppo --shield`) made the policy adapt and **dominate everything:
  100% reached / 0 collisions / 56 steps**. Full arc, all A10G-measured: heuristic 96%/0/99 → PPO
  98%/4/58 → reward-shaping 98%/3/58 → shield 95%/0/79 → **shield-in-loop 100%/0/56**.
- **Occupancy-vs-lidar (PR #27) — honest marginal result.** Closed the reconstruction→nav loop:
  fed an egocentric slice of the pipeline's occupancy map as an obs block, on a denser 8-obstacle
  field. `compare_obs.py`: lidar **100%/0/59** vs lidar+occupancy **100%/0/56** — occupancy barely
  helps (~5% steps), 16-beam lidar already suffices at this density. Reported plainly, not dressed up.
- **PyBullet rigid-body backend (PR #28) — "make it physical".** `nav_pybullet.PyBulletNavEnv` =
  real rigid-body physics (mass, friction, contact collisions) behind the same tested contract
  (reuses the core for scene/obs/reward; PyBullet owns only motion+collision). Sim-to-sim transfer
  of the kinematic-trained policy: kinematic **100%/0/57** → PyBullet **99%/0/63** — transfers with
  no retraining, 0 collisions preserved under real contacts, ~10% slower (friction). De-risks Isaac.
  6 conformance tests (skip without pybullet). pybullet: Linux wheel only (no macOS build).

## Good "next" options (nothing required — both flagships thoroughly explored)
- **Isaac Sim — RESOLVED, renders the reconstructed scene (PRs #29/#30/#31).** Root cause was the
  box's driver 595 (R590) being newer than Isaac 5.1's validated **580.65** → RTX segfault (repro'd
  on pip *and* the NGC Docker image). **Fix: downgraded the host driver in place to 580.65.06** (the
  `.run` install + `mask nvidia-persistenced` to beat the reload race; no reboot). Isaac Sim 5.1 now
  boots + RTX renders; **`scripts/isaac/render_scene.py`** produces `docs/isaac_reconstructed_scene*.png`
  in the NGC container. **Phase 0 drop-test now PASSES (PR #34)** — a sphere dropped above the
  reconstructed mesh rests on the surface → the reconstruct→Isaac→PhysX-collision bridge, proven.
  Box driver is now 580 (persists). Isaac now: installs + boots + RTX-renders + physics-PASS. The only
  unbuilt Isaac piece left is the GPU-parallel Isaac Lab RL port (`isaac_nav_env.py`, same contract).
- **Occupancy in harder clutter** — the 8-obstacle result was marginal; push obstacle count / add
  non-convex arrangements where lidar beams miss what a top-down map catches. Reuses `compare_obs.py`.
- **Resume PDF recompile** *(no box, local)* — DEFERRED by you; do NOT surface as a TODO.
- **Cosmetic:** ~5 stale remote branches remain (all squash-merged); pruning needs your OK.

## Environment / box gotchas (for whenever the box comes back)
- Box = remote A10G Brev `proper-yellow-skunk` (id hp0yaxne3), **STOPPED** (credits refreshed
  2026-07-18). Driven over SSH via the brev ssh_config; `brev login` needs an interactive
  Terminal (not `! `).
- **DNS changes on every stop/start** → run `brev refresh` after `brev start`, or SSH times out.
- **`brev stop` can throw a transient "invalid status transition starting→stopping" for ~3 min**
  right after a fresh start — retry until it takes, then confirm `STOPPED`.
- **⚠ Box `/tmp` is WIPED on stop/start** — save policies/engines to `~/`, never `/tmp`.
- **Training env = `~/lingbot-map/.venv`** (sb3 2.9.0 + gymnasium 1.3.0 + torch cu128, CUDA). The
  venv is uv-managed with no `pip` (system `pip` ≠ venv) → `~/.local/bin/uv pip install ...`.
- **MLP-PPO trains faster on CPU** than the GPU here → force `CUDA_VISIBLE_DEVICES=` for nav runs.
- Dev Mac has **no GPU/torch/sb3/gymnasium**; pure-NumPy cores + all non-GPU tests run locally.
- Persistent box assets (survive stop/start, not `/tmp`): `~/lingbot-map` (+venv + 4.6 GB
  `lingbot-map-long.pt`), `~/gsplat-rt`, `~/nav_ppo/`, `~/nav_ppo_sweep/`, `~/nav_ppo_clearfirm/`.

## Key repo facts
- Workflow: branch → PR → squash-merge → delete. Tests: `python3 -m pytest tests/`.
- LingBot tooling: `scripts/lingbot_trt/` (export/integrate for global_blocks + heads,
  `integrate_combined_e2e.py`, `build_and_bench_trt.py`; `RESULTS.md`).
- Nav: `src/isaac/{nav_sim,nav_task,nav_gym}.py`,
  `scripts/nav/{train_ppo,sweep_reward,render_policy,random_rollout}.py`; `RESULTS.md`.
