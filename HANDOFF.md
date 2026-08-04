# gsplat-rt — session handoff (updated 2026-08-04)

Human-readable "pick up here." Persistent context also lives in Claude memory (`MEMORY.md`
auto-loads each session); this is the plain-English summary of where things stand.

## 2026-08-04 — Option A (digital-twin nav) BUILT + merged to main, read this first
Built the "something cool" in-repo capstone, reframed on your pushback (*"not a webcam scan —
how would I teach an AV in a room?"*): the AV-honest use of splats is a **digital twin** —
reconstruct a scene into an occupancy grid, then test a driving policy *inside* it, closed-loop.
Three milestones, all pure-NumPy/OpenCV + Three.js, **Mac-only (no box)**, all **merged to main**:
- **Occupancy-grid world** (`src/isaac/grid_world.py`, #36): `GridWorld` gives the nav env
  clearance/collision/lidar against an arbitrary metric occupancy grid, folded into `nav_sim`
  additively (`cfg.grid_world`). Same policy + shield now run over reconstructed *shape*.
- **Ackermann car + braking shield + DWA planner** (`src/isaac/car_sim.py`, `driving_scene.py`,
  `scripts/nav/drive_scene.py`, #38): `BicycleNavEnv` (can't pivot at v=0) + `car_safety_shield`
  (brakes) + `car_dwa_action` (a Dynamic-Window-Approach local planner — a reactive gap-follower
  provably wedges a non-holonomic car at the shield's keep-out shell; DWA rolls out feasible arcs
  and doesn't). Demo: `docs/car_digital_twin.{mp4,gif,png}` (0 collisions, 196 steps).
- **In-browser closed loop** (`src/viz/nav_runner.py`, `web_viewer.py` `/api/nav[_scene]`,
  `static/viewer.js` nav layer, `run_viewer.py --nav`, #38): the Three.js viewer renders the
  occupancy grid in 3-D and animates the shielded car driving it live. Verified headless-Chromium
  (0 JS errors) → `docs/nav_browser.png`. Run: `python scripts/run_viewer.py --nav`.
- **Real reconstructed-scene loader** (`src/isaac/nurec_scene.py`, #40; README #41): a surface
  **mesh → occupancy** projector so a real drive reconstructed by NVIDIA **NuRec** (real-to-sim
  3DGS, shipped as USDZ + surface mesh + `.xodr`) becomes a scene the shielded car drives. Marks
  a cell occupied only where geometry sits in the car-height band (road below / gantry above are
  ignored). `.obj/.ply` via trimesh, `.usd/.usdz` via OpenUSD (`pxr`) — both lazy-imported (both
  already on the Mac). Verified on synthetic meshes + an authored OpenUSD `.usda` round-trip. Run:
  `python scripts/nav/drive_scene.py --mesh scene.usdz`.

Suite now **276 passed, 6 skipped** (was 229; +62 tests). Full detail in memory
`gsplat-digital-twin-nav.md`. Everything below (2026-08-01 pivot) still governs.

### NEXT STEP on gsplat-rt — drive a REAL NuRec clip (Mac-only, no box)
Everything is laid out; the only missing piece is one download (an interactive HF step, so it's a
**you-step**), then it's one command. Decided AGAINST NVIDIA's Alpamayo-R1 *model* (11B end-to-end
camera→trajectory planner — wrong shape for our interpretable shield+planner, and box-only);
adopted the NuRec *dataset* instead.
1. Accept terms: <https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec>
2. `hf auth login` (token: <https://hf.co/settings/tokens>) — CLI is installed (`hf` 1.26).
3. `scripts/fetch_nurec.sh --list` → pick a clip dir → `scripts/fetch_nurec.sh '<clip_dir>/**'`
   (downloads ~2 GB to `~/nurec`, then prints the exact `drive_scene.py --mesh …` command).
4. If the car drives through walls, the mesh is Y-up → add `--mesh-up-axis 1`.
Box-only, deferred: rendering the actual 3DGS splat layers, and loading the USDZ into Isaac.

### Portfolio framing (agreed 2026-08-04) — pairs with `~/Documents/kitti-nav`
One thesis across both repos: **a hard safety shield wrapped around a learned planner**, shown from
both ends of the AV stack. kitti-nav = the onboard path (stereo VO + lidar BEV → shielded PPO
planner on real KITTI, 78%/0 collisions); gsplat-rt = the digital-twin path (reconstruct a scene →
drive that same shielded car *inside* it, in-browser; NuRec-ready). Rules for any blurb: NuRec is
"ready to ingest / loader verified, real clip next" (capability, not done); Alpamayo only as a
scoped-out *decision*, never a feature; every figure **measured, not assumed**. gsplat-rt reads as
the *cool/ambitious* piece, kitti-nav as the *rigorous/applied* one — the contrast is the point.

**→ SWITCHING TO kitti-nav:** AV/portfolio work continues in `~/Documents/kitti-nav` (its own
`HANDOFF.md`). gsplat-rt is at a clean stopping point — `main` green, 0 open PRs; the only open
gsplat-rt thread is the NuRec download above, do-able anytime.

## 2026-08-01 — strategic pivot, read this first
**"I don't want to make this a portfolio piece, I just want to make something cool."** This
supersedes any resume/interview framing for this repo going forward — don't weigh "is this
defensible in an interview" when scoping work here. Keep the engineering discipline (measure,
don't hallucinate numbers, correct claims down) — that's good practice independent of motive,
not a portfolio artifact.

Also established this session: **splats are not what real AVs use for live driving decisions.**
Production AV stacks run on lidar/occupancy/BEV grids + detection nets; Gaussian splatting's
real role in AV work is offline — reconstructing recorded drives into replayable "digital twin"
scenes for closed-loop testing (Street Gaussians / DrivingGaussian / UniSim-style), not an
onboard real-time representation. gsplat-rt's own nav policy already reflects this correctly:
it consumes occupancy/lidar, never splats directly — splats are the rendering layer only.

**Two directions are on the table, not yet decided. Both are scoped to need NO Brev box** —
credits status is unconfirmed/likely still out as of this date; don't propose box-only steps as
actionable until credits are confirmed refreshed (see box section below).

### Option A — Robotics, off what's already built here (no box)
The idea we landed on: an all-software, buy-nothing capstone. Extend the existing browser
viewer (`scripts/run_viewer.py`, stdlib server + Three.js, already streams live splats) so you
can scan a real room with a webcam, watch it splat into 3D live, then **drop the already-trained
shield-in-loop nav policy into that exact reconstructed geometry in-browser** via WebGL physics
(e.g. cannon-es) — watch a simulated robot navigate the real shape of your actual room, live, no
sim-to-real gap because the room geometry *is* real.
- Reuses: `src/isaac/nav_sim.py` (`safety_shield`, `predict_pose`, `clearance_at` — the exact
  math to port to JS or serve from a small local inference endpoint), the trained shield-in-loop
  policy weights, the existing splat/occupancy export path, `run_viewer.py`'s server+viewer split.
- Open design question not yet resolved: run policy inference client-side (port the small MLP to
  JS/ONNX.js) vs. keep it server-side (stdlib server already exists, just add an action endpoint
  the browser polls/streams). Server-side is almost certainly less work — the policy net is tiny.
- Secondary/lower-priority thread: `src/isaac/isaac_nav_env.py` is a real stub (every method
  `raise NotImplementedError`) for the full GPU-parallel Isaac Lab RL port. It's Mac-*writable*
  (you can draft the vectorized wiring against `nav_task.py`'s tested reward/obs contract) but not
  Mac-*runnable or verifiable* (needs `isaaclab`/`isaacsim`/torch-CUDA) — treat as prep-only, box
  verification deferred, not the main thread.

### Option B — Branch into a new repo for AV work (no box, to start)
Reuses the transferable parts of gsplat-rt's stack, retargeted at driving scale:
- **SLAM front-end**: the ORB visual-odometry path is pure CPU/OpenCV, no GPU needed — can run
  directly against KITTI odometry sequences on the Mac. (SuperPoint+LightGlue TensorRT path stays
  box-gated for later; start with ORB.)
- **Occupancy/BEV, not splats**: the AV repo's live decision representation should be
  occupancy/BEV grids from the start, matching what real stacks do — splats (if kept at all) are
  an offline scene-reconstruction/testing artifact, not the live path.
- **Safety shield concept ports directly**: `nav_sim.safety_shield`'s one-step-lookahead
  collision-filter pattern generalizes to a bicycle/Ackermann kinematic model instead of
  diff-drive — same idea (wrap any learned policy with a hard collision filter), new dynamics.
- Concrete Mac-only first steps: (1) new repo, (2) KITTI odometry benchmark using the existing
  ORB frontend code as a starting point (adapt, don't copy blindly — dataset format differs from
  TUM), (3) design the bicycle-model nav env + shield before touching any GPU-dependent piece.

**DECIDED 2026-08-01: Option B.** Work moved to a new repo — `~/Documents/kitti-nav`
(public: <https://github.com/matthewhamilton3141/kitti-nav>), which has its own `HANDOFF.md`.
As of 2026-08-02 it is complete through stereo VO, lidar BEV occupancy, a braking-aware
safety shield running on real lidar, and a PPO planner behind that shield (78% success / 0
collisions on real KITTI geometry). Continue *that* repo's handoff for AV work.

**Option A is not dead, just not chosen first.** The browser splat-scan + WebGL-physics idea
(webcam-scan your room, drop the trained shield-in-loop nav policy into the reconstructed
geometry in-browser) is still the best "something cool" candidate in *this* repo, and still
needs no box. Its full scoping is in the Option A section above and remains accurate.

One cross-repo finding worth knowing here: **kitti-nav could not reproduce this repo's
shield-in-the-loop capstone.** Training a policy through the shield strictly dominated here
(100%/0/56 vs 98%/4/58), but on driving dynamics it was statistically indistinguishable from
bolting the shield on at evaluation (5 seeds; +1.1%, 95% CI [−0.5, +2.7], p = 0.14). The
likely reason is that this repo's shield *cost* the policy real performance when bolted on
(98%/4 → 95%/0, 58 → 79 steps) — that penalty is what in-loop training recovered — whereas
the driving shield is already nearly free. Nothing here needs correcting; the result stands
in its own setting.

## Current state (as of last code changes, 2026-07-18 — unchanged this session, discussion-only)
- **`main` is clean, 0 open PRs.** Everything below is merged and measured.
- Full test suite green: **229 passed, 6 skipped** (re-run 2026-08-01; GPU/dataset rows skip
  cleanly off-box).
- **A10G box credit status: unclear/likely still out.** Memory has conflicting notes from
  2026-07-18 (exhausted vs. refreshed same day); user confirmed 2026-08-01 as "still out/unknown."
  **Do not propose box-only work as actionable until the user confirms credits are refreshed.**

## The two flagship arcs (both complete, all merged to main, unchanged since 2026-07-18)

### LingBot-Map → TensorRT study (PRs #15–#17, #20, #21, #23)
A VGGT-style streaming-reconstruction foundation model taken to TensorRT on the A10G, every
figure reproduced on the box. Writeup: `scripts/lingbot_trt/RESULTS.md` (Stages 0–7).
- Profiled first: `global_blocks` 45% + DPT/camera heads 17.5% dominate — *not* the frame blocks
  (which is why the naive Stage-4 frame-block swap only moved the whole model ~1.08×).
- `global_blocks` (stateful, complex-RoPE + growing KV cache): 1.53× per-block → **1.069×** e2e.
- **DPT head** (static): 2.93× per-head → **1.098×** e2e, parity verified (1.19%, 0 NaN).
- **Stage 7 capstone — both levers stacked: 1.187× whole-model** (7.69 → 9.13 fps, parity 3.23%,
  0 NaN). The two disjoint slices (62.7% of runtime in TRT) **compound** above either alone.

### M7 nav RL flagship (PRs #16, #18, #19, #22, #24–#28)
Backend-agnostic diff-drive navigation (pure-NumPy core, CPU-trainable). Writeup:
`scripts/nav/RESULTS.md`.
- `DiffDriveNavEnv` core + lidar gap-following baseline + egocentric occupancy obs + per-episode
  obstacle randomization; `nav_gym.NavGymEnv` (only gymnasium dep) + `scripts/nav/train_ppo.py`.
- Full arc, all A10G-measured: heuristic 96%/0/99 → PPO 98%/4/58 → reward-shaping 98%/3/58
  (can't hit 0) → **hard safety shield** 95%/0/79 → **shield-in-the-loop 100%/0/56** (dominates
  everything). `nav_sim.safety_shield` = one-step-lookahead collision filter over any policy.
- Occupancy-vs-lidar (PR #27): marginal (lidar alone already suffices at this density).
- PyBullet rigid-body backend (PR #28): sim-to-sim transfer, kinematic 100%/0/57 → PyBullet
  99%/0/63, no retraining needed. De-risked Isaac's physics realism.

### Isaac Sim (PRs #29–#31, #34)
Root cause of an early RTX-headless blocker was a driver mismatch (box driver 595 vs Isaac 5.1's
validated 580.65) — fixed by downgrading in place. Isaac Sim now: installs + boots + RTX-renders
+ **Phase 0 drop-test PASSES** (a sphere dropped on the reconstructed mesh rests on it — the
reconstruct→Isaac→PhysX-collision bridge, proven). The only unbuilt Isaac piece is the
GPU-parallel Isaac Lab RL port (`isaac_nav_env.py`) — see Option A above, now de-prioritized
relative to the browser-physics idea.

## Environment / box gotchas (for whenever the box is confirmed usable again)
- Box = remote A10G Brev `proper-yellow-skunk` (id hp0yaxne3). Credit status unconfirmed as of
  2026-08-01 — verify with the user before any box step.
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
- Nav: `src/isaac/{nav_sim,nav_task,nav_gym,nav_pybullet}.py`,
  `scripts/nav/{train_ppo,sweep_reward,render_policy,random_rollout,compare_obs,eval_shield,
  eval_pybullet}.py`; `RESULTS.md`.
- Viewer: `scripts/run_viewer.py` (stdlib server + Three.js, live webcam splat streaming) —
  the likely base for Option A.
