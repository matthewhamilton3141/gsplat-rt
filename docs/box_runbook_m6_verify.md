# A10G box runbook — M6 end-to-end pose-tracking verify

Goal: prove the wired `pose_tracking='superpoint', pose_backend='tensorrt'` path
runs **inside the live PipelineManager** on the GPU box (not just eval_odometry),
tracks a non-identity trajectory, and holds the latency budget. Everything below
is paste-into-the-box; the Mac cannot run any of it.

Prereqs already true on a bootstrapped box (`bash scripts/brev_setup.sh`): TensorRT
+ onnxruntime-gpu in the system env, TUM fr1/desk fetched, depth engine built.

**Two hard-won environment facts (2026-07-09, fresh box) — read before pasting:**
- **Interpreter:** brev's Jupyter terminal resolves bare `python3` to an interpreter
  WITHOUT the deps (no numpy/tensorrt_libs). The stack lives under
  `~/.local/lib/python3.10`, so use **`python3.10`** explicitly on this box.
- **TensorRT major must be 10.** onnxruntime-gpu 1.2x's TensorRT EP links
  `libnvinfer.so.10`. If `pip` pulled TensorRT 11 (`libnvinfer.so.11`), the EP fails
  and silently drops to CUDA. Fix: `pip install "tensorrt>=10,<11"` and **rebuild the
  depth engines** (`rm models/depth_engine*.engine; python3.10 src/depth/compile_trt.py
  --fp16`) — a TRT-11-built engine won't deserialize under TRT-10 (depth would fall
  back to mock). Now pinned in requirements.txt + brev_setup.sh so fresh boxes are OK.

## 0. One-time: get the fused ONNX onto the box

The fused SuperPoint+LightGlue ONNX is **not committed**. Produce it (idempotent):

```bash
cd ~/gsplat-rt
bash scripts/export_sp_lg.sh          # -> models/sp_lg_tum.onnx  (uses isolated uv env)
ls -la models/sp_lg_tum.onnx          # confirm it exists
```

## 1. Sanity: SLAM ATE via the TensorRT front-end (fast, isolates the engine)

This is the already-verified path — run it first to confirm the engine + provider
still load before touching the pipeline. Expect ~3.5 cm ATE, ~7 ms/frame engine.

```bash
cd ~/gsplat-rt
python3.10 scripts/eval_odometry.py --frontend superpoint --provider tensorrt \
    --sp-onnx models/sp_lg_tum.onnx --max-frames 200
# PASS looks like: ATE-RMSE ~3.3-3.5 cm, 200/200 PnP-ok  (2026-07-09: 3.3 cm)
```

The TensorRT EP needs libnvinfer on `LD_LIBRARY_PATH` or it fails with
`libnvinfer.so.10: cannot open` and drops to CUDA/CPU. This **IS** required
(an earlier note here claimed the system env didn't need it — that was a stale,
pre-reprovision box; the 2026-07-09 fresh box confirmed it is needed). Always:
```bash
export LD_LIBRARY_PATH=$(python3.10 -c 'import os,tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))'):$LD_LIBRARY_PATH
```

## 2. THE VERIFY: superpoint pose tracking inside the live pipeline

Drive a TUM RGB sequence (or a webcam/clip) through PipelineManager with the pose
provider active. On the box, depth comes from the real DepthAnything TRT engine,
so the superpoint front-end back-projects through metric-ish depth and produces a
moving pose — the whole point of the end-to-end wiring.

The pipeline's source goes through `cv2.VideoCapture`, which will NOT read TUM's
timestamp-named PNGs — pack them into an mp4 first:

```bash
cd ~/gsplat-rt
ffmpeg -framerate 30 -pattern_type glob \
    -i 'data/tum/rgbd_dataset_freiburg1_desk/rgb/*.png' \
    -c:v libx264 -pix_fmt yuv420p /tmp/tum_fr1_desk.mp4
```

```bash
# --realtime plays at frame rate; drop it to run as fast as the box allows and
# read the sustained FPS.
python3.10 scripts/run_live.py \
    --source /tmp/tum_fr1_desk.mp4 \
    --pose-tracking superpoint --pose-backend tensorrt \
    --pose-onnx models/sp_lg_tum.onnx \
    --duration 30 --ascii-map
```

Confirm in the output:
- banner prints `Pose tracking: superpoint (tensorrt)`
- the log line `Pose provider: SuperPoint+LightGlue (models/sp_lg_tum.onnx, providers=[...Tensorrt...])`
  — i.e. it did NOT silently coast at identity
- the ascii occupancy map resolves into a coherent scene (not a single frustum
  blob stacked at the origin — that blob is the identity-fusion failure mode)
- sustained `fps=` — the whole pipeline (depth + superpoint pose + TSDF) should
  stay near/above 30. If pose tracking pushes it under budget, note the number;
  the fused ONNX re-runs SuperPoint on both frames per pair (2× extractor) and is
  the first thing to optimise.

## 3. Report back

Paste the banner + a few status lines + the final `Done. frames=… fps…` summary.
What we need to record (and only then put in README/memory):
- did superpoint actually load the TensorRT EP (not coast at identity)?
- sustained end-to-end FPS with pose tracking on vs `--pose-tracking none`
- whether the map is coherent

Correct any doc number **down** to what the box shows — never assume.

## NEXT RUN: map-coherence verify (intrinsics + metric scale)

Status after 2026-07-09: the TensorRT pose path runs live end-to-end at ~30 fps
(verified), but the live TUM *map* was NOISE — `run_live` used a generic FOV
camera (forcing fx==fy) on TUM's 640x480→518x518 non-uniform resize, and depth is
monocular/relative. Two fixes now exist to try together:
- `--tum-intrinsics` — real freiburg1 intrinsics, rescaled to depth space (fx!=fy).
- `--metric-scale-monocular` — cross-frame scale consistency for relative depth.

Requires branch **feat/run-live-source-intrinsics** (the intrinsics flags aren't
on main yet). On a fresh box, checkout it BEFORE `brev_setup.sh`:
```bash
git clone https://github.com/matthewhamilton3141/gsplat-rt.git ~/gsplat-rt
cd ~/gsplat-rt && git checkout feat/run-live-source-intrinsics
bash scripts/brev_setup.sh    # main has the tensorrt<11 pin; now also installs ffmpeg
```
Then (after export_sp_lg.sh + the ffmpeg pack from section 2; LD_LIBRARY_PATH set):
```bash
python3.10 scripts/run_live.py --source /tmp/tum_fr1_desk.mp4 \
    --pose-tracking superpoint --pose-backend tensorrt --pose-onnx models/sp_lg_tum.onnx \
    --tum-intrinsics --metric-scale-monocular \
    --realtime --loop --duration 30
# banner must show: "Camera intrinsics: TUM fr1 ..." and "Monocular scale reference active"
```
PASS = `output/live_scene_occupancy.png` / `_splat_preview.png` show a coherent
desk scene (view in Jupyter file browser), not the origin-blob/noise from before.
If still noisy, the residual is monocular-depth scale drift — fall back to
`scripts/reconstruct_tum.py` (real depth + real intrinsics) for a clean visual.

Reminders that cost time tonight: use **`python3.10`** (brev's bare python3 lacks
deps); run inside **tmux** (SSH drops); **don't kill a run early** — first-frame
TRT-engine build/warmup can look frozen for 60-90s; drop **`--ascii-map`** in a
Jupyter terminal (ANSI clears don't render — looks hung).
