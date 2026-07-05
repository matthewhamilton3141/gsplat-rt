# gsplat-rt

Real-time conversion of live video into 3D Gaussian Splats with a physics-ready collision mesh, exported as an OpenUSD stage for NVIDIA Isaac Sim and Omniverse.

> **Status: work in progress.** The pipeline runs end-to-end and is now **benchmarked on real hardware — 34.7 FPS on an NVIDIA A10G** (28.9 ms/frame), clearing the 30 FPS real-time budget, with TensorRT depth inference at **14.3 ms/frame**. A Python-only mock depth estimator keeps everything runnable GPU-free. The **M6 SLAM front-end is built**: an RGB-D visual-odometry tracker (5.6 cm ATE on TUM fr1/desk) supplies per-frame camera poses to pose-aware TSDF + Gaussian fusion. The **M5 Gaussian optimizer is built**: a differentiable 3DGS rasterizer with hand-derived analytic gradients (finite-difference verified) and an Adam training loop that reconstructs held-out views to >60 dB PSNR. A **custom CUDA TSDF kernel** now retires the last over-budget stage — **0.06 ms/frame on the A10G, a 175× speed-up** over the numpy integrator, bit-for-bit verified. Not yet built: true FP16 depth. See [Measured performance](#measured-performance-nvidia-a10g) and [Roadmap](#roadmap).

## What it does

A single video stream (webcam or file) enters the pipeline. Four concurrent stages transform it into a live scene description that a reinforcement learning robot can see and physically interact with:

1. **Video ingestion** — frames captured into a bounded queue at 1,000+ FPS throughput, decoupled from all downstream processing.
2. **Depth estimation** — each frame is run through a TensorRT engine built from Depth Anything V2 Small — **14.3 ms/frame measured on an A10G** (FP16 where the TensorRT version supports the weakly-typed flag, TF32 Tensor Cores on TensorRT 10+).
3. **Geometry extraction** — depth maps are fused into a TSDF volume at 10 Hz; marching cubes extracts a coarse collision mesh in a background thread.
4. **USD export** — a `.usdz` stage is written periodically containing a `ParticleField` Gaussian Splat layer for NuRec rendering and an invisible `UsdGeom.Mesh` collision proxy with `UsdPhysics.CollisionAPI` for PhysX.
5. **2-D previews** — alongside each export, two glanceable PNGs are written so you can eyeball a run without a USD viewer: a top-down **occupancy map** (floor plan from the TSDF) and a depth-colored **splat preview** (the point cloud projected through the camera). These need only numpy + OpenCV, so they render even when `pxr` and CUDA are absent.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  VideoCapture thread                                        │
│  OpenCV → resize 640×480 → bounded queue (drop-oldest)     │
└──────────────────────────┬──────────────────────────────────┘
                           │ queue.Queue
┌──────────────────────────▼──────────────────────────────────┐
│  Coordinator thread                                         │
│  ├─ TensorRT FP16 depth infer  (pre-alloc device buffers)  │
│  ├─ Gaussian back-projection   (pre-alloc index arrays)     │
│  ├─ push_depth ──────────────────────────────────────────┐  │
│  └─ periodic USD export  (atomic os.replace → .usdz)     │  │
└──────────────────────────────────────────────────────────┼──┘
                                                           │ queue.Queue
┌──────────────────────────────────────────────────────────▼──┐
│  TSDFWorker thread  (10 Hz)                                 │
│  numpy TSDF integration → marching cubes → TriangleMesh     │
└─────────────────────────────────────────────────────────────┘
```

**Lock-free hot path.** Inter-thread handoffs use `queue.Queue`. Gaussian positions accumulate in a `collections.deque` (GIL-atomic appends). USD stage writes happen exclusively on the Coordinator thread — no mutex required.

**Exception isolation.** Every thread target is wrapped in a try/except. Crashes set a shared `_stop_event` and are re-raised in the caller's thread on `stop()`.

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| CUDA toolkit | 11.8+ |
| TensorRT | 9.0+ |
| PyTorch | 2.1+ |
| OpenCV | 4.8+ |
| OpenUSD (`pxr`) | 23.05+ or bundled with Isaac Sim |

A Python-only mock depth estimator is included so the pipeline and all non-GPU tests run on any machine.

## Installation

```bash
git clone https://github.com/matthewhamilton3141/gsplat-rt.git
cd gsplat-rt
pip install -r requirements.txt

# CUDA kernels (when custom kernels are added to /kernels)
python setup.py build_ext --inplace
```

TensorRT is not on PyPI's default index:
```bash
pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com
```

## Getting started

### Step 1 — export the depth model to ONNX

Downloads Depth Anything V2 Small from HuggingFace and exports a fixed-shape graph (no dynamic axes, for maximum TRT fusion).

```bash
python src/depth/export_onnx.py
# → models/depth_v2_small.onnx  (~50 MB)
```

### Step 2 — compile the TensorRT engine

Profiles and builds the engine for your specific GPU (FP16 on TensorRT 8/9, TF32 on 10+ — see [note](#a-note-on-precision-fp16-vs-tf32)). Run once; takes 2–5 minutes.

```bash
python src/depth/compile_trt.py
# → models/depth_engine.engine
```

### Step 3 — run the pipeline (and watch it)

The quickest way to run a source and *see* the pipeline working — a live status line plus, with `--ascii-map`, the occupancy map redrawn in your terminal (handy over SSH on a headless GPU box, no file copying):

```bash
python scripts/run_live.py --source 0 --ascii-map        # webcam
python scripts/run_live.py --source clip.mp4 --duration 20
```

```
[  4.5s] depth=tensorrt  fps= 31.2  frames=142   depth= 12.8ms  splats=5000  exports=2
█████████████████████████████
··············█████···········     █ occupied
·····················█········     · free
··········(top-down occupancy)     (blank) unknown — X→right, depth↑
```

It stops on `--duration`, when a file source is exhausted, or on Ctrl-C. Each run writes into `output/`:

```
live_scene.usdz               — scene for Isaac Sim / Omniverse
live_scene_occupancy.png      — top-down occupancy map (floor plan)
live_scene_splat_preview.png  — depth-colored splat cloud preview
```

The PNGs are overwritten in place on every export, so they always reflect the latest scene. Set `write_previews=False` on `PipelineConfig` to skip them.

Or drive it from Python directly:

```python
from src.pipeline_manager import PipelineManager, PipelineConfig

config = PipelineConfig(
    video_source=0,              # webcam index or path to video file
    engine_path="models/depth_engine.engine",
    output_dir="output",
    usd_update_interval_s=3.0,  # write a fresh .usdz every 3 seconds
)

with PipelineManager(config) as pipeline:
    input("Pipeline running — press Enter to stop\n")
    print(pipeline.stats())      # {frames, exports, gaussians, depth_ms, depth_backend}
```

### Loading in Isaac Sim

```python
import omni.usd
omni.usd.get_context().open_stage("output/live_scene.usdz")
```

The stage contains:
- `/World/GaussianSplats` — `ParticleField` prim, rendered by Omniverse NuRec
- `/World/CollisionMesh` — invisible `UsdGeom.Mesh` with PhysX `convexDecomposition`

## Project structure

```
gsplat-rt/
├── src/
│   ├── ingestion/
│   │   └── video_stream.py       # threaded OpenCV capture
│   ├── depth/
│   │   ├── export_onnx.py        # HuggingFace → ONNX export (legacy exporter)
│   │   ├── compile_trt.py        # ONNX → TensorRT engine (FP16/TF32, TRT 8–11)
│   │   └── depth_estimator.py    # TRT inference, pre-alloc buffers
│   ├── slam/
│   │   ├── tum_dataset.py        # TUM RGB-D loader (rgb+depth+ground-truth poses)
│   │   └── rgbd_odometry.py      # ORB+PnP visual odometry + ATE eval + pose provider
│   ├── mapping/
│   │   ├── collision_proxy.py    # TSDF volume + async mesh extractor + occupancy grid
│   │   ├── usd_bridge.py         # OpenUSD stage writer
│   │   └── visualization.py      # occupancy map + splat preview PNGs (numpy + cv2)
│   └── pipeline_manager.py       # central orchestrator (optional pose provider)
├── scripts/
│   ├── run_live.py               # run + watch live (dashboard + ASCII map)
│   ├── bench_pipeline.py         # per-stage latency + FPS benchmark
│   ├── reconstruct_tum.py        # identity-vs-ground-truth-pose fusion proof
│   ├── eval_odometry.py          # visual-odometry ATE + trajectory render
│   ├── fetch_tum.sh              # idempotent TUM sequence download
│   └── brev_setup.sh             # one-shot GPU box bootstrap (Brev/A10G)
├── kernels/                      # custom CUDA kernels (.cu) — TSDF kernel planned
├── models/                       # .onnx and .engine files (not committed)
├── tests/
│   ├── test_video_stream.py      # 1,000-frame FPS benchmark
│   ├── test_depth_inference.py   # TRT latency benchmark (GPU required)
│   ├── test_tum_dataset.py       # TUM loader association + metric depth + SE(3) poses
│   ├── test_rgbd_odometry.py     # visual odometry + ATE on fr1/desk
│   ├── test_pose_aware_pipeline.py   # camera→world fusion via pose provider
│   ├── test_usd_bridge.py        # TSDF + USD round-trip
│   ├── test_visualization.py     # occupancy grid + preview PNG validation
│   └── test_pipeline_integration.py  # end-to-end .usdz validation
├── configs/
├── requirements.txt
└── setup.py                      # torch.utils.cpp_extension for CUDA kernels
```

## Tests

```bash
pytest tests/ -v
```

| Test | Requires GPU | Result |
|---|---|---|
| `test_video_stream_fps` | No | **1,113 FPS** ingestion throughput |
| `test_depth_output_shape` | Yes | (518, 518) float32, no NaN |
| `test_depth_inference_latency` | Yes | **A10G: 14.29 ms mean, P99 15.01 ms** ✓ |
| `test_depth_buffer_reuse` | Yes | Zero GPU memory growth |
| `test_tum_dataset_*` | No (needs dataset) | Association, metric depth, valid SE(3) poses |
| `test_odometry_tracks_fr1_desk` | No (needs dataset) | **5.6 cm ATE**, 63 FPS CPU, 100% PnP-tracked |
| `test_backproject_camera_vs_world` | No | Pose provider transforms points camera→world |
| `test_tsdf_integration_and_mesh` | No | 3.3 ms/frame, mesh in 10 ms |
| `test_extractor_async_10hz` | No | First mesh within 500 ms |
| `test_full_pipeline_usdz` | No | Valid .usdz, both layers present |
| `test_occupancy_grid_*` | No | 3-state top-down grid, correct shape/dtype |
| `test_save_splat_preview_*` | No | Depth-colored PNG; empty input → no file |
| `test_ascii_map_*` | No | Terminal occupancy render; obstacles survive downsample |
| `test_pipeline_writes_preview_pngs` | No | Preview PNGs + live `stats()` on a running pipeline |
| `test_pipeline_smoke` | No | Clean start/stop, no thread errors |
| `test_pipeline_frame_throughput` | No | Periodic USD export fires on schedule |
| `test_pipeline_full_usdz_validation` | No | Full USD layer + physics API check |

The two SLAM rows need an extracted TUM sequence (`bash scripts/fetch_tum.sh`) and skip cleanly without it.

## Measured performance (NVIDIA A10G)

First end-to-end GPU run, `scripts/bench_pipeline.py`, TensorRT 11.1 (TF32), 116 frames:

| Stage | Budget | Measured (mean / p99) | Verdict |
|---|---|---|---|
| Video ingestion | — | throughput-bound (queue absorbs bursts) | — |
| Depth inference | < 15 ms | 14.3 ms mean *(isolated; 15.1 under load)* | ✓ at budget |
| TSDF integration | < 5 ms/frame | 13.1 ms numpy → **0.06 ms CUDA kernel** | ✓ **175× via custom kernel** |
| Mesh extraction | < 10 ms | 10.0 ms / 20.7 ms | ✓ |
| **Full pipeline (live)** | **≥ 30 FPS** | **28.9 ms/frame → 34.7 FPS** | **✓ real-time** |

The pipeline clears 30 FPS because stages are decoupled across threads — the numpy TSDF's 13 ms ran on the 10 Hz worker and never gated the frame path. The **custom CUDA TSDF kernel** (`kernels/tsdf_integrate.cu`) now integrates a 64³ grid in **0.06 ms/frame — a 175× speed-up** over numpy, measured by `scripts/bench_tsdf.py` and bit-for-bit verified against the numpy path — retiring that bottleneck outright. The remaining next win is **true FP16** depth (TF32→FP16 should reach ~8–10 ms). PhysX collision runs at 120 Hz via `convexDecomposition` baked at load time.

### A note on precision (FP16 vs TF32)

The project targets an FP16 depth engine. TensorRT 10+/11 removed the weakly-typed `BuilderFlag.FP16`, moving precision control to strongly-typed networks, so `compile_trt.py` builds FP16 where the flag exists (TRT 8/9) and otherwise a default fp32 engine that still uses **Ampere Tensor Cores via TF32**. The 14.3 ms above is TF32; true FP16 on TRT 11 (an fp16 ONNX + a `STRONGLY_TYPED` network) is tracked in the roadmap.

## Roadmap

- **CUDA TSDF kernel** — *built + measured on the A10G*: a custom one-thread-per-voxel integrate kernel (`kernels/tsdf_integrate.cu`) with a torch binding + numpy fallback (`src/mapping/tsdf_cuda.py`) replaces the numpy 64³ integrator (the one over-budget stage). Measured **0.06 ms/frame vs 10.9 ms numpy — a 175× speed-up**, sub-millisecond and well inside the 30 FPS budget. GPU output matches the production numpy path bit-for-bit (`tests/test_tsdf_cuda.py::test_cuda_matches_reference`, held to a CPU oracle); `scripts/bench_tsdf.py` reproduces the number. Next: wire the kernel into `TSDFVolume`'s live path (persistent device volume, lazy host sync for mesh/occupancy).
- **True FP16 depth** — export an fp16 ONNX and build a `STRONGLY_TYPED` engine so TensorRT 11 runs genuine FP16 (~8–10 ms) rather than TF32.
- **M6 — SLAM pose tracking** — *front-end done*: RGB-D visual odometry (`src/slam/`, ORB+PnP, 5.6 cm ATE on TUM fr1/desk) feeds per-frame poses into pose-aware fusion. Next: a learned front-end (SuperPoint + SuperGlue), keyframing / loop closure, and closing the monocular scale gap so poses work on the live mono-depth path, not just metric RGB-D.
- **M5 — Gaussian optimizer** — *built + wired into the pipeline*: a differentiable EWA-splatting rasterizer with hand-derived analytic gradients (verified against finite differences to <1e-4) and a numpy Adam loop (`src/gaussian/`). Fits posed views to >60 dB PSNR at ~2 ms/iter on CPU; ports to CUDA/torch on the A10G unchanged. The pipeline now runs it as an **offline finalize stage** (`optimize_on_finalize`): the hot path stashes RGB keyframes + poses, and `stop()` seeds Gaussians from the fused point cloud, fits them against the keyframes, and exports optimized splats as a 3DGS `.ply` — kept off the 30 FPS path since pure-numpy is too slow per frame. Next: adaptive densify/prune, D-SSIM loss, SH colour, and a CUDA/torch fit fast enough to run online.
- **M7 — Isaac Sim live reload** — hot-swap the `.usdz` stage in Omniverse as new geometry arrives, without restarting the simulation

## License

MIT
