# gsplat-rt

Real-time conversion of live video into 3D Gaussian Splats with a physics-ready collision mesh, exported as an OpenUSD stage for NVIDIA Isaac Sim and Omniverse.

> **Status: work in progress.** The pipeline runs end-to-end today (ingestion → depth → TSDF → USD + 2-D previews) and the CPU-side stages are benchmarked. The TensorRT depth path is validated on Ampere GPUs but the GPU latency figures below are **targets**, not yet a published measured run; a Python-only mock depth estimator keeps everything runnable on any machine. The Gaussian optimizer and SLAM pose tracking (see [Roadmap](#roadmap)) are not built yet — accumulated splats are currently raw back-projected points at identity pose.

## What it does

A single video stream (webcam or file) enters the pipeline. Four concurrent stages transform it into a live scene description that a reinforcement learning robot can see and physically interact with:

1. **Video ingestion** — frames captured into a bounded queue at 1,000+ FPS throughput, decoupled from all downstream processing.
2. **Depth estimation** — each frame is run through a TensorRT FP16 engine built from Depth Anything V2 Small, targeting under 15 ms per frame on Ampere+ GPUs.
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

### Step 2 — compile the TensorRT FP16 engine

Profiles and builds the engine for your specific GPU. Run once; takes 2–5 minutes.

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
│   │   ├── export_onnx.py        # HuggingFace → ONNX export
│   │   ├── compile_trt.py        # ONNX → TensorRT FP16 engine
│   │   └── depth_estimator.py    # TRT inference, pre-alloc buffers
│   ├── mapping/
│   │   ├── collision_proxy.py    # TSDF volume + async mesh extractor + occupancy grid
│   │   ├── usd_bridge.py         # OpenUSD stage writer
│   │   └── visualization.py      # occupancy map + splat preview PNGs (numpy + cv2)
│   └── pipeline_manager.py       # central orchestrator
├── scripts/
│   ├── run_live.py               # run + watch live (dashboard + ASCII map)
│   ├── bench_pipeline.py         # per-stage latency + FPS benchmark
│   └── brev_setup.sh             # one-shot GPU box bootstrap (Brev/A10G)
├── kernels/                      # custom CUDA kernels (.cu)
├── models/                       # .onnx and .engine files (not committed)
├── tests/
│   ├── test_video_stream.py      # 1,000-frame FPS benchmark
│   ├── test_depth_inference.py   # TRT latency benchmark (GPU required)
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
| `test_depth_inference_latency` | Yes | Mean < 15 ms, P99 < 20 ms |
| `test_depth_buffer_reuse` | Yes | Zero GPU memory growth |
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

The GPU-required rows above are the pass criteria the tests assert against on Ampere hardware; the specific latency numbers are the target thresholds, not a published measured run.

## Performance targets

| Stage | Budget | Notes |
|---|---|---|
| Video ingestion | — | Throughput-bound; queue absorbs bursts |
| Depth inference | < 15 ms | TRT FP16, Ampere Tensor Cores |
| TSDF integration | < 5 ms/frame | 64³ grid, numpy vectorised |
| Mesh extraction | < 10 ms | scikit-image marching cubes |
| Full pipeline | < 33 ms | 30 FPS end-to-end budget |
| PhysX collision | 120 Hz | convexDecomposition baked at load time |

## Roadmap

- **M5 — Gaussian optimizer** — differentiable 3DGS optimization on the accumulated point cloud (`src/gaussian/` is stubbed)
- **M6 — SLAM pose tracking** — wire in SuperPoint + SuperGlue or ORB-SLAM3 so the TSDF integrates multi-view depth with correct camera poses
- **M7 — Isaac Sim live reload** — hot-swap the `.usdz` stage in Omniverse as new geometry arrives, without restarting the simulation

## License

MIT
