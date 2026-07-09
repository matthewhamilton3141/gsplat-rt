# Architecture

Detailed threaded architecture, concurrency design, USD schema, and repository
layout. See the [README](../README.md) for the high-level overview.

## Thread topology

```
┌─────────────────────────────────────────────────────────────┐
│  VideoCapture thread                                        │
│  OpenCV → resize 640×480 → bounded queue (drop-oldest)     │
└──────────────────────────┬──────────────────────────────────┘
                           │ queue.Queue
┌──────────────────────────▼──────────────────────────────────┐
│  Coordinator thread                                         │
│  ├─ TensorRT FP16 depth infer  (pre-alloc device buffers)  │
│  ├─ metric-scale align (opt.)   relative → metric depth     │
│  ├─ VO pose provider (opt.)     ORB / SuperPoint+LightGlue  │
│  ├─ Gaussian back-projection   (camera → world via pose)    │
│  ├─ push_depth ──────────────────────────────────────────┐  │
│  └─ periodic USD export  (atomic os.replace → .usdz)     │  │
└──────────────────────────────────────────────────────────┼──┘
                                                           │ queue.Queue
┌──────────────────────────────────────────────────────────▼──┐
│  TSDFWorker thread  (10 Hz)                                 │
│  CUDA TSDF integrate (numpy fallback) → marching cubes      │
└─────────────────────────────────────────────────────────────┘
```

**Lock-free hot path.** Inter-thread handoffs use `queue.Queue`. Gaussian positions
accumulate in a `collections.deque` (GIL-atomic appends). USD stage writes happen
exclusively on the Coordinator thread — no mutex required.

**Exception isolation.** Every thread target is wrapped in a try/except. Crashes set
a shared `_stop_event` and are re-raised in the caller's thread on `stop()`. Optional
stages (metric-scale, pose provider) that fail to build are caught and the pipeline
coasts (identity pose / relative depth) rather than crashing.

## USD stage schema

A periodic `.usdz` is written atomically (`os.replace`). Loaded in Isaac Sim:

```python
import omni.usd
omni.usd.get_context().open_stage("output/live_scene.usdz")
```

- `/World/GaussianSplats` — `ParticleField` prim, rendered by Omniverse NuRec.
- `/World/CollisionMesh` — invisible `UsdGeom.Mesh` with PhysX `convexDecomposition`
  (`UsdPhysics.CollisionAPI`); collision runs at 120 Hz via decomposition baked at load.

## Repository structure

```
gsplat-rt/
├── src/
│   ├── ingestion/
│   │   └── video_stream.py       # threaded OpenCV capture
│   ├── depth/
│   │   ├── export_onnx.py        # HuggingFace → ONNX export (+ --fp16 uniform graph)
│   │   ├── compile_trt.py        # ONNX → TensorRT engine (FP16 strongly-typed / TF32)
│   │   ├── depth_estimator.py    # dtype-aware TRT inference, pre-alloc buffers
│   │   └── metric_scale.py       # relative→metric depth aligner (DPT scale+shift)
│   ├── slam/
│   │   ├── tum_dataset.py        # TUM RGB-D loader (rgb+depth+ground-truth poses)
│   │   ├── rgbd_odometry.py      # pluggable-frontend VO + PnP + keyframing + ATE eval
│   │   ├── pose_graph.py         # SE(3) pose-graph optimiser (loop-closure back-end)
│   │   ├── superpoint_lightglue.py  # SuperPoint+LightGlue ONNX front-end (onnxruntime)
│   │   └── monocular_scale.py    # two-view triangulation + cross-frame scale propagation
│   ├── mapping/
│   │   ├── collision_proxy.py    # TSDF volume + async mesh extractor + occupancy grid
│   │   ├── tsdf_cuda.py          # CUDA TSDF kernel wrapper + numpy oracle/fallback
│   │   ├── usd_bridge.py         # OpenUSD stage writer
│   │   └── visualization.py      # occupancy map + points/splat preview PNGs (numpy + cv2)
│   ├── gaussian/                 # differentiable 3DGS optimizer
│   │   ├── rasterizer.py         # EWA-splatting forward + analytic-gradient backward
│   │   ├── ssim.py               # SSIM + analytic D-SSIM gradient
│   │   ├── densify.py            # Adaptive Density Control (clone/split/prune)
│   │   └── optimizer.py          # Adam fit loop / finalize stage
│   ├── viz/
│   │   ├── scene_source.py       # pipeline/.ply/synthetic → SceneSnapshot (+ PLY reader)
│   │   ├── web_viewer.py         # stdlib ThreadingHTTPServer + JSON scene feed
│   │   ├── viser_viewer.py       # viser 3-D viewer (upright, orbit-able) — optional dep
│   │   └── static/               # Three.js SPA (index.html + viewer.js)
│   └── pipeline_manager.py       # central orchestrator (depth, scale, pose, TSDF, USD)
├── kernels/
│   └── tsdf_integrate.cu         # custom CUDA TSDF integrate kernel (175× over numpy)
├── scripts/
│   ├── run_live.py               # run + watch live (dashboard + ASCII map)
│   ├── run_viewer.py             # live 3-D browser viewer (splats + occupancy)
│   ├── view_scene.py             # viser viewer entry (--ply / --demo)
│   ├── lingbot_trt/              # LingBot-Map → TensorRT study (export + build/bench + RESULTS.md)
│   ├── bench_pipeline.py         # per-stage latency + FPS benchmark
│   ├── bench_depth.py            # TF32 vs FP16 depth latency + fidelity
│   ├── bench_tsdf.py             # numpy vs CUDA TSDF integrate speed-up
│   ├── eval_odometry.py          # visual-odometry ATE + trajectory render
│   ├── eval_metric_scale.py      # metric-scale AbsRel/RMSE (synthetic + --tum)
│   ├── export_sp_lg.sh           # regenerate the SuperPoint+LightGlue ONNX
│   ├── reconstruct_tum.py        # identity-vs-ground-truth-pose fusion proof
│   ├── fetch_tum.sh              # idempotent TUM sequence download
│   └── brev_setup.sh             # one-shot GPU box bootstrap (Brev/A10G)
├── models/                       # .onnx and .engine files (not committed)
├── tests/                        # 177 tests (pytest); GPU/dataset rows skip cleanly
├── configs/
├── requirements.txt
└── setup.py                      # torch.utils.cpp_extension for CUDA kernels
```
