# Benchmarks & tests

All figures measured on an **NVIDIA A10G**; each has a reproducing script. See the
[precision note](precision.md) for the FP16-vs-TF32 detail.

## Measured performance (NVIDIA A10G)

`scripts/bench_pipeline.py`, TensorRT 11.1, **FP16 depth engine + CUDA TSDF kernel**,
500 frames:

| Stage | Budget | Measured (mean / p99) | Verdict |
|---|---|---|---|
| Video ingestion | — | throughput-bound (queue absorbs bursts) | — |
| Depth inference | < 15 ms | **8.7 ms** / 18.2 ms *(TF32 14.2 → FP16 6.3 ms isolated, 2.24×)* | ✓ **true FP16** |
| TSDF integration | < 5 ms/frame | 13.1 ms numpy → **0.30 ms CUDA kernel** | ✓ **175× via custom kernel** |
| Mesh extraction | < 10 ms | 7.7 ms / 12.3 ms | ✓ |
| **Full pipeline (live)** | **≥ 30 FPS** | **12.1 ms/frame → 82.7 FPS** | **✓ 2.75× real-time** |

Two custom-optimisation wins compound here. The **custom CUDA TSDF kernel**
(`kernels/tsdf_integrate.cu`) integrates a 64³ grid in **0.06 ms/frame — a 175×
speed-up** over numpy (`scripts/bench_tsdf.py`, bit-for-bit verified against the numpy
path); wired into `TSDFVolume`'s live path (GPU-resident volume, lazy host sync) it
measures **0.30 ms/frame in-pipeline**. The **true-FP16 depth engine** halves the depth
stage. Together they lift end-to-end throughput from the 34.7 FPS TF32/numpy baseline
to **82.7 FPS**. Stages are decoupled across threads, and PhysX collision runs at 120 Hz
via `convexDecomposition` baked at load time.

## Reproducing

```bash
python scripts/bench_pipeline.py                       # per-stage latency + FPS
python scripts/bench_depth.py                          # TF32 vs FP16 depth
python scripts/bench_tsdf.py                           # numpy vs CUDA TSDF
python scripts/eval_odometry.py --frontend superpoint --provider tensorrt   # ATE
python scripts/eval_metric_scale.py --tum              # metric-scale AbsRel/RMSE
```

## Test suite

```bash
pytest tests/ -v
```

177 tests; the bulk run GPU-free (mock depth) on any machine. Representative rows:

| Test | Requires | Result |
|---|---|---|
| `test_video_stream_fps` | — | **1,113 FPS** ingestion throughput |
| `test_depth_inference_latency` | GPU | **A10G: 6.3 ms FP16** (2.24× over TF32, corr 0.99996) |
| `test_depth_buffer_reuse` | GPU | Zero GPU memory growth |
| `test_cuda_matches_reference` | GPU | CUDA TSDF == numpy oracle, bit-for-bit |
| `test_odometry_tracks_fr1_desk` | dataset | **5.7 cm ATE**, 100% PnP-tracked |
| `test_pairwise_branch_recovers_known_translation` | — | learned-matcher branch recovers a known motion |
| `test_keyframe_odometry` / `test_pose_graph` | — | keyframing + SE(3) pose-graph loop closure |
| `test_pose_provider_pipeline` | — | config auto-wires ORB/learned VO into the live pipeline |
| `test_metric_scale_*` | — | scale+shift fit, propagation, pipeline hook (51 tests) |
| `test_gaussian_backward` / `test_ssim` / `test_gaussian_densify` | — | analytic gradients FD-verified; D-SSIM; clone/split/prune |
| `test_tsdf_integration_and_mesh` | — | numpy TSDF + mesh extraction |
| `test_full_pipeline_usdz` | — | valid .usdz, both layers present |
| `test_pipeline_smoke` | — | clean start/stop, no thread errors |

Dataset rows need an extracted TUM sequence (`bash scripts/fetch_tum.sh`) and skip
cleanly without it; GPU rows skip without CUDA/TensorRT.
