# gsplat-rt

**Real-time monocular video → 3D Gaussian Splats + a physics-ready collision mesh for NVIDIA Isaac Sim.**

gsplat-rt turns a live RGB stream into a 3D scene a robot can both *see* and *touch*: a Gaussian-Splat representation for rendering, an occupancy/collision mesh for physics, and an OpenUSD stage ready to drop into Isaac Sim / Omniverse. It runs the full depth → pose → fusion → export loop in real time on a single GPU, and every performance number below is measured on an NVIDIA A10G — not estimated. A pure-Python mock depth path keeps the whole pipeline (and all non-GPU tests) runnable with no GPU at all.

![Live monocular reconstruction of a TUM desk sequence as 3D Gaussian splats](docs/reconstruction_desk.png)

*Real pipeline output — a monocular RGB stream (TUM fr1/desk) reconstructed into 3D Gaussian splats, with real source intrinsics + cross-frame metric scale.*

![Orbiting turntable of the TUM desk reconstruction](docs/reconstruction_turntable.gif)

*Orbiting turntable of the same reconstruction (`scripts/render_turntable.py` — a GPU-free numpy point-splat renderer). [Full-quality MP4 →](docs/reconstruction_turntable.mp4)*

## Headline results *(measured, NVIDIA A10G)*

| Metric | Result |
|---|---|
| End-to-end pipeline | **82.7 FPS** (12.1 ms/frame) — 2.75× the 30 FPS budget |
| TensorRT FP16 depth engine | **6.3 ms/frame** — 2.24× over TF32, output correlation 0.99996 |
| Custom CUDA TSDF fusion | **0.06 ms/frame** — 175× over numpy, bit-for-bit verified |
| Visual odometry (SuperPoint + LightGlue) | **3.5 cm ATE** on TUM fr1/desk — 7.4 ms/frame via TensorRT |
| Monocular metric depth | AbsRel **2.66 → 0.049** (relative → metric, δ<1.25 = 0.97) |
| Test suite | **177 tests**, GPU/dataset rows skip cleanly off-box |

## What it does

A single video stream (webcam or file) enters the pipeline; decoupled stages turn it into a live, physics-ready scene:

1. **Ingest** — frames captured into a bounded queue (1,000+ FPS throughput), decoupled from all downstream work.
2. **Depth** — each frame runs a TensorRT FP16 engine (Depth Anything V2) — genuine strongly-typed FP16 at 6.3 ms/frame.
3. **Track** *(optional)* — a visual-odometry front-end supplies a per-frame camera pose so geometry fuses in a coherent world frame.
4. **Fuse** — depth is integrated into a TSDF volume by a custom CUDA kernel; marching cubes extracts a collision mesh in the background.
5. **Export** — a periodic `.usdz` stage carries a Gaussian-Splat layer for rendering + an invisible collision mesh with `UsdPhysics.CollisionAPI` for PhysX.
6. **Visualize** — glanceable 2-D previews (occupancy floor plan + splat render) and a live browser 3-D viewer, all GPU-free.

## Architecture

```
 RGB stream
     │  bounded queue (drop-oldest)
     ▼
┌─────────────────────────────────────────────┐
│ Coordinator                                 │
│  TensorRT FP16 depth  →  metric-scale (opt) │
│         │                     │             │
│  VO pose (opt) ──────────────►│             │
│         ▼                     ▼             │
│  CUDA TSDF fusion  →  marching-cubes mesh   │
│         └──────► Gaussian splats ───────────┤
└─────────────────────────┬───────────────────┘
                          ▼
         OpenUSD (.usdz) + previews + web viewer
```

Stages run on separate threads and communicate through bounded queues, so depth inference, fusion, and export never block each other. The volume stays resident on the GPU — only depth crosses PCIe per frame. Full thread topology, concurrency design, and the USD schema are in [`docs/architecture.md`](docs/architecture.md).

## Quickstart

```bash
# 1. install  (Python 3.10+, CUDA 11.8+, TensorRT 10+)
git clone https://github.com/matthewhamilton3141/gsplat-rt.git && cd gsplat-rt
pip install -r requirements.txt
pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com   # not on default PyPI
python setup.py build_ext --inplace                                  # builds the CUDA TSDF kernel

# 2. build the depth engine (ONNX → TensorRT FP16)
python src/depth/export_onnx.py --fp16
python src/depth/compile_trt.py --fp16

# 3. run the live pipeline (webcam) with the dashboard
python scripts/run_live.py --source 0 --ascii-map

# 4. watch it form in 3D (browser, no Omniverse needed)
python scripts/run_viewer.py --source 0        # then open http://localhost:8000
```

No GPU? Everything above runs against the mock depth estimator — the pipeline, previews, and viewer all work on a laptop. Each run writes a `.usdz` scene plus occupancy / splat preview PNGs into `output/`.

**Camera tracking (SLAM front-end).** Score the visual-odometry front-ends or enable one live:

```bash
python scripts/eval_odometry.py --frontend superpoint --provider tensorrt   # 3.5 cm ATE
# in the pipeline: PipelineConfig(pose_tracking="superpoint", pose_backend="tensorrt")
```

**Interactive 3-D.** `scripts/run_viewer.py` (stdlib Three.js — press `A` to toggle points ↔ anisotropic splats) or `scripts/view_scene.py --ply … | --demo` (a [viser](https://github.com/nerfstudio-project/viser) viewer with auto-upright framing).

**Isaac Sim.** Load the exported `output/live_scene.usdz` as a payload; the Gaussian layer renders and the collision mesh drives PhysX.

## Key capabilities

- **TensorRT FP16 depth** — a strongly-typed FP16 engine (uniform-fp16 ONNX, retargeted internal casts) at 6.3 ms/frame, 2.24× over TF32 with 0.99996 output correlation. dtype-aware runner handles either engine. → [`docs/precision.md`](docs/precision.md)
- **Custom CUDA TSDF fusion** — a one-thread-per-voxel integrate kernel, 0.06 ms/frame (175× over numpy), matched bit-for-bit against a CPU oracle. Lives on the GPU; numpy fallback off-box.
- **Monocular metric scale** — DPT-protocol scale+shift with robust IRLS + two-view triangulation anchor + cross-frame propagation, cutting AbsRel 2.66 → 0.049 on TUM and holding scale drift at ~0% over a run. One flag turns a pure monocular stream metric. → [`docs/metric_scale.md`](docs/metric_scale.md)
- **Learned SLAM front-end + back-end** — SuperPoint + LightGlue visual odometry (3.5 cm ATE, 18× faster via TensorRT than onnxruntime) behind a pluggable `Frontend`, plus keyframing and a dependency-free SE(3) pose-graph optimizer that closes loops and distributes drift. → [`docs/slam.md`](docs/slam.md)
- **Gaussian optimizer** — a differentiable 3DGS rasterizer with hand-derived analytic gradients (finite-difference verified), the full `(1−λ)L1 + λ D-SSIM` loss, and Adaptive Density Control; runs as an offline finalize stage exporting optimized `.ply` splats. → [`docs/gaussian_optimizer.md`](docs/gaussian_optimizer.md)
- **Visualization + export** — OpenUSD `.usdz` (splats + collision proxy), 2-D occupancy/splat previews, and browser viewers (stdlib Three.js viewer + an optional viser viewer with auto-upright framing).

**Measured SLAM accuracy** — estimated trajectory (orange) vs TUM fr1/desk ground truth (green):

| ORB + PnP | SuperPoint + LightGlue |
|:---:|:---:|
| ![ORB + PnP visual-odometry trajectory](docs/odometry_orb.png) | ![SuperPoint + LightGlue visual-odometry trajectory](docs/odometry_superpoint_lightglue.png) |
| 5.7 cm ATE | **3.6 cm ATE** (3.5 cm via TensorRT) |

## Testing & reliability

177 tests (pytest); GPU- and dataset-dependent rows skip cleanly off-box. The pipeline degrades gracefully by design — no CUDA/TensorRT falls back to numpy + a mock depth estimator, so the full system runs and is verifiable on a CPU-only laptop. Numerics are held to references (CUDA TSDF bit-for-bit vs a CPU oracle; analytic gradients vs finite differences). Full performance table + test matrix: [`docs/benchmarks.md`](docs/benchmarks.md).

## Roadmap

- **Loop detection** — descriptor retrieval + geometric verification to feed the pose-graph optimizer real loop constraints (front-end + back-end are done).
- **Isaac Sim live reload** — hot-swap the `.usdz` stage as new geometry arrives, without restarting the sim.
- **Gaussian optimizer** — SH view-dependent colour and an online CUDA fit.

## Related experiments

### Reconstructed-scene navigation (RL flagship)

![A PPO policy navigating random obstacle fields via lidar](docs/nav_ppo_policy.gif)

*A PPO policy (trained from scratch, 1.5M steps) driving to the goal through randomized obstacle fields using a 16-beam lidar observation — [full MP4 →](docs/nav_ppo_policy.mp4).*

The endpoint of the pipeline is a scene a robot can *act* in, so this milestone closes the loop: a differential-drive agent that learns to navigate. Built backend-agnostic — a pure-NumPy env + task core (`src/isaac/nav_sim.py`, fully unit-tested on a laptop, 22 tests) with a thin `gymnasium` shell, so a PyBullet / Isaac Lab port is an adapter, not a rewrite.

- **Hand-written baseline:** a lidar follow-the-gap controller — 35/40 random scenes solved, **0 collisions** (vs a blind go-to-goal's 12/40, 28 collisions).
- **Learned policy (PPO, stable-baselines3):** **98%** reached on held-out random scenes (matches the heuristic) while reaching **~40% faster** — the classic efficiency-vs-safety trade a learned policy discovers (it trades the heuristic's perfect safety for a small 2% collision rate). Measured, A10G.

### LingBot-Map → TensorRT (optimization study)

A separate, measurement-driven study taking a **VGGT-style streaming reconstruction foundation model** to TensorRT — every figure reproduced on an A10G, nothing assumed. The honest arc:

- **Profiled first.** A CUDA-event runtime split found the KV-cache `global_blocks` (45%) + DPT/camera heads (17.5%) dominate — *not* the frame blocks. This is why the initial Stage-4 experiment (all 24 frame blocks → TRT) only moved the whole model ~1.08×: the per-block 1.76× fusion win doesn't translate when you optimize the wrong thing.
- **Then optimized the real bottlenecks, per component → measured whole-model:** the stateful `global_blocks` (complex-RoPE + growing KV cache) at **1.53× per block → 1.069× whole-model**, and the static **DPT head at 2.93× per head → 1.098× whole-model** (parity-verified, 0 NaN). Notably the *smaller* head chunk won *more* end-to-end — because static work translates cleanly where the dynamic-cache blocks pay integration overhead.
- **Stacked both levers → 1.187× whole-model** (7.69 → 9.13 fps, parity 3.23%, 0 NaN). Because the two targets are *disjoint* runtime slices (45% + 17.5% = 62.7% now in TensorRT), their savings **compound** — the combined number beats either alone, the mirror image of the per-block dilution above.
- Getting correct numbers took real engineering: dynamic optimization profiles (batch *and* cache-length), current-stream execution (fixing a precision-independent NaN race), a real-valued RoPE refactor to escape complex-dtype ONNX, and engine-authoritative I/O binding.

See [`scripts/lingbot_trt/RESULTS.md`](scripts/lingbot_trt/RESULTS.md).

## Tech stack

Python · PyTorch · TensorRT · custom CUDA · OpenUSD · OpenCV · NumPy · stable-baselines3 / Gymnasium (RL) — targeting NVIDIA Isaac Sim / Omniverse.

## License

MIT
