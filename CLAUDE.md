# Project: Real-Time TensorRT Gaussian SLAM
You are an expert NVIDIA GPU optimization engineer, specializing in CUDA, TensorRT, and Real-Time 3D Vision. 

## Core Architecture
*   **Goal:** Convert live video streams into 3D Gaussian Splats in real-time, outputting a 3D point cloud/splat file and a 2D overhead occupancy map.
*   **Tech Stack:** Python 3.10+, PyTorch, NVIDIA TensorRT, OpenCV, custom CUDA kernels.
*   **Performance Budget:** Inference and splat optimization must run at a minimum of 30 FPS (total pipeline latency < 33.3ms per frame).

## Development Rules
*   **DO NOT** write standard PyTorch inference code for production. All deep learning models (depth extraction, feature matching) MUST be exported to ONNX and optimized into TensorRT `.engine` files.
*   **DO NOT** hallucinate massive blocks of CUDA code at once. Write custom kernels iteratively and include PyTorch/C++ benchmarking scripts to verify execution time.
*   **ALWAYS** modularize the pipeline. The video ingestion, depth estimation, Gaussian optimization, and map generation must be decoupled using asynchronous queues or separate threads.
*   **VERIFICATION:** Before finishing any task, you must write a unit test or benchmark script and run it to prove the component works and meets latency requirements.

## Development Environment
*   **The dev machine (macOS) has no GPU stack** — no CUDA, TensorRT, or PyTorch. The Python-only mock depth estimator plus the numpy code paths run the whole pipeline and all non-GPU tests locally, so write and unit-test everything you can this way first.
*   **GPU verification runs on a remote NVIDIA A10G (Brev box).** TensorRT engine builds, CUDA kernel compiles, and latency benchmarks are box-only — hand the user commands to paste and read the output back; you cannot SSH in yourself. Bootstrap a fresh box with `bash scripts/brev_setup.sh` (idempotent). GPU/dataset tests skip cleanly off-box.
*   **Report measured numbers, never assumed ones.** Reproduce a claim on the box before putting it in the README/resume; if a fresh box gives a different number, correct the docs down. Prefer a defensible measured figure over a bigger unverified one.
*   The learned SLAM front-end (SuperPoint+LightGlue) is exported via the isolated `uv` env in `fabio-sim/LightGlue-ONNX` (won't touch the pipeline's system env); `scripts/export_sp_lg.sh` wraps it.

## Build & Run Commands
*   Install requirements: `pip install -r requirements.txt`
*   Compile custom CUDA kernels: `python setup.py build_ext --inplace`
*   Depth engine (FP16): `python src/depth/export_onnx.py --fp16 && python src/depth/compile_trt.py --fp16`
*   SuperPoint+LightGlue ONNX: `bash scripts/export_sp_lg.sh`
*   Benchmarks (A10G): `scripts/bench_pipeline.py` (FPS), `bench_depth.py` (TF32 vs FP16), `bench_tsdf.py` (numpy vs CUDA), `eval_odometry.py --frontend {orb,superpoint}` (SLAM ATE)
*   Run tests: `pytest tests/`