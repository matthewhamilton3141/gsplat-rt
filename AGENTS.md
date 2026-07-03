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

## Build & Run Commands
*   Install requirements: `pip install -r requirements.txt`
*   Compile custom CUDA kernels: `python setup.py build_ext --inplace`
*   Run tests: `pytest tests/`