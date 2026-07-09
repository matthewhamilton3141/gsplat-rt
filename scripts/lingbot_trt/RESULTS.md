# LingBot-Map → TensorRT: Results

TensorRT optimization of [LingBot-Map](https://github.com/Robbyant/lingbot-map)
(Ant Group / Robbyant — a VGGT-style feed-forward streaming 3D-reconstruction
foundation model). Goal: take the model's PyTorch inference and make it faster on an
NVIDIA A10G via ONNX export + TensorRT, and report **measured** numbers only.

All figures below were reproduced on the box; nothing is assumed. Tooling is in this
directory (`export_probe.py`, `build_and_bench_trt.py`).

## Environment
- **GPU:** NVIDIA A10G (Ampere, sm86), 22.5 GB usable, driver CUDA 13.2
- **Stack:** isolated `uv` venv, PyTorch (cu128), TensorRT 10.x, onnxruntime; **no
  FlashInfer** (its CUDA paged-KV attention is optional — every attention path has an
  `F.scaled_dot_product_attention` fallback, so we run and export the SDPA path)
- **Model:** `lingbot-map-long.pt` (4.63 GB). Architecture: DINOv2/ViT `patch_embed`
  → aggregator (`frame_blocks` + KV-cache `global_blocks`, alternating) → DPT head
  (dense depth) + camera head (pose)

## Stage 0 — PyTorch baseline
`demo.py --mode windowed --window_size 16 --use_sdpa`, 205 frames (fps-10 sample of a
613-frame TUM `freiburg1_desk` clip), 518×392:

| metric | value |
|---|---|
| throughput | **0.49 fps** (205 frames / 418.6 s) |
| per window | 2.20 s (190 windows) |
| precision | aggregator bf16, heads fp32 |
| GPU peak | 13.35 GB (of 22.5) |

Reconstruction: coherent, dense, colored point cloud + solved camera trajectory
(viewed in viser). Note 0.49 fps on a datacenter GPU ≈ the model's Apple-MPS
ballpark — because it's 16-bit SDPA with no fused kernels. That gap is the target.

## Optimization target: one aggregator frame block
Frame blocks are fixed-shape and **cacheless** (the `global_blocks` carry a
cross-window KV cache — dynamic control flow, out of scope for this pass). Exported
`aggregator.frame_blocks.0` via a **forward hook that captures the block's real
inputs** (no shape guessing):
- inputs: `x` (8, 1042, 1024) float + `pos` (8, 1042, 2) **int64** (RoPE index)
- output: (8, 1042, 1024)

## Stage 1 — ONNX export (correctness)
| exporter | parity (max abs diff, torch fp32 vs onnxruntime) |
|---|---|
| classic TorchScript, opset 18 | **7.6e-06** ✓ |

Findings that shaped the toolchain:
- The **dynamo** exporter's ONNX broke TensorRT's `OnnxParser` (weight-import failure)
  even though onnxruntime loaded it fine → use the **classic** TorchScript exporter,
  whose weights TRT parses reliably.
- RoPE indexes an embedding table with `pos`, so integer index tensors must **not**
  be cast to float during export prep.

## Stage 2 — FP16 TensorRT engine (speed)
| build | latency / block (median, 200 runs) | speedup vs bf16 | parity vs fp32 |
|---|---|---|---|
| PyTorch bf16 (baseline) | 9.23 ms | 1.00× | — |
| TRT weakly-typed FP16 | 5.47 ms | 1.69× | 1.5e-2 |
| **TRT strongly-typed FP16** | **5.25 ms** | **1.76×** | 1.6e-2 |

Strongly-typed (native fp16 ONNX, fp16 I/O) shaved the ~4% boundary-cast overhead off
weakly-typed. **Key point:** the baseline is already bf16 (16-bit), so this 1.76× is a
pure kernel-**fusion** win at equal precision — not a precision win.

## Stage 3 — INT8 (measured, does not pay off)
INT8 via `IInt8MinMaxCalibrator` (MinMax, calibrated from a real captured activation
tensor — millions of range samples):

| build | latency / block | parity |
|---|---|---|
| TRT INT8 (implicit calibration) | 5.46 ms | 1.6e-2 |

**No gain over fp16, because INT8 never engaged.** TensorRT 10.1 reported "Missing
scale and zero-point … fall back to non-int8" for nearly every tensor, and flagged the
calibrator API as deprecated ("superseded by explicit quantization"). So the block ran
as fp16 underneath — identical speed and accuracy.

Two reasons INT8 isn't the answer here, both worth stating:
1. **Path:** implicit calibration is dead in TRT 10; real INT8 needs **explicit QDQ**
   nodes baked into the ONNX (TRT Model-Optimizer / ORT quant) — a much larger effort.
2. **Ceiling:** the block is **memory-bound and non-GEMM-heavy** (LayerNorm, RoPE
   gathers, attention softmax all stay ≥fp16). INT8 only accelerates the large GEMMs,
   which are not the bottleneck — the realistic upside is ~10–20%.

**Conclusion: FP16 (1.76×) is the per-block sweet spot** on the standard TensorRT paths.

## Methodology
- **Input capture:** a forward pre-hook grabs the block's actual `(args, kwargs)` during
  a real `inference_windowed` call, so the exported module sees exactly the tensors it
  runs on — no hand-constructed dummy shapes.
- **TRT I/O:** torch CUDA tensors as engine buffers via `set_tensor_address` +
  `execute_async_v3` on a private stream (no pycuda) — the project's depth-engine idiom.
- **Timing:** CUDA events, 200 timed runs after 50 warmup; median/mean/p95 reported.
- **Parity:** onnxruntime reference; fp16/INT8 tolerances noted per stage.

## What's next (not done)
- **End-to-end integration:** wire the fp16 engine into the aggregator and measure
  whole-model fps. Expected to be *diluted* vs 1.76× — the frame blocks are only part
  of the compute, and the stateful `global_blocks` + heads stay in PyTorch.
- **The hard win:** the KV-cache `global_blocks` (dynamic control flow) — the real
  remaining latency, and the genuinely difficult export.
