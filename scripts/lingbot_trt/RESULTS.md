# LingBot-Map → TensorRT: Results

TensorRT optimization of [LingBot-Map](https://github.com/Robbyant/lingbot-map)
(Ant Group / Robbyant — a VGGT-style feed-forward streaming 3D-reconstruction
foundation model). Goal: take the model's PyTorch inference and make it faster on an
NVIDIA A10G via ONNX export + TensorRT, and report **measured** numbers only.

All figures below were reproduced on the box; nothing is assumed. Tooling is in this
directory (`export_probe.py`, `build_and_bench_trt.py`).

![LingBot-Map reconstruction — dense point cloud + solved camera trajectory](lingbot_reconstruction.png)

*LingBot-Map's Stage-0 output on the A10G: a handheld desk clip reconstructed into a
dense point cloud with the solved camera trajectory (frustums), rendered in viser.*

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

## Methodology (Stages 1–3, per-block)
- **Input capture:** a forward pre-hook grabs the block's actual `(args, kwargs)` during
  a real `inference_windowed` call, so the exported module sees exactly the tensors it
  runs on — no hand-constructed dummy shapes.
- **TRT I/O:** torch CUDA tensors as engine buffers via `set_tensor_address` +
  `execute_async_v3` on a private stream (no pycuda) — fine here because the standalone
  block has its inputs pre-synced (Stage 4 end-to-end needed the current stream instead).
- **Timing:** CUDA events, 200 timed runs after 50 warmup; median/mean/p95 reported.
- **Parity:** onnxruntime reference; fp16/INT8 tolerances noted per stage.

## Stage 4 — end-to-end integration (the per-block win does NOT translate)
Swapped **all 24** aggregator frame blocks for TRT engines and measured whole-model
fps vs the bf16 PyTorch baseline (`integrate_e2e.py`, 48-frame windowed run, real TUM
frames). Every input shape, block count, and output structure is discovered at runtime
via forward hooks — no hand-coded shapes.

| build | whole-model fps | speedup | reconstruction parity (rel) | non-finite |
|---|---|---|---|---|
| bf16 PyTorch (baseline) | 7.69 | 1.00× | — | 0 |
| **TRT strongly-typed bf16** | **8.31** | **1.08×** | 13.69% | 0 |
| TRT weakly-typed bf16 | 6.78 | 0.88× | 8.17% | 0 |

After the fixes below, engine **engagement is 100%** (2592–2616 engine calls, 0 torch
fallbacks). The headline: **the isolated 1.76× per-block fusion win (Stage 2) does not
survive end-to-end.** The best whole-model result is ~1.08×, and only by accepting
~14% precision drift; keeping precision faithful (fp32 LayerNorm/softmax, weakly-typed)
tightens parity to ~8% but goes *net slower* (0.88×). The frame blocks are simply not
the bottleneck — the stateful `global_blocks` (KV-cache) + DPT/camera heads stay in
PyTorch and dominate the runtime. The 8–14% parity is not error but TRT-bf16 vs
PyTorch-autocast-bf16 diverging over 24 non-associative fp accumulations.

### What it took to get a *correct* end-to-end number (the real engineering)
Each of these was measured, not assumed — the naive swap was wrong in four separate ways:
1. **Dynamic-batch profiles (11% → 100% engagement).** Frame blocks are called at
   *variable* batch (≤ 8: scale-frame / window / partial passes). A static engine built
   for one shape engaged only 11% of calls (89% fell back to torch → the "speedup" was
   just the baseline). Fix: export a dynamic batch axis + build one `1..max_batch`
   optimization profile per block (`max_batch` auto-detected from the capture pass).
2. **Stream-race NaN (precision-independent).** Running the engine on a *private* CUDA
   stream mid-forward raced the default-stream ops still producing the input → the
   engine read stale memory → ~40% NaN, byte-identical in fp16 and bf16. Fix: run on
   torch's **current stream** so the engine is ordered with the surrounding ops.
3. **fp16 is a red herring here.** Peak frame-block input activation is only **268.7**
   (fp16 caps at 65504) — measured, so the NaN was never fp16 overflow. bf16 and fp16
   gave identical NaN, confirming it was structural (the stream race), not numeric.
4. **Mixed precision must be matched.** The block runs LayerNorm/softmax in **fp32**
   under `autocast(bf16)`. A strongly-typed all-bf16 engine forces them to bf16 → 13.69%
   drift; a weakly-typed engine (fp32 ONNX + BF16 builder flag, norms stay fp32) mirrors
   autocast → 8.17%, but is slower.
5. **Shared device memory.** 24 weakly-typed (fp32) contexts want ~1.2 GB each →
   28 GB > the A10G's 22.5 GB. The blocks run sequentially on one stream, so they share
   **one** device-memory scratch (sized to the largest engine) — 24×1.2 GB → 1×1.2 GB.

## Stage 5, Step 1.5 — measured runtime split (the go/no-go gate)
Before writing any `global_blocks` export, `runtime_split.py` CUDA-event-timed one real
`inference_windowed` (48 frames, window 16, mean of 3 runs on the A10G) to settle where the
time actually goes — a perfect block engine is worthless if it isn't the bottleneck (the
Stage-4 lesson). And a **structure probe** (`probe_global_blocks.py`) first confirmed the KV
cache is **functional** (passed as an arg, not mutating module state) → the cleanest export
branch; the dynamic axis is the sequence/token length (prefill 8336 → decode 1042 tokens).

| group | ms / windowed forward | share |
|---|---|---|
| **global_blocks** (KV-cache attention) | **2820.7** | **45.2%** |
| frame_blocks (Stage-4 target) | 1398.2 | 22.4% |
| heads (DPT depth + camera) | 1090.7 | 17.5% |
| rest (patch embed / RoPE / norm / proj / overhead) | 937.5 | 15.0% |
| **TOTAL** | **6247.1** | 100% |

**Verdict: GO.** `global_blocks` are the single largest cost at **45.2%** — this is exactly
why Stage 4's frame-block engine (22.4% of runtime) only moved the whole model ~1.08×. The
honest Amdahl framing: a *perfect* (zero-cost) `global_blocks` caps whole-model at **1.82×**
(6247 / 3426); at a Stage-4-like **1.76× per-block** kernel speedup the realistic whole-model
gain is **~1.24×** (gb 2820→1603 ms), rising to ~1.43× if the cache engine hits 3×. So this
is the first LingBot target with a defensible whole-model win — because it's where the time
is, not just where the fusion looked good in isolation. (⚠ export snag noted for Step 2: `pos`
is `complex128` RoPE — TRT/ONNX have no complex dtype, so RoPE must be applied as real cos/sin
ops or precomputed outside the engine.)

## Stage 5, Step 2 — `global_blocks` export LANDED + measured (A10G)
`export_global_block.py` exports one `global_blocks[i]` (an `SDPAAttention` block) to a
functional ONNX and `build_and_bench_trt.py` builds + benches it. The block is
`f(x, pos, k_in, v_in) -> (out, k_out, v_out)`; the 8-frame cache grows to 9 in-graph
(`torch.cat` on the frame axis), returned as explicit outputs.

**The complex-RoPE snag was smaller than feared.** Phase-1 discovery showed `pos`
(`complex128` freqs_cis) is consumed **only** by `apply_rotary_emb`, which already runs
real arithmetic and just reads `pos.real`/`pos.imag`. Fix: feed `pos` as a real `[…,2]`
(cos, sin) tensor and monkeypatch `apply_rotary_emb` to a real-input variant — **no complex
op survives in the graph.** (The scaffold's first wrapper was wrong twice: it rebuilt a
`torch.complex` *inside* the graph, re-adding the unexportable op, and fed the post-mutation
9-frame cache so the graph would `cat` to 10 — fixed to feed the pre-write 8-frame snapshot.)

Measured, one block (index 0, first-decode call: 1042 tokens, 8→9 frame cache, mean of 200):

| config | ms / block (median) | note |
|---|---|---|
| TRT **fp32/TF32** | 8.50 | no tensor cores on the big SDPA |
| **torch bf16** (production baseline) | **2.76** | bf16 autocast, complex RoPE |
| **TRT fp16** | **2.28** | parity vs ORT-fp32 = 2.0e-2 (ok for fp16) |

- Export parity (fp32 wrapper vs bf16 capture): `mean|Δ|=3.2e-5`, `max|Δ|=1.1e-2` (bf16 noise).
- **Per-block TRT-fp16 vs torch-bf16 = 1.21×** — **below** the 1.76× that Step-1.5 assumed.
- Whole-model Amdahl with the *measured* 1.21×: `1/(0.548 + 0.452/1.21) ≈` **1.09×**
  (ceiling unchanged at 1.82×). **Correcting the Step-1.5 ~1.24× projection down** — it
  assumed a Stage-4-like 1.76× per-block kernel gain; the real global-block gain is 1.21×.

## What's next (not done)
- **Raise the per-block gain before integrating.** The fp16 engine has **fp32 I/O with
  boundary cast nodes** (weakly-typed). Levers, cheapest first: (a) a true fp16 ONNX +
  `--strongly-typed` to drop the casts; (b) INT8 (calibration npz already dumped by the
  exporter → `--calib-npz`); (c) the KV-cache-engine idea. Re-measure per-block, then
  recompute the whole-model projection.
- **Integrate all 24 blocks** (`integrate_e2e.py` pattern) + whole-model fps vs the bf16
  baseline with the NaN-aware parity harness — only worth it once the per-block gain and
  the dynamic-cache-length engine (currently static at 9 frames) are settled.
- **Caveat to carry:** the static engine bakes the cache conditionals for one length
  (two `TracerWarning`s on the skip-append branch); a streaming engine needs the
  `--dynamic-cache` axis + an optimization profile over cache history.

## Methodology (Stage 4, end-to-end)
- **TRT I/O:** torch CUDA tensors as engine buffers via `set_tensor_address` +
  `execute_async_v3` on torch's **current stream** (no pycuda) — ordered with the model.
- **Timing:** whole-model `inference_windowed`, warmup then timed iters, fps = frames/s.
- **Parity:** baseline vs TRT full reconstruction on identical real frames; NaN-aware
  (per-side non-finite counts + max abs diff over jointly-finite elements).
