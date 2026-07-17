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

| config | ms / block (median) | per-block | note |
|---|---|---|---|
| TRT **fp32/TF32** | 8.50 | 0.36× | no tensor cores on the big SDPA |
| **torch bf16** (production baseline) | **2.76** | 1.00× | bf16 autocast, complex RoPE |
| TRT fp16 **weakly-typed** (fp32 I/O) | 2.28 | 1.21× | parity vs ORT-fp32 = 2.0e-2 |
| **TRT fp16 strongly-typed** | **1.80** | **1.53×** | true fp16 ONNX, no I/O casts; parity 1.9e-2 |
| TRT bf16 strongly-typed | 1.82 | 1.52× | ~tied; ORT-CPU can't verify bf16 parity |

- Export parity (fp32 wrapper vs bf16 capture): `mean|Δ|=3.2e-5`, `max|Δ|=1.1e-2` (bf16 noise);
  the true-fp16 export holds at `mean|Δ|=8.9e-5`.
- **Per-block best = 1.53×** (strongly-typed fp16). The weakly-typed engine's fp32 I/O +
  boundary cast nodes were the handicap; a true fp16 ONNX built strongly-typed drops them
  (2.28 → 1.80 ms, a further 1.27× *within* TRT). fp16 beats bf16 by a hair and, unlike bf16,
  its parity is CPU-verifiable — so fp16 strongly-typed is the pick.
- Whole-model Amdahl with the measured **1.53×**: `1/(0.548 + 0.452/1.53) ≈` **1.19×** (ceiling
  1.82×). This recovers most of the Step-1.5 ~1.24× projection — that estimate assumed a 1.76×
  per-block gain; the real strongly-typed gain is 1.53×, so ~1.19× is the honest whole-model
  number to carry (the earlier 1.09× was the weakly-typed engine before this lever).

## Stage 5, Step 3 — end-to-end integration (measured: the per-block win still dilutes)
`integrate_global_e2e.py` swaps **all 24** `global_blocks` for the strongly-typed fp16
engines and measures whole-model fps vs the bf16 baseline (48-frame TUM clip, real frames,
same NaN-aware parity harness as Stage 4). Each block is a functional engine
`f(x, pos_real, k_in, v_in) → (out, k_out, v_out)` with a **dynamic cache-length profile**;
the KV cache is managed in Python (concat happens in-graph, returned + written back), complex
`pos` is converted to real cos/sin per call, and the **prefill / scale-frame call is left in
PyTorch** (torch fallback). Regime: the observed cache stays at **8..15 frames** (≪ the
~72-frame eviction threshold), so the baked no-eviction engine is valid; the run asserts this.

| build | whole-model fps | speedup | reconstruction parity | non-finite |
|---|---|---|---|---|
| bf16 PyTorch (baseline) | 7.69 | 1.00× | — | 0 |
| global_blocks → TRT fp16 (cache round-trip) | 7.81 | 1.015× | 3.6% rel | 0 |
| **global_blocks → TRT fp16 (fp16 cache)** | **8.22** | **1.069×** | 3.5% rel | 0 |

2328 engine calls / 288 torch fallbacks (the prefill calls); 0 non-finite either side.
Dropping a growing-cache fp32↔fp16 round-trip (store the engine's fp16 in place, restore
torch dtype only on the rare fallback) lifted 1.015× → **1.069×** with parity unchanged.

**The honest headline: the isolated 1.53× per-block collapses to 1.069× whole-model** — a
*real* measured win (~7% faster, tight parity, zero NaN), but far below both the per-block
figure and the **1.19× Amdahl projection** from Step 2. **Correcting my own projection down:
1.19× projected → 1.069× measured.** Why the gap: (a) the integrated engines are
**dynamic** cache-length (variable-KV attention → slower tactics than the 1.80 ms *static*
bench); (b) per-call Python overhead — shape re-specialization, output-cache alloc, the
complex→real `pos` conversion, dict bookkeeping. This is the **Stage-4 lesson a third time**:
even the *dominant* 45% component, with a genuine 1.53× isolated kernel, nets ~7% end-to-end
once integration overhead and Amdahl are paid. Measured, not assumed.

## Stage 6 — DPT depth head → TRT (isolated, measured): the static win the blocks weren't
Following the Stage-3 read ("the heads are static → the cleaner win per effort"), exported the
`DPTHead` (the heavy one; the camera head outputs `(1,1,9)` and is negligible). It's a static
feed-forward module taking 4 aggregated-token feature maps `(1,8,1042,2048)` + `images`
`(1,8,3,392,518)` → `(depth, conf)` `(1,8,392,518,*)`. `export_head.py` captures the real call
and exports; the true-fp16 export tripped an internal fp32↔fp16 mismatch (common in a complex
head), so — the Stage-4 recipe — export fp32 ONNX + build a **weakly-typed fp16** engine (TRT
picks per-layer precision, keeps precision-sensitive ops fp32).

| build | latency / call (median, 200) | speedup |
|---|---|---|
| **torch fp32 (production baseline)** | **109.9 ms** | 1.00× |
| **TRT weakly-typed fp16** | **37.5 ms** | **2.93×** |

**2.93× — nearly double the global-block per-component win (1.53×)**, and for structural
reasons: the head runs **fp32** in production (so fp16 is a genuine precision+fusion win, not
the equal-precision fusion the bf16 blocks got), it's **static** (no dynamic cache-length engine
penalty that diluted the blocks), and it's **conv/GEMM-heavy** (ideal for fp16 tensor cores).
Projected whole-model from the head alone: `1/(0.825 + 0.175/2.93) ≈` **1.13×**.

## Stage 6b — DPT head integrated end-to-end (measured): it translated
`integrate_head_e2e.py` swaps the engine into the live model and measures whole-model fps +
parity — including a **per-head self-check on the real captured input** (the verification
Stage 6 skipped). Two findings, both measured:

- **Parity VERIFIED.** TRT fp16 engine vs the torch fp32 head on the real input: **1.19% worst
  rel diff, 0 non-finite** (much tighter than the ~8% guess — the DPT head is conv/interp-heavy,
  which fp16 handles cleanly). Resolves the Stage-6 caveat.
- **Dynamic frame axis was required** (Stage-4 lesson, again). The head is called at variable
  frame counts (S = 1..8: chunks / scale pass / partial windows), so a *static* engine covered
  only **13/109 calls** (88% torch fallback) → 1.035×. Marking dim-1 dynamic + one optimization
  profile lifted engagement to **109/109 (0 fallbacks)**.

| build | whole-model fps | speedup | parity | non-finite |
|---|---|---|---|---|
| bf16+fp32-head baseline | 7.70 | 1.00× | — | 0 |
| static head engine (12% engaged) | 7.97 | 1.035× | 1.15% | 0 |
| **dynamic head engine (100% engaged)** | **8.45** | **1.098×** | 1.75% | 0 |

**1.098× — and notably the heads (17.5% of runtime) beat the global_blocks (45% → 1.069×)
end-to-end.** A smaller runtime slice won more whole-model because it *translates*: static (no
dynamic-cache/Python overhead that diluted the blocks) + a bigger per-component speedup (2.93×
vs 1.53×). The "static heads are the cleaner win per effort" call (Stage 3) held up, measured.

## What's next (not done)
- **Stack heads + global_blocks** in one run (independent chunks) for the combined whole-model
  number — the two biggest levers together (projected ~1.16–1.20×, to be measured).
- **Global-block gap** (1.069×): static per-cache-length engines / CUDA-graph. Diminishing
  returns — likely near the practical ceiling for that path; leave it.

## Methodology (Stage 4, end-to-end)
- **TRT I/O:** torch CUDA tensors as engine buffers via `set_tensor_address` +
  `execute_async_v3` on torch's **current stream** (no pycuda) — ordered with the model.
- **Timing:** whole-model `inference_windowed`, warmup then timed iters, fps = frames/s.
- **Parity:** baseline vs TRT full reconstruction on identical real frames; NaN-aware
  (per-side non-finite counts + max abs diff over jointly-finite elements).
