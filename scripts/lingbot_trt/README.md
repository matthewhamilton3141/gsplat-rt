# LingBot-Map → TensorRT (Track B)

Making [LingBot-Map](https://github.com/Robbyant/lingbot-map) (VGGT-style streaming
3D reconstruction) fast on the A10G. The upstream model runs ~0.49 fps in bf16 SDPA
(measured, `--window_size 16`, box `brev-hp0yaxne3`) — the optimization target.

These scripts run **on the box**, inside the lingbot-map venv (`~/lingbot-map/.venv`),
against a clone of the upstream repo (`--lingbot-root ~/lingbot-map`). Nothing here is
Mac-testable (no torch on the dev machine): the workflow is hand-a-command / read the
output back, and iterate.

## Architecture (why the plan is shaped this way)
LingBot-Map = DINOv2/ViT `patch_embed` → aggregator (`frame_blocks` + `global_blocks`,
alternating) → DPT head (dense depth→points) + camera head (pose). The heavy compute
is the aggregator. Key facts found in recon:
- **FlashInfer is optional** — every attention path has an `F.scaled_dot_product_attention`
  fallback; run with `--use_sdpa`. So we export the SDPA path, not a custom CUDA kernel.
- `frame_blocks` are fixed-shape and **cacheless**; `global_blocks` carry a cross-window
  **KV cache** (dynamic — the hard part). `demo.py`'s `compile_model` torch.compiles the
  frame blocks + patch_embed blocks, confirming they're the fixed-shape hot modules.

## Stages (all DONE — measured numbers + full analysis in `RESULTS.md`)
- **Stage 0 — baseline:** `demo.py --mode windowed --window_size 16 --use_sdpa`
  → 205 frames / 418.6 s ≈ 0.49 fps, peak 13.35 GB. Coherent, dense, colored recon.
- **Stage 1 — ONNX export probe (`export_probe.py`):** export ONE frame block to ONNX
  with numeric parity (7.6e-6) via a forward hook that captures its real inputs. Findings:
  use the **classic** TorchScript exporter (dynamo's ONNX broke TRT's parser); keep the
  RoPE `pos` index tensor **int** (don't cast to float).
- **Stage 2 — FP16 TensorRT engine (`build_and_bench_trt.py`):** strongly-typed fp16
  engine, **1.76× per block** vs the bf16 baseline — a pure kernel-**fusion** win at equal
  precision (the baseline is already bf16).
- **Stage 3 — INT8:** does not pay off — implicit calibration is dead in TRT 10, and the
  block is memory-bound / non-GEMM-heavy, so INT8 never engages. fp16 is the per-block
  sweet spot.
- **Stage 4 — end-to-end integration (`integrate_e2e.py`):** swap **all 24** frame blocks
  and measure whole-model fps. **The per-block 1.76× does NOT translate:** best whole-model
  is ~1.08× (strongly-typed bf16, ~14% drift); faithful precision (weakly-typed, fp32
  norms) is net *slower* (0.88×). Frame blocks aren't the bottleneck — the KV-cache
  `global_blocks` + heads dominate. Getting a *correct* number took five measured fixes
  (dynamic-batch profiles, current-stream execution, ruling out fp16 overflow, autocast
  mixed-precision matching, shared TRT device memory) — see `RESULTS.md`.

## What's next (not done)
The real target is the KV-cache `global_blocks` (dynamic control flow) — the genuinely
hard, stateful export and the only path to a meaningful whole-model speedup.

## Run (Stage 4, end-to-end)
```bash
cd ~/lingbot-map && source .venv/bin/activate
python ~/gsplat-rt/scripts/lingbot_trt/integrate_e2e.py \
    --model_path checkpoints/lingbot-map-long.pt \
    --lingbot-root ~/lingbot-map \
    --window-size 16 --frames 48 --precision bf16   # add --weakly-typed for tight parity
```
Engines cache under `--engine-dir` (tagged by precision/mode), so builds are one-time.
Reports engagement (engine calls vs torch fallbacks), NaN-aware parity, and whole-model
speedup vs the bf16 baseline.
