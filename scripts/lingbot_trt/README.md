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

## Stages
- **Stage 0 — baseline (DONE):** `demo.py --mode windowed --window_size 16 --use_sdpa`
  → 205 frames / 418.6 s ≈ 0.49 fps, peak 13.35 GB. Coherent, dense, colored recon.
- **Stage 1 — ONNX export probe (IN PROGRESS):** `export_probe.py` — export ONE
  self-contained **frame block** to ONNX with numeric parity, via a forward hook that
  captures the block's real inputs (no shape guessing). De-risks the ONNX→TRT toolchain.
  First pass is expected to surface an unsupported op (RoPE complex ops / SDPA) — that's
  the signal for what to handle next.
- **Stage 2 — FP16 TensorRT engine:** build a TRT engine from the parity-checked ONNX,
  measure a single block's latency, then export the full frame-block stack.
- **Stage 3 — rewire + INT8:** swap the TRT engine into the aggregator forward (engine =
  block compute, Python keeps the KV-cache orchestration + DPT/camera heads), measure
  end-to-end vs the 0.49 fps baseline; then INT8 calibration (A10G is Ampere → INT8, not FP8).

## Run (Stage 1)
```bash
cd ~/lingbot-map && source .venv/bin/activate
python ~/gsplat-rt/scripts/lingbot_trt/export_probe.py \
    --model_path checkpoints/lingbot-map-long.pt \
    --target aggregator.frame_blocks.0 \
    --window-size 16 --onnx-out /tmp/frame_block0.onnx
```
Report back: the captured input shapes, and either the parity number (PASS) or the
export error (which op broke).
