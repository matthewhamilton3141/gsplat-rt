# Design — LingBot-Map `global_blocks` (KV-cache) → TensorRT (Stage 5)

Status: **scoping / not started.** This is the "real win" flagged in
[`scripts/lingbot_trt/RESULTS.md`](../scripts/lingbot_trt/RESULTS.md). It is the hard,
uncertain part; this doc scopes it so the next box session is efficient. Box-only work.

## Why this, and why it's hard
Stage 4 measured that swapping all 24 **frame blocks** to TensorRT nets only ~1.08×
whole-model — the frame blocks aren't the bottleneck. What's left in PyTorch and
dominating runtime is (a) the **`global_blocks`**, which carry a **cross-window KV
cache**, and (b) the DPT/camera heads. The `global_blocks` are the interesting target
because they're attention over a *growing* cache — the genuinely hard export.

Hard for three specific reasons, all of which Stage 4's frame-block toolchain did **not**
have to face:
1. **Dynamic sequence length.** A frame block was fixed-shape (only batch varied). A
   global block attends over the accumulated KV cache, so its effective key/value length
   **grows across windows**. That's a dynamic axis on the *sequence*, not just batch —
   more optimization-profile range, and a real risk that TRT picks poor tactics at the
   extremes.
2. **State across calls.** The KV cache is *carried between* `inference_windowed`
   windows (`clean_kv_cache()` resets it). A TRT engine is a pure function; the cache
   must be lifted out and managed in Python (pass cached K/V in, get updated K/V out) —
   or the block refactored to expose it.
3. **Data-dependent control flow.** VGGT-style caches often have first-window-vs-later
   branches, eviction, or masking. Control flow doesn't export to a static graph; each
   branch may need its own engine or a Python-side split.

## Step 1 (do first): characterize, don't assume — `probe_global_blocks.py`
The exact `global_block.forward` signature and cache mechanism live in the on-box
`lingbot_map` source, so we **discover** them at runtime instead of guessing (same move
that de-risked the frame blocks in Stage 1). The probe hooks every `global_blocks[i]`
during one real windowed run and reports:
- the full `(args, kwargs)` structure per call — tensors vs baked scalars/None;
- each tensor's shape + dtype, and **which dim drifts across calls** (the cache axis);
- whether the block owns **mutating buffers** (⇒ the cache is module state) or the cache
  is **passed as an arg** (⇒ functional, much easier to export).

Run:
```bash
cd ~/lingbot-map && source .venv/bin/activate
python ~/gsplat-rt/scripts/lingbot_trt/probe_global_blocks.py \
    --model_path checkpoints/lingbot-map-long.pt --lingbot-root ~/lingbot-map \
    --window-size 16 --frames 48
```
Its output picks the branch below. **Everything after Step 1 is contingent on it.**

## Step 2 — export strategy (decided by the probe)
Reuse everything that worked in Stage 4 (`integrate_e2e.py`): runtime input capture via
hooks, classic TorchScript ONNX export (dynamo broke TRT's parser), keep int index
tensors int, **current-stream** execution (a private stream raced the model → NaN),
**weakly-typed bf16** to match autocast (fp32 LayerNorm/softmax), **shared TRT device
memory** across contexts, and the NaN-aware parity + per-block self-check harness.

- **If the cache is functional (passed as args):** wrap the block as
  `f(current_tokens, cached_k, cached_v, pos) -> (out, new_k, new_v)`; build ONE engine
  with a dynamic profile over the cache-length axis `min=1 .. max=(max cache seen in the
  probe)`; keep the concat/evict in Python. Cleanest path.
- **If the cache is module state (mutating buffers):** first refactor a functional
  wrapper that takes the cache in and returns it out (no in-place buffer writes) — only
  then is it exportable. Larger change; validate parity on the wrapper before TRT.
- **If there's first-window/later branching:** export the steady-state (cache-present)
  path as the engine; let the first window (empty cache) run in PyTorch (rare, cheap) —
  mirrors Stage 4's torch fallback for off-profile shapes.

### Step 2 first pass — `export_global_block.py` (two-phase, box-only)
Step 1.5 measured GO (`global_blocks` = **45.2%** of runtime; RESULTS.md). The export scaffold
`scripts/lingbot_trt/export_global_block.py` runs in two phases so the cheap discovery is never
lost to a wrapper bug:
- **Phase 1 (always completes):** prints `inspect.getsource(type(block).forward)` + its source
  file, the captured decode-call `(args, kwargs)`, the full `kv_cache` dict (per-key shape/dtype),
  **which cache keys change across the call** (the slot this block writes), the output signature,
  every complex tensor, and dumps the real I/O to `.npz`. This is what we still lack: the exact
  `forward` body, the cache-return contract, and the RoPE application — none of it guessed.
- **Phase 2 (attempt):** functional wrapper `f(tokens, cache_slot…, pos_real, pos_imag) ->
  (out, new_cache…)`, complex `pos` split to real/imag at the boundary, classic-TorchScript
  export. **Expected to fail on the complex RoPE op** — that error names the real-cos/sin refactor.

First box run (discovery only, then attempt):
```bash
cd ~/lingbot-map && source .venv/bin/activate
python ~/gsplat-rt/scripts/lingbot_trt/export_global_block.py \
    --model_path checkpoints/lingbot-map-long.pt --lingbot-root ~/lingbot-map \
    --index 0 --capture-call 1 --dump-io /tmp/gblock0_io.npz   # add --no-export for Phase 1 only
```
**Wrapper assumptions to confirm against the Phase-1 source dump** (fix before trusting Phase 2):
the block reads/writes only its own slot inside the passed `kv_cache` dict (keyed by `global_idx`),
and returns its main activation as the output. The probe showed the *same* full dict passed to all
24 blocks → the cache is shared, per-block-keyed (`k_{idx}`/`v_{idx}` + `_special`).

## Step 3 — integrate + measure (honest)
Same structure as Stage 4: swap all global blocks, verify with the self-check + NaN-aware
end-to-end parity on **real** frames, and report whole-model fps vs the bf16 baseline.

**Expectation management:** this may *still* not yield a big whole-model win — if the DPT/
camera heads are a large fraction of runtime, even a perfect `global_blocks` engine is
diluted (the Stage 4 lesson, again). So **Step 1.5 = a coarse runtime breakdown**: time
`global_blocks` vs `frame_blocks` vs heads with CUDA-event timers around each group in one
windowed run, *before* investing in the export. If the heads dominate, the honest move is
to pivot the effort to the heads (static, far easier to export) rather than the cache.

## Box-inspection TODOs (unknown until Step 1 runs)
- [ ] Exact `global_block.forward(*args, **kwargs)` signature; which tensor is the cache.
- [ ] Cache storage: arg/kwarg vs module buffer; dtype; how it's concatenated/evicted.
- [ ] The cache-length range across a 48-frame run (sets the profile max).
- [ ] Attention mask / causal / sliding-window behavior (affects export correctness).
- [ ] First-window special-casing (empty-cache branch).
- [ ] Coarse runtime split global_blocks : frame_blocks : heads (is this even worth it?).

## Cost / risk
Box-heavy and uncertain — genuinely research-y, unlike the frame blocks. Do Step 1 (the
probe, ~2 min) and Step 1.5 (the runtime breakdown) **first and cheaply**; only commit to
the full export if the breakdown says the `global_blocks` are actually where the time is.
Prefer a defensible "measured that it isn't worth it" over a big unverified push.
