#!/usr/bin/env python3
"""Stage 5 (characterization): discover how the aggregator's KV-cache `global_blocks`
are actually called, so we can decide whether/how to export them to TensorRT.

Stage 4 proved the frame blocks aren't the whole-model bottleneck — the stateful
`global_blocks` (cross-window KV cache) + the DPT/camera heads are. Before writing any
export, we need to KNOW the shape of the problem, and the only source of truth is the
running model. This script hooks every `global_blocks[i]` during a real
`inference_windowed` call and reports, per block and per call:

  - the full (args, kwargs) structure: which are tensors vs baked scalars/None
  - each tensor's shape + dtype, and WHICH dims change across calls (the cache axis)
  - whether the block owns stateful buffers/attributes that mutate between windows
    (i.e. where the KV cache actually lives — an arg passed in, or state held inside)

That answer decides the export strategy (see design_lingbot_global_blocks_trt.md):
a dynamic profile over a *growing sequence* axis, and whether the cache must be lifted
out of the module (functional export) or can stay a Python-managed concat.

Box-only (needs torch + the lingbot_map package + the checkpoint). Nothing here builds
an engine yet — it's pure, read-only discovery, so it's cheap and safe to run first.

Example:
    cd ~/lingbot-map && source .venv/bin/activate
    python ~/gsplat-rt/scripts/lingbot_trt/probe_global_blocks.py \
        --model_path checkpoints/lingbot-map-long.pt --lingbot-root ~/lingbot-map \
        --window-size 16 --frames 48
"""

import argparse
import os
import sys

# Reuse the Stage-1 helper set (same directory) so model-arg construction matches
# exactly what export_probe / integrate_e2e feed the loader.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_probe import _model_args                                   # noqa: E402


def _describe(v, torch):
    """One-line description of a captured value (tensor -> shape/dtype, else type)."""
    if torch.is_tensor(v):
        return f"T{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}"
    if v is None:
        return "None"
    if isinstance(v, (int, float, bool, str)):
        return f"{type(v).__name__}({v})"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{len(v)}]"
    if isinstance(v, dict):
        return f"dict{{{','.join(v)}}}"
    return type(v).__name__


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5: characterize aggregator global_blocks")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lingbot-root", default=os.path.expanduser("~/lingbot-map"))
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--max-calls", type=int, default=6,
                    help="how many calls per block to print in full (the rest summarized)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.lingbot_root))
    import torch
    from demo import load_model

    if not torch.cuda.is_available():
        print("ERROR: CUDA required (box-only).")
        return 2
    device = torch.device("cuda")

    model = load_model(_model_args(args), device).eval()
    agg = getattr(model, "aggregator", None)
    if agg is None:
        print("ERROR: model has no .aggregator.")
        return 2
    if torch.cuda.get_device_capability()[0] >= 8:
        model.aggregator = model.aggregator.to(dtype=torch.bfloat16)   # mirrors demo.py
        agg = model.aggregator

    gblocks = getattr(agg, "global_blocks", None)
    if gblocks is None:
        # discover the real attribute name if it differs from the docs
        cand = [n for n, _ in agg.named_children()]
        print(f"ERROR: aggregator has no .global_blocks. children = {cand}")
        return 2
    n_blocks = len(gblocks)
    print(f"aggregator has {n_blocks} global blocks "
          f"(+ {len(getattr(agg, 'frame_blocks', []))} frame blocks)")

    # snapshot each block's buffers so we can see which mutate between windows (= the
    # KV cache, if it's held as module state rather than passed as an arg).
    def buf_sig(mod):
        out = {}
        for name, b in mod.named_buffers(recurse=True):
            if torch.is_tensor(b):
                out[name] = (tuple(b.shape), str(b.dtype))
        return out

    log = {i: [] for i in range(n_blocks)}          # idx -> list of per-call records
    buf_before = {i: buf_sig(gblocks[i]) for i in range(n_blocks)}

    handles = []
    for i, blk in enumerate(gblocks):
        def mk(idx):
            def pre(mod, a, kw):
                rec = {
                    "args": [_describe(v, torch) for v in a],
                    "kwargs": {k: _describe(v, torch) for k, v in kw.items()},
                }
                log[idx].append(rec)
            return pre
        handles.append(blk.register_forward_pre_hook(mk(i), with_kwargs=True))

    print(f"\nrunning one windowed forward ({args.frames} frames, window {args.window_size}) ...")
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    torch.manual_seed(0)
    imgs = torch.rand(args.frames, 3, args.height, args.width, device=device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.inference_windowed(imgs, window_size=args.window_size, overlap_size=0,
                                 num_scale_frames=args.num_scale_frames, keyframe_interval=1)
    for h in handles:
        h.remove()
    buf_after = {i: buf_sig(gblocks[i]) for i in range(n_blocks)}

    # ---- report ------------------------------------------------------------------
    print("\n=== per-block call structure (block 0 shown in full; others summarized) ===")
    for i in range(n_blocks):
        calls = log[i]
        print(f"\n[global_block {i}] called {len(calls)}x")
        show = calls if i == 0 else calls[:1]
        for j, rec in enumerate(show[: args.max_calls]):
            print(f"  call {j}: args={rec['args']} kwargs={rec['kwargs']}")
        # which tensor arg dims vary across calls -> the growing cache / seq axis
        if len(calls) > 1:
            first, last = calls[0]["args"], calls[-1]["args"]
            deltas = [f"arg{k}: {a} -> {b}" for k, (a, b) in enumerate(zip(first, last))
                      if a != b]
            if deltas:
                print(f"  shape drift first->last call: {deltas}")

    print("\n=== mutating buffers (candidate KV-cache state held inside the module) ===")
    any_state = False
    for i in range(n_blocks):
        changed = [n for n in buf_after[i]
                   if buf_before[i].get(n) != buf_after[i].get(n)]
        if changed:
            any_state = True
            print(f"[global_block {i}] buffers changed after the run: "
                  f"{[(n, buf_before[i].get(n), buf_after[i].get(n)) for n in changed]}")
    if not any_state:
        print("no module buffers changed -> the KV cache is passed as an ARG / kwarg "
              "(functional), not held as module state. Good: easier to export.")

    print("\nNext: pick the export strategy in design_lingbot_global_blocks_trt.md using "
          "the varying-dim ('cache axis') and the state answer above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
