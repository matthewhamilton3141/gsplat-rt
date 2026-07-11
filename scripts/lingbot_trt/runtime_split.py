#!/usr/bin/env python3
"""Stage 5, Step 1.5 (the go/no-go gate): coarse GPU runtime split of a windowed forward.

`probe_global_blocks.py` answered the *structure* question (functional KV cache, dynamic
sequence axis) but not the one that actually decides whether Stage 5 is worth doing: **where
does the time go?** The whole Stage-4 lesson (RESULTS.md) is that a perfect block engine gets
diluted if it isn't the bottleneck — swapping all 24 frame blocks to TensorRT netted only
~1.08× whole-model. So before writing any `global_blocks` export we measure, with CUDA-event
timers, how one real `inference_windowed` splits across:

  - aggregator **global_blocks**  (the cross-window KV-cache attention — Stage 5's target)
  - aggregator **frame_blocks**   (already shown not to be the bottleneck in Stage 4)
  - the model's **heads**          (DPT depth / camera / etc. — top-level non-aggregator children)
  - **rest**                       (patch embed, RoPE, norms, projections, Python overhead) =
                                   total − the sum of the hooked groups

Decision rule (design_lingbot_global_blocks_trt.md): if `global_blocks` are a large fraction,
the export is worth the research cost; if the **heads** dominate, pivot to them (static, far
easier to export) rather than the cache. A defensible "measured that it isn't worth it" beats
a big unverified push.

CUDA-event timing (not wall clock): a start/end event pair is recorded around every hooked
module call on the current stream; after one synchronize we sum GPU elapsed time per group.
This attributes async kernel time correctly even though frame/global blocks interleave.

Box-only (needs torch + CUDA + the lingbot_map package + the checkpoint). Read-only: builds
no engine. Example:
    cd ~/lingbot-map && source .venv/bin/activate
    python ~/gsplat-rt/scripts/lingbot_trt/runtime_split.py \
        --model_path checkpoints/lingbot-map-long.pt --lingbot-root ~/lingbot-map \
        --window-size 16 --frames 48 --iters 3 --warmup 1
"""

import argparse
import os
import sys

# Reuse the Stage-1 helper set (same directory) so model-arg construction matches exactly
# what export_probe / integrate_e2e / probe_global_blocks feed the loader.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_probe import _model_args  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 Step 1.5: coarse runtime split")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lingbot-root", default=os.path.expanduser("~/lingbot-map"))
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--iters", type=int, default=3, help="timed windowed runs (averaged)")
    ap.add_argument("--warmup", type=int, default=1, help="untimed warmup runs (cuDNN/TRT autotune)")
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

    # ---- decide which modules belong to which timing group -----------------------------
    # global_blocks / frame_blocks live *inside* the aggregator; the heads are the model's
    # other top-level children. We hook only these group-level modules (never a parent AND
    # its child), so no call is double-counted; "rest" is recovered as total − sum(groups).
    groups: dict[str, list] = {"global_blocks": [], "frame_blocks": [], "heads": []}
    for name in ("global_blocks", "frame_blocks"):
        mods = getattr(agg, name, None)
        if mods is None:
            cand = [n for n, _ in agg.named_children()]
            print(f"WARN: aggregator has no .{name}; children = {cand}")
            continue
        groups[name] = list(mods)
    head_names = []
    for name, child in model.named_children():
        if child is agg:
            continue
        groups["heads"].append(child)
        head_names.append(name)
    print(f"aggregator: {len(groups['global_blocks'])} global blocks, "
          f"{len(groups['frame_blocks'])} frame blocks")
    print(f"heads (top-level non-aggregator children): {head_names}")

    # module id -> group label, for the shared hook to attribute each call.
    label_of = {}
    for label, mods in groups.items():
        for m in mods:
            label_of[id(m)] = label

    # Per-run accumulator of (start_event, end_event) pairs, keyed by group label.
    pairs: dict[str, list] = {}

    def pre_hook(mod, inp):
        s = torch.cuda.Event(enable_timing=True)
        s.record()
        # stash the start event on the module for the matching post-hook (calls don't nest
        # within a single module, so a single slot is safe).
        mod._rt_start = s

    def post_hook(mod, inp, out):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        pairs.setdefault(label_of[id(mod)], []).append((mod._rt_start, e))

    handles = []
    for label, mods in groups.items():
        for m in mods:
            handles.append(m.register_forward_pre_hook(pre_hook))
            handles.append(m.register_forward_hook(post_hook))

    torch.manual_seed(0)
    imgs = torch.rand(args.frames, 3, args.height, args.width, device=device)

    def run_once():
        if hasattr(model, "clean_kv_cache"):
            model.clean_kv_cache()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.inference_windowed(imgs, window_size=args.window_size, overlap_size=0,
                                     num_scale_frames=args.num_scale_frames, keyframe_interval=1)

    print(f"\nwarmup ({args.warmup}) ...")
    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    # ---- timed runs --------------------------------------------------------------------
    print(f"timing ({args.iters} windowed runs, {args.frames} frames, window {args.window_size}) ...")
    totals = {k: 0.0 for k in list(groups) + ["total"]}
    for _ in range(args.iters):
        pairs.clear()
        whole_s = torch.cuda.Event(enable_timing=True)
        whole_e = torch.cuda.Event(enable_timing=True)
        whole_s.record()
        run_once()
        whole_e.record()
        torch.cuda.synchronize()
        for label, plist in pairs.items():
            totals[label] += sum(s.elapsed_time(e) for s, e in plist)
        totals["total"] += whole_s.elapsed_time(whole_e)

    n = args.iters
    total_ms = totals["total"] / n
    accounted = sum(totals[k] / n for k in groups)
    rest_ms = total_ms - accounted

    # ---- report ------------------------------------------------------------------------
    print("\n=== per-group GPU time per windowed forward (mean over "
          f"{n} runs, {args.frames} frames) ===")
    rows = [(k, totals[k] / n) for k in ("global_blocks", "frame_blocks", "heads")]
    rows.append(("rest (embed/RoPE/norm/proj/overhead)", rest_ms))
    width = max(len(k) for k, _ in rows)
    for label, ms in rows:
        pct = 100.0 * ms / total_ms if total_ms else 0.0
        print(f"  {label:<{width}}  {ms:8.2f} ms  {pct:5.1f}%")
    print(f"  {'TOTAL':<{width}}  {total_ms:8.2f} ms  100.0%")

    gb = totals["global_blocks"] / n
    gb_pct = 100.0 * gb / total_ms if total_ms else 0.0
    heads = totals["heads"] / n
    heads_pct = 100.0 * heads / total_ms if total_ms else 0.0
    print("\n=== go/no-go (design_lingbot_global_blocks_trt.md Step 1.5) ===")
    print(f"global_blocks = {gb_pct:.1f}% of runtime; heads = {heads_pct:.1f}%.")
    if gb_pct >= 40:
        print("-> global_blocks DOMINATE: the KV-cache export is worth the research cost.")
    elif heads_pct >= gb_pct:
        print("-> heads dominate global_blocks: PIVOT the export effort to the heads "
              "(static, far easier) — see the doc's expectation-management note.")
    else:
        print("-> mixed: even a perfect global_blocks engine is capped near its % share "
              "(Amdahl). Weigh that ceiling before committing to the export.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
