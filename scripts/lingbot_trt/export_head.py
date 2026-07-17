#!/usr/bin/env python3
"""Stage 6: export the DPT depth head to a functional ONNX + torch baseline (box-only).

The heads are ~17.5% of runtime and, unlike the stateful `global_blocks`, are **static**
feed-forward — the easiest remaining TRT target and one that should translate *better* than
the global blocks did (no dynamic cache-length engine penalty). `DPTHead` is the heavy one
(dense depth 392×518); the camera head is tiny (negligible runtime). This does the isolated
piece: capture the head's real call via a hook, torch-time it (the fp32 baseline — the heads
run fp32 under the aggregator's autocast), and export a functional ONNX. Then
`build_and_bench_trt.py --onnx <out> --strongly-typed` builds + benches the engine.

Two-phase like `export_global_block.py`: Phase 1 (discovery: shapes + torch bench) always
completes; Phase 2 (ONNX export) is separate so a wrapper bug can't erase the discovery.

    python ~/gsplat-rt/scripts/lingbot_trt/export_head.py \
        --model_path checkpoints/lingbot-map-long.pt --lingbot-root ~/lingbot-map \
        --precision fp16 --bench-torch 100 --onnx-out /tmp/dpt_head.fp16.onnx
"""

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_probe import _model_args, _flatten_tensors           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Export the DPT depth head to ONNX")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lingbot-root", default=os.path.expanduser("~/lingbot-map"))
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--head", default="depth_head", help="model attribute of the head module")
    ap.add_argument("--onnx-out", default="/tmp/dpt_head.onnx")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--bench-torch", type=int, default=0, metavar="ITERS")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.lingbot_root))
    import torch
    from demo import load_model

    if not torch.cuda.is_available():
        print("ERROR: CUDA required (box-only).")
        return 2
    device = torch.device("cuda")
    model = load_model(_model_args(args), device).eval()
    if torch.cuda.get_device_capability()[0] >= 8:
        model.aggregator = model.aggregator.to(dtype=torch.bfloat16)   # heads stay fp32
    head = getattr(model, args.head, None)
    if head is None:
        print(f"ERROR: model has no attribute '{args.head}'")
        return 2

    # --- capture the head's real call during one windowed run ---
    cap = {}

    def pre(m, a, kw):
        cap.setdefault("in", (a, kw))

    def post(m, a, kw, out):
        cap.setdefault("out", out)

    h1 = head.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = head.register_forward_hook(post, with_kwargs=True)
    torch.manual_seed(0)
    imgs = torch.rand(args.frames, 3, args.height, args.width, device=device)
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.inference_windowed(imgs, window_size=args.window_size, overlap_size=0,
                                 num_scale_frames=args.num_scale_frames, keyframe_interval=1)
    h1.remove(); h2.remove()

    if "in" not in cap:
        print("ERROR: head was never called during the windowed run.")
        return 2

    print("=" * 78)
    print(f"PHASE 1 — DISCOVERY: {args.head} = {type(head).__name__}")
    print("=" * 78)
    a, kw = cap["in"]
    # positional arg 0 is the aggregated-tokens list; images/patch_start_idx are kwargs.
    feats = list(a[0]) if isinstance(a[0], (list, tuple)) else [a[0]]
    const_kw = {k: v for k, v in kw.items() if not torch.is_tensor(v)}
    tensor_kw = {k: v for k, v in kw.items() if torch.is_tensor(v)}
    print("  feature list:", [tuple(t.shape) for t in feats],
          "dtypes", [str(t.dtype).replace("torch.", "") for t in feats])
    print("  tensor kwargs:", {k: (tuple(v.shape), str(v.dtype).replace("torch.", ""))
                               for k, v in tensor_kw.items()})
    print("  const kwargs:", const_kw)
    out = cap.get("out")
    print("  output:", [tuple(t.shape) for t in _flatten_tensors(out)],
          f"(raw type {type(out).__name__})")

    # --- torch baseline (fp32 head on the captured input) ---
    if args.bench_torch:
        n = args.bench_torch
        with torch.no_grad():
            for _ in range(max(5, n // 4)):
                head(feats, **kw)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(n):
                s.record()
                head(feats, **kw)
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
        ts = np.array(ts)
        print(f"\n[torch] {args.head} fwd (fp32) over {n}: median {np.median(ts):.3f} ms | "
              f"mean {ts.mean():.3f} ms | p95 {np.percentile(ts, 95):.3f} ms")

    if args.no_export:
        return 0

    # --- PHASE 2: functional export ---
    print("\n" + "=" * 78)
    print(f"PHASE 2 — EXPORT ({args.precision})")
    print("=" * 78)
    export_dt = torch.float16 if args.precision == "fp16" else torch.float32
    head2 = copy.deepcopy(head).to(export_dt).eval()
    feats_dt = [t.to(export_dt) for t in feats]
    tkw_dt = {k: v.to(export_dt) for k, v in tensor_kw.items()}
    n_feat = len(feats_dt)
    kw_order = list(tensor_kw.keys())            # tensor kwargs, in a fixed order
    tensor_ins = tuple(feats_dt) + tuple(tkw_dt[k] for k in kw_order)

    class Wrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *ins):
            feats_in = list(ins[:n_feat])
            tkw_in = {k: ins[n_feat + i] for i, k in enumerate(kw_order)}
            out = self.mod(feats_in, **tkw_in, **const_kw)
            return tuple(_flatten_tensors(out))

    wrapper = Wrapper(head2).eval()
    with torch.no_grad():
        ref = wrapper(*tensor_ins)
    print(f"reference forward OK — {len(ref)} outputs: {[tuple(t.shape) for t in ref]}")

    in_names = [f"feat{i}" for i in range(n_feat)] + list(kw_order)
    out_names = ["depth", "conf"][:len(ref)] + [f"out{i}" for i in range(2, len(ref))]
    print(f"exporting -> {args.onnx_out} (opset {args.opset}) ...")
    try:
        torch.onnx.export(wrapper, tensor_ins, args.onnx_out, input_names=in_names,
                          output_names=out_names, opset_version=args.opset,
                          do_constant_folding=True, dynamo=False)
    except Exception as ex:
        print(f"\nONNX EXPORT FAILED: {type(ex).__name__}: {ex}")
        import traceback
        traceback.print_exc()
        return 1
    print(f"EXPORT OK -> {args.onnx_out}")
    print("Next: build_and_bench_trt.py --onnx <this> --strongly-typed  (fp16 engine + bench)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
