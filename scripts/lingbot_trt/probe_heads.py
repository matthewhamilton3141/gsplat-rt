#!/usr/bin/env python3
"""Discover the DPT depth + camera head structure/IO — scoping for a heads→TRT export.

The heads are ~17.5% of runtime (RESULTS.md runtime split) and, unlike the stateful
`global_blocks`, they are **static** feed-forward modules → the easiest remaining TRT win.
Before exporting anything we discover — never guess — their `forward` signature, real
captured IO (shapes/dtypes), and output structure, via forward hooks during one real
windowed run. Box-only. Prints the material a `export_head.py` / integration would need.

    python ~/gsplat-rt/scripts/lingbot_trt/probe_heads.py \
        --model_path checkpoints/lingbot-map-long.pt --lingbot-root ~/lingbot-map --frames 16
"""

import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_probe import _model_args, _flatten_tensors           # noqa: E402

_HEAD_KEYS = ("head", "dpt", "depth", "camera", "pose", "predict")


def _desc(v, torch):
    if torch.is_tensor(v):
        return f"T{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}"
    if v is None:
        return "None"
    if isinstance(v, (int, float, bool, str)):
        return f"{type(v).__name__}({v})"
    if isinstance(v, dict):
        return f"dict{{{list(v)[:6]}}}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{len(v)}]"
    return type(v).__name__


def main() -> int:
    ap = argparse.ArgumentParser(description="Discover DPT/camera head structure + IO")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lingbot-root", default=os.path.expanduser("~/lingbot-map"))
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.lingbot_root))
    import torch
    from demo import load_model

    if not torch.cuda.is_available():
        print("ERROR: CUDA required (box-only).")
        return 2
    device = torch.device("cuda")
    model = load_model(_model_args(args), device).eval()

    print("top-level children:", [n for n, _ in model.named_children()])

    # shallowest modules whose name segment matches a head keyword (skip if an ancestor matched)
    heads = {}
    for name, mod in model.named_modules():
        if not name:
            continue
        segs = name.split(".")
        if any(any(k in s.lower() for k in _HEAD_KEYS) for s in segs):
            if any(name != h and name.startswith(h + ".") for h in heads):
                continue                              # an ancestor is already a head
            heads[name] = mod
    print("head candidates:", {n: type(m).__name__ for n, m in heads.items()})
    if not heads:
        print("no head-like modules found; inspect top-level children above.")
        return 2

    cap = {}
    handles = []
    for name, mod in heads.items():
        def mk(name):
            def pre(m, a, kw):
                cap.setdefault(name, {})
                cap[name].setdefault("in", (a, kw))
            def post(m, a, kw, out):
                cap.setdefault(name, {})["out"] = out
            return pre, post
        pre, post = mk(name)
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(mod.register_forward_hook(post, with_kwargs=True))

    torch.manual_seed(0)
    imgs = torch.rand(args.frames, 3, args.height, args.width, device=device)
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.inference_windowed(imgs, window_size=args.window_size, overlap_size=0,
                                 num_scale_frames=args.num_scale_frames, keyframe_interval=1)
    for h in handles:
        h.remove()

    for name, mod in heads.items():
        print("\n" + "=" * 78)
        print(f"=== {name}  ({type(mod).__name__}) ===")
        if name not in cap or "in" not in cap[name]:
            print("  NOT CALLED during the windowed run")
            continue
        a, kw = cap[name]["in"]
        print("  args:  ", [_desc(x, torch) for x in a])
        print("  kwargs:", {k: _desc(v, torch) for k, v in kw.items()})
        out = cap[name].get("out")
        print("  out:   ", [_desc(t, torch) for t in _flatten_tensors(out)],
              f"(raw type {type(out).__name__})")
        try:
            src = inspect.getsource(type(mod).forward)
            print("  forward source (first 1800 chars):\n", src[:1800])
        except (OSError, TypeError) as e:
            print(f"  (no forward source: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
