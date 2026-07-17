#!/usr/bin/env python3
"""Stage 7: global_blocks + DPT head together — the combined whole-model number.

The capstone of the TRT study: swap the two biggest measured levers in ONE run and measure the
combined whole-model fps. Individually (48-frame TUM clip, A10G): global_blocks 45.2% → 1.069×;
DPT head 17.5% → 1.098×. Since they're independent runtime chunks, together they should compound
— this measures it (measured, not projected). Reuses the two tested integrations wholesale:
`integrate_global_e2e` (stateful blocks, dynamic cache) + `integrate_head_e2e` (static head,
dynamic frame axis). Box-only.

    python ~/gsplat-rt/scripts/lingbot_trt/integrate_combined_e2e.py \
        --model_path checkpoints/lingbot-map-long.pt --lingbot-root ~/lingbot-map --frames 48
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_probe import _model_args, _flatten_tensors           # noqa: E402
from build_and_bench_trt import build_engine                      # noqa: E402
from integrate_e2e import _load_frames, _time_inference           # noqa: E402
import integrate_global_e2e as G                                  # noqa: E402
import integrate_head_e2e as H                                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 7: global_blocks + DPT head TRT swap")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lingbot-root", default=os.path.expanduser("~/lingbot-map"))
    ap.add_argument("--engine-dir", default="/tmp/lingbot_global_engines")
    ap.add_argument("--head-engine-dir", default="/tmp/lingbot_head_engines")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--tum-dir", default=os.path.expanduser(
        "~/gsplat-rt/data/tum/rgbd_dataset_freiburg1_desk/rgb"))
    ap.add_argument("--head", default="depth_head")
    ap.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--workspace-gb", type=int, default=10)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.lingbot_root))
    import torch
    import tensorrt as trt
    from demo import load_model

    if not torch.cuda.is_available():
        print("ERROR: CUDA required (box-only).")
        return 2
    device = torch.device("cuda")
    os.makedirs(args.engine_dir, exist_ok=True)
    os.makedirs(args.head_engine_dir, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)

    model = load_model(_model_args(args), device).eval()
    if torch.cuda.get_device_capability()[0] >= 8:
        model.aggregator = model.aggregator.to(dtype=torch.bfloat16)
    global_blocks = model.aggregator.global_blocks
    head = getattr(model, args.head)
    imgs = _load_frames(args.tum_dir, args.frames, args.height, args.width, device)

    # --- ONE capture pass: hook the global blocks AND the head together ---
    print("capture pass: global blocks + head in one windowed forward ...")
    g_handles = G._capture_hooks(global_blocks, torch)
    hcap = {}

    def hpre(m, a, kw):
        if "in" not in hcap:
            hcap["in"] = (a, kw)
        a0 = a[0]
        feats = list(a0) if isinstance(a0, (list, tuple)) else [a0]
        hcap.setdefault("s_seen", []).append(int(feats[0].shape[1]))

    def hpost(m, a, kw, out):
        if "out" not in hcap:
            hcap["out"] = out

    hh = [head.register_forward_pre_hook(hpre, with_kwargs=True),
          head.register_forward_hook(hpost, with_kwargs=True)]
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.inference_windowed(imgs, window_size=args.window_size, overlap_size=0,
                                 num_scale_frames=args.num_scale_frames, keyframe_interval=1)
    for h in g_handles + hh:
        h.remove()

    captured = sorted(i for i in range(len(global_blocks)) if "out" in G._CAPTURE.get(i, {}))
    if not captured or "in" not in hcap:
        print(f"ERROR: capture incomplete (global {len(captured)} blocks, head "
              f"{'ok' if 'in' in hcap else 'MISSING'}).")
        return 2
    kmax_all = max(G._CAPTURE[i]["kmax"] for i in captured)
    print(f"global blocks with decode: {len(captured)}; cache range up to {kmax_all}; "
          f"head S range {min(hcap['s_seen'])}..{max(hcap['s_seen'])}")

    # --- baseline ---
    print("\n== baseline (bf16 aggregator + fp32 head) ==")
    base_fps, base_out = _time_inference(model, imgs, args, args.warmup, args.iters)
    print(f"baseline whole-model: {base_fps:.3f} fps")

    # --- build + swap global blocks (reuse the Stage-5 Step-3 path) ---
    print("\n== build + swap global blocks ==")
    for i in captured:
        G._export_build_global(i, global_blocks[i], args.precision, args.engine_dir,
                               args.opset, args.workspace_gb, logger, trt, torch)
        torch.cuda.empty_cache()
    runtime = trt.Runtime(logger)
    g_eng, max_dm = {}, 0
    for i in captured:
        eb = G._export_build_global(i, global_blocks[i], args.precision, args.engine_dir,
                                    args.opset, args.workspace_gb, logger, trt, torch)
        g_eng[i] = eb
        e = runtime.deserialize_cuda_engine(eb)
        max_dm = max(max_dm, getattr(e, "device_memory_size", 0))
        del e
    torch.cuda.empty_cache()
    shared = torch.empty(max(max_dm, 1), dtype=torch.uint8, device="cuda")
    g_blocks = []
    for i in captured:
        tb = G._make_trt_global_block(global_blocks[i], g_eng[i], i, G._CAPTURE[i]["kmin"],
                                      G._CAPTURE[i]["kmax"], trt, logger, torch,
                                      devmem=(shared.data_ptr(), shared.numel()))
        global_blocks[i] = tb
        g_blocks.append(tb)
    print(f"swapped {len(g_blocks)} global blocks")

    # --- build + swap the DPT head (reuse the Stage-6b path) ---
    print("\n== build + swap DPT head ==")
    s_seen = hcap["s_seen"]
    s_min, s_max = min(s_seen), max(s_seen)
    onnx_path = os.path.join(args.head_engine_dir, f"{args.head}.fp32.dyn.onnx")
    eng_path = os.path.join(args.head_engine_dir, f"{args.head}.fp16w.dyn.engine")
    in_names, out_names, n_feat, kw_order, const_kw, ex = H._export_head_onnx(
        head, hcap, onnx_path, args.opset, torch)
    profiles = {}
    for i, name in enumerate(in_names):
        shp = list(ex[i].shape)
        profiles[name] = (tuple([shp[0], s_min] + shp[2:]),
                          tuple([shp[0], s_max] + shp[2:]),
                          tuple([shp[0], s_max] + shp[2:]))
    if os.path.exists(eng_path):
        with open(eng_path, "rb") as f:
            head_eng = f.read()
        print(f"loaded cached head engine {eng_path}")
    else:
        head_eng = build_engine(onnx_path, fp16=True, bf16=False, strongly_typed=False,
                                int8=False, calibrator=None, workspace_gb=args.workspace_gb,
                                logger=logger, trt=trt, dynamic_profiles=profiles)
        with open(eng_path, "wb") as f:
            f.write(head_eng)
    trt_head = H._make_trt_head(head, head_eng, in_names, out_names, hcap["out"], s_max,
                                trt, logger, torch)
    setattr(model, args.head, trt_head)
    print("swapped DPT head")

    # --- TRT run (both swapped) ---
    print(f"\n== TRT (global blocks + head, {args.precision}) ==")
    trt_fps, trt_out = _time_inference(model, imgs, args, args.warmup, args.iters)
    g_hit = sum(b.n_trt for b in g_blocks)
    g_fb = sum(b.n_fallback for b in g_blocks)
    print(f"TRT whole-model: {trt_fps:.3f} fps  (global engine calls {g_hit}/fallbacks {g_fb}; "
          f"head calls {trt_head.n_trt}/fallbacks {trt_head.n_fallback})")

    # --- parity ---
    bt, tt = _flatten_tensors(base_out), _flatten_tensors(trt_out)
    if len(bt) == len(tt) and bt:
        b_bad = sum(int((~torch.isfinite(b)).sum()) for b in bt)
        t_bad = sum(int((~torch.isfinite(t)).sum()) for t in tt)
        n_tot = sum(b.numel() for b in bt)
        max_err, rng = 0.0, 1.0
        for b, t in zip(bt, tt):
            bf, tf = b.float(), t.float()
            m = torch.isfinite(bf) & torch.isfinite(tf)
            if m.any():
                max_err = max(max_err, float((bf[m] - tf[m]).abs().max()))
                rng = max(rng, float(bf[m].abs().max()))
        print(f"parity: baseline non-finite {b_bad}/{n_tot}, TRT non-finite {t_bad}/{n_tot}")
        print(f"parity: max abs diff over finite = {max_err:.3e}  (rel {max_err / rng:.2%})")
    else:
        print(f"parity: output structure differed ({len(bt)} vs {len(tt)})")

    speedup = trt_fps / base_fps if base_fps else float("nan")
    print(f"\n=== COMBINED whole-model speedup: {speedup:.3f}x "
          f"({base_fps:.2f} -> {trt_fps:.2f} fps) === "
          f"[global 45.2% + head 17.5% = 62.7% of runtime in TRT]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
