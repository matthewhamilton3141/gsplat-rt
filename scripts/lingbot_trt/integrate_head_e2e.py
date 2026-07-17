#!/usr/bin/env python3
"""Stage 6b: integrate the DPT depth-head TRT engine end-to-end + verify parity.

Stage 6 measured the head at **2.93× isolated** (109.9 → 37.5 ms). This swaps it into the
live model and measures (a) whole-model fps vs the bf16 baseline and (b) parity — including a
**per-head self-check on the real captured input** (engine vs the torch fp32 head), which is
the parity verification Stage 6 skipped (ORT-CPU was too slow on the ~1 GB inputs).

Much simpler than the global-block integration: the head is **static** (no KV cache, no
complex RoPE, no dynamic cache-length engine). The engine is fixed-shape; any off-shape call
(e.g. a short final chunk) falls back to the torch head. Weakly-typed fp16 → **fp32 I/O**, so
inputs/outputs need no casting (the head already runs fp32).

    python ~/gsplat-rt/scripts/lingbot_trt/integrate_head_e2e.py \
        --model_path checkpoints/lingbot-map-long.pt --lingbot-root ~/lingbot-map --frames 48
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_probe import _model_args, _flatten_tensors           # noqa: E402
from build_and_bench_trt import build_engine, _trt_to_torch       # noqa: E402
from integrate_e2e import _load_frames, _time_inference           # noqa: E402


def _capture_head(model, head, imgs, args, torch):
    """Run one windowed forward, capturing the head's real call (feature list + images +
    const kwargs) and its output structure."""
    cap = {}

    def pre(m, a, kw):
        if "in" not in cap:
            cap["in"] = (a, kw)

    def post(m, a, kw, out):
        if "out" not in cap:
            cap["out"] = out

    h1 = head.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = head.register_forward_hook(post, with_kwargs=True)
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.inference_windowed(imgs, window_size=args.window_size, overlap_size=0,
                                 num_scale_frames=args.num_scale_frames, keyframe_interval=1)
    h1.remove(); h2.remove()
    return cap


def _export_head_onnx(head, cap, onnx_path, opset, torch):
    """Export the head (fp32) as f(*feats, *tensor_kwargs) -> flat outputs. Returns
    (in_names, feat_count, tensor_kw_keys, const_kw, example_inputs)."""
    a, kw = cap["in"]
    feats = list(a[0]) if isinstance(a[0], (list, tuple)) else [a[0]]
    const_kw = {k: v for k, v in kw.items() if not torch.is_tensor(v)}
    tkw = {k: v for k, v in kw.items() if torch.is_tensor(v)}
    kw_order = list(tkw.keys())
    n_feat = len(feats)
    ex = tuple(feats) + tuple(tkw[k] for k in kw_order)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *ins):
            fs = list(ins[:n_feat])
            tk = {k: ins[n_feat + i] for i, k in enumerate(kw_order)}
            return tuple(_flatten_tensors(self.mod(fs, **tk, **const_kw)))

    wrapper = ExportWrapper(head).eval()
    with torch.no_grad():
        ref = wrapper(*ex)
    in_names = [f"feat{i}" for i in range(n_feat)] + list(kw_order)
    out_names = [f"out{i}" for i in range(len(ref))]
    print(f"exporting head ONNX ({len(ref)} outputs {[tuple(t.shape) for t in ref]}) ...")
    torch.onnx.export(wrapper, ex, onnx_path, input_names=in_names, output_names=out_names,
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    return in_names, out_names, n_feat, kw_order, const_kw, ex


def _make_trt_head(orig, engine_bytes, in_names, out_names, out_template, trt, logger, torch):
    """nn.Module replacing the head: runs the fixed-shape engine when the call matches, else
    falls back to the torch head. Weakly-typed fp16 has fp32 I/O so no casting is needed."""
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("create_execution_context returned None (GPU OOM?)")

    in_dtypes, out_dtypes, exp_shapes = {}, {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dt = _trt_to_torch(trt, engine.get_tensor_dtype(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            in_dtypes[name] = dt
            exp_shapes[name] = tuple(engine.get_tensor_shape(name))
        else:
            out_dtypes[name] = dt

    class TRTHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.orig = orig
            self.n_trt = 0
            self.n_fallback = 0

        def forward(self, *args, **kwargs):
            a0 = args[0]
            feats = list(a0) if isinstance(a0, (list, tuple)) else [a0]
            tkw = {k: v for k, v in kwargs.items() if torch.is_tensor(v)}
            # assemble inputs in the exported name order
            named = {}
            for i, f in enumerate(feats):
                named[f"feat{i}"] = f
            named.update(tkw)
            if any(n not in named or tuple(named[n].shape) != exp_shapes[n] for n in in_names):
                self.n_fallback += 1                     # off-shape (e.g. short final chunk)
                return self.orig(*args, **kwargs)

            held = []
            for name in in_names:
                t = named[name].to(in_dtypes[name]).contiguous()
                held.append(t)
                context.set_input_shape(name, tuple(t.shape))
                context.set_tensor_address(name, t.data_ptr())
            out_bufs = {n: torch.empty(tuple(context.get_tensor_shape(n)), dtype=out_dtypes[n],
                                       device="cuda") for n in out_names}
            for n in out_names:
                context.set_tensor_address(n, out_bufs[n].data_ptr())
            stream = torch.cuda.current_stream()
            context.execute_async_v3(stream_handle=stream.cuda_stream)
            self.n_trt += 1
            # rebuild the head's original output structure (tuple/list) from the engine outputs
            outs = [out_bufs[n] for n in out_names]
            from integrate_e2e import _rebuild
            return _rebuild(out_template, iter(outs))

    return TRTHead()


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 6b: end-to-end DPT-head TRT swap")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lingbot-root", default=os.path.expanduser("~/lingbot-map"))
    ap.add_argument("--engine-dir", default="/tmp/lingbot_head_engines")
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
    logger = trt.Logger(trt.Logger.WARNING)

    model = load_model(_model_args(args), device).eval()
    if getattr(model, "aggregator", None) is not None and \
            torch.cuda.get_device_capability()[0] >= 8:
        model.aggregator = model.aggregator.to(dtype=torch.bfloat16)
    head = getattr(model, args.head)

    imgs = _load_frames(args.tum_dir, args.frames, args.height, args.width, device)

    print("capture pass: grab the head's real call ...")
    cap = _capture_head(model, head, imgs, args, torch)
    if "in" not in cap:
        print("ERROR: head never called.")
        return 2
    a, kw = cap["in"]
    feats = list(a[0]) if isinstance(a[0], (list, tuple)) else [a[0]]
    print(f"head call: {len(feats)} feats {[tuple(t.shape) for t in feats]} + "
          f"{ {k: tuple(v.shape) for k, v in kw.items() if torch.is_tensor(v)} }")

    # --- export fp32 ONNX + build weakly-typed fp16 engine ---
    onnx_path = os.path.join(args.engine_dir, f"{args.head}.fp32.onnx")
    eng_path = os.path.join(args.engine_dir, f"{args.head}.fp16w.engine")
    in_names, out_names, n_feat, kw_order, const_kw, ex = _export_head_onnx(
        head, cap, onnx_path, args.opset, torch)
    if os.path.exists(eng_path):
        with open(eng_path, "rb") as f:
            engine_bytes = f.read()
        print(f"loaded cached engine {eng_path}")
    else:
        engine_bytes = build_engine(onnx_path, fp16=True, bf16=False, strongly_typed=False,
                                    int8=False, calibrator=None, workspace_gb=args.workspace_gb,
                                    logger=logger, trt=trt)
        with open(eng_path, "wb") as f:
            f.write(engine_bytes)
        print(f"engine saved -> {eng_path}")

    trt_head = _make_trt_head(head, engine_bytes, in_names, out_names, cap["out"],
                              trt, logger, torch)

    # --- per-head self-check on the REAL captured input (the parity verification) ---
    print("\n== per-head self-check: TRT fp16 engine vs torch fp32 head (captured input) ==")
    with torch.no_grad():
        ref = _flatten_tensors(trt_head.orig(*a, **kw))
        got = _flatten_tensors(trt_head(*a, **kw))
    torch.cuda.synchronize()
    worst, nonfinite = 0.0, 0
    for r, g in zip(ref, got):
        rf, gf = r.float(), g.float()
        nonfinite += int((~torch.isfinite(gf)).sum())
        m = torch.isfinite(rf) & torch.isfinite(gf)
        if m.any():
            rng = max(float(rf[m].abs().max()), 1e-6)
            worst = max(worst, float((rf[m] - gf[m]).abs().max()) / rng)
    print(f"self-check: worst rel diff = {worst:.3%}  (TRT non-finite {nonfinite}) "
          f"-> {'PARITY OK' if worst < 0.10 and nonfinite == 0 else 'INSPECT'}")

    # --- baseline, swap, TRT run, whole-model parity ---
    print("\n== baseline (bf16 aggregator + fp32 head) ==")
    base_fps, base_out = _time_inference(model, imgs, args, args.warmup, args.iters)
    print(f"baseline whole-model: {base_fps:.3f} fps")

    setattr(model, args.head, trt_head)
    print(f"\n== TRT ({args.head} fp16) ==")
    trt_fps, trt_out = _time_inference(model, imgs, args, args.warmup, args.iters)
    print(f"TRT whole-model: {trt_fps:.3f} fps  (engine calls {trt_head.n_trt}, "
          f"fallbacks {trt_head.n_fallback})")

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
    print(f"\n=== whole-model speedup: {speedup:.3f}x ({base_fps:.2f} -> {trt_fps:.2f} fps) === "
          f"[DPT head = 17.5% of runtime; head alone caps ~1.13x]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
