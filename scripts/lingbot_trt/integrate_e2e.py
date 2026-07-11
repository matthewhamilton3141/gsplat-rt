#!/usr/bin/env python3
"""Stage 4: end-to-end integration — swap *every* aggregator frame block for an
FP16 TensorRT engine and measure whole-model fps vs the PyTorch bf16 baseline.

Stages 1-3 proved the toolchain on ONE frame block (1.76x per block, fp16). This
closes the study honestly: the per-block win is diluted at the whole-model level
because the stateful KV-cache `global_blocks` and the DPT/camera heads stay in
PyTorch. This script produces that whole-model number instead of assuming it.

What it does (all discovered at runtime — no hand-coded shapes or block counts):
  1. Load the model exactly as demo.py does (bf16 aggregator on sm80+).
  2. One short windowed forward with hooks on every `aggregator.frame_blocks[i]`
     to capture each block's REAL inputs (the tensors + baked-in non-tensor args)
     and its output STRUCTURE (so the swap is transparent to the caller).
  3. Per block: export a true-fp16 ONNX (fp16 weights + I/O, int index tensors
     kept) from a deepcopy, then build a strongly-typed FP16 engine. Engines are
     cached to --engine-dir and reused across runs.
  4. Hot-swap each frame block for a TRTBlock wrapper (torch fallback for any
     off-profile / partial-window shape), then time inference_windowed and compare
     the full reconstruction against the baseline run (numeric parity).

Runs on the A10G box inside the lingbot-map venv (needs torch + tensorrt + the
lingbot_map package). Nothing here is Mac-testable (no torch on the dev machine).
Reuses the Stage-1/2 helpers (export_probe, build_and_bench_trt) so the proven
capture + build idioms aren't duplicated.

Example:
    cd ~/lingbot-map && source .venv/bin/activate
    python ~/gsplat-rt/scripts/lingbot_trt/integrate_e2e.py \
        --model_path checkpoints/lingbot-map-long.pt \
        --lingbot-root ~/lingbot-map \
        --engine-dir /tmp/lingbot_frame_engines \
        --window-size 16 --frames 48
"""

import argparse
import copy
import os
import sys
import time

import numpy as np

# Reuse the proven Stage-1/2 helpers (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_and_bench_trt import build_engine, _trt_to_torch          # noqa: E402
from export_probe import _model_args, _flatten_tensors               # noqa: E402


# ---------------------------------------------------------------------------
# Output-structure rebuild (the inverse of export_probe._flatten_tensors)
# ---------------------------------------------------------------------------

def _rebuild(template, flat_iter):
    """Rebuild an output with `template`'s structure, tensor leaves taken in order
    from `flat_iter` (the engine outputs). Non-tensor leaves are kept as-is."""
    import torch
    if torch.is_tensor(template):
        return next(flat_iter)
    if isinstance(template, tuple):
        return tuple(_rebuild(x, flat_iter) for x in template)
    if isinstance(template, list):
        return [_rebuild(x, flat_iter) for x in template]
    if isinstance(template, dict):
        return {k: _rebuild(v, flat_iter) for k, v in template.items()}
    return template


def _slots_and_tensors(args, kwargs):
    """Split a captured (args, kwargs) call into ordered tensor slots + values.
    Mirrors export_probe: tensors become graph inputs, everything else is baked in."""
    import torch
    slots, tensors = [], []
    for i, v in enumerate(args):
        if torch.is_tensor(v):
            slots.append(("arg", i)); tensors.append(v)
    for k, v in kwargs.items():
        if torch.is_tensor(v):
            slots.append(("kwarg", k)); tensors.append(v)
    return slots, tensors


# ---------------------------------------------------------------------------
# TRT-backed frame block (drop-in replacement, torch fallback)
# ---------------------------------------------------------------------------

def _make_trt_block(orig, engine_bytes, slots, in_shapes, max_batch, out_template,
                    trt, logger):
    """Build an nn.Module that runs `engine_bytes` (a dynamic-batch fp16 engine) for
    the frame block. The batch dim (dim 0) is dynamic up to `max_batch`; the trailing
    dims are fixed. Falls back to `orig` (the real bf16 block) only when a call's
    trailing dims differ from the captured block or its batch exceeds the profile."""
    import torch

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    in_names, out_names, in_dtypes, out_dtypes = [], [], [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dt = _trt_to_torch(trt, engine.get_tensor_dtype(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            in_names.append(name); in_dtypes.append(dt)
        else:
            out_names.append(name); out_dtypes.append(dt)
    # fixed trailing shape per input (everything but the dynamic batch dim 0)
    in_tails = [tuple(s[1:]) for s in in_shapes]
    out_tmpl_dtypes = [t.dtype for t in _flatten_tensors(out_template)]

    class TRTBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.orig = orig
            self.n_trt = 0
            self.n_fallback = 0

        def forward(self, *args, **kwargs):
            # assemble tensor inputs in the captured slot order
            try:
                tin = [args[k] if kind == "arg" else kwargs[k] for kind, k in slots]
            except (IndexError, KeyError):
                self.n_fallback += 1
                return self.orig(*args, **kwargs)
            # dynamic batch (dim 0) is fine; trailing dims must match and batch must
            # be within the built profile.
            if any(tuple(t.shape[1:]) != tail or t.shape[0] > max_batch
                   for t, tail in zip(tin, in_tails)):
                self.n_fallback += 1                      # off-profile shape
                return self.orig(*args, **kwargs)

            held = []                                     # keep cast tensors alive
            for name, dt, t in zip(in_names, in_dtypes, tin):
                x = t.to(dt).contiguous() if t.dtype != dt else t.contiguous()
                held.append(x)
                context.set_input_shape(name, tuple(x.shape))
                context.set_tensor_address(name, x.data_ptr())
            # output shapes depend on the batch we just set -> allocate now
            out_bufs = [torch.empty(tuple(context.get_tensor_shape(n)), dtype=d,
                                    device="cuda")
                        for n, d in zip(out_names, out_dtypes)]
            for n, b in zip(out_names, out_bufs):
                context.set_tensor_address(n, b.data_ptr())
            # Run on torch's CURRENT stream so the engine is ordered after the ops
            # that produced `x` and before the ops that consume the output. A private
            # stream races the model's default-stream work end-to-end (the input is
            # read before it's ready) -> garbage/NaN, precision-independent.
            stream = torch.cuda.current_stream()
            context.execute_async_v3(stream_handle=stream.cuda_stream)
            outs = [b.to(td) for b, td in zip(out_bufs, out_tmpl_dtypes)]
            self.n_trt += 1
            return _rebuild(out_template, iter(outs))

    return TRTBlock()


# ---------------------------------------------------------------------------
# Per-block ONNX export + engine build (fp16, strongly-typed)
# ---------------------------------------------------------------------------

def _export_and_build(idx, block, slots, tensors, max_batch, precision, weakly_typed,
                      engine_dir, opset, workspace_gb, logger, trt):
    """Export block `idx` to a dynamic-batch ONNX and build an engine with a
    batch-1..max_batch optimization profile. `precision` is "bf16" or "fp16".

    Two build modes:
    - strongly-typed (default): export a true-`precision` ONNX; every op runs at that
      precision. Simple, but forces LayerNorm/softmax to low precision too, which the
      model runs in fp32 under autocast -> ~14% end-to-end drift.
    - weakly-typed (weakly_typed=True): export an fp32 ONNX and set the precision as a
      builder *flag*; TRT keeps precision-sensitive layers in fp32 and matmuls in
      bf16/fp16 — mirrors the model's autocast, so parity stays tight.
    Returns engine bytes. Caches both artefacts under engine_dir."""
    import torch

    # weakly-typed keeps the ONNX in fp32 (TRT picks per-layer precision); strongly-typed
    # bakes the target dtype into the ONNX itself.
    dtype = (torch.float32 if weakly_typed else
             (torch.bfloat16 if precision == "bf16" else torch.float16))
    # tag encodes precision + mode + ".dyn" so no two caches ever collide.
    tag = f"{precision}{'w' if weakly_typed else ''}.dyn"
    onnx_path = os.path.join(engine_dir, f"frame_block{idx}.{tag}.onnx")
    eng_path = os.path.join(engine_dir, f"frame_block{idx}.{tag}.engine")
    if os.path.exists(eng_path):
        with open(eng_path, "rb") as f:
            print(f"[block {idx}] loaded cached engine {eng_path}")
            return f.read()

    # deepcopy so we never mutate the live (bf16) model — we still need it for the
    # baseline run and for the fallback path.
    blk = copy.deepcopy(block).to(dtype).eval()

    def _cast(t):
        return t.to(dtype) if (torch.is_tensor(t) and t.is_floating_point()) else t

    # rebuild the exact captured call, cast float tensors to `dtype`, keep int index
    # tensors (RoPE `pos` indexes an embedding table — casting breaks F.embedding).
    a_full = [_cast(v) for v in _CAPTURE[idx]["args"]]
    kw_full = {k: _cast(v) for k, v in _CAPTURE[idx]["kwargs"].items()}
    tin = [t.to(dtype) if t.is_floating_point() else t for t in tensors]

    class Wrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *tins):
            a, kw = list(a_full), dict(kw_full)
            for (kind, key), t in zip(slots, tins):
                if kind == "arg":
                    a[key] = t
                else:
                    kw[key] = t
            return _flatten_tensors(self.mod(*a, **kw))

    wrapper = Wrapper(blk).eval()
    in_names = [f"in{i}" for i in range(len(tin))]
    with torch.no_grad():
        n_out = len(wrapper(*tin))
    out_names = [f"out{i}" for i in range(n_out)]

    # dim 0 (frame count) varies across scale-frame / window / partial calls -> mark
    # it dynamic on every I/O so one engine covers batch 1..max_batch.
    dyn_axes = {n: {0: "batch"} for n in in_names + out_names}

    print(f"[block {idx}] exporting {tag} ONNX (dynamic batch) -> {onnx_path}")
    torch.onnx.export(wrapper, tuple(tin), onnx_path, input_names=in_names,
                      output_names=out_names, opset_version=opset,
                      do_constant_folding=True, dynamo=False, dynamic_axes=dyn_axes)
    profiles = {n: ((1, *tuple(t.shape[1:])),
                    (max_batch, *tuple(t.shape[1:])),
                    (max_batch, *tuple(t.shape[1:])))
                for n, t in zip(in_names, tin)}
    engine_bytes = build_engine(onnx_path, fp16=(precision == "fp16"),
                                bf16=(precision == "bf16"),
                                strongly_typed=not weakly_typed, int8=False,
                                calibrator=None, workspace_gb=workspace_gb,
                                logger=logger, trt=trt, dynamic_profiles=profiles)
    with open(eng_path, "wb") as f:
        f.write(engine_bytes)
    print(f"[block {idx}] engine saved -> {eng_path}")
    return engine_bytes


_CAPTURE = {}   # idx -> {"args","kwargs","out"} (module-level so Wrapper closures see it)


def _load_frames(tum_dir, n, h, w, device):
    """Load `n` real RGB frames (sorted) from a TUM `rgb/` dir as (n,3,h,w) in [0,1].
    Real images matter for parity: garbage/random input makes the DPT + camera heads
    emit NaN/Inf, so the baseline itself goes NaN and the compare is meaningless.
    Falls back to random noise (with a warning) if the dir is unusable."""
    import torch
    import glob
    files = sorted(glob.glob(os.path.join(tum_dir, "*.png")) +
                   glob.glob(os.path.join(tum_dir, "*.jpg")))
    if not files:
        print(f"WARNING: no frames in {tum_dir} — falling back to random input "
              f"(parity may be NaN). Pass --tum-dir to a TUM rgb/ folder.")
        torch.manual_seed(0)
        return torch.rand(n, 3, h, w, device=device)
    from PIL import Image
    files = (files * ((n // len(files)) + 1))[:n]         # tile if the clip is short
    arrs = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((w, h))
        arrs.append(torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0))
    print(f"loaded {n} real frames from {tum_dir}")
    return torch.stack(arrs).permute(0, 3, 1, 2).contiguous().to(device)


def _time_inference(model, imgs, args, warmup, iters):
    """Run inference_windowed `warmup`+`iters` times, return (fps, last_result)."""
    import torch
    n = imgs.shape[0]

    def one():
        if hasattr(model, "clean_kv_cache"):
            model.clean_kv_cache()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return model.inference_windowed(
                imgs, window_size=args.window_size, overlap_size=0,
                num_scale_frames=args.num_scale_frames, keyframe_interval=1)

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.time()
    result = None
    for _ in range(iters):
        result = one()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return n / dt, result


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4: end-to-end frame-block TRT swap")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lingbot-root", default=os.path.expanduser("~/lingbot-map"))
    ap.add_argument("--engine-dir", default="/tmp/lingbot_frame_engines")
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--frames", type=int, default=48,
                    help="frames in the timed clip (loaded from --tum-dir; parity "
                         "compares the baseline and TRT runs on this identical input)")
    ap.add_argument("--tum-dir",
                    default=os.path.expanduser(
                        "~/gsplat-rt/data/tum/rgbd_dataset_freiburg1_desk/rgb"),
                    help="real RGB frames for the timed/parity run (real input keeps "
                         "the heads from emitting NaN; falls back to random if missing)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--precision", choices=["bf16", "fp16"], default="bf16",
                    help="engine compute precision for matmuls. bf16 matches the "
                         "aggregator's own dtype.")
    ap.add_argument("--weakly-typed", action="store_true",
                    help="build weakly-typed (fp32 ONNX + precision as a builder flag) "
                         "so TRT keeps LayerNorm/softmax in fp32 like the model's "
                         "autocast — tight parity. Default (off) is strongly-typed, "
                         "which forces those ops to low precision (~14% drift).")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="only swap the first N frame blocks (0 = all; for quick smoke)")
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
    if getattr(model, "aggregator", None) is None:
        print("ERROR: model has no .aggregator.")
        return 2
    if torch.cuda.get_device_capability()[0] >= 8:
        model.aggregator = model.aggregator.to(dtype=torch.bfloat16)   # mirrors demo.py

    frame_blocks = model.aggregator.frame_blocks
    n_blocks = len(frame_blocks)
    print(f"aggregator has {n_blocks} frame blocks")

    # --- capture real inputs + output structure for every frame block ----------
    imgs = _load_frames(args.tum_dir, args.frames, args.height, args.width, device)

    handles = []
    for i, blk in enumerate(frame_blocks):
        def mk(idx):
            def pre(mod, a, kw):
                import torch as _t
                _CAPTURE.setdefault(idx, {})
                if "args" not in _CAPTURE[idx]:
                    _CAPTURE[idx]["args"], _CAPTURE[idx]["kwargs"] = a, kw
                fts = [v for v in list(a) + list(kw.values())
                       if _t.is_tensor(v) and v.is_floating_point()]
                # track the largest batch (dim 0) this block is ever called with, so
                # the dynamic profile's max covers scale-frame / window / partial calls.
                b = max((v.shape[0] for v in list(a) + list(kw.values())
                         if _t.is_tensor(v)), default=0)
                _CAPTURE[idx]["max_batch"] = max(_CAPTURE[idx].get("max_batch", 0), b)
                # peak input magnitude -> shows whether activations exceed fp16's 65504.
                amax = max((float(v.abs().max()) for v in fts), default=0.0)
                _CAPTURE[idx]["amax"] = max(_CAPTURE[idx].get("amax", 0.0), amax)

            def post(mod, a, kw, out):
                _CAPTURE.setdefault(idx, {})
                if "out" not in _CAPTURE[idx]:
                    _CAPTURE[idx]["out"] = out
            return pre, post
        pre, post = mk(i)
        handles.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(blk.register_forward_hook(post, with_kwargs=True))

    print("capture pass: one windowed forward to grab per-block io ...")
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.inference_windowed(imgs, window_size=args.window_size, overlap_size=0,
                                 num_scale_frames=args.num_scale_frames, keyframe_interval=1)
    for h in handles:
        h.remove()

    missing = [i for i in range(n_blocks) if "out" not in _CAPTURE.get(i, {})]
    if missing:
        print(f"ERROR: blocks never called during capture: {missing}")
        return 2
    shp0 = [tuple(t.shape) for t in _slots_and_tensors(_CAPTURE[0]["args"],
                                                       _CAPTURE[0]["kwargs"])[1]]
    max_batch = max(_CAPTURE[i]["max_batch"] for i in range(n_blocks))
    peak_amax = max(_CAPTURE[i]["amax"] for i in range(n_blocks))
    print(f"block-0 tensor input shapes: {shp0}  (max batch seen across blocks: "
          f"{max_batch})")
    mode = f"{args.precision}{'-weak' if args.weakly_typed else '-strong'}"
    print(f"peak frame-block input |activation| across blocks: {peak_amax:.1f}  "
          f"(fp16 max = 65504 -> {'OVERFLOWS fp16' if peak_amax > 65504 else 'fits fp16'}; "
          f"building {mode} engines)")

    # --- baseline (all-torch bf16) ---------------------------------------------
    print("\n== baseline (bf16 PyTorch) ==")
    base_fps, base_out = _time_inference(model, imgs, args, args.warmup, args.iters)
    print(f"baseline whole-model: {base_fps:.3f} fps")

    # --- build engines + swap every frame block --------------------------------
    n_swap = n_blocks if args.max_blocks <= 0 else min(args.max_blocks, n_blocks)
    trt_blocks = []
    for i in range(n_swap):
        slots, tensors = _slots_and_tensors(_CAPTURE[i]["args"], _CAPTURE[i]["kwargs"])
        in_shapes = [tuple(t.shape) for t in tensors]
        eng = _export_and_build(i, frame_blocks[i], slots, tensors, max_batch,
                                args.precision, args.weakly_typed, args.engine_dir,
                                args.opset, args.workspace_gb, logger, trt)
        tb = _make_trt_block(frame_blocks[i], eng, slots, in_shapes, max_batch,
                             _CAPTURE[i]["out"], trt, logger)
        frame_blocks[i] = tb
        trt_blocks.append(tb)
        # release the per-block export deepcopy + fragmentation so the next build's
        # autotuner has room (weakly-typed builds probe fp32 + bf16 tactics and OOM
        # otherwise, with the 4.4 GB model + prior engine contexts resident).
        torch.cuda.empty_cache()
    print(f"\nswapped {n_swap}/{n_blocks} frame blocks -> TRT {args.precision}")

    # --- per-block self-check: engine vs orig on the block's OWN captured input --
    # Synced, no cross-stream race and no per-window input variation -> isolates
    # "is each engine numerically correct" from end-to-end swap/interaction issues.
    print("\n== per-block self-check (engine vs orig on captured input) ==")
    worst, worst_i, sc_nonfinite = 0.0, -1, 0
    for i, tb in enumerate(trt_blocks):
        a, kw = _CAPTURE[i]["args"], _CAPTURE[i]["kwargs"]
        # the reference must run under the same autocast the baseline used — the block
        # is mixed-precision (bf16 matmuls, fp32 LayerNorm/softmax), not pure bf16.
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ref = _flatten_tensors(tb.orig(*a, **kw))
            got = _flatten_tensors(tb(*a, **kw))
        torch.cuda.synchronize()
        for r, g in zip(ref, got):
            rf, gf = r.float(), g.float()
            sc_nonfinite += int((~torch.isfinite(gf)).sum())
            m = torch.isfinite(rf) & torch.isfinite(gf)
            if m.any():
                d = float((rf[m] - gf[m]).abs().max())
                if d > worst:
                    worst, worst_i = d, i
    print(f"self-check: worst per-block max abs diff = {worst:.3e} (block {worst_i}), "
          f"TRT non-finite = {sc_nonfinite}")

    # --- TRT run ---------------------------------------------------------------
    print(f"\n== TRT (frame blocks {args.precision}) ==")
    trt_fps, trt_out = _time_inference(model, imgs, args, args.warmup, args.iters)
    hit = sum(b.n_trt for b in trt_blocks)
    fb = sum(b.n_fallback for b in trt_blocks)
    print(f"TRT whole-model: {trt_fps:.3f} fps   "
          f"(engine calls {hit}, torch fallbacks {fb})")

    # --- parity: compare the two full reconstructions --------------------------
    # NaN-aware: report each side's non-finite counts (tells us whether a NaN comes
    # from the baseline itself vs the fp16 TRT path) and measure the diff over the
    # elements that are finite on BOTH sides.
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
        print(f"parity: baseline non-finite {b_bad}/{n_tot}, TRT non-finite "
              f"{t_bad}/{n_tot}")
        print(f"parity: max abs diff over finite elems = {max_err:.3e}  "
              f"(rel {max_err / rng:.2%})")
    else:
        print(f"parity: output structure differed ({len(bt)} vs {len(tt)} tensors)")

    speedup = trt_fps / base_fps if base_fps else float("nan")
    print(f"\n=== whole-model speedup: {speedup:.3f}x "
          f"({base_fps:.2f} -> {trt_fps:.2f} fps) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
