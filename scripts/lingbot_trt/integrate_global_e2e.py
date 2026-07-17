#!/usr/bin/env python3
"""Stage 5 Step 3: end-to-end `global_blocks` TRT swap + whole-model measurement.

Turns the per-block win from `export_global_block.py` (strongly-typed fp16 = 1.53x on one
block) into a *measured* whole-model number — the honest capstone of the LingBot study, and
the direct test of the Stage-4 lesson that per-block wins can evaporate on integration.

What's different from Stage 4 (`integrate_e2e.py`, frame blocks):
  * The global block is `SDPAAttention`: it attends over a **growing KV cache** carried in a
    Python dict, and applies **complex-RoPE** (`pos` is complex128). So the engine is
    functional — `f(x, pos_real, k_in, v_in) -> (out, k_out, v_out)` — with:
      - `pos` split to a real [..., 2] (cos, sin) tensor + a real-input `apply_rotary_emb`
        (monkeypatched only during export tracing; the runtime engine bakes it in);
      - a **dynamic cache-length** optimization profile (the cache frame axis, dim 2, grows
        across decode calls) rather than Stage 4's dynamic batch;
      - the cache concat done *in-graph*, returned as explicit outputs and written back into
        the dict by the wrapper.
  * The **prefill / scale-frame call** (`num_frames > 1`, multi-frame, empty-cache branch) is
    left in PyTorch (torch fallback) — only the steady-state decode path is an engine.

Regime validity: the exported engine bakes the decode keyframe path with **no eviction** (the
sliding-window/scale-frame eviction is a no-op until the cache exceeds ~sliding_window+scale
frames). For the default 48-frame clip the cache stays under that, so it's correct; the run
asserts the observed max cache length stayed under the eviction threshold and warns otherwise.

Box-only (CUDA + TensorRT + the lingbot-map checkpoint). Example:
    python ~/gsplat-rt/scripts/lingbot_trt/integrate_global_e2e.py \
        --model_path checkpoints/lingbot-map-long.pt --lingbot-root ~/lingbot-map \
        --frames 48 --precision fp16
"""

import argparse
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_probe import _flatten_tensors, _model_args           # noqa: E402
from build_and_bench_trt import build_engine, _trt_to_torch       # noqa: E402
from integrate_e2e import _load_frames, _time_inference           # noqa: E402


# Per-block capture: idx -> {args, kwargs, k_in, v_in (pre-write clones), kmin, kmax, out}
_CAPTURE = {}


def _install_real_rope(torch):
    """Swap `attention.apply_rotary_emb` for a real-input variant (used during export only).

    The original already runs real arithmetic and only reads freqs.real/.imag; this variant
    accepts either the complex freqs (baseline/fallback) or the real [..., 2] (cos, sin) tensor
    we feed the engine, so no complex op enters the traced graph. Returns the original to
    restore afterwards. `attention` imports the symbol by value, so we patch that module.
    """
    import lingbot_map.layers.attention as attn

    orig = attn.apply_rotary_emb

    def _real(t, freqs):
        if freqs.is_complex():
            cos = freqs.real.to(t.dtype)
            sin = freqs.imag.to(t.dtype)
        else:
            cos = freqs[..., 0].to(t.dtype)
            sin = freqs[..., 1].to(t.dtype)
        t1, t2 = t[..., 0::2], t[..., 1::2]
        return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).reshape(t.shape)

    attn.apply_rotary_emb = _real
    return orig


def _real_pos(pos, torch, dtype):
    """Complex freqs_cis -> real [..., 2] (cos, sin) graph input."""
    return torch.stack([pos.real, pos.imag], dim=-1).to(dtype)


def _export_build_global(idx, block, precision, engine_dir, opset, workspace_gb,
                         logger, trt, torch):
    """Export global block `idx` (decode path) to a functional ONNX with a dynamic cache-length
    profile and build a strongly-typed `precision` engine. Returns engine bytes (cached)."""
    cap = _CAPTURE[idx]
    export_dt = torch.float16 if precision == "fp16" else torch.bfloat16
    tag = f"{precision}.gdyn"
    onnx_path = os.path.join(engine_dir, f"global_block{idx}.{tag}.onnx")
    eng_path = os.path.join(engine_dir, f"global_block{idx}.{tag}.engine")
    if os.path.exists(eng_path):
        with open(eng_path, "rb") as f:
            print(f"[gblock {idx}] loaded cached engine {eng_path}")
            return f.read()

    blk = copy.deepcopy(block).to(export_dt).eval()
    device = next(blk.parameters()).device
    kkey, vkey = f"k_{idx}", f"v_{idx}"

    x_in = cap["args"][0].to(device=device, dtype=export_dt)
    pos_real = _real_pos(cap["kwargs"]["pos"], torch, torch.float32).to(device)
    k_in = cap["k_in"].to(device=device, dtype=export_dt)
    v_in = cap["v_in"].to(device=device, dtype=export_dt)
    tensor_ins = (x_in, pos_real, k_in, v_in)
    const_kw = {kk: vv for kk, vv in cap["kwargs"].items()
                if not (torch.is_tensor(vv) or isinstance(vv, dict))}

    class Wrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, x, pos_r, k_cur, v_cur):
            kv = {kkey: k_cur, vkey: v_cur, f"{kkey}_special": None, f"{vkey}_special": None}
            out = self.mod(x, pos=pos_r, kv_cache=kv, **const_kw)
            outs = list(_flatten_tensors(out))
            return tuple(outs) + (kv[kkey], kv[vkey])

    wrapper = Wrapper(blk).eval()
    in_names = ["x", "pos", f"{kkey}_in", f"{vkey}_in"]
    out_names = ["out", f"{kkey}_out", f"{vkey}_out"]
    # cache history axis (dim 2) is dynamic on the k/v in and out.
    dyn = {f"{kkey}_in": {2: "hist"}, f"{vkey}_in": {2: "hist"},
           f"{kkey}_out": {2: "hist1"}, f"{vkey}_out": {2: "hist1"}}

    orig_rope = _install_real_rope(torch)
    try:
        with torch.no_grad():
            n_out = len(wrapper(*tensor_ins))
        assert n_out == 3, f"expected (out,k,v), got {n_out} outputs"
        print(f"[gblock {idx}] exporting {tag} ONNX (dyn cache {cap['kmin']}..{cap['kmax']}) "
              f"-> {onnx_path}")
        torch.onnx.export(wrapper, tensor_ins, onnx_path, input_names=in_names,
                          output_names=out_names, dynamic_axes=dyn, opset_version=opset,
                          do_constant_folding=True, dynamo=False)
    finally:
        import lingbot_map.layers.attention as attn
        attn.apply_rotary_emb = orig_rope

    # profile: x, pos fixed; k/v range over the observed decode cache length.
    kmn, kmx = cap["kmin"], cap["kmax"]
    kop = int(cap["k_in"].shape[2])
    def _kv_shape(frames):
        s = list(k_in.shape); s[2] = frames; return tuple(s)
    profiles = {
        "x": (tuple(x_in.shape),) * 3,
        "pos": (tuple(pos_real.shape),) * 3,
        f"{kkey}_in": (_kv_shape(kmn), _kv_shape(kop), _kv_shape(kmx)),
        f"{vkey}_in": (_kv_shape(kmn), _kv_shape(kop), _kv_shape(kmx)),
    }
    engine_bytes = build_engine(onnx_path, fp16=(precision == "fp16"),
                                bf16=(precision == "bf16"), strongly_typed=True, int8=False,
                                calibrator=None, workspace_gb=workspace_gb, logger=logger,
                                trt=trt, dynamic_profiles=profiles)
    with open(eng_path, "wb") as f:
        f.write(engine_bytes)
    print(f"[gblock {idx}] engine saved -> {eng_path}")
    return engine_bytes


def _make_trt_global_block(orig, engine_bytes, idx, kmin, kmax, trt, logger, torch, devmem):
    """nn.Module replacing global_blocks[idx]. Runs the engine for steady-state decode calls
    (cache present, num_frames==1, cache length in [kmin,kmax]); falls back to the real torch
    block for the prefill/scale-frame call and any off-profile shape."""
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if devmem is not None:
        ptr, size = devmem
        try:
            context = engine.create_execution_context_without_device_memory()
        except Exception:
            context = engine.create_execution_context(
                trt.ExecutionContextAllocationStrategy.USER_MANAGED)
        try:
            context.set_device_memory(ptr, size)
        except TypeError:
            context.set_device_memory(ptr)
    else:
        context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("create_execution_context returned None (GPU OOM?)")

    in_names, out_names, in_dtypes, out_dtypes = [], [], [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dt = _trt_to_torch(trt, engine.get_tensor_dtype(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            in_names.append(name); in_dtypes.append(dt)
        else:
            out_names.append(name); out_dtypes.append(dt)
    kkey, vkey = f"k_{idx}", f"v_{idx}"

    class TRTGlobalBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.orig = orig
            self.n_trt = 0
            self.n_fallback = 0

        def forward(self, *args, **kwargs):
            kv = kwargs.get("kv_cache")
            nf = kwargs.get("num_frames")
            pos = kwargs.get("pos")
            k_cur = kv.get(kkey) if isinstance(kv, dict) else None
            # steady-state decode only; everything else -> the real block.
            if (nf is None or nf > 1 or pos is None or k_cur is None
                    or k_cur.shape[2] < kmin or k_cur.shape[2] > kmax):
                self.n_fallback += 1
                return self.orig(*args, **kwargs)

            x = args[0]
            v_cur = kv[vkey]
            k_dt, v_dt = k_cur.dtype, v_cur.dtype   # torch keeps k=fp32, v=bf16
            # inputs in the exported order: x, pos_real, k_in, v_in
            tin = [x, _real_pos(pos, torch, torch.float32), k_cur, v_cur]
            held = []
            for name, dt, t in zip(in_names, in_dtypes, tin):
                xt = t.to(dt).contiguous() if t.dtype != dt else t.contiguous()
                held.append(xt)
                context.set_input_shape(name, tuple(xt.shape))
                context.set_tensor_address(name, xt.data_ptr())
            out_bufs = [torch.empty(tuple(context.get_tensor_shape(n)), dtype=d, device="cuda")
                        for n, d in zip(out_names, out_dtypes)]
            for n, b in zip(out_names, out_bufs):
                context.set_tensor_address(n, b.data_ptr())
            stream = torch.cuda.current_stream()
            context.execute_async_v3(stream_handle=stream.cuda_stream)
            # out_names order = [out, k_out, v_out]; write the grown cache back in the dict's
            # native dtype so any later torch-fallback (prefill) cat sees the expected type.
            res = {n: b for n, b in zip(out_names, out_bufs)}
            kv[kkey] = res[f"{kkey}_out"].to(k_dt)
            kv[vkey] = res[f"{vkey}_out"].to(v_dt)
            self.n_trt += 1
            return res["out"].to(x.dtype)

    return TRTGlobalBlock()


def _capture_hooks(global_blocks, torch):
    """Register pre/post hooks that grab the first steady-state decode call per block plus the
    min/max cache length seen (for the profile). Returns the handle list to remove later."""
    handles = []
    for i, blk in enumerate(global_blocks):
        def mk(i):
            def pre(mod, a, kw):
                gi = int(kw.get("global_idx", i))
                kv = kw.get("kv_cache")
                nf = kw.get("num_frames")
                if not isinstance(kv, dict):
                    return
                kcur = kv.get(f"k_{gi}")
                if nf != 1 or kcur is None:      # only steady-state decode calls
                    return
                rec = _CAPTURE.setdefault(gi, {})
                frames = int(kcur.shape[2])
                rec["kmin"] = min(rec.get("kmin", 10 ** 9), frames)
                rec["kmax"] = max(rec.get("kmax", 0), frames)
                if "args" not in rec:            # first decode = the export template
                    rec["args"], rec["kwargs"] = a, kw
                    rec["k_in"] = kcur.detach().clone()
                    rec["v_in"] = kv[f"v_{gi}"].detach().clone()
            def post(mod, a, kw, out):
                gi = int(kw.get("global_idx", i))
                if gi in _CAPTURE and "out" not in _CAPTURE[gi] and "args" in _CAPTURE[gi]:
                    _CAPTURE[gi]["out"] = out
            return pre, post
        pre, post = mk(i)
        handles.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(blk.register_forward_hook(post, with_kwargs=True))
    return handles


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 Step 3: end-to-end global_block TRT swap")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lingbot-root", default=os.path.expanduser("~/lingbot-map"))
    ap.add_argument("--engine-dir", default="/tmp/lingbot_global_engines")
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--tum-dir", default=os.path.expanduser(
        "~/gsplat-rt/data/tum/rgbd_dataset_freiburg1_desk/rgb"))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--precision", choices=["fp16", "bf16"], default="fp16",
                    help="engine precision (strongly-typed). fp16 was the per-block winner.")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=0, help="swap only first N (0=all)")
    ap.add_argument("--evict-threshold", type=int, default=72,
                    help="cache frames above which the baked no-eviction engine is invalid "
                         "(sliding_window + scale_frames); the run warns if kmax exceeds it")
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
    if getattr(model, "aggregator", None) is None or \
            getattr(model.aggregator, "global_blocks", None) is None:
        print("ERROR: model.aggregator.global_blocks not found.")
        return 2
    if torch.cuda.get_device_capability()[0] >= 8:
        model.aggregator = model.aggregator.to(dtype=torch.bfloat16)

    global_blocks = model.aggregator.global_blocks
    n_blocks = len(global_blocks)
    print(f"aggregator has {n_blocks} global blocks")

    imgs = _load_frames(args.tum_dir, args.frames, args.height, args.width, device)

    # --- capture pass ---
    handles = _capture_hooks(global_blocks, torch)
    print("capture pass: one windowed forward to grab per-block decode io + cache range ...")
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.inference_windowed(imgs, window_size=args.window_size, overlap_size=0,
                                 num_scale_frames=args.num_scale_frames, keyframe_interval=1)
    for h in handles:
        h.remove()

    captured = sorted(i for i in range(n_blocks) if "out" in _CAPTURE.get(i, {}))
    missing = [i for i in range(n_blocks) if i not in captured]
    if missing:
        print(f"WARNING: no steady-state decode captured for blocks {missing} — they will "
              f"stay in torch (only blocks with a decode call get an engine).")
    if not captured:
        print("ERROR: no global block saw a steady-state decode call; nothing to swap.")
        return 2
    kmax_all = max(_CAPTURE[i]["kmax"] for i in captured)
    kmin_all = min(_CAPTURE[i]["kmin"] for i in captured)
    print(f"decode cache length range across blocks: {kmin_all}..{kmax_all} frames")
    if kmax_all > args.evict_threshold:
        print(f"⚠ WARNING: max cache {kmax_all} > eviction threshold {args.evict_threshold}; "
              f"the baked no-eviction engine may be INCORRECT for the longest calls. Use fewer "
              f"--frames or fold eviction into the wrapper before trusting parity.")
    else:
        print(f"✓ max cache {kmax_all} <= {args.evict_threshold}: no-eviction regime, engine valid.")

    # --- baseline ---
    print("\n== baseline (bf16 PyTorch) ==")
    base_fps, base_out = _time_inference(model, imgs, args, args.warmup, args.iters)
    print(f"baseline whole-model: {base_fps:.3f} fps")

    n_swap = len(captured) if args.max_blocks <= 0 else min(args.max_blocks, len(captured))
    swap_ids = captured[:n_swap]

    # --- phase 1: build + cache every engine (no contexts resident) ---
    print("\n== phase 1: build + cache every global-block engine ==")
    for i in swap_ids:
        _export_build_global(i, global_blocks[i], args.precision, args.engine_dir,
                             args.opset, args.workspace_gb, logger, trt, torch)
        torch.cuda.empty_cache()

    # --- phase 2: load engines, size one shared device-memory scratch ---
    print("\n== phase 2: load engines + shared device memory ==")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng_bytes, max_dm = {}, 0
    for i in swap_ids:
        eb = _export_build_global(i, global_blocks[i], args.precision, args.engine_dir,
                                 args.opset, args.workspace_gb, logger, trt, torch)  # cache hit
        eng_bytes[i] = eb
        e = runtime.deserialize_cuda_engine(eb)
        max_dm = max(max_dm, getattr(e, "device_memory_size", 0))
        del e
    torch.cuda.empty_cache()
    shared = torch.empty(max(max_dm, 1), dtype=torch.uint8, device="cuda")
    print(f"shared TRT device memory: {max_dm / 1e6:.0f} MB (one buffer for {n_swap} blocks)")

    # --- swap ---
    trt_blocks = []
    for i in swap_ids:
        tb = _make_trt_global_block(global_blocks[i], eng_bytes[i], i,
                                    _CAPTURE[i]["kmin"], _CAPTURE[i]["kmax"], trt, logger, torch,
                                    devmem=(shared.data_ptr(), shared.numel()))
        global_blocks[i] = tb
        trt_blocks.append((i, tb))
    print(f"\nswapped {n_swap}/{n_blocks} global blocks -> TRT {args.precision}")

    # --- per-block self-check: engine vs orig on the captured decode input (copied cache) ---
    print("\n== per-block self-check (engine vs orig on captured decode input) ==")
    worst, worst_i, sc_nonfinite = 0.0, -1, 0
    for i, tb in trt_blocks:
        a, kw = _CAPTURE[i]["args"], _CAPTURE[i]["kwargs"]
        # copy the kv dict so orig's in-place cache write doesn't corrupt the shared state,
        # and rebuild this block's slot from the pre-write clone so both see identical input.
        def fresh_kw():
            k2 = dict(kw)
            kv2 = dict(kw["kv_cache"])
            kv2[f"k_{i}"] = _CAPTURE[i]["k_in"].to(device)
            kv2[f"v_{i}"] = _CAPTURE[i]["v_in"].to(device)
            k2["kv_cache"] = kv2
            return k2
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ref = _flatten_tensors(tb.orig(*a, **fresh_kw()))
            got = _flatten_tensors(tb(*a, **fresh_kw()))
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

    # --- TRT run ---
    print(f"\n== TRT (global blocks {args.precision}) ==")
    trt_fps, trt_out = _time_inference(model, imgs, args, args.warmup, args.iters)
    hit = sum(tb.n_trt for _, tb in trt_blocks)
    fb = sum(tb.n_fallback for _, tb in trt_blocks)
    print(f"TRT whole-model: {trt_fps:.3f} fps   (engine calls {hit}, torch fallbacks {fb})")

    # --- parity: full reconstruction, NaN-aware ---
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
        print(f"parity: max abs diff over finite elems = {max_err:.3e}  (rel {max_err / rng:.2%})")
    else:
        print(f"parity: output structure differed ({len(bt)} vs {len(tt)} tensors)")

    speedup = trt_fps / base_fps if base_fps else float("nan")
    print(f"\n=== whole-model speedup: {speedup:.3f}x "
          f"({base_fps:.2f} -> {trt_fps:.2f} fps) === "
          f"[global_blocks are 45.2% of runtime; Amdahl ceiling 1.82x]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
