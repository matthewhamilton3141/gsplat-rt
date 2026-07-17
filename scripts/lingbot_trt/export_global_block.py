#!/usr/bin/env python3
"""Stage 5, Step 2 — export ONE aggregator `global_block` (functional KV cache) to ONNX.

The runtime split (Step 1.5) said GO: `global_blocks` are 45.2% of the windowed forward,
the single dominant cost. The probe (Step 1) said the KV cache is *functional* (passed as a
kwarg dict, not mutating module state) — the clean export branch — and flagged the export
snag: `pos` is `complex128` RoPE, and TensorRT/ONNX have no complex dtype.

Three things the frame-block toolchain (`export_probe.py`) never had to handle, and why this
is its own script:
  1. **The `kv_cache` dict** — a kwarg, so `export_probe` bakes it as a *constant*. For a
     functional export it must become dynamic tensor I/O: this block's cache slot goes in as
     inputs and the *updated* slot comes out as outputs.
  2. **`complex128` `pos`** — no ONNX/TRT complex dtype. Handled at the graph boundary as two
     real tensors (real, imag) recombined inside the wrapper; the RoPE op *inside* the block
     is still complex and is the expected first-pass export failure — which is the point.
  3. **Output = out + updated cache** — the block returns/writes new k/v; we must discover the
     exact signature, not guess it.

Because we can't write a correct wrapper until we've *read* the real `forward`, this script is
deliberately two-phase and Phase 1 always finishes first:

  PHASE 1 (discovery, always runs, saves before Phase 2 can crash):
    - print `inspect.getsource(type(block).forward)` + the class's source file path
    - the captured call's full (args, kwargs) structure (tensors vs baked scalars/None)
    - the `kv_cache` dict: every key -> shape/dtype
    - which cache keys CHANGE across the call (before vs after) == the slot this block writes
    - the output structure, and whether any output tensor IS an updated-cache tensor
    - every complex-dtype tensor (the RoPE inputs to refactor)
    - dump the real captured inputs + outputs to .npz (for later TRT parity)

  PHASE 2 (export attempt, wrapped — a bug here cannot lose Phase 1):
    - build a functional wrapper: f(tokens, cache_slot..., pos_real, pos_imag) -> (out, new_k, ...)
    - torch reference forward, then classic-TorchScript ONNX export
    - on failure, name the unsupported op (expect the complex RoPE) — that names the refactor

Box-only (torch + CUDA + lingbot_map + checkpoint). Nothing Mac-testable. Example:
    cd ~/lingbot-map && source .venv/bin/activate
    python ~/gsplat-rt/scripts/lingbot_trt/export_global_block.py \
        --model_path checkpoints/lingbot-map-long.pt --lingbot-root ~/lingbot-map \
        --index 0 --capture-call 1 --onnx-out /tmp/global_block0.onnx \
        --dump-io /tmp/global_block0_io.npz
"""

import argparse
import inspect
import os
import sys

# Reuse the Stage-1 helpers (same directory): model-arg construction + the tensor-flatten
# used everywhere else, so this stays consistent with export_probe / integrate_e2e.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_probe import _flatten_tensors, _model_args  # noqa: E402


class _Captured(Exception):
    """Raised inside the post-hook to abort the forward once we have the target call."""


def _snapshot_cache(kv, torch):
    """Clone the tensor entries of a kv_cache dict -> {name: (shape, dtype, cpu_clone)}."""
    if kv is None or not isinstance(kv, dict):
        return {}
    snap = {}
    for name, v in kv.items():
        if torch.is_tensor(v):
            snap[name] = (tuple(v.shape), str(v.dtype), v.detach().to("cpu", copy=True))
    return snap


def _changed_keys(before, after, torch):
    """Cache keys new or modified across the call = the slot this block writes."""
    changed = []
    for name, (shp, dt, tens) in after.items():
        if name not in before:
            changed.append(name)
            continue
        bshp, bdt, btens = before[name]
        if bshp != shp or bdt != dt or not torch.equal(btens.float(), tens.float()):
            changed.append(name)
    return changed


def _describe(v, torch):
    if torch.is_tensor(v):
        return f"T{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}"
    if v is None:
        return "None"
    if isinstance(v, (int, float, bool, str)):
        return f"{type(v).__name__}({v})"
    if isinstance(v, dict):
        return f"dict{{{len(v)} keys}}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{len(v)}]"
    return type(v).__name__


def _to_numpy(t, torch):
    """numpy() cast that survives dtypes numpy can't hold natively.

    The v-cache is bfloat16 (no numpy equivalent) -> up-cast to float32 first, which is
    what we compare against for parity anyway. complex128 (RoPE `pos`) numpy handles fine.
    """
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 Step 2: export one global_block")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lingbot-root", default=os.path.expanduser("~/lingbot-map"))
    ap.add_argument("--index", type=int, default=0, help="which global_blocks[i] to export")
    ap.add_argument("--capture-call", type=int, default=1,
                    help="which call of THIS block to export (0=prefill 8-frame, 1=first decode)")
    ap.add_argument("--window-size", type=int, default=16)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--height", type=int, default=392)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--onnx-out", default="/tmp/global_block.onnx")
    ap.add_argument("--dump-io", default=None, help="save captured inputs+outputs to this .npz")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--no-export", action="store_true", help="Phase 1 discovery only, skip export")
    ap.add_argument("--dynamic-cache", action="store_true",
                    help="mark the cache history dim (dim 2) dynamic; default static so "
                         "build_and_bench_trt.py's fixed-shape path benches it directly")
    ap.add_argument("--bench-torch", type=int, default=0, metavar="ITERS",
                    help="time the original torch block (bf16 autocast, complex RoPE) on the "
                         "captured inputs = the production baseline the TRT engine replaces")
    ap.add_argument("--half", choices=["fp16", "bf16"], default=None,
                    help="export a TRUE half-precision ONNX (activations+weights+cache in this "
                         "dtype, pos stays fp32) so build_and_bench_trt.py --strongly-typed drops "
                         "the fp32 I/O boundary casts; default fp32 ONNX")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.lingbot_root))
    import numpy as np
    import torch
    from demo import load_model

    if not torch.cuda.is_available():
        print("ERROR: CUDA required (box-only).")
        return 2
    device = torch.device("cuda")

    model = load_model(_model_args(args), device).eval()
    agg = getattr(model, "aggregator", None)
    if agg is None or getattr(agg, "global_blocks", None) is None:
        print("ERROR: model.aggregator.global_blocks not found.")
        return 2
    if torch.cuda.get_device_capability()[0] >= 8:
        model.aggregator = model.aggregator.to(dtype=torch.bfloat16)   # mirrors demo.py
        agg = model.aggregator
    block = agg.global_blocks[args.index]

    # ============================ PHASE 1: DISCOVERY ================================
    print("=" * 78)
    print(f"PHASE 1 — DISCOVERY: aggregator.global_blocks[{args.index}] "
          f"= {type(block).__name__}")
    print("=" * 78)
    try:
        src_file = inspect.getsourcefile(type(block))
        print(f"class source file: {src_file}")
        print("\n--- type(block).forward source ---")
        print(inspect.getsource(type(block).forward))
    except (OSError, TypeError) as e:
        print(f"(could not read forward source: {e})")

    # Capture the target call (a decode call by default = the steady-state we export).
    counter = {"n": 0}
    cap = {}

    def pre(mod, a, kw):
        if counter["n"] == args.capture_call:
            cap["args"], cap["kwargs"] = a, kw
            cap["cache_before"] = _snapshot_cache(kw.get("kv_cache"), torch)

    def post(mod, a, kw, out):
        if counter["n"] == args.capture_call:
            cap["out"] = out
            cap["cache_after"] = _snapshot_cache(kw.get("kv_cache"), torch)
            raise _Captured
        counter["n"] += 1

    h1 = block.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = block.register_forward_hook(post, with_kwargs=True)
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    torch.manual_seed(0)
    imgs = torch.rand(args.frames, 3, args.height, args.width, device=device)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.inference_windowed(imgs, window_size=args.window_size, overlap_size=0,
                                     num_scale_frames=args.num_scale_frames, keyframe_interval=1)
    except _Captured:
        pass
    finally:
        h1.remove(); h2.remove()

    if "args" not in cap:
        print(f"ERROR: block was called < {args.capture_call + 1} times; lower --capture-call.")
        return 2

    print("\n--- captured call structure ---")
    print(f"args:   {[_describe(v, torch) for v in cap['args']]}")
    print(f"kwargs: {{{', '.join(f'{k}: {_describe(v, torch)}' for k, v in cap['kwargs'].items())}}}")

    kv = cap["kwargs"].get("kv_cache")
    print("\n--- kv_cache dict ---")
    if isinstance(kv, dict):
        print(f"{len(kv)} entries. per-key shape/dtype:")
        for name, v in kv.items():
            print(f"  {name}: {_describe(v, torch)}")
    else:
        print(f"kv_cache is not a dict: {_describe(kv, torch)}  (revisit the wrapper below)")

    changed = _changed_keys(cap["cache_before"], cap["cache_after"], torch)
    print("\n--- cache keys this block WRITES (changed before->after) ---")
    for name in changed:
        b = cap["cache_before"].get(name)
        a = cap["cache_after"][name]
        print(f"  {name}: {b[0] if b else 'NEW'} -> {a[0]}  ({a[1]})")
    if not changed:
        print("  (none changed — cache may be returned in the output, not written in place)")

    print("\n--- output structure ---")
    out_flat = _flatten_tensors(cap["out"])
    print(f"raw output type: {type(cap['out']).__name__}; "
          f"{len(out_flat)} tensor leaf/leaves: {[tuple(t.shape) for t in out_flat]}")

    complex_inputs = [(_slot_name(i, k), v) for i, (k, v) in enumerate(_iter_call(cap, torch))
                      if torch.is_tensor(v) and v.is_complex()]
    print("\n--- complex-dtype tensors (RoPE — must go real for ONNX/TRT) ---")
    for name, v in complex_inputs:
        print(f"  {name}: {tuple(v.shape)} {v.dtype}")
    if not complex_inputs:
        print("  (none — good; the complex-RoPE snag may not apply to this call)")

    if args.dump_io:
        dump = {}
        for i, (k, v) in enumerate(_iter_call(cap, torch)):
            if torch.is_tensor(v):
                dump[f"in__{_slot_name(i, k)}"] = _to_numpy(v, torch)
        for i, t in enumerate(out_flat):
            dump[f"out__{i}"] = _to_numpy(t, torch)
        for name in changed:
            dump[f"newcache__{name}"] = _to_numpy(cap["cache_after"][name][2], torch)
        np.savez(args.dump_io, **dump)
        print(f"\ndumped captured I/O -> {args.dump_io} ({len(dump)} arrays)")

    print("\nPHASE 1 complete. If PHASE 2 fails, the above is what we need to finish the export.")

    if args.bench_torch:
        _bench_torch_block(block, cap, args.bench_torch, np, torch)

    if args.no_export:
        return 0

    # ============================ PHASE 2: EXPORT ATTEMPT ===========================
    # NOTE: this wrapper encodes assumptions the Phase-1 dump must confirm — chiefly that
    # the block reads/writes only its own slot (keyed by `global_idx`) inside the passed
    # kv_cache dict, and returns its main activation as the output. Adjust against the
    # printed forward source. The export is EXPECTED to surface the complex RoPE op on the
    # first pass — that error names the exact refactor (real cos/sin) for the next pass.
    print("\n" + "=" * 78)
    print("PHASE 2 — FUNCTIONAL EXPORT ATTEMPT")
    print("=" * 78)
    try:
        rc = _attempt_export(block, cap, changed, args, torch)
    except Exception as e:  # never let a wrapper bug erase Phase 1
        print(f"\nPHASE 2 wrapper/export raised {type(e).__name__}: {e}")
        print("Phase 1 discovery above is intact — use it to correct the wrapper.")
        return 1
    return rc


def _iter_call(cap, torch):
    """Yield (key, value) over the captured call: positional args as ints, kwargs as names."""
    for i, v in enumerate(cap["args"]):
        yield (i, v)
    for k, v in cap["kwargs"].items():
        yield (k, v)


def _slot_name(i, k):
    return f"arg{k}" if isinstance(k, int) else str(k)


def _bench_torch_block(block, cap, iters, np, torch):
    """Time the ORIGINAL torch block (complex RoPE, bf16 autocast) on the captured inputs.

    This is the production baseline the TRT engine replaces — same block, same shapes, same
    dtype path as the live windowed inference. Uses `_skip_append` so the cache doesn't grow
    across iterations (compute is equivalent to the keyframe path minus the store/evict).
    """
    device = next(block.parameters()).device
    x = cap["args"][0]
    base_kw = dict(cap["kwargs"])
    cb = cap["cache_before"]

    def make_kv():
        kv = {name: t.to(device) for name, (_, _, t) in cb.items()}
        kv["_skip_append"] = True
        return kv

    def call():
        kw = dict(base_kw)
        kw["kv_cache"] = make_kv()          # built OUTSIDE the timed region below
        return kw

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(max(10, iters // 4)):
            block(x, **call())
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            kw = call()                      # cache build excluded from timing
            torch.cuda.synchronize()
            s.record()
            block(x, **kw)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    ts = np.array(ts)
    print(f"\n[torch] block fwd (bf16 autocast, complex RoPE) over {iters} runs: "
          f"median {np.median(ts):.3f} ms | mean {ts.mean():.3f} ms | "
          f"p95 {np.percentile(ts, 95):.3f} ms")


def _report_parity(label, a, b, np):
    """Print max/mean abs + relative diff and NaN counts for two same-shape arrays."""
    if a.shape != b.shape:
        print(f"  parity[{label}]: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    diff = np.abs(a64 - b64)
    rel = diff / (np.abs(b64) + 1e-6)
    print(f"  parity[{label}]: max|Δ|={np.nanmax(diff):.3e} mean|Δ|={np.nanmean(diff):.3e} "
          f"maxrel={np.nanmax(rel):.3e}  nan(a/b)={int(np.isnan(a).sum())}/{int(np.isnan(b).sum())}")


def _attempt_export(block, cap, changed, args, torch):
    """Functional float32 export of one SDPA global_block, complex RoPE removed.

    Phase-1 discovery pinned the exact contract:
      * `pos` (complex128 freqs_cis) is consumed ONLY by apply_rotary_emb, which needs just
        cos/sin (= pos.real / pos.imag). We feed it as a real [..., 2] tensor and swap in a
        real-input apply_rotary_emb -> no complex op survives in the graph.
      * The block reads/writes only its own slot k_{idx}/v_{idx}; here it appends the current
        frame (dim-2 8->9). We feed the PRE-write cache (cache_before) so the graph's cat
        produces the grown cache as an explicit output.
    Exported in float32 for a portable ONNX; TRT still builds fp16/bf16 from it.
    """
    import numpy as np
    import lingbot_map.layers.attention as _attn_mod

    # Export dtype: default fp32 (portable ONNX, TRT builds fp16/bf16 from it with a builder
    # flag = fp32 I/O + boundary casts). --half fp16|bf16 emits a TRUE half ONNX so a
    # --strongly-typed engine drops those I/O casts — the per-block lever. pos stays fp32
    # (RoPE cos/sin are precision-sensitive; only a couple of small tensors).
    export_dt = {None: torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.half]

    device = next(block.parameters()).device
    block = block.to(dtype=export_dt).eval()
    idx = int(cap["kwargs"].get("global_idx", args.index))
    kkey, vkey = f"k_{idx}", f"v_{idx}"

    cb = cap["cache_before"]
    if kkey not in cb:
        print(f"ERROR: {kkey} not in cache_before ({list(cb)[:6]}...); cannot form cache input.")
        return 2

    # graph inputs on device (pre-write cache = 8 frames). Activations/cache in export_dt;
    # pos kept fp32 for RoPE stability.
    x_in = cap["args"][0].to(device=device, dtype=export_dt)
    pos = cap["kwargs"]["pos"]
    pos_real = torch.stack([pos.real, pos.imag], dim=-1).to(device=device, dtype=torch.float32)
    k_in = cb[kkey][2].to(device=device, dtype=export_dt)
    v_in = cb[vkey][2].to(device=device, dtype=export_dt)
    tensor_ins = (x_in, pos_real, k_in, v_in)
    print(f"export dtype = {export_dt}")

    # baked scalar kwargs (everything that isn't a tensor or the kv_cache dict)
    const_kw = {kk: vv for kk, vv in cap["kwargs"].items()
                if not (torch.is_tensor(vv) or isinstance(vv, dict))}

    def _rope_real(t, freqs):  # real-arithmetic RoPE; freqs = [..., D//2, 2] (cos, sin)
        cos = freqs[..., 0].to(t.dtype)
        sin = freqs[..., 1].to(t.dtype)
        t1, t2 = t[..., 0::2], t[..., 1::2]
        return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).reshape(t.shape)

    class Wrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, x, pos_r, k_cur, v_cur):
            kv = {kkey: k_cur, vkey: v_cur,
                  f"{kkey}_special": None, f"{vkey}_special": None}
            out = self.mod(x, pos=pos_r, kv_cache=kv, **const_kw)
            outs = list(_flatten_tensors(out))
            return tuple(outs) + (kv[kkey], kv[vkey])

    wrapper = Wrapper(block).eval()
    _orig_are = _attn_mod.apply_rotary_emb
    _attn_mod.apply_rotary_emb = _rope_real
    try:
        with torch.no_grad():
            ref = wrapper(*tensor_ins)
        print(f"reference forward OK — {len(ref)} outputs: {[tuple(t.shape) for t in ref]}")

        if cap.get("out") is not None:
            _report_parity("block-out fp32-wrapper vs bf16-capture",
                           ref[0].float().cpu().numpy(),
                           _to_numpy(_flatten_tensors(cap["out"])[0], torch), np)

        n_main = len(ref) - 2
        out_names = ([f"out{i}" for i in range(n_main)] if n_main != 1 else ["out"]) \
            + [f"{kkey}_out", f"{vkey}_out"]
        in_names = ["x", "pos", f"{kkey}_in", f"{vkey}_in"]
        dyn = {f"{kkey}_in": {2: "hist"}, f"{vkey}_in": {2: "hist"},
               f"{kkey}_out": {2: "hist1"}, f"{vkey}_out": {2: "hist1"}} \
            if args.dynamic_cache else None
        print(f"exporting -> {args.onnx_out} (opset {args.opset}) ...")
        try:
            torch.onnx.export(
                wrapper, tensor_ins, args.onnx_out,
                input_names=in_names, output_names=out_names, dynamic_axes=dyn,
                opset_version=args.opset, do_constant_folding=True, dynamo=False,
            )
        except Exception as e:
            print(f"\nONNX EXPORT FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print("The op it names is the next thing to make real/traceable. Report it back.")
            return 1
    finally:
        _attn_mod.apply_rotary_emb = _orig_are

    io_path = os.path.splitext(args.onnx_out)[0] + "_io.npz"
    np.savez(io_path,
             x=_to_numpy(x_in, torch), pos=_to_numpy(pos_real, torch),
             k_in=_to_numpy(k_in, torch), v_in=_to_numpy(v_in, torch),
             out=_to_numpy(ref[0], torch),
             k_out=_to_numpy(ref[-2], torch), v_out=_to_numpy(ref[-1], torch))
    print(f"EXPORT OK -> {args.onnx_out}  (+ real I/O {io_path})")
    print("Next: build_and_bench_trt.py (fp32 + fp16), NaN-aware parity vs this npz, then "
          "integrate all 24 blocks (integrate_e2e.py pattern).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
