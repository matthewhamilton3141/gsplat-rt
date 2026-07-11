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
                dump[f"in__{_slot_name(i, k)}"] = v.detach().cpu().numpy()
        for i, t in enumerate(out_flat):
            dump[f"out__{i}"] = t.detach().cpu().numpy()
        for name in changed:
            dump[f"newcache__{name}"] = cap["cache_after"][name][2].numpy()
        np.savez(args.dump_io, **dump)
        print(f"\ndumped captured I/O -> {args.dump_io} ({len(dump)} arrays)")

    print("\nPHASE 1 complete. If PHASE 2 fails, the above is what we need to finish the export.")
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


def _attempt_export(block, cap, changed, args, torch):
    """Build the functional wrapper and export it. Kept separate so Phase 1 is safe."""
    # Graph inputs = every tensor in the call EXCEPT the kv_cache dict (handled specially),
    # with complex tensors split into (real, imag) real inputs at the boundary.
    in_specs = []          # (key, is_complex)
    tensor_ins = []        # the real torch tensors fed to export
    for k, v in _iter_call(cap, torch):
        if not torch.is_tensor(v):
            continue
        if v.is_complex():
            in_specs.append((k, True))
            tensor_ins.append(torch.view_as_real(v)[..., 0].contiguous())  # real part
            tensor_ins.append(torch.view_as_real(v)[..., 1].contiguous())  # imag part
        else:
            in_specs.append((k, False))
            tensor_ins.append(v)

    # This block's cache slot: pass the changed (written) keys IN as current-cache inputs so
    # the concat/grow happens against real tensors; return them OUT as the updated cache.
    cache_keys = list(changed)
    kv = cap["kwargs"].get("kv_cache")
    for name in cache_keys:
        tensor_ins.append(kv[name])

    baked_args = list(cap["args"])
    baked_kwargs = dict(cap["kwargs"])

    class Wrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *tins):
            it = iter(tins)
            a = list(baked_args)
            kwd = dict(baked_kwargs)
            for (key, is_cplx) in in_specs:
                if is_cplx:
                    re, im = next(it), next(it)
                    val = torch.complex(re.to(torch.float64), im.to(torch.float64))
                else:
                    val = next(it)
                if isinstance(key, int):
                    a[key] = val
                else:
                    kwd[key] = val
            # rebuild the kv_cache with this block's slot taken from the inputs
            new_kv = dict(kv) if isinstance(kv, dict) else {}
            for name in cache_keys:
                new_kv[name] = next(it)
            if isinstance(kv, dict):
                kwd["kv_cache"] = new_kv
            out = self.mod(*a, **kwd)
            outs = list(_flatten_tensors(out))
            # append the updated cache tensors as explicit outputs (functional contract)
            for name in cache_keys:
                if name in new_kv and torch.is_tensor(new_kv[name]):
                    outs.append(new_kv[name])
            return tuple(outs)

    wrapper = Wrapper(block).eval()
    with torch.no_grad():
        ref = wrapper(*tensor_ins)
    print(f"reference forward OK — {len(ref)} output tensor(s): {[tuple(t.shape) for t in ref]}")

    in_names = [f"in{i}" for i in range(len(tensor_ins))]
    out_names = [f"out{i}" for i in range(len(ref))]
    print(f"exporting -> {args.onnx_out} (opset {args.opset}, classic TorchScript) ...")
    try:
        torch.onnx.export(
            wrapper, tuple(tensor_ins), args.onnx_out,
            input_names=in_names, output_names=out_names,
            opset_version=args.opset, do_constant_folding=True, dynamo=False,
        )
    except Exception as e:
        print(f"\nONNX EXPORT FAILED: {type(e).__name__}: {e}")
        print("Expected first-pass outcome — the op it names (likely the complex RoPE / "
              "SDPA) is the exact thing to refactor to real ops. Report it back.")
        return 1
    print("EXPORT OK. Next: build TRT (build_and_bench_trt.py) + NaN-aware parity vs the "
          "dumped I/O, then integrate all 24 blocks (integrate_e2e.py pattern).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
