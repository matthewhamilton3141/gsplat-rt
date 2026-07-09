#!/usr/bin/env python3
"""Stage 1 ONNX-export probe for LingBot-Map submodules (A10G box).

Goal of Stage 1: get *one* self-contained transformer block of the aggregator out
to ONNX with numeric parity, which de-risks the whole ONNX→TensorRT toolchain on
this model before we take on the stateful KV-cache global blocks.

Why a "probe": the aggregator's global-attention blocks carry a cross-window KV
cache (dynamic control flow — hard to trace). The **frame blocks** are fixed-shape
and cacheless (they're what demo.py's `compile_model` torch.compiles), so they are
the right first target. We don't hand-build dummy inputs for an internal module —
we register a forward hook, run one aggregator forward, and **capture the real
inputs** the block is actually called with, then export exactly that.

Runs on the box inside the lingbot-map venv (needs torch + the lingbot_map package).
Nothing here is Mac-testable (no torch on the dev machine); expect to iterate on the
box output — ONNX export of a research transformer usually surfaces an unsupported
op (RoPE complex ops / SDPA) on the first pass, which is exactly what we want to see.

Example:
    cd ~/lingbot-map && source .venv/bin/activate
    python ~/gsplat-rt/scripts/lingbot_trt/export_probe.py \
        --model_path checkpoints/lingbot-map-long.pt \
        --lingbot-root ~/lingbot-map \
        --target aggregator.frame_blocks.0 \
        --window-size 16 --onnx-out /tmp/frame_block0.onnx
"""

import argparse
import os
import sys
import types


# ---------------------------------------------------------------------------
# Model construction (mirrors demo.load_model without importing demo's main)
# ---------------------------------------------------------------------------

def _model_args(a) -> types.SimpleNamespace:
    """The subset of demo.py args that load_model reads, with demo defaults."""
    return types.SimpleNamespace(
        mode="windowed",
        model_path=a.model_path,
        image_size=a.image_size,
        patch_size=a.patch_size,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        num_scale_frames=a.num_scale_frames,
        use_sdpa=True,                       # SDPA path (no flashinfer), our TRT target
        camera_num_iterations=4,
    )


def _resolve(module, dotted: str):
    """Resolve 'aggregator.frame_blocks.0' to the submodule (ints index ModuleList)."""
    obj = module
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Input capture via forward hook
# ---------------------------------------------------------------------------

class _Captured(Exception):
    """Raised inside the hook to abort the forward once inputs are grabbed."""


def _flatten_tensors(out):
    """Flatten an arbitrary block output into a tuple of tensors (drops the rest)."""
    import torch
    flat = []

    def rec(x):
        if torch.is_tensor(x):
            flat.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                rec(y)
        elif isinstance(x, dict):
            for y in x.values():
                rec(y)
    rec(out)
    return tuple(flat)


def main() -> int:
    ap = argparse.ArgumentParser(description="LingBot-Map ONNX export probe")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lingbot-root", default=os.path.expanduser("~/lingbot-map"),
                    help="path to the cloned Robbyant/lingbot-map repo (for imports)")
    ap.add_argument("--target", default="aggregator.frame_blocks.0",
                    help="dotted path to the submodule to export")
    ap.add_argument("--window-size", type=int, default=16, help="frames in the probe window")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--onnx-out", default="/tmp/lingbot_block.onnx")
    ap.add_argument("--opset", type=int, default=18)   # 18+: Split w/ num_outputs (dynamo exporter)
    ap.add_argument("--height", type=int, default=392, help="preprocessed H (canonical crop)")
    ap.add_argument("--width", type=int, default=518, help="preprocessed W")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.lingbot_root))
    import numpy as np
    import torch
    from demo import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(_model_args(args), device).eval()

    # Mirror demo.py's dtype handling so we capture the exact real code path:
    # bf16 aggregator (sm80+) run under autocast; calling the aggregator directly
    # skips inference_windowed's setup and trips a float-vs-int index error.
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (mirrors demo.py)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    target = _resolve(model, args.target)
    print(f"Target module: {args.target} = {type(target).__name__}")

    # --- capture the real inputs the target is called with ---------------------
    captured = {}

    def hook(mod, inputs, kwargs):
        captured["args"] = inputs
        captured["kwargs"] = kwargs
        raise _Captured

    handle = target.register_forward_pre_hook(hook, with_kwargs=True)

    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    imgs = torch.rand(args.window_size, 3, args.height, args.width, device=device)
    try:
        with torch.no_grad(), torch.amp.autocast(device.type, dtype=dtype):
            model.inference_windowed(
                imgs, window_size=args.window_size, overlap_size=0,
                num_scale_frames=args.num_scale_frames, keyframe_interval=1,
            )
    except _Captured:
        pass
    finally:
        handle.remove()

    if "args" not in captured:
        print("ERROR: target was never called during the aggregator forward.")
        return 2

    # Export in fp32 for a clean parity check; TensorRT does its own fp16 later.
    # Cast ONLY floating tensors — integer/index tensors (e.g. RoPE's `pos`, which
    # indexes an embedding table) must keep their dtype or F.embedding breaks.
    def _cast(t):
        return t.float() if (torch.is_tensor(t) and t.is_floating_point()) else t

    target = target.float()
    a_list = [_cast(x) for x in captured["args"]]
    kw = {k: _cast(v) for k, v in captured["kwargs"].items()}

    # Which entries are tensors → graph inputs; everything else is baked in.
    slots, tensor_inputs = [], []
    for i, v in enumerate(a_list):
        if torch.is_tensor(v):
            slots.append(("arg", i)); tensor_inputs.append(v)
    for k, v in kw.items():
        if torch.is_tensor(v):
            slots.append(("kwarg", k)); tensor_inputs.append(v)
    print(f"Captured {len(tensor_inputs)} tensor inputs:")
    for slot, t in zip(slots, tensor_inputs):
        print(f"  {slot}: shape={tuple(t.shape)} dtype={t.dtype}")

    class Wrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *tins):
            a = list(a_list)
            kwd = dict(kw)
            for slot, t in zip(slots, tins):
                if slot[0] == "arg":
                    a[slot[1]] = t
                else:
                    kwd[slot[1]] = t
            return _flatten_tensors(self.mod(*a, **kwd))

    wrapper = Wrapper(target).eval()
    with torch.no_grad():
        ref = wrapper(*tensor_inputs)
    print(f"Reference forward OK — {len(ref)} tensor output(s): "
          f"{[tuple(t.shape) for t in ref]}")

    # --- export ----------------------------------------------------------------
    in_names = [f"in{i}" for i in range(len(tensor_inputs))]
    out_names = [f"out{i}" for i in range(len(ref))]
    print(f"Exporting → {args.onnx_out} (opset {args.opset}) ...")
    try:
        torch.onnx.export(
            wrapper, tuple(tensor_inputs), args.onnx_out,
            input_names=in_names, output_names=out_names,
            opset_version=args.opset, do_constant_folding=True,
        )
    except Exception as e:
        print(f"\nONNX EXPORT FAILED: {type(e).__name__}: {e}")
        print("This is the expected first-pass outcome — the error names the op / "
              "control-flow to handle (often RoPE complex ops or SDPA). Report it back.")
        return 1
    print("Export OK.")

    # --- parity check ----------------------------------------------------------
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping parity check "
              "(export succeeded). `uv pip install onnxruntime` to enable.")
        return 0

    sess = ort.InferenceSession(args.onnx_out, providers=["CPUExecutionProvider"])
    feeds = {n: t.detach().cpu().numpy() for n, t in zip(in_names, tensor_inputs)}
    ort_out = sess.run(None, feeds)
    max_err = max(float(np.abs(r.detach().cpu().numpy() - o).max())
                  for r, o in zip(ref, ort_out))
    print(f"Parity: max abs diff (torch fp32 vs ORT) = {max_err:.3e}")
    print("PASS" if max_err < 1e-3 else "WARN: parity looser than 1e-3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
