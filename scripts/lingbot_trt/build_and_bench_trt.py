#!/usr/bin/env python3
"""Stage 2: build an FP16 TensorRT engine from an exported ONNX and benchmark it.

Takes the parity-checked ONNX from Stage 1 (`export_probe.py`), builds a TensorRT
engine (weakly-typed FP16 — TRT runs the GEMMs/attention in fp16 from an fp32 ONNX),
measures its latency with CUDA events, and checks numeric parity against onnxruntime.
Reuses the project's box-proven TRT idioms (src/depth/{compile_trt,depth_estimator}):
torch CUDA tensors as I/O buffers via set_tensor_address + execute_async_v3, no pycuda.

Runs on the box. Needs `tensorrt` in the env:  uv pip install "tensorrt>=10,<11"

Example:
    python ~/gsplat-rt/scripts/lingbot_trt/build_and_bench_trt.py \
        --onnx /tmp/frame_block0.onnx --engine-out /tmp/frame_block0.fp16.engine
"""

import argparse
import time

import numpy as np


def _trt_to_torch(trt, dt):
    import torch
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT64: torch.int64,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.INT8: torch.int8,
    }[dt]


def build_engine(onnx_path: str, fp16: bool, workspace_gb: int, logger, trt) -> bytes:
    builder = trt.Builder(logger)
    flags = 0
    ndcf = trt.NetworkDefinitionCreationFlag
    if hasattr(ndcf, "EXPLICIT_BATCH"):            # required TRT<10, harmless flag name
        flags |= 1 << int(ndcf.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
        print("[trt] precision: FP16 (weakly-typed flag; internals in fp16)")
    else:
        print("[trt] precision: FP32/TF32")

    print("[trt] building serialized engine (first build can take a minute)...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None — check GPU mem / logs")
    print(f"[trt] built in {time.time() - t0:.1f}s")
    return bytes(serialized)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build + benchmark a TensorRT engine")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine-out", default=None, help="save the serialized engine here")
    ap.add_argument("--no-fp16", action="store_true", help="build FP32/TF32 instead of FP16")
    ap.add_argument("--workspace-gb", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--no-parity", action="store_true", help="skip the onnxruntime check")
    args = ap.parse_args()

    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA required.")
        return 2

    logger = trt.Logger(trt.Logger.WARNING)
    engine_bytes = build_engine(args.onnx, not args.no_fp16, args.workspace_gb, logger, trt)
    if args.engine_out:
        with open(args.engine_out, "wb") as f:
            f.write(engine_bytes)
        print(f"[trt] engine saved → {args.engine_out}")

    # --- set up execution: torch CUDA tensors as I/O buffers -------------------
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    buffers, inputs, outputs = {}, [], []
    rng = np.random.default_rng(0)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_in = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        shape = tuple(engine.get_tensor_shape(name))
        dt = _trt_to_torch(trt, engine.get_tensor_dtype(name))
        if is_in:
            if dt.is_floating_point:
                t = torch.randn(shape, dtype=dt, device="cuda")
            else:                                   # index/bool inputs: zeros are safe
                t = torch.zeros(shape, dtype=dt, device="cuda")
            context.set_input_shape(name, shape)
            inputs.append(name)
        else:
            t = torch.empty(shape, dtype=dt, device="cuda")
            outputs.append(name)
        buffers[name] = t
        context.set_tensor_address(name, t.data_ptr())
        print(f"[trt] {'in ' if is_in else 'out'} {name}: {shape} {dt}")

    def _run():
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

    # --- benchmark -------------------------------------------------------------
    for _ in range(args.warmup):
        _run()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iters):
        starter.record(stream)
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream_handle=stream.cuda_stream)
        ender.record(stream)
        stream.synchronize()
        times.append(starter.elapsed_time(ender))    # ms
    times = np.array(times)
    print(f"\n[trt] latency over {args.iters} runs: "
          f"median {np.median(times):.3f} ms | mean {times.mean():.3f} ms | "
          f"p95 {np.percentile(times, 95):.3f} ms")

    # --- parity vs onnxruntime -------------------------------------------------
    if not args.no_parity:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[parity] onnxruntime not installed — skipping.")
            return 0
        _run()                                       # ensure outputs populated
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        feeds = {n: buffers[n].detach().cpu().numpy() for n in inputs}
        ort_out = sess.run(outputs, feeds)
        max_err = max(float(np.abs(buffers[o].detach().cpu().numpy() - r).max())
                      for o, r in zip(outputs, ort_out))
        print(f"[parity] max abs diff (TRT fp16 vs ORT fp32) = {max_err:.3e}  "
              f"({'ok for fp16' if max_err < 5e-2 else 'HIGH — inspect'})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
