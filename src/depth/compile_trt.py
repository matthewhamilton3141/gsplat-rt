"""Compile the Depth Anything V2 ONNX model into a TensorRT FP16 engine.

Requirements: tensorrt>=9.0.0, CUDA toolkit, NVIDIA GPU.

Usage:
    python src/depth/compile_trt.py
    python src/depth/compile_trt.py --onnx models/depth_v2_small.onnx \
                                    --engine models/depth_engine.engine \
                                    --workspace-gb 4

Build time is typically 2-5 minutes the first time (TRT performs layer
profiling). The resulting .engine file is GPU-architecture-specific and
must be regenerated if you move to a different GPU family.
"""

import argparse
import os
import time

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DEFAULT_ONNX_PATH = os.path.join(_ROOT, "models", "depth_v2_small.onnx")
DEFAULT_ENGINE_PATH = os.path.join(_ROOT, "models", "depth_engine.engine")

# Fixed I/O shapes matching export_onnx.py
INPUT_H = 518
INPUT_W = 518
FIXED_SHAPE = (1, 3, INPUT_H, INPUT_W)


def build_engine(
    onnx_path: str = DEFAULT_ONNX_PATH,
    engine_path: str = DEFAULT_ENGINE_PATH,
    workspace_gb: int = 4,
) -> None:
    try:
        import tensorrt as trt
    except ImportError:
        raise SystemExit(
            "TensorRT is not installed. "
            "pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com"
        )

    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}\n"
            "Run: python src/depth/export_onnx.py"
        )

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    print(f"[compile] TensorRT {trt.__version__}")
    print(f"[compile] ONNX   : {onnx_path}")
    print(f"[compile] Engine : {engine_path}")
    print(f"[compile] Workspace: {workspace_gb} GiB")

    builder = trt.Builder(logger)
    # EXPLICIT_BATCH was removed in TensorRT 10 (explicit batch is the only mode,
    # flags=0). On TRT 8/9 the flag is still required. Guard so this builds on
    # whichever version the NGC index resolves to.
    network_flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # Workspace: TRT allocates scratch memory here for layer intermediates
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    # Precision, portable across TensorRT major versions.
    #
    # TRT 8/9 (weakly typed): set BuilderFlag.FP16 to let TRT run GEMM/conv on
    #   Tensor Cores in half precision, keeping precision-sensitive layers in fp32.
    # TRT 10/11: the weakly-typed FP16/INT8 flags were removed — precision now
    #   comes from strongly-typed networks (the ONNX's own dtypes). With our fp32
    #   ONNX the engine builds in fp32, which on Ampere still uses Tensor Cores via
    #   TF32 (enabled by default). True FP16 there means an fp16 ONNX + a
    #   STRONGLY_TYPED network — a follow-up if TF32 misses the latency budget.
    if hasattr(trt.BuilderFlag, "FP16"):
        if hasattr(builder, "platform_has_fast_fp16") and not builder.platform_has_fast_fp16:
            print("[compile] WARNING: GPU does not report fast FP16; engine will be slower")
        config.set_flag(trt.BuilderFlag.FP16)
        print("[compile] Precision: FP16")
    else:
        print("[compile] Precision: default fp32 (TF32 Tensor Cores on Ampere) — "
              "weakly-typed FP16 flag absent in TensorRT %s" % trt.__version__)

    with open(onnx_path, "rb") as f:
        raw = f.read()
    if not parser.parse(raw):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errors}")

    # Single fixed-shape optimization profile (no dynamic batching)
    profile = builder.create_optimization_profile()
    inp_name = network.get_input(0).name
    profile.set_shape(inp_name, FIXED_SHAPE, FIXED_SHAPE, FIXED_SHAPE)
    config.add_optimization_profile(profile)

    print("[compile] Building engine (this may take several minutes) …")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None — check GPU memory and logs")
    elapsed = time.time() - t0

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f"[compile] Done in {elapsed:.1f}s — {engine_path}  ({size_mb:.1f} MB)")

    # Cleanup: Python GC must release these before the logger is destroyed
    del parser, network, config, builder


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build TensorRT FP16 depth engine")
    ap.add_argument("--onnx", default=DEFAULT_ONNX_PATH)
    ap.add_argument("--engine", default=DEFAULT_ENGINE_PATH)
    ap.add_argument("--workspace-gb", type=int, default=4)
    args = ap.parse_args()
    build_engine(args.onnx, args.engine, args.workspace_gb)
