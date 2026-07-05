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
# Strongly-typed FP16 pair (built from the fp16 ONNX; see export_onnx.to_fp16).
DEFAULT_FP16_ONNX_PATH = os.path.join(_ROOT, "models", "depth_v2_small_fp16.onnx")
DEFAULT_FP16_ENGINE_PATH = os.path.join(_ROOT, "models", "depth_engine_fp16.engine")

# Fixed I/O shapes matching export_onnx.py
INPUT_H = 518
INPUT_W = 518
FIXED_SHAPE = (1, 3, INPUT_H, INPUT_W)


def make_network_flags(trt_module, strongly_typed: bool) -> int:
    """Compute the createNetwork flag bitmask, portable across TRT versions.

    - EXPLICIT_BATCH: required on TRT 8/9, removed (implicit default) on TRT 10+.
    - STRONGLY_TYPED (TRT 10+): the network's precision comes from the ONNX's own
      dtypes rather than builder flags — this is how true FP16 is expressed once
      the weakly-typed ``BuilderFlag.FP16`` was removed. Requires an fp16 ONNX.

    Pure/side-effect-free so it can be unit-tested with a fake ``trt`` module on
    machines without TensorRT installed.
    """
    flags = 0
    ndcf = trt_module.NetworkDefinitionCreationFlag
    if hasattr(ndcf, "EXPLICIT_BATCH"):
        flags |= 1 << int(ndcf.EXPLICIT_BATCH)
    if strongly_typed:
        if not hasattr(ndcf, "STRONGLY_TYPED"):
            raise ValueError(
                "strongly_typed=True requires TensorRT 10+ (STRONGLY_TYPED "
                f"network flag absent). Build the default engine instead.")
        flags |= 1 << int(ndcf.STRONGLY_TYPED)
    return flags


def build_engine(
    onnx_path: str = DEFAULT_ONNX_PATH,
    engine_path: str = DEFAULT_ENGINE_PATH,
    workspace_gb: int = 4,
    strongly_typed: bool = False,
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
    # Network creation flags (EXPLICIT_BATCH / STRONGLY_TYPED) are computed in a
    # version-portable, unit-testable helper.
    network_flags = make_network_flags(trt, strongly_typed)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # Workspace: TRT allocates scratch memory here for layer intermediates
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    # Precision, portable across TensorRT major versions.
    #
    # strongly_typed (TRT 10/11, from an fp16 ONNX): the network dictates its own
    #   precision — TRT runs conv/GEMM in the ONNX's declared fp16. Builder
    #   precision flags are invalid on a strongly-typed network, so we set none.
    # TRT 8/9 (weakly typed): set BuilderFlag.FP16 to let TRT run GEMM/conv on
    #   Tensor Cores in half precision, keeping precision-sensitive layers in fp32.
    # TRT 10/11 with the fp32 ONNX (default): the weakly-typed FP16 flag was
    #   removed; the engine builds fp32, which on Ampere still uses Tensor Cores
    #   via TF32. True FP16 there is the strongly_typed path above.
    if strongly_typed:
        print("[compile] Precision: FP16 (strongly-typed — from the fp16 ONNX's dtypes)")
    elif hasattr(trt.BuilderFlag, "FP16"):
        if hasattr(builder, "platform_has_fast_fp16") and not builder.platform_has_fast_fp16:
            print("[compile] WARNING: GPU does not report fast FP16; engine will be slower")
        config.set_flag(trt.BuilderFlag.FP16)
        print("[compile] Precision: FP16 (weakly-typed flag)")
    else:
        print("[compile] Precision: default fp32 (TF32 Tensor Cores on Ampere) — "
              "weakly-typed FP16 flag absent in TensorRT %s; use --fp16 for true FP16"
              % trt.__version__)

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
    ap = argparse.ArgumentParser(description="Build a TensorRT depth engine")
    ap.add_argument("--onnx", default=None,
                    help="ONNX path (default: fp16 ONNX under --fp16, else fp32)")
    ap.add_argument("--engine", default=None,
                    help="Output engine path (default depends on --fp16)")
    ap.add_argument("--workspace-gb", type=int, default=4)
    ap.add_argument("--fp16", action="store_true",
                    help="Build a true-FP16 strongly-typed engine from the fp16 "
                         "ONNX (TensorRT 10+). Run export_onnx.py --fp16 first.")
    args = ap.parse_args()

    # --fp16 flips the default onnx/engine pair to the fp16 artifacts, but an
    # explicit --onnx/--engine still wins.
    onnx_path = args.onnx or (DEFAULT_FP16_ONNX_PATH if args.fp16 else DEFAULT_ONNX_PATH)
    engine_path = args.engine or (DEFAULT_FP16_ENGINE_PATH if args.fp16 else DEFAULT_ENGINE_PATH)
    build_engine(onnx_path, engine_path, args.workspace_gb, strongly_typed=args.fp16)
