"""Benchmark: measure TensorRT depth inference latency.

Pass conditions (matches 15ms pipeline budget):
  - Mean latency < 15ms
  - P99  latency < 20ms  (allows for occasional DRAM pressure spikes)

Run:
    pytest tests/test_depth_inference.py -v -s
    # or directly:
    python tests/test_depth_inference.py

Skips gracefully when the compiled engine or CUDA is absent so that
CI on non-GPU machines still passes the overall test suite.
"""

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ENGINE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "depth_engine.engine")
WARMUP_RUNS = 100
BENCH_RUNS = 500
BUDGET_MEAN_MS = 15.0
BUDGET_P99_MS = 20.0


# ---------------------------------------------------------------------------
# Fixtures / skip guards
# ---------------------------------------------------------------------------

def _skip_if_no_engine():
    if not os.path.exists(ENGINE_PATH):
        pytest.skip(
            f"Engine not built: {ENGINE_PATH}\n"
            "  1. python src/depth/export_onnx.py\n"
            "  2. python src/depth/compile_trt.py"
        )


def _skip_if_no_cuda():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available on this machine")
    except ImportError:
        pytest.skip("PyTorch not installed")


def _skip_if_no_trt():
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        pytest.skip("TensorRT not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Random BGR uint8 frame, seeded for reproducibility."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _report(latencies: np.ndarray, depth_shape: tuple) -> float:
    mean_ms = latencies.mean()
    p50_ms = np.percentile(latencies, 50)
    p95_ms = np.percentile(latencies, 95)
    p99_ms = np.percentile(latencies, 99)
    min_ms = latencies.min()
    max_ms = latencies.max()

    print(f"\n{'─'*40}")
    print(f"  Depth Inference Latency ({len(latencies)} runs)")
    print(f"{'─'*40}")
    print(f"  Output shape : {depth_shape}")
    print(f"  Min          : {min_ms:.2f} ms")
    print(f"  Mean         : {mean_ms:.2f} ms   budget={BUDGET_MEAN_MS}ms  {'✓ PASS' if mean_ms < BUDGET_MEAN_MS else '✗ FAIL'}")
    print(f"  P50          : {p50_ms:.2f} ms")
    print(f"  P95          : {p95_ms:.2f} ms")
    print(f"  P99          : {p99_ms:.2f} ms   budget={BUDGET_P99_MS}ms  {'✓ PASS' if p99_ms < BUDGET_P99_MS else '✗ FAIL'}")
    print(f"  Max          : {max_ms:.2f} ms")
    print(f"{'─'*40}\n")
    return mean_ms


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_depth_output_shape():
    """Smoke test: engine produces the expected (518, 518) output."""
    _skip_if_no_cuda()
    _skip_if_no_trt()
    _skip_if_no_engine()

    from depth.depth_estimator import DepthEstimator, INPUT_H, INPUT_W
    frame = _make_dummy_frame()

    with DepthEstimator(ENGINE_PATH) as est:
        depth = est.infer(frame)

    assert depth.shape == (INPUT_H, INPUT_W), f"Bad shape: {depth.shape}"
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all(), "Depth map contains NaN or Inf"


def test_depth_inference_latency():
    """Full latency benchmark: warmup + 500 timed runs, assert mean < 15ms."""
    _skip_if_no_cuda()
    _skip_if_no_trt()
    _skip_if_no_engine()

    from depth.depth_estimator import DepthEstimator
    import torch

    frame = _make_dummy_frame()
    latencies = np.empty(BENCH_RUNS, dtype=np.float64)

    with DepthEstimator(ENGINE_PATH) as est:
        # Warmup: let TRT select optimal kernels and fill CUDA context caches
        print(f"\n  Warming up ({WARMUP_RUNS} runs) …", end="", flush=True)
        for _ in range(WARMUP_RUNS):
            est.infer(frame)
        print(" done")

        # Benchmark: use CUDA events for GPU-accurate timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i in range(BENCH_RUNS):
            start_event.record()
            depth = est.infer(frame)
            end_event.record()
            torch.cuda.synchronize()
            latencies[i] = start_event.elapsed_time(end_event)  # milliseconds

    mean_ms = _report(latencies, depth.shape)

    assert mean_ms < BUDGET_MEAN_MS, (
        f"Mean {mean_ms:.2f}ms exceeds budget {BUDGET_MEAN_MS}ms. "
        "Check GPU load, FP16 support, or consider a smaller input resolution."
    )
    p99_ms = float(np.percentile(latencies, 99))
    assert p99_ms < BUDGET_P99_MS, (
        f"P99 {p99_ms:.2f}ms exceeds budget {BUDGET_P99_MS}ms."
    )


def test_depth_buffer_reuse():
    """Verify that running multiple inferences does not allocate new GPU memory.

    If buffers leak we'd see CUDA OOM on long-running streams. This test checks
    that peak reserved memory stays flat after the first inference.
    """
    _skip_if_no_cuda()
    _skip_if_no_trt()
    _skip_if_no_engine()

    import torch
    from depth.depth_estimator import DepthEstimator

    frame = _make_dummy_frame()

    with DepthEstimator(ENGINE_PATH) as est:
        est.infer(frame)                               # allocate everything
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_reserved()

        for _ in range(50):
            est.infer(frame)
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_reserved()

    assert mem_after <= mem_before, (
        f"GPU memory grew during repeated inference: "
        f"{mem_before/1e6:.1f}MB → {mem_after/1e6:.1f}MB"
    )


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running depth inference benchmark …")
    try:
        test_depth_output_shape()
        print("Shape test: PASS")
        test_depth_buffer_reuse()
        print("Buffer reuse test: PASS")
        test_depth_inference_latency()
    except pytest.skip.Exception as e:
        print(f"SKIPPED: {e}")
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
