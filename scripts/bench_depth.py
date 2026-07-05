"""Benchmark the depth engines: default (TF32 on Ampere) vs true FP16.

Times `DepthEstimator.infer` for each engine that exists and, when both are
present, reports the FP16 speed-up and the output fidelity vs the TF32 engine
(max abs diff + correlation) so we can confirm half precision didn't wreck the
depth map.

The FP16 engine is built strongly-typed from an fp16 ONNX with fp32 I/O bindings
(see export_onnx.to_fp16), so `DepthEstimator` runs it unchanged.

Usage (on a CUDA box, after brev_setup.sh):
    python scripts/bench_depth.py --frames 200
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from depth.compile_trt import DEFAULT_ENGINE_PATH, DEFAULT_FP16_ENGINE_PATH


def _bench_engine(engine_path, frames, warmup):
    """Return (latency_stats_ms, stacked_depths) for one engine, or None."""
    from depth.depth_estimator import DepthEstimator

    rng = np.random.default_rng(0)
    inputs = [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
              for _ in range(max(frames, warmup))]

    est = DepthEstimator(engine_path)
    for i in range(warmup):
        est.infer(inputs[i % len(inputs)])

    times, depths = [], []
    for i in range(frames):
        frame = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        d = est.infer(frame)
        times.append((time.perf_counter() - t0) * 1e3)
        if i < 8:                       # keep a few for the fidelity check
            depths.append(d)
    t = np.array(times)
    stats = dict(mean=t.mean(), p50=float(np.median(t)),
                 p99=float(np.percentile(t, 99)), min=t.min())
    return stats, np.stack(depths)


def _fidelity(a, b):
    """Max abs diff + Pearson correlation between two depth stacks."""
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    max_abs = float(np.max(np.abs(a - b)))
    denom = a.std() * b.std()
    corr = float(np.mean((a - a.mean()) * (b - b.mean())) / denom) if denom > 0 else 0.0
    return max_abs, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    engines = [("TF32 (default)", DEFAULT_ENGINE_PATH),
               ("FP16 (strongly-typed)", DEFAULT_FP16_ENGINE_PATH)]

    print(f"Depth engine benchmark — {args.frames} frames "
          f"({args.warmup} warmup)\n")
    print(f"{'engine':<24} {'mean':>8} {'p50':>8} {'p99':>8} {'min':>8}")

    results = {}
    for label, path in engines:
        if not os.path.exists(path):
            print(f"{label:<24}   (engine not found: {path})")
            continue
        stats, depths = _bench_engine(path, args.frames, args.warmup)
        results[label] = (stats, depths)
        print(f"{label:<24} {stats['mean']:7.2f}m {stats['p50']:7.2f}m "
              f"{stats['p99']:7.2f}m {stats['min']:7.2f}m")

    if len(results) == 2:
        tf32 = results["TF32 (default)"][0]["mean"]
        fp16 = results["FP16 (strongly-typed)"][0]["mean"]
        print(f"\nspeed-up: {tf32 / fp16:.2f}x   ({tf32:.2f} → {fp16:.2f} ms)")
        max_abs, corr = _fidelity(results["TF32 (default)"][1],
                                  results["FP16 (strongly-typed)"][1])
        print(f"fidelity vs TF32: max|Δ|={max_abs:.4f}  corr={corr:.5f}")
        budget = "PASS" if fp16 < 15.0 else "OVER"
        print(f"15 ms depth budget: FP16 {budget}")


if __name__ == "__main__":
    main()
