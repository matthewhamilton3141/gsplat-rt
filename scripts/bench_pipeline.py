"""End-to-end pipeline benchmark: per-stage latency + total FPS vs the 30 FPS budget.

Measures two layers:

  1. Stage micro-benchmarks (isolated, warmed up):
       depth inference        budget < 15 ms   (TRT engine; mock on non-GPU machines)
       Gaussian backprojection            —    (numpy today; CUDA kernel in M5)
       TSDF integration       budget <  5 ms
       mesh extraction        budget < 10 ms
       USD export                         —    (off the hot path, informational)

  2. Live pipeline throughput: run PipelineManager on a synthetic video and
     report effective FPS (budget >= 30).

Results are printed as a table and written to a JSON artifact for the README.

Usage:
    python3 scripts/bench_pipeline.py                          # 300 frames, output/bench_results.json
    python3 scripts/bench_pipeline.py --frames 500 --out results.json
    python3 scripts/bench_pipeline.py --strict                 # exit 1 on any budget miss

On machines without CUDA/TensorRT the depth stage uses the mock estimator —
the run is still useful for the CPU-side stages but is labeled accordingly.
"""

import argparse
import json
import os
import platform
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline_manager import PipelineConfig, PipelineManager, _MockDepthEstimator

BUDGETS_MS = {
    "depth_inference": 15.0,
    "tsdf_integration": 5.0,
    "mesh_extraction": 10.0,
}
FPS_BUDGET = 30.0


def _percentiles(samples_ms):
    a = np.asarray(samples_ms)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p99_ms": float(np.percentile(a, 99)),
        "runs": len(a),
    }


def _make_video(path: str, n_frames: int, fps: float = 60.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    rng = np.random.default_rng(7)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
    writer.release()


def bench_stages(manager: PipelineManager, runs: int, warmup: int) -> dict:
    """Isolated per-stage latencies using the manager's own components."""
    results = {}
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

    # --- Depth inference ---
    est = manager._depth_estimator
    for _ in range(warmup):
        depth = est.infer(frame)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        depth = est.infer(frame)
        samples.append((time.perf_counter() - t0) * 1e3)
    results["depth_inference"] = _percentiles(samples)

    # --- Gaussian backprojection ---
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        manager._backproject_gaussians(depth)
        samples.append((time.perf_counter() - t0) * 1e3)
    results["backprojection"] = _percentiles(samples)

    # --- TSDF integration + mesh extraction (fresh volume, direct calls) ---
    from mapping.collision_proxy import TSDFVolume

    tsdf = TSDFVolume(
        voxel_size=manager._config.tsdf_voxel_size,
        grid_dim=manager._config.tsdf_grid_dim,
    )
    for _ in range(warmup):
        tsdf.integrate(depth, manager._camera_k, None)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tsdf.integrate(depth, manager._camera_k, None)
        samples.append((time.perf_counter() - t0) * 1e3)
    results["tsdf_integration"] = _percentiles(samples)

    samples = []
    for _ in range(max(runs // 10, 5)):
        t0 = time.perf_counter()
        tsdf.extract_mesh()
        samples.append((time.perf_counter() - t0) * 1e3)
    results["mesh_extraction"] = _percentiles(samples)

    # --- USD export (informational; skipped without pxr) ---
    if manager._usd_bridge is not None:
        samples = []
        for _ in range(max(runs // 20, 3)):
            t0 = time.perf_counter()
            manager._trigger_usd_export()
            samples.append((time.perf_counter() - t0) * 1e3)
        results["usd_export"] = _percentiles(samples)

    return results


def bench_throughput(cfg: PipelineConfig, n_frames: int, timeout_s: float) -> dict:
    """Run the live pipeline and measure the coordinator's processing rate.

    A file source is read at disk speed (1000+ FPS), so the capture thread
    exhausts the video quickly and the drop-oldest queue discards what the
    coordinator can't keep up with. The meaningful number is therefore the
    processing rate over the window while frames were flowing — timing stops
    when the count stalls (source exhausted), not at a fixed frame target.
    """
    STALL_S = 0.5

    manager = PipelineManager(cfg)
    manager.start()
    try:
        deadline = time.monotonic() + timeout_s
        # Wait for the first processed frame so startup cost isn't counted
        while manager.frames_processed == 0 and time.monotonic() < deadline:
            time.sleep(0.002)
        t0 = time.monotonic()
        f0 = manager.frames_processed
        last_count, last_t = f0, t0
        while time.monotonic() < deadline:
            time.sleep(0.005)
            cur = manager.frames_processed
            now = time.monotonic()
            if cur != last_count:
                last_count, last_t = cur, now
            elif now - last_t > STALL_S:
                break                      # source exhausted
            if cur - f0 >= n_frames:
                last_t = now
                break
        elapsed = last_t - t0
        frames = last_count - f0
    finally:
        manager.stop(flush_usd=False)

    fps = frames / elapsed if elapsed > 0 else 0.0
    return {
        "frames": frames,
        "elapsed_s": round(elapsed, 3),
        "fps": round(fps, 1),
        "frame_time_ms": round(1e3 / fps, 2) if fps > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="gsplat-rt pipeline benchmark")
    ap.add_argument("--frames", type=int, default=300, help="frames for the throughput run")
    ap.add_argument("--runs", type=int, default=200, help="iterations per stage benchmark")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default=os.path.join("output", "bench_results.json"))
    ap.add_argument("--engine", default=None,
                    help="TensorRT engine path (e.g. models/depth_engine_fp16.engine "
                         "for the FP16 pipeline); defaults to PipelineConfig's engine")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any budget is missed")
    args = ap.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
    # 4x the target: the capture thread reads at disk speed and drops what the
    # coordinator can't take, so the video must outlast the measurement window
    _make_video(video_path, args.frames * 4)

    try:
        engine_kw = {"engine_path": args.engine} if args.engine else {}
        cfg = PipelineConfig(
            video_source=video_path,
            output_dir=tempfile.mkdtemp(prefix="bench_usd_"),
            usd_update_interval_s=3.0,
            **engine_kw,
        )

        # Stage benchmarks reuse a started-then-idle manager's components
        manager = PipelineManager(cfg)
        manager.start()
        try:
            manager._stop_event.set()  # park the coordinator; we drive stages directly
            manager._coordinator_thread.join(timeout=5.0)
            stages = bench_stages(manager, args.runs, args.warmup)
            depth_is_mock = isinstance(manager._depth_estimator, _MockDepthEstimator)
        finally:
            manager.stop(flush_usd=False)

        # Fresh output dir: UsdBridge's Stage.CreateNew refuses to overwrite the
        # .usda left behind by the stage-benchmark manager
        throughput_cfg = PipelineConfig(
            video_source=video_path,
            output_dir=tempfile.mkdtemp(prefix="bench_usd_tp_"),
            usd_update_interval_s=3.0,
            **engine_kw,
        )
        throughput = bench_throughput(throughput_cfg, args.frames, timeout_s=120.0)

        gpu = "none (mock depth)"
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "platform": platform.platform(),
            "gpu": gpu,
            "depth_estimator": "mock" if depth_is_mock else "tensorrt",
            "stages": stages,
            "throughput": throughput,
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

        # ---- Human-readable summary ----
        print(f"\ngsplat-rt benchmark — GPU: {gpu} — depth: {report['depth_estimator']}")
        print(f"{'stage':<20} {'mean':>9} {'p50':>9} {'p99':>9} {'budget':>9}  verdict")
        failures = []
        for name, stats in stages.items():
            budget = BUDGETS_MS.get(name)
            verdict = ""
            if budget is not None:
                ok = stats["mean_ms"] < budget
                verdict = "PASS" if ok else "FAIL"
                if not ok:
                    failures.append(f"{name}: {stats['mean_ms']:.2f}ms > {budget}ms")
            budget_s = f"<{budget:.0f}ms" if budget else "—"
            print(
                f"{name:<20} {stats['mean_ms']:>7.2f}ms {stats['p50_ms']:>7.2f}ms "
                f"{stats['p99_ms']:>7.2f}ms {budget_s:>9}  {verdict}"
            )

        fps = throughput["fps"]
        fps_ok = fps >= FPS_BUDGET
        if not fps_ok:
            failures.append(f"throughput: {fps} FPS < {FPS_BUDGET}")
        print(
            f"{'pipeline (live)':<20} {throughput['frame_time_ms'] or float('nan'):>7.2f}ms"
            f"{'':>10}{'':>10} {'>=30fps':>9}  "
            f"{'PASS' if fps_ok else 'FAIL'}  ({fps} FPS over {throughput['frames']} frames)"
        )
        print(f"\nResults written to {args.out}")

        if depth_is_mock:
            print("NOTE: depth stage used the mock estimator — GPU numbers require a")
            print("      TensorRT engine (run scripts/brev_setup.sh on a GPU box).")

        if failures and args.strict:
            print("\nBudget failures:\n  " + "\n  ".join(failures))
            return 1
        return 0
    finally:
        os.unlink(video_path)


if __name__ == "__main__":
    sys.exit(main())
