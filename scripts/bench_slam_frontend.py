"""Isolated latency benchmark for the SuperPoint+LightGlue SLAM front-end.

Unlike eval_odometry.py — which times the whole odometry loop (image decode,
PnP, trajectory bookkeeping) and is an *accuracy* harness — this times just the
learned matcher on the GPU, so the number is comparable to the depth/TSDF
micro-benchmarks and to the 30 FPS pose budget.

Two figures are reported per image pair:
    * infer      raw ``session.run`` (pure TensorRT/CUDA engine time)
    * match_pair the full front-end call (preprocess + infer + numpy post-proc),
                 i.e. what RGBDOdometry actually pays per frame.

The active providers are printed so a silent CPU/CUDA fallback (which would make
the "TensorRT" number meaningless) is visible rather than hidden.

Usage (A10G):
    python scripts/bench_slam_frontend.py --provider tensorrt \
        --sp-onnx models/sp_lg_tum.onnx --warmup 10 --iters 100
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.tum_dataset import TUMDataset
from slam.superpoint_lightglue import SuperPointLightGlueFrontend, ort_providers


def _summarize(name, times_ms):
    times_ms = sorted(times_ms)
    n = len(times_ms)
    mean = statistics.fmean(times_ms)
    median = statistics.median(times_ms)
    p95 = times_ms[min(n - 1, int(round(0.95 * (n - 1))))]
    print(f"{name:11s}: {mean:6.2f} ms mean  {median:6.2f} median  "
          f"{p95:6.2f} p95  {min(times_ms):6.2f} min   ({1e3 / mean:5.1f} fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="data/tum/rgbd_dataset_freiburg1_desk")
    ap.add_argument("--sp-onnx", default="models/sp_lg_tum.onnx",
                    help="fused SuperPoint+LightGlue ONNX")
    ap.add_argument("--provider", choices=["cuda", "tensorrt", "cpu"], default="tensorrt",
                    help="onnxruntime execution provider; tensorrt builds/caches "
                         "an FP16 TRT engine next to the ONNX")
    ap.add_argument("--warmup", type=int, default=10,
                    help="untimed iterations (TRT engine build + cuDNN autotune "
                         "land here, not in the reported numbers)")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    ds = TUMDataset(args.seq)
    if len(ds.frames) < 2:
        raise SystemExit(f"need >=2 frames in {args.seq}, found {len(ds.frames)}")
    K = ds.intrinsics
    rgb0, rgb1 = ds.frames[0].load_rgb(), ds.frames[1].load_rgb()

    fe = SuperPointLightGlueFrontend(args.sp_onnx, height=K.height, width=K.width,
                                     providers=ort_providers(args.provider, args.sp_onnx))
    print(f"Front-end : SuperPoint+LightGlue ONNX  (providers={fe.providers})")
    if args.provider == "tensorrt" and "TensorrtExecutionProvider" not in fe.providers:
        print("WARNING: TensorRT EP not active — falling back; the numbers below "
              "are NOT the TensorRT engine latency.")

    # Preprocessed model input, built once so preprocessing is excluded from the
    # raw-infer timing (it is included in match_pair).
    import numpy as np
    images = np.stack([fe._preprocess(rgb0), fe._preprocess(rgb1)], axis=0)

    # Warmup: first TRT run builds/loads the cached engine; also lets cuDNN
    # autotune settle so the timed loop measures steady state.
    for _ in range(args.warmup):
        fe._sess.run(None, {fe._in: images})
        fe.match_pair(rgb0, rgb1)

    infer_ms, call_ms = [], []
    for _ in range(args.iters):
        t = time.perf_counter()
        fe._sess.run(None, {fe._in: images})
        infer_ms.append((time.perf_counter() - t) * 1e3)
    for _ in range(args.iters):
        t = time.perf_counter()
        fe.match_pair(rgb0, rgb1)
        call_ms.append((time.perf_counter() - t) * 1e3)

    print(f"Sequence  : {os.path.basename(args.seq)}  "
          f"({K.width}x{K.height}, {args.iters} iters, {args.warmup} warmup)")
    _summarize("infer", infer_ms)
    _summarize("match_pair", call_ms)


if __name__ == "__main__":
    main()
