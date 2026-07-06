"""Quantify how metric the scale-aligned depth becomes.

Two modes:

  --synthetic   (default, runs anywhere incl. the numpy-only dev box)
      Build a metric depth map, derive an affine-invariant disparity prediction
      of it (α·(1/z)+β with random α,β — Depth Anything's invariance), align a
      sparse subset back to metric, and report the error on the *full* map. Proves
      the aligner recovers metric depth from a purely relative prediction.

  --tum <seq_dir>   (needs a CUDA box: TRT depth engine + TUM RGB-D sequence)
      For each frame: run the depth engine on the RGB, align its relative output
      to the sensor depth (the metric reference), and accumulate the standard
      monocular-depth metrics across the sequence — the honest measurement of how
      well the pipeline's live depth becomes metric.

Metrics (standard KITTI/NYU depth protocol):
  AbsRel = mean(|z_hat - z| / z)      RMSE = sqrt(mean((z_hat - z)^2))
  delta1 = fraction with max(z_hat/z, z/z_hat) < 1.25   (higher is better)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from depth.metric_scale import DepthScaleAligner, sample_dense_reference  # noqa: E402


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Standard depth metrics over valid (gt>0, pred>0) pixels."""
    pred = np.asarray(pred, np.float64).ravel()
    gt = np.asarray(gt, np.float64).ravel()
    m = (gt > 0) & (pred > 0) & np.isfinite(pred)
    p, g = pred[m], gt[m]
    if p.size == 0:
        return dict(absrel=float("nan"), rmse=float("nan"), delta1=0.0, n=0)
    absrel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    delta1 = float(np.mean(ratio < 1.25))
    return dict(absrel=absrel, rmse=rmse, delta1=delta1, n=int(p.size))


def _fmt(tag: str, m: dict) -> str:
    return (f"{tag:<22} AbsRel={m['absrel']:.4f}  RMSE={m['rmse']:.4f} m  "
            f"delta<1.25={m['delta1']:.3f}  (n={m['n']})")


def run_synthetic(seed: int = 0, h: int = 240, w: int = 320) -> None:
    rng = np.random.default_rng(seed)
    z_true = 0.5 + 5.5 * rng.random((h, w))                 # metres, [0.5, 6]
    alpha, beta = rng.uniform(0.5, 8.0), rng.uniform(-1.0, 1.0)
    pred_rel = alpha * (1.0 / z_true) + beta                # affine-invariant disparity

    # Naive baseline: treat the relative prediction as if it were metric depth.
    print(_fmt("relative-as-metric", depth_metrics(pred_rel, z_true)))

    # Align a sparse subset, apply to the whole frame.
    pv, rv = sample_dense_reference(pred_rel, z_true, max_points=500, rng=rng)
    aligner = DepthScaleAligner(space="disparity", smoothing=0.0)
    aligner.fit(pv, rv)
    metric = aligner.transform(pred_rel)
    print(_fmt("scale-aligned", depth_metrics(metric, z_true)))
    print(f"\nrecovered scale/shift: s={aligner.params.scale:.4f} "
          f"t={aligner.params.shift:.4f}  (fit RMS resid={aligner.last_residual:.2e})")


def run_tum(seq_dir: str, engine: str, space: str, max_frames: int) -> None:
    from depth.depth_estimator import DepthEstimator
    from slam.tum_dataset import TUMDataset      # loader from M6

    est = DepthEstimator(engine)
    ds = TUMDataset(seq_dir)
    aligner = DepthScaleAligner(space=space, smoothing=0.0)

    naive, aligned = [], []
    for i, frame in enumerate(ds):
        if i >= max_frames:
            break
        rgb = frame.load_rgb()
        gt = frame.load_depth()                         # metric sensor depth (m)
        rel = est.infer(rgb)
        if rel.shape != gt.shape:
            import cv2
            gt = cv2.resize(gt, (rel.shape[1], rel.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        pv, rv = sample_dense_reference(rel, gt, max_points=2000)
        aligner.fit(pv, rv)
        metric = aligner.transform(rel)
        naive.append(depth_metrics(rel, gt))
        aligned.append(depth_metrics(metric, gt))

    def _avg(rows, k):
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.mean(vals)) if vals else float("nan")

    for tag, rows in (("relative-as-metric", naive), ("scale-aligned", aligned)):
        print(_fmt(tag, dict(absrel=_avg(rows, "absrel"), rmse=_avg(rows, "rmse"),
                             delta1=_avg(rows, "delta1"),
                             n=sum(r["n"] for r in rows))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tum", metavar="SEQ_DIR",
                    help="TUM RGB-D sequence dir (needs a CUDA box)")
    ap.add_argument("--engine", default="models/depth_engine.engine")
    ap.add_argument("--space", default="disparity", choices=["disparity", "depth"])
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.tum:
        print(f"TUM metric-scale eval — {args.tum}\n")
        run_tum(args.tum, args.engine, args.space, args.max_frames)
    else:
        print("Synthetic metric-scale eval (affine-invariant disparity)\n")
        run_synthetic(args.seed)


if __name__ == "__main__":
    main()
