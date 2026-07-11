# Monocular metric scale

*Built + measured on the A10G (TUM fr1/desk).*

Depth Anything emits *relative* (affine-invariant) depth, so the live monocular path
can't produce a metric map — every downstream consumer (`_backproject_gaussians`, the
TSDF push, `RGBDOdometry`) assumes the depth value is metric `z`. `src/depth/metric_scale.py`
closes the gap with the DPT/MiDaS protocol: a per-frame scale+shift fit
(`metric_disparity ≈ s·d + t`) with robust Huber-IRLS reweighting and a temporal EMA so
scale stays coherent frame-to-frame; it coasts on the last good scale through reference
dropouts.

Two anchor sources: a **two-view triangulation** reference for the live monocular path
(`src/slam/monocular_scale.py` — essential-matrix relative pose → triangulated
metric-consistent depths) and the **RGB-D/TUM sensor depth** for validation. Wired as a
config-flagged pipeline stage (`metric_scale_enabled`, off by default → existing metric
runs unchanged) between depth inference and the consumers.

## Results

`scripts/eval_metric_scale.py` on synthetic data: naive relative-as-metric **AbsRel 0.94
→ aligned AbsRel 0.00** (perfect recovery).

**On real TUM `freiburg1_desk`** (measured, A10G): naive relative-as-metric **AbsRel
2.66, δ<1.25 0.05 → scale-aligned AbsRel 0.049, RMSE 0.148 m, δ<1.25 0.97** — a 54× error
reduction, competitive with published metric-depth methods.

## Cross-frame scale propagation

`ScalePropagator` defeats monocular scale drift: `recoverPose` gives a *unit* baseline per
pair, so each pair's depths live in their own gauge — landmarks shared with the previous
pair pin the new baseline into one running global gauge (robust-median ratio). A
`--propagation` demo over a varying-speed trajectory: naive per-pair scale drifts **35.7%**
while propagated holds at **0.0%**. This leaves one genuinely-unobservable global factor
(the reconstruction's absolute size), pinned by the `anchor` (known first baseline →
metres, or 1.0 → consistent arbitrary gauge).

The stage is wired end-to-end: `metric_scale_enabled=True, metric_scale_monocular=True`
auto-builds the triangulation+propagation reference from the camera intrinsics, so a **pure
monocular stream produces metric depth with one flag** (no injection). 51 tests
(`tests/test_metric_scale*.py`, `test_monocular_scale.py`).
