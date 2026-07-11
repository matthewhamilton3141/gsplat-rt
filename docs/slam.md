# SLAM front-end & back-end

*Learned front-end done + TensorRT-accelerated; keyframing + pose-graph loop-closure
back-end built.*

## Visual odometry front-end

RGB-D visual odometry (`src/slam/`) feeds per-frame poses into pose-aware fusion. The
detect+match step is a pluggable `Frontend`; the ORB+PnP baseline (5.7 cm ATE on TUM
fr1/desk) has a **SuperPoint + LightGlue** learned upgrade that drops in via a pairwise
`match_pair` branch — the fused SuperPoint+LightGlue ONNX (from
[LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)) jointly matches an image
pair, and its correspondences back-project through the previous depth into the same
RANSAC-PnP. On TUM fr1/desk (200 frames, A10G) it cuts **ATE-RMSE 5.7 → 3.6 cm** (median
4.7 → 2.9, all frames PnP-tracked) — the estimated trajectory (orange) hugs ground truth
(green) far more tightly:

| ORB + PnP — 5.7 cm ATE | SuperPoint + LightGlue — **3.6 cm ATE** |
|:---:|:---:|
| ![ORB visual-odometry trajectory](odometry_orb.png) | ![SuperPoint+LightGlue visual-odometry trajectory](odometry_superpoint_lightglue.png) |

Run either with `scripts/eval_odometry.py --frontend {orb,superpoint}` (add `--provider
tensorrt` for the compiled engine). A **TensorRT FP16 engine** takes the learned front-end
from 132 ms/frame (onnxruntime CUDA) to **7.4 ms/frame** — an 18× speed-up, well inside the
real-time budget — while preserving accuracy (**3.5 cm ATE via TRT vs 3.6 cm via
onnxruntime**; FP16 leaves match quality intact). It is wired into the live pipeline via
`PipelineConfig.pose_tracking="superpoint"`, and the **full live tracked pipeline (FP16
depth + SuperPoint+LightGlue TRT pose + TSDF) sustains ~30 FPS on the A10G** (GPU-verified
end-to-end).

**Live map coherence** is closed: `run_live --tum-intrinsics --metric-scale-monocular`
feeds real source intrinsics (`fx≠fy`, rescaled to depth space) + cross-frame metric scale,
and with the upright auto-framed previews the live TUM map renders a recognizable, colored
desk instead of an origin blob.

## Keyframing + pose-graph loop closure

Frame-to-frame odometry drifts monotonically. The SLAM back-end (design +
staging in [`design_keyframing_loop_closure.md`](design_keyframing_loop_closure.md))
adds:

- **Stage 1 — keyframing** (`rgbd_odometry.py`): a `KeyframeDB`, keyframe-triggered
  insertion (translation/rotation/inlier thresholds), and frame-to-**keyframe** tracking —
  less short-term drift, and the fused SuperPoint ONNX only re-extracts the keyframe once.
  Opt-in; `keyframe=False` is the untouched, byte-identical frame-to-frame baseline.
- **Stage 3 — SE(3) pose-graph optimisation** (`pose_graph.py`): a dependency-free numpy
  SE(3) manifold (closed-form exp/log via Rodrigues + the left Jacobian) and a damped
  Gauss-Newton solve. `RGBDOdometry.optimize_keyframes(loop_edges)` distributes accumulated
  drift around a loop by minimising relative-pose residuals — measured to reduce
  end-of-trajectory drift in `tests/test_pose_graph.py`.
- **Remaining — Stage 2 loop detection**: descriptor retrieval + geometric verification to
  produce the loop constraints the optimiser consumes (retrieval quality is a box/TUM
  measurement).
