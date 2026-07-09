# Design note — keyframing & loop closure for the SLAM front-end

Status: **design only, not implemented.** Prep for M6 remaining item #2.

## Why

Today `src/slam/rgbd_odometry.py` is pure **frame-to-frame** visual odometry:
each frame's pose is `prev_pose @ relative`, with a constant-velocity fallback on
a degenerate PnP step. There is no drift correction — error accumulates
monotonically along the trajectory. On fr1/desk (200 frames, small loop) ATE is
already good (3.5 cm with SuperPoint+LightGlue), but on longer sequences or a
live walk-around the map will smear when the camera revisits a place. Keyframing
+ loop closure is what turns "good odometry" into "SLAM."

## Current structures to build on

- `RGBDOdometry.track(rgb, depth)` → `TrackResult(pose, n_inliers, n_matches, ok)`;
  keeps `self.trajectory` (list of 4×4), `self._pose`, `self._last_rel`.
- Pluggable `Frontend` (ORBFrontend / SuperPointLightGlueFrontend) exposing either
  `detect/match` or `match_pair(rgb0, rgb1) -> (uv0, uv1)`.
- `OdometryPoseProvider.__call__(frame, depth) -> 4×4` adapts it for the pipeline.

## Proposed staging (smallest useful first)

### Stage 1 — Keyframe selection (no optimisation yet)
A cheap, standalone win that also cuts front-end cost.
- Add a `KeyframeDB`: stores `(id, pose, keypoints, descriptors, depth_thumb)`.
- Insert a keyframe when any of: translation since last KF > τ_t (e.g. 10 cm),
  rotation > τ_r (e.g. 10°), or tracked-inlier ratio drops below a floor.
- Track new frames against the **current keyframe** (frame-to-keyframe), not the
  immediately previous frame — reduces short-term drift and is cheaper for the
  fused SuperPoint ONNX (extract KF once, reuse).
- Testable on Mac with synthetic poses; no GPU needed.

### Stage 2 — Loop detection
- **ORB path:** bag-of-words (DBoW2-style) over ORB descriptors, or a simple
  cosine-similarity retrieval over aggregated descriptors for a first cut.
- **SuperPoint path:** global descriptor retrieval (NetVLAD-lite / mean-pooled
  SuperPoint) → shortlist candidate KFs → geometric verification by running the
  existing `match_pair` + RANSAC-PnP between current KF and candidate.
- A candidate is a loop if geometric verification yields enough inliers AND the
  candidate is temporally distant (avoid matching neighbours).

### Stage 3 — Pose-graph optimisation
- Nodes = keyframe poses; edges = odometry constraints (consecutive KFs) + loop
  constraints (verified matches).
- Optimise with `g2o`/`gtsam` if available; a dependency-free fallback is a small
  Gauss-Newton / Levenberg on SE(3) (`scipy.optimize` + a manual SE(3) manifold)
  so the Mac dev path keeps working without native SLAM libs.
- On loop closure, re-run the optimiser and rewrite `self.trajectory`; the
  pipeline's map (`TSDFVolume`) would need a re-integration pass from corrected
  KF poses — treat map correction as a follow-up (expensive; only at loop events).

## Scope guard

- Full bundle adjustment and dense map deformation are **out of scope** for M6.
- Keep everything behind the `Frontend`/provider seam so ORB (Mac) and SuperPoint
  (box) share one code path.
- Unit-test Stages 1 & 3 on Mac with synthetic correspondences (mirror
  `tests/test_pairwise_odometry.py`); Stage 2 retrieval quality is a box/TUM
  measurement.

## Suggested first PR

Stage 1 only: `KeyframeDB` + keyframe-triggered insertion + frame-to-keyframe
tracking, ORB default, byte-identical ATE gate on fr1/desk. Small, Mac-testable,
and independently useful even before loop closure lands.
