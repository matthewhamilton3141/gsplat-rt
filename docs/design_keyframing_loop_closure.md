# Design note — keyframing & loop closure for the SLAM front-end

Status: **Stage 1 implemented** (keyframing + frame-to-keyframe tracking, opt-in,
ORB default, Mac-tested — `tests/test_keyframe_odometry.py`). Stages 2–3 (loop
detection, pose-graph optimisation) remain. Prep for M6 remaining item #2.

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

### Stage 1 — Keyframe selection (no optimisation yet) — ✅ DONE
A cheap, standalone win that also cuts front-end cost.
- `KeyframeDB` + `Keyframe(id, pose, depth, xy/des or rgb)` in `rgbd_odometry.py`.
- `RGBDOdometry(..., keyframe=True, kf_trans_thresh=0.10, kf_rot_thresh_deg=10,
  kf_min_inlier_ratio=0.5)`: inserts a keyframe when a well-tracked frame moves past
  the translation/rotation threshold or its inlier ratio drops below the floor.
- Tracks each frame against the **current keyframe** (`_track_keyframe`), for both
  front-end kinds (detect/match caches keypoints+descriptors; pairwise caches rgb).
  `OdometryPoseProvider(**kwargs)` already threads `keyframe=True` through.
- Opt-in: `keyframe=False` is the untouched, byte-identical frame-to-frame baseline.
- Mac-tested on a synthetic scene (`tests/test_keyframe_odometry.py`, 4 tests):
  DB semantics, KF0, insertion-as-camera-moves + trajectory tracks GT, off ⇒ no KFs.

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
