# Design note — keyframing & loop closure for the SLAM front-end

Status: **Stages 1 & 3 implemented** (keyframing + frame-to-keyframe tracking, and
the SE(3) pose-graph optimisation back-end — both Mac-tested,
`tests/test_keyframe_odometry.py` + `tests/test_pose_graph.py`). **Stage 2** (loop
*detection* — descriptor retrieval + geometric verification) is the remaining gap:
its retrieval quality needs box/TUM, and it's what feeds real loop edges to the
Stage-3 optimiser. Prep for M6 remaining item #2.

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

### Stage 3 — Pose-graph optimisation — ✅ DONE
- `src/slam/pose_graph.py`: nodes = keyframe poses; edges = odometry constraints
  (consecutive KFs, via `from_keyframes`) + loop constraints (verified matches).
- **Dependency-free**: a hand-rolled SE(3) manifold (closed-form exp/log via
  Rodrigues + the left Jacobian) and a damped Gauss-Newton solve with a numerical
  Jacobian in pure numpy — no g2o/gtsam/scipy, so the Mac dev path just works. Node 0
  is fixed (gauge). Small keyframe graphs ⇒ dense solve is plenty.
- `RGBDOdometry.optimize_keyframes(loop_edges)` builds the graph from its keyframes,
  optimises, and writes corrected poses back to the `Keyframe`s.
- Mac-tested (`tests/test_pose_graph.py`): SE(3) exp/log round-trip, recovery of a
  consistent graph from a drifted guess, **loop closure reduces end-of-trajectory
  drift**, and the odometry integration.
- Still a follow-up: propagating corrected KF poses to the per-frame trajectory and
  re-integrating the `TSDFVolume` map (expensive; only at loop events).

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
