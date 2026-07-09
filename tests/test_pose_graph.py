"""SE(3) pose-graph optimisation — the loop-closure back-end (src/slam/pose_graph.py).

Pure-numpy, Mac-testable: the SE(3) manifold round-trips, a consistent graph is
recovered from a drifted initial guess, and — the point of the whole thing — adding
a loop-closure constraint reduces end-of-trajectory drift.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from slam.pose_graph import PoseGraph, se3_exp, se3_log, _inv, from_keyframes  # noqa: E402


def _rand_se3(rng, t_scale=0.3, r_scale=0.3):
    xi = np.concatenate([rng.uniform(-t_scale, t_scale, 3),
                         rng.uniform(-r_scale, r_scale, 3)])
    return se3_exp(xi)


def _gt_trajectory(n, rng):
    """A wandering camera path (accumulated random small motions)."""
    poses = [np.eye(4)]
    for _ in range(n - 1):
        poses.append(poses[-1] @ _rand_se3(rng))
    return poses


# ---------------------------------------------------------------------------
# SE(3) manifold
# ---------------------------------------------------------------------------

def test_se3_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        # keep |phi| < pi so the rotation doesn't wrap (log's principal branch)
        xi = np.concatenate([rng.uniform(-2, 2, 3), rng.uniform(-1, 1, 3)])
        assert np.allclose(se3_log(se3_exp(xi)), xi, atol=1e-9)


def test_se3_exp_is_valid_transform():
    rng = np.random.default_rng(1)
    T = se3_exp(rng.uniform(-1, 1, 6))
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)          # orthonormal
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)       # proper rotation
    assert np.allclose(T[3], [0, 0, 0, 1])


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def test_recovers_consistent_graph_from_drifted_init():
    """Exact edges + a fixed anchor ⇒ Gauss-Newton pulls a perturbed guess back to
    ground truth (residuals → 0)."""
    rng = np.random.default_rng(2)
    gt = _gt_trajectory(6, rng)

    g = PoseGraph()
    for p in gt:
        g.add_node(p)
    for i in range(len(gt) - 1):                               # exact odometry
        g.add_edge(i, i + 1, _inv(gt[i]) @ gt[i + 1])
    g.add_edge(len(gt) - 1, 0, _inv(gt[-1]) @ gt[0])           # exact loop

    for k in range(1, len(gt)):                                # perturb all but anchor
        g.nodes[k] = gt[k] @ _rand_se3(rng, 0.15, 0.15)

    assert g.chi2() > 1e-3                                     # starts inconsistent
    g.optimize(iters=40)
    assert g.chi2() < 1e-10                                    # solved
    for est, truth in zip(g.nodes, gt):
        assert np.linalg.norm(est[:3, 3] - truth[:3, 3]) < 1e-3


def test_loop_closure_reduces_drift():
    """Noisy odometry drifts; an exact loop constraint distributes the error and
    brings the final pose closer to ground truth."""
    rng = np.random.default_rng(7)
    n = 8
    gt = _gt_trajectory(n, rng)

    # Noisy relative measurements; chain them → a drifting initial estimate.
    noisy_rel = [(_inv(gt[i]) @ gt[i + 1]) @ _rand_se3(rng, 0.02, 0.02)
                 for i in range(n - 1)]
    est = [np.eye(4)]
    for Z in noisy_rel:
        est.append(est[-1] @ Z)

    drift_before = np.linalg.norm(est[-1][:3, 3] - gt[-1][:3, 3])

    # Odometry-only graph: the chained estimate already satisfies its edges, so
    # optimisation leaves the drift in place.
    g_odo = PoseGraph()
    for p in est:
        g_odo.add_node(p)
    for i in range(n - 1):
        g_odo.add_edge(i, i + 1, noisy_rel[i])
    g_odo.optimize()
    drift_odo = np.linalg.norm(g_odo.nodes[-1][:3, 3] - gt[-1][:3, 3])

    # Same graph + an accurate loop closure between the last and first keyframe.
    g_loop = PoseGraph()
    for p in est:
        g_loop.add_node(p)
    for i in range(n - 1):
        g_loop.add_edge(i, i + 1, noisy_rel[i])
    g_loop.add_edge(n - 1, 0, _inv(gt[-1]) @ gt[0], info=10.0)
    g_loop.optimize()
    drift_loop = np.linalg.norm(g_loop.nodes[-1][:3, 3] - gt[-1][:3, 3])

    assert drift_odo > 1e-3                     # odometry-only can't fix it
    assert drift_loop < 0.5 * drift_before      # loop closure meaningfully corrects


def test_from_keyframes_zero_residual_without_loops():
    """A graph built from keyframe poses has exactly-satisfied odometry edges."""
    rng = np.random.default_rng(3)
    gt = _gt_trajectory(5, rng)
    g = from_keyframes(gt)
    assert len(g.nodes) == 5 and len(g.edges) == 4
    assert g.chi2() < 1e-12


def test_odometry_optimize_keyframes_writes_corrected_poses():
    """RGBDOdometry.optimize_keyframes runs the back-end and rewrites KF poses:
    a drifted final keyframe snaps back once an accurate loop edge is added."""
    from mapping.collision_proxy import CameraIntrinsics
    from slam.rgbd_odometry import RGBDOdometry, Keyframe

    rng = np.random.default_rng(11)
    gt = _gt_trajectory(6, rng)
    odo = RGBDOdometry(CameraIntrinsics(500, 500, 320, 240, 640, 480), keyframe=True)
    # Populate keyframes directly: ground truth, but the last one drifted.
    depth = np.zeros((4, 4), np.float32)
    for i, p in enumerate(gt):
        pose = p if i < len(gt) - 1 else p @ _rand_se3(rng, 0.2, 0.0)
        odo.keyframes.add(Keyframe(id=i, pose=pose, depth=depth))
    before = np.linalg.norm(odo.keyframes.current.pose[:3, 3] - gt[-1][:3, 3])

    odo.optimize_keyframes(loop_edges=[(len(gt) - 1, 0, _inv(gt[-1]) @ gt[0], 10.0)])
    after = np.linalg.norm(odo.keyframes.current.pose[:3, 3] - gt[-1][:3, 3])
    assert after < before and after < 0.1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
