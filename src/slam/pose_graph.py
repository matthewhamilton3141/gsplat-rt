"""SE(3) pose-graph optimisation — the loop-closure back-end (Stage 3).

Turns keyframe odometry (which drifts monotonically) into a globally-consistent
map: nodes are keyframe camera-to-world poses, edges are relative-pose measurements
(odometry between consecutive keyframes + verified loop closures). Minimising the
edge residuals in SE(3) distributes the accumulated error around the loop.

Dependency-free: a hand-rolled SE(3) manifold (closed-form exp/log via Rodrigues +
the left Jacobian) and a damped Gauss-Newton solve in pure numpy — so the Mac dev
path needs no g2o/gtsam. Small keyframe graphs, so a dense solve with a numerical
Jacobian is plenty.

Convention: poses are 4x4 camera-to-world SE(3); a tangent vector is
``xi = [rho(3), phi(3)]`` (translation part, rotation part); retraction is the
right perturbation ``T ← T · exp(xi)``.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# SE(3) manifold
# ---------------------------------------------------------------------------

def _skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1e-8:
        return np.eye(3) + W
    return (np.eye(3) + (np.sin(theta) / theta) * W
            + ((1.0 - np.cos(theta)) / theta**2) * (W @ W))


def _so3_log(R: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * w
    return (theta / (2.0 * np.sin(theta))) * w


def _left_jacobian(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1e-8:
        return np.eye(3) + 0.5 * W
    a = (1.0 - np.cos(theta)) / theta**2
    b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) + a * W + b * (W @ W)


def _left_jacobian_inv(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = _skew(w)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * W
    c = (1.0 / theta**2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * W + c * (W @ W)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Tangent [rho, phi] → 4x4 SE(3)."""
    rho, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = _so3_exp(phi)
    T[:3, 3] = _left_jacobian(phi) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """4x4 SE(3) → tangent [rho, phi]."""
    phi = _so3_log(T[:3, :3])
    rho = _left_jacobian_inv(phi) @ T[:3, 3]
    return np.concatenate([rho, phi])


def _inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ---------------------------------------------------------------------------
# Pose graph
# ---------------------------------------------------------------------------

class PoseGraph:
    """A graph of SE(3) poses tied by relative-pose measurements.

    ``add_node`` / ``add_edge`` build it; ``optimize`` runs damped Gauss-Newton and
    rewrites the node poses in place. Node 0 is fixed by default (gauge freedom).
    """

    def __init__(self):
        self.nodes: List[np.ndarray] = []
        self.edges: List[Tuple[int, int, np.ndarray, float]] = []
        self.fixed: Set[int] = {0}

    def add_node(self, pose: np.ndarray) -> int:
        self.nodes.append(np.asarray(pose, dtype=np.float64).copy())
        return len(self.nodes) - 1

    def add_edge(self, i: int, j: int, measurement: np.ndarray, info: float = 1.0) -> None:
        """Edge i→j: ``measurement`` is the observed relative pose ``T_i^-1 · T_j``."""
        self.edges.append((i, j, np.asarray(measurement, dtype=np.float64).copy(), float(info)))

    def _residuals(self) -> np.ndarray:
        res = []
        for i, j, Z, info in self.edges:
            pred = _inv(self.nodes[i]) @ self.nodes[j]
            res.append(np.sqrt(info) * se3_log(_inv(Z) @ pred))
        return np.concatenate(res) if res else np.zeros(0)

    def chi2(self) -> float:
        r = self._residuals()
        return float(r @ r)

    def optimize(self, iters: int = 30, tol: float = 1e-10, eps: float = 1e-6) -> float:
        """Damped Gauss-Newton over the free nodes. Returns the final chi-squared.

        Numerical Jacobian (dense) — fine for the small keyframe graphs this targets.
        """
        free = [k for k in range(len(self.nodes)) if k not in self.fixed]
        if not free or not self.edges:
            return self.chi2()
        nf6 = len(free) * 6

        # errstate mutes the spurious float BLAS subnormal warnings on macOS Accelerate.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return self._gauss_newton(free, nf6, iters, tol, eps)

    def _gauss_newton(self, free, nf6, iters, tol, eps) -> float:
        for _ in range(iters):
            r = self._residuals()
            J = np.zeros((r.shape[0], nf6))
            for col, k in enumerate(free):
                base = self.nodes[k].copy()
                for d in range(6):
                    dv = np.zeros(6)
                    dv[d] = eps
                    self.nodes[k] = base @ se3_exp(dv)
                    J[:, col * 6 + d] = (self._residuals() - r) / eps
                self.nodes[k] = base

            H = J.T @ J + 1e-9 * np.eye(nf6)          # tiny LM damping
            dx = -np.linalg.solve(H, J.T @ r)
            for col, k in enumerate(free):
                self.nodes[k] = self.nodes[k] @ se3_exp(dx[col * 6:col * 6 + 6])
            if float(np.linalg.norm(dx)) < tol:
                break
        return self.chi2()


def from_keyframes(poses: List[np.ndarray],
                   loop_edges: Optional[List[Tuple[int, int, np.ndarray, float]]] = None
                   ) -> PoseGraph:
    """Build a pose graph from an ordered list of keyframe poses.

    Adds a node per keyframe and an **odometry edge** between consecutive keyframes
    (measurement = their current relative pose, so at construction the odometry
    residuals are zero). ``loop_edges`` are ``(i, j, measurement, info)`` closures
    from Stage 2 — the constraints that actually pull the drift out.
    """
    g = PoseGraph()
    for p in poses:
        g.add_node(p)
    for i in range(len(poses) - 1):
        g.add_edge(i, i + 1, _inv(g.nodes[i]) @ g.nodes[i + 1])
    for (i, j, meas, info) in (loop_edges or []):
        g.add_edge(i, j, meas, info)
    return g
