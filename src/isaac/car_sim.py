"""Kinematic-bicycle (Ackermann) car on top of the diff-drive nav core — driving dynamics.

Testing a policy inside a *reconstructed driving scene* needs car dynamics, not a diff-drive
puck: a car steers (front-wheel angle) rather than commands a yaw rate, and crucially **cannot
turn in place** — its heading only changes while it is moving. That single fact is what makes a
driving safety shield a *braking* shield (a stuck car brakes; it can't pivot out).

This module reuses the whole nav stack unchanged. The kinematic bicycle step is exactly the
unicycle step once you convert the command: for wheelbase `L`, steering angle `δ` and speed `v`,
the yaw rate is `ω = v/L·tan δ`, and integrating `(v, ω)` is byte-identical to the diff-drive
`predict_pose`. So `BicycleNavEnv` subclasses `DiffDriveNavEnv`, overriding only the action
semantics; collision, lidar, the occupancy grid, the shared `world_clearance` (circles + walls +
reconstructed `GridWorld`), reward and observation are all inherited. At `v = 0` the yaw rate is
0 — the model can't pivot, correctly.

Pure NumPy; no torch/GPU/gymnasium. Metric: metres, m/s, radians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nav_sim import DiffDriveNavEnv, NavSimConfig, predict_pose, world_clearance
from .nav_task import ACT_DIM


@dataclass
class CarSimConfig(NavSimConfig):
    """Nav world + reward config extended with kinematic-bicycle parameters.

    Inherits every geometry/world field (bounds, obstacles, `grid_world`, robot_radius,
    safety_margin, dt, sensors, ...) so the shared `world_clearance` and the inherited env
    machinery work unchanged. `max_ang_vel` (inherited) must stay above the maximum yaw rate
    `max_speed/wheelbase·tan(max_steer)` or the inherited integrator would clip cornering.
    """

    max_speed: float = 1.0        # |forward speed| clip (m/s)
    max_steer: float = 0.5        # |steering angle| clip (rad); ~28.6°
    wheelbase: float = 0.5        # distance between axles (m)


def bicycle_yaw_rate(speed: float, steer: float, wheelbase: float) -> float:
    """Yaw rate (rad/s) of a kinematic bicycle at `speed` with front-wheel angle `steer`."""
    return float(speed) / wheelbase * np.tan(float(steer))


def predict_pose_bicycle(robot_xy: np.ndarray, heading: float, speed: float, steer: float,
                         wheelbase: float, dt: float) -> np.ndarray:
    """Next (x, y) of a kinematic bicycle — reuses the unicycle `predict_pose` via the yaw rate.

    Shared by the env step and `car_safety_shield` so their one-step lookahead integrates
    exactly as the sim does (an unsound shield predicts differently than it drives).
    """
    ang = bicycle_yaw_rate(speed, steer, wheelbase)
    return predict_pose(robot_xy, heading, speed, ang, dt)


def car_safety_shield(action: np.ndarray, robot_xy: np.ndarray, heading: float,
                      obstacles: np.ndarray, cfg: "CarSimConfig") -> np.ndarray:
    """Braking one-step-lookahead safety filter over a commanded (speed, steer) action.

    Scales the commanded speed down to the largest fraction whose *predicted* next pose (with
    the commanded steering, and the yaw rate that speed actually produces) keeps clearance ≥
    `cfg.safety_margin`; if none is safe, commands **speed 0 — a hard brake**. A stopped car
    cannot move and so cannot collide, which is the car analogue of the diff-drive shield's
    "rotate in place" (a car can't rotate in place, so its only safe fallback is to stop). The
    steering command always passes through. Runtime layer over *any* policy; never retrains.
    """
    a = np.asarray(action, float).reshape(-1)
    speed = float(np.clip(a[0], -cfg.max_speed, cfg.max_speed))
    steer = float(np.clip(a[1], -cfg.max_steer, cfg.max_steer))
    obs = np.asarray(obstacles, float).reshape(-1, 3) if obstacles is not None and len(obstacles) \
        else np.zeros((0, 3), float)
    for s in np.linspace(1.0, 0.0, 11):          # largest safe speed fraction wins
        nxt = predict_pose_bicycle(robot_xy, heading, s * speed, steer, cfg.wheelbase, cfg.dt)
        if world_clearance(nxt, obs, cfg) >= cfg.safety_margin:
            return np.array([s * speed, steer], np.float32)
    return np.array([0.0, steer], np.float32)     # boxed in: brake to a stop


def car_dwa_action(robot_xy: np.ndarray, heading: float, goal: np.ndarray,
                   obstacles: np.ndarray, cfg: "CarSimConfig", horizon: int = 20,
                   n_speeds: int = 6, n_steers: int = 15) -> np.ndarray:
    """Dynamic-Window-Approach local planner for the car — a short-horizon rollout controller.

    A reactive gap-follower can't drive a non-holonomic car through obstacles: it glides onto the
    shield's keep-out shell and freezes, unable to commit a feasible turn while braked. DWA fixes
    that by planning over *dynamically feasible arcs*: it samples a grid of constant `(speed,
    steer)` commands, rolls each one `horizon` steps forward under the exact bicycle model, throws
    out any arc that ever loses clearance, and scores the survivors by progress toward the goal,
    kept clearance, and forward speed (a stall penalty). Because it evaluates whole trajectories
    rather than the instantaneous heading, it naturally arcs *around* an obstacle instead of
    wedging against it — and it only ever considers arcs the car can actually drive.

    Plans over the world geometry directly (circles + walls + reconstructed `GridWorld` via
    `world_clearance`) — i.e. over the reconstructed map, exactly the digital-twin premise. Pair
    with `car_safety_shield` for the hard one-step guarantee on top.
    """
    robot_xy = np.asarray(robot_xy, float)
    goal = np.asarray(goal, float)
    obs = np.asarray(obstacles, float).reshape(-1, 3) if obstacles is not None and len(obstacles) \
        else np.zeros((0, 3), float)
    dt, L = cfg.dt, cfg.wheelbase
    speeds = np.linspace(cfg.max_speed / n_speeds, cfg.max_speed, n_speeds)  # forward, no dead stop
    steers = np.linspace(-cfg.max_steer, cfg.max_steer, n_steers)
    clear_cap = 0.6                                   # clearance beyond this is "plenty of room"

    # Roll every dynamically-feasible arc out and record its objectives; the winner is chosen by
    # a *normalised* weighted sum (standard DWA) so no single term dominates by raw scale — the
    # fix that makes the planner robust to the horizon / geometry instead of hand-tuned weights.
    cand, progress, clear, vel = [], [], [], []
    for v in speeds:
        for st in steers:
            ang = bicycle_yaw_rate(v, st, L)
            xy, h, min_clear, feasible = robot_xy.copy(), heading, np.inf, True
            for _ in range(horizon):
                xy = predict_pose(xy, h, v, ang, dt)
                h = h + ang * dt
                c = world_clearance(xy, obs, cfg)
                min_clear = min(min_clear, c)
                if c < cfg.safety_margin:             # arc grazes the keep-out shell -> reject
                    feasible = False
                    break
            if not feasible:
                continue
            cand.append((v, st))
            progress.append(float(np.linalg.norm(goal - robot_xy) - np.linalg.norm(goal - xy)))
            clear.append(min(min_clear, clear_cap))
            vel.append(v)

    if not cand:
        # Every forward arc is blocked — brake and let the (shielded) car sit; a rare dead-end.
        return np.array([0.0, 0.0], np.float32)

    def _norm(a):
        a = np.asarray(a, float)
        lo, hi = a.min(), a.max()
        return np.zeros_like(a) if hi - lo < 1e-9 else (a - lo) / (hi - lo)

    # Progress dominates (reach the goal), clearance keeps the car centred in gaps rather than
    # hugging a surface (what stopped it stalling at the curb), speed breaks ties toward motion.
    scores = 1.0 * _norm(progress) + 0.4 * _norm(clear) + 0.15 * _norm(vel)
    v, st = cand[int(np.argmax(scores))]
    return np.array([v, st], np.float32)


class BicycleNavEnv(DiffDriveNavEnv):
    """A kinematic-bicycle car navigating to a goal amid obstacles / reconstructed geometry.

    Action is `(speed, steering_angle)` instead of the diff-drive `(linear, angular)`. Everything
    else — reset, collision, lidar, occupancy grid, reward, termination, the `GridWorld` bridge —
    is inherited from `DiffDriveNavEnv`. The stored `_lin_vel`/`_ang_vel` (which feed the
    observation) become the car's forward speed and realised yaw rate.
    """

    def __init__(self, cfg: CarSimConfig | None = None):
        super().__init__(cfg or CarSimConfig())
        # Action space is (speed, steering angle), not (linear, angular) velocity.
        self.action_low = np.array([-self.cfg.max_speed, -self.cfg.max_steer], np.float32)
        self.action_high = np.array([self.cfg.max_speed, self.cfg.max_steer], np.float32)
        max_yaw = self.cfg.max_speed / self.cfg.wheelbase * np.tan(self.cfg.max_steer)
        if max_yaw > self.cfg.max_ang_vel + 1e-9:
            raise ValueError(
                f"max_ang_vel ({self.cfg.max_ang_vel}) must exceed the bicycle's max yaw rate "
                f"({max_yaw:.3f} = max_speed/wheelbase·tan(max_steer)) or cornering is clipped")

    def step(self, action: np.ndarray):
        """Advance one control step from a (speed, steering) command."""
        a = np.asarray(action, float).reshape(-1)
        speed = float(np.clip(a[0], -self.cfg.max_speed, self.cfg.max_speed))
        steer = float(np.clip(a[1], -self.cfg.max_steer, self.cfg.max_steer))
        ang = bicycle_yaw_rate(speed, steer, self.cfg.wheelbase)
        # Delegate the integration + collision + reward to the (tested) diff-drive step, which
        # integrates (speed, ang) identically to the bicycle model.
        return super().step([speed, ang])


# Sanity: the car action dimensionality matches the shared task's action dim.
assert ACT_DIM == 2
