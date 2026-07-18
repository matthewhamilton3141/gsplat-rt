"""PyBullet rigid-body backend for the nav task — the roadmap's "build the flagship in a real
physics engine" step (Phase 4), and the bridge toward the Isaac Lab port.

Same `reset`/`step` contract, observation, reward, and termination as the pure-NumPy
`DiffDriveNavEnv` — but the kinematic unicycle integration is replaced by a **real rigid-body
cylinder in PyBullet**: mass, friction, and contact-resolved collisions instead of an analytic
"stop dead on overlap". Everything that *isn't* the dynamics is reused verbatim from the tested
core: an internal `DiffDriveNavEnv` samples the scene (so a given seed yields the identical
start / goal / obstacle field — essential for a fair kinematic↔physics transfer comparison) and
supplies the observation (goal frame + lidar + occupancy) and the `nav_task` reward/termination.
PyBullet owns only how the robot *moves* and *collides*.

Optional dep: `pybullet` (installs from a prebuilt wheel on Linux/the A10G box; the macOS source
build is flaky, so this backend is Linux/box-verified). Import is lazy and guarded so the pure
core and its tests keep running anywhere; `tests/test_nav_pybullet.py` skips when pybullet is
absent, exactly like the GPU/dataset rows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .nav_sim import DiffDriveNavEnv, NavSimConfig, clearance_at, safety_shield
from .nav_task import (
    OBS_DIM, distance_to_goal, reached_goal,
    reward as nav_reward, terminated as nav_terminated, truncated as nav_truncated,
)

# Physics substeps per control step: PyBullet integrates best at a small fixed timestep, so we
# run the control `dt` as N substeps of `dt/N` rather than one coarse step (stabler contacts).
_SUBSTEPS = 12
_ROBOT_HEIGHT = 0.2          # cylinder height (m); the robot is a low disc
_WALL_HEIGHT = 0.5
_WALL_THICK = 0.1


class PyBulletNavEnv:
    """Rigid-body diff-drive nav env with the `DiffDriveNavEnv` contract (single agent).

    Motion model: the robot is a dynamic cylinder driven by a commanded base velocity
    (`lin` along its heading, `ang` about +z); PyBullet resolves contacts, so driving into an
    obstacle is stopped/deflected by the solver rather than by an analytic check. A collision is
    any contact between the robot and an obstacle/wall body (ground contact excluded).
    """

    def __init__(self, cfg: Optional[NavSimConfig] = None):
        import pybullet as p                      # lazy: keep the pure core import-free of pybullet
        self._p = p
        self.cfg = cfg or NavSimConfig()
        # Internal kinematic env: scene sampling + observation + reward/termination (all reused).
        self._kin = DiffDriveNavEnv(self.cfg)
        self.obs_dim = self._kin.obs_dim
        self.act_dim = self._kin.act_dim
        self.action_low = self._kin.action_low
        self.action_high = self._kin.action_high
        self._cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self._cid)
        p.setTimeStep(self.cfg.dt / _SUBSTEPS, physicsClientId=self._cid)
        self._robot = None
        self._solid_ids: set[int] = set()         # obstacle + wall body ids (collision counts)

    # -- world construction -------------------------------------------------------------------
    def _build_world(self) -> None:
        p, cid = self._p, self._cid
        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(self.cfg.dt / _SUBSTEPS, physicsClientId=cid)
        self._solid_ids = set()

        # Ground plane (a large static box; avoids needing pybullet_data's plane.urdf).
        ground_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[50, 50, 0.5],
                                            physicsClientId=cid)
        p.createMultiBody(0, ground_col, basePosition=[0, 0, -0.5], physicsClientId=cid)

        # Four static walls around the bounds.
        x_min, y_min, x_max, y_max = self.cfg.bounds
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        t, wh = _WALL_THICK, _WALL_HEIGHT / 2
        for hx, hy, px, py in ((w / 2 + t, t, cx, y_min - t), (w / 2 + t, t, cx, y_max + t),
                               (t, h / 2 + t, x_min - t, cy), (t, h / 2 + t, x_max + t, cy)):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, wh],
                                         physicsClientId=cid)
            wid = p.createMultiBody(0, col, basePosition=[px, py, wh], physicsClientId=cid)
            self._solid_ids.add(wid)

        # Obstacle cylinders (static).
        for (ox, oy, orad) in self._kin._obstacles:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(orad),
                                         height=_WALL_HEIGHT, physicsClientId=cid)
            oid = p.createMultiBody(0, col, basePosition=[float(ox), float(oy), _WALL_HEIGHT / 2],
                                    physicsClientId=cid)
            self._solid_ids.add(oid)

        # The robot: a dynamic cylinder of radius `robot_radius`.
        rcol = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.cfg.robot_radius,
                                      height=_ROBOT_HEIGHT, physicsClientId=cid)
        self._robot = p.createMultiBody(1.0, rcol, basePosition=[self._kin._x, self._kin._y,
                                                                 _ROBOT_HEIGHT / 2],
                                        physicsClientId=cid)
        self._set_yaw(self._kin._heading)

    def _set_yaw(self, yaw: float) -> None:
        p, cid = self._p, self._cid
        pos, _ = p.getBasePositionAndOrientation(self._robot, physicsClientId=cid)
        quat = p.getQuaternionFromEuler([0, 0, yaw], physicsClientId=cid)
        p.resetBasePositionAndOrientation(self._robot, pos, quat, physicsClientId=cid)

    # -- pose read-back -----------------------------------------------------------------------
    def _read_pose(self) -> tuple[float, float, float]:
        p, cid = self._p, self._cid
        (x, y, _), quat = p.getBasePositionAndOrientation(self._robot, physicsClientId=cid)
        yaw = p.getEulerFromQuaternion(quat, physicsClientId=cid)[2]
        return float(x), float(y), float(yaw)

    def _sync_kin(self, x, y, yaw, lin, ang) -> None:
        """Mirror the physical pose into the kinematic env so its obs/lidar/occupancy is correct."""
        self._kin._x, self._kin._y, self._kin._heading = x, y, yaw
        self._kin._lin_vel, self._kin._ang_vel = lin, ang

    def _collided(self) -> bool:
        p, cid = self._p, self._cid
        for cp in p.getContactPoints(bodyA=self._robot, physicsClientId=cid):
            if cp[2] in self._solid_ids:          # cp[2] = bodyB; ignore the ground
                return True
        return False

    # -- Gymnasium-style API ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        self._kin.reset(seed)                     # samples goal / start / obstacle field
        self._build_world()
        self._sync_kin(self._kin._x, self._kin._y, self._kin._heading, 0.0, 0.0)
        return self._kin._make_obs(), self._info(collided=False, reached=False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        p, cid = self._p, self._cid
        action = np.asarray(action, float).reshape(-1)
        lin = float(np.clip(action[0], -self.cfg.max_lin_vel, self.cfg.max_lin_vel))
        ang = float(np.clip(action[1], -self.cfg.max_ang_vel, self.cfg.max_ang_vel))

        collided = False
        for _ in range(_SUBSTEPS):
            _, _, yaw = self._read_pose()
            vx, vy = lin * np.cos(yaw), lin * np.sin(yaw)
            # Commanded planar velocity; z-lin held at 0 so gravity/contacts keep it on the ground
            # plane, z-ang = the turn rate. Contacts during the substep deflect/stop x,y.
            p.resetBaseVelocity(self._robot, [vx, vy, 0.0], [0.0, 0.0, ang], physicsClientId=cid)
            p.stepSimulation(physicsClientId=cid)
            if self._collided():
                collided = True

        x, y, yaw = self._read_pose()
        self._sync_kin(x, y, yaw, lin, ang)

        self._kin._step += 1
        curr_dist = distance_to_goal(self._kin.robot_xy, self._kin._goal)
        reached = reached_goal(self._kin.robot_xy, self._kin._goal, self.cfg.task)
        clearance = clearance_at(self._kin.robot_xy, self._kin._obstacles, self.cfg.bounds,
                                 self.cfg.robot_radius)
        r = nav_reward(self._kin._prev_dist, curr_dist, collided, reached, self.cfg.task,
                       clearance=clearance)
        self._kin._prev_dist = curr_dist

        term = nav_terminated(reached, collided)
        trunc = nav_truncated(self._kin._step, self.cfg.task)
        return self._kin._make_obs(), r, term, trunc, self._info(collided, reached)

    def _info(self, collided: bool, reached: bool) -> dict:
        return {
            "robot_xy": self._kin.robot_xy,
            "heading": self._kin._heading,
            "goal_xy": self._kin._goal.copy(),
            "distance": distance_to_goal(self._kin.robot_xy, self._kin._goal),
            "collided": collided,
            "reached": reached,
            "step": self._kin._step,
        }

    def close(self) -> None:
        try:
            self._p.disconnect(physicsClientId=self._cid)
        except Exception:                          # already disconnected / shutting down
            pass


def shielded_action(env: "PyBulletNavEnv", action: np.ndarray) -> np.ndarray:
    """Apply the safety shield using the physical robot's current pose (for eval-time wrapping)."""
    return safety_shield(action, env._kin.robot_xy, env._kin._heading, env._kin._obstacles,
                         env.cfg)
