"""Isaac Lab adapter for the reconstructed-scene nav task (M7 groundwork milestone 4).

Thin binding: the task *math* (reward / observation / termination) lives in the tested,
framework-agnostic `nav_task` module; this file only wires it into Isaac Lab's vectorized
`DirectRLEnv` — load the pipeline's exported scene as the world, spawn N differential-drive
robots, sense collisions, and drive PPO (rsl_rl) via `isaac_setup.sh`'s Isaac Lab.

── UNVERIFIED SCAFFOLD (box-only) ────────────────────────────────────────────────────────
Not runnable on the dev Mac (no isaaclab/isaacsim/torch-CUDA) and NOT yet run on the box.
Isaac Lab's DirectRLEnv API + asset configs shift across versions, so the `TODO`s below are
the real wiring work for the first box session. The reward/obs formulas are the durable
part and are locked by tests/test_nav_task.py — keep this adapter matching them.
Prereq order (see building-plans/gsplat-rt-m7-isaac-sim-training.md): only after RL basics
(Phase 3) + a working PyBullet nav flagship (Phase 4). Also requires a metric, Z-up stage
(run scripts/isaac/phase0_smoke.py first; resolve the Y-up→Z-up finding).
──────────────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass

import torch  # noqa: F401  (Isaac Lab is torch-vectorized; present only on the box)

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # box-only
from isaaclab.scene import InteractiveSceneCfg          # box-only
from isaaclab.utils import configclass                  # box-only

from .nav_task import ACT_DIM, OBS_DIM, NavTaskConfig


@configclass
class ReconstructedNavEnvCfg(DirectRLEnvCfg):
    """Config for the nav env. TODO: fill Isaac Lab sim/scene/robot fields for your version."""
    # --- RL spaces (from the shared task core) ---
    num_actions: int = ACT_DIM       # (linear vel cmd, angular vel cmd)
    num_observations: int = OBS_DIM
    num_states: int = 0

    # --- task tuning (shared with PyBullet via nav_task) ---
    task: NavTaskConfig = NavTaskConfig()

    # --- scene / assets (TODO: real Isaac Lab configs) ---
    usd_scene_path: str = ""         # the pipeline's exported .usdz (Z-up, metric)
    num_envs: int = 256              # start small on 24 GB; scale once stable
    env_spacing: float = 8.0
    episode_length_s: float = 20.0
    # TODO: scene: InteractiveSceneCfg = ...   (add the reconstructed mesh + a robot asset,
    #       e.g. a Jetbot/differential-drive articulation Isaac Lab ships)
    # TODO: sim: SimulationCfg = ...           (dt, gravity=(0,0,-9.81) — needs Z-up stage)
    # TODO: a ContactSensorCfg on the robot chassis for the collision term


class ReconstructedNavEnv(DirectRLEnv):
    """PPO nav in the reconstructed scene. Reward/obs mirror `nav_task` (single source of truth)."""

    cfg: ReconstructedNavEnvCfg

    def __init__(self, cfg: ReconstructedNavEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # TODO: cache handles to the robot articulation, contact sensor, and per-env goals.
        # self._goals = self._sample_goals()          # (num_envs, 2) on self.device
        # self._prev_dist = self._distance_to_goal()  # for the progress reward

    # -- scene ---------------------------------------------------------------------------
    def _setup_scene(self):
        # TODO: add_reference_to_stage(self.cfg.usd_scene_path, ".../Scene"); spawn robots;
        #       clone across envs; register the contact sensor. See phase0_smoke.py for the
        #       reference-loading pattern and the up-axis handling.
        raise NotImplementedError("wire the reconstructed scene + robot asset (box session)")

    # -- step ----------------------------------------------------------------------------
    def _pre_physics_step(self, actions: "torch.Tensor"):
        self._actions = actions.clone()

    def _apply_action(self):
        # TODO: map self._actions (lin, ang) -> wheel joint velocity targets on the robot.
        raise NotImplementedError("map action -> differential-drive wheel targets")

    # -- MDP: these MUST match nav_task.py (tested), just vectorized in torch --------------
    def _get_observations(self) -> dict:
        # TODO: build the OBS_DIM vector per env from robot pose/vel + goal, exactly as
        #       nav_task.observation (goal in robot frame, dist, heading cos/sin, lin/ang vel).
        raise NotImplementedError("vectorized nav_task.observation")

    def _get_rewards(self) -> "torch.Tensor":
        # TODO: r = progress_weight*(prev_dist-curr_dist) - time_penalty
        #           - collision_penalty*collided + success_bonus*reached   (== nav_task.reward)
        #       then update self._prev_dist = curr_dist.
        raise NotImplementedError("vectorized nav_task.reward")

    def _get_dones(self):
        # TODO: terminated = reached | collided ; truncated = episode_length exceeded.
        raise NotImplementedError("vectorized nav_task.terminated / truncated")

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # TODO: reset robot pose, resample goals, reset self._prev_dist for env_ids.
        raise NotImplementedError("reset robots + goals")
