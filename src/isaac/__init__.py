"""M7 Isaac Sim / Isaac Lab groundwork for gsplat-rt.

`nav_task` holds the framework-agnostic navigation task logic (reward, observation,
termination) — pure functions with no Isaac dependency, unit-tested on CPU and reusable
by a PyBullet env (roadmap Phase 4) and an Isaac Lab env (Phase 5). `isaac_nav_env` is the
thin Isaac Lab adapter that wires that core into a DirectRLEnv (box-only).
"""
