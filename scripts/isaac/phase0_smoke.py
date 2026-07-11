#!/usr/bin/env python3
"""M7 Phase 0 smoke test — is the pipeline's exported scene a valid Isaac physics world?

The single most valuable de-risk before any RL work, and it needs no RL: load the
`.usdz` your pipeline exports (Gaussian-splat layer + UsdPhysics collision mesh) into a
headless Isaac Sim on the A10G, drop a dynamic sphere above the reconstructed mesh, step
PhysX, and assert the sphere **comes to rest on the surface** — it doesn't fall through
(no collision) and doesn't explode (bad cooking). If it passes, the reconstruct→sim
bridge is proven end to end; if it fails, that's a `collision_proxy.py` / `usd_bridge.py`
export bug to fix now, while it's fresh.

Two guards baked in from the earlier milestones:
  1. Runs the CPU-only `validate_isaac_stage` (milestone 1) FIRST and aborts on any ERROR
     — no point booting Isaac on a stage we already know is malformed.
  2. Reads the stage up-axis and orients gravity along it, so the collider is tested on
     its own terms; separately reports if the axis isn't Isaac's Z-up (the known export
     finding) since Isaac Lab tasks assume Z-up.

── UNVERIFIED SCAFFOLD ──────────────────────────────────────────────────────────────────
Not yet run on the box (Isaac Sim isn't installed on the dev Mac). Isaac Sim's Python API
moved namespaces across versions (`omni.isaac.core` → `isaacsim.core.api` in 4.x); this
tries the new path then falls back. Confirm the API against your installed version; expect
first-run iteration. The physics/assert logic is the durable part.
──────────────────────────────────────────────────────────────────────────────────────────

Usage (on the box, inside the Isaac venv):
    source ~/isaacsim-venv/bin/activate
    python ~/gsplat-rt/scripts/isaac/phase0_smoke.py --usdz /path/to/scene.usdz
"""

import argparse
import os
import sys

_GSPLAT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")

# Axis name -> (index, +unit vector). Isaac's world convention is Z-up.
_AXIS = {"X": (0, (1.0, 0.0, 0.0)), "Y": (1, (0.0, 1.0, 0.0)), "Z": (2, (0.0, 0.0, 1.0))}


def _preflight(usdz: str) -> str:
    """Milestone-1 validator as a pre-flight; returns the stage up-axis. Aborts on ERROR."""
    sys.path.insert(0, os.path.abspath(_GSPLAT_SRC))
    from mapping.usd_isaac_check import validate_isaac_stage, summarize  # noqa: E402

    issues = validate_isaac_stage(usdz)
    for i in issues:
        print(i)
    print(f"pre-flight: {summarize(issues)}")
    if any(i.severity == "ERROR" for i in issues):
        raise SystemExit("stage has ERRORs — fix the export before the Isaac smoke test.")
    up = next((i.message.split("=")[-1].strip() for i in issues if i.code == "up_axis"), "Z")
    return up.upper()


def _import_isaac_core():
    """Return (World, DynamicSphere, add_reference_to_stage), tolerating the 4.x rename."""
    try:                                            # Isaac Sim 4.x
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicSphere
        from isaacsim.core.utils.stage import add_reference_to_stage
    except ImportError:                             # Isaac Sim <= 2023.x
        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicSphere
        from omni.isaac.core.utils.stage import add_reference_to_stage
    return World, DynamicSphere, add_reference_to_stage


def main() -> int:
    ap = argparse.ArgumentParser(description="M7 Phase 0: drop-test the reconstructed scene")
    ap.add_argument("--usdz", required=True, help="exported scene (.usdz/.usd) to test")
    ap.add_argument("--steps", type=int, default=300, help="physics steps to simulate")
    ap.add_argument("--drop-height", type=float, default=1.0,
                    help="metres above the mesh top to spawn the sphere")
    ap.add_argument("--radius", type=float, default=0.1, help="sphere radius (m)")
    ap.add_argument("--settle-speed", type=float, default=0.05,
                    help="max speed (m/s) to count as 'at rest'")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    up_axis = "Z" if args.skip_validate else _preflight(args.usdz)
    if up_axis != "Z":
        print(f"NOTE: stage is {up_axis}-up, not Isaac's Z-up. Testing the collider along "
              f"its own up-axis; Isaac Lab (Z-up) will need the stage re-oriented first.")
    axis_idx, up_vec = _AXIS.get(up_axis, _AXIS["Z"])

    # Boot Isaac headless BEFORE importing core modules (SimulationApp sets up the runtime).
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        import numpy as np
        World, DynamicSphere, add_reference_to_stage = _import_isaac_core()

        world = World(stage_units_in_meters=1.0)
        # gravity along -up so the collider is tested in its authored orientation
        world.get_physics_context().set_gravity(tuple(-9.81 * c for c in up_vec))

        # reference the exported scene; its collision mesh already carries CollisionAPI
        add_reference_to_stage(usd_path=os.path.abspath(args.usdz), prim_path="/World/Scene")

        # find the mesh's top along the up-axis so we spawn just above it
        from pxr import UsdGeom, Usd
        stage = world.stage
        top = 0.0
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                bbox = UsdGeom.Imageable(prim).ComputeWorldBound(
                    Usd.TimeCode.Default(), UsdGeom.Tokens.default_).ComputeAlignedRange()
                top = max(top, bbox.GetMax()[axis_idx])
        spawn = [0.0, 0.0, 0.0]
        spawn[axis_idx] = top + args.drop_height

        sphere = world.scene.add(DynamicSphere(
            prim_path="/World/DropSphere", name="drop_sphere",
            position=np.array(spawn), radius=args.radius, mass=1.0))

        world.reset()
        start_h = float(sphere.get_world_pose()[0][axis_idx])
        print(f"mesh top(up-axis) = {top:.3f} m; sphere spawned at {start_h:.3f} m")

        last_h, at_rest_for = start_h, 0
        for step in range(args.steps):
            world.step(render=False)
            pos = sphere.get_world_pose()[0]
            h = float(pos[axis_idx])
            if not np.all(np.isfinite(pos)):
                print("FAIL: sphere position went non-finite (bad mesh cooking).")
                return 1
            speed = abs(h - last_h) / max(world.get_physics_dt(), 1e-6)
            last_h = h
            at_rest_for = at_rest_for + 1 if speed < args.settle_speed else 0
            if at_rest_for >= 20:                   # ~stable for 20 steps
                break

        rest_h = last_h
        floor = top                                 # rest should be ~ mesh top + radius
        print(f"sphere final height = {rest_h:.3f} m (mesh top {floor:.3f} m, "
              f"radius {args.radius})")

        # PASS: rested near/above the surface (didn't tunnel through), and settled.
        fell_through = rest_h < floor - 5 * args.radius
        settled = at_rest_for >= 20
        if fell_through:
            print("FAIL: sphere fell through the mesh — collision not active "
                  "(check CollisionAPI / approximation / mesh winding).")
            return 1
        if not settled:
            print("FAIL: sphere never came to rest in the step budget "
                  "(unstable contact or wrong gravity axis).")
            return 1
        print("PASS: reconstructed mesh is a valid collidable Isaac stage — "
              "sphere rested on the surface. Reconstruct→sim bridge proven.")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
