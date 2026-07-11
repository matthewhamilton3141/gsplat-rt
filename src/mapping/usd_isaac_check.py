"""Isaac-readiness validator for exported USD stages (M7 groundwork).

Before a `.usd`/`.usdz` from `UsdBridge` is ever loaded into Isaac Sim on the GPU box,
this module checks it against the invariants Isaac/PhysX actually care about — so an
export bug is caught on a laptop (pxr / `usd-core` runs CPU-only) instead of burning an
Isaac session. It operationalizes the M7 plan's "traps": metric scale, stage up-axis, and
the collision-mesh setup.

Severity:
  ERROR — will break the Isaac import or the physics (no collision, degenerate mesh,
          NaN verts, out-of-range faces, missing stage metadata).
  WARN  — loads, but is probably wrong and worth a conscious look (non-Z up-axis vs
          Isaac's Z-up world; non-metric units; implausible scene extent = a likely
          metric-scale bug; missing collision approximation).
  INFO  — fine, just reported (up-axis value, scene extent, prim counts).

Use as a library (`validate_isaac_stage`) or a CLI (`python -m src.mapping.usd_isaac_check
scene.usdz`); the CLI exits non-zero iff any ERROR is found, so it drops into a pre-flight
check or CI. No GPU, no Isaac needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

# PhysX collision-shape approximations PhysX/Isaac accept on a triangle mesh.
_VALID_APPROX = {
    "none", "convexHull", "convexDecomposition", "meshSimplification",
    "boundingCube", "boundingSphere", "sdf", "sphereFill",
}


@dataclass
class Issue:
    severity: str          # "ERROR" | "WARN" | "INFO"
    code: str              # short stable identifier, e.g. "no_collision_api"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message}"


def _require_pxr():
    try:
        from pxr import Usd  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "OpenUSD (pxr) is required to validate a USD stage.\n"
            "  pip install usd-core\n"
            "  or use the pxr bundled with Isaac Sim / Omniverse Kit."
        ) from e


def validate_isaac_stage(
    usd_path: str,
    *,
    expect_up_axis: str = "Z",
    max_extent_m: float = 100.0,
    min_extent_m: float = 0.05,
) -> List[Issue]:
    """Validate that the USD stage at `usd_path` is ready to import into Isaac Sim.

    Returns a list of `Issue`s (possibly empty). `expect_up_axis` is Isaac's world
    convention ("Z"); `min/max_extent_m` bound the plausible scene size in metres —
    a room-scale reconstruction outside that range signals a metric-scale problem.
    Raises only if pxr is unavailable or the stage cannot be opened at all.
    """
    _require_pxr()
    from pxr import Usd, UsdGeom, UsdPhysics

    issues: List[Issue] = []

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"could not open USD stage: {usd_path}")

    # --- stage metadata -------------------------------------------------------
    if not stage.GetDefaultPrim().IsValid():
        issues.append(Issue("ERROR", "no_default_prim",
                            "stage has no valid defaultPrim — Isaac references the "
                            "default prim on add-reference; import will be empty."))

    up = UsdGeom.GetStageUpAxis(stage)
    issues.append(Issue("INFO", "up_axis", f"stage upAxis = {up}"))
    if str(up).upper() != expect_up_axis.upper():
        issues.append(Issue("WARN", "up_axis_mismatch",
                            f"stage is {up}-up but Isaac Sim's world is "
                            f"{expect_up_axis}-up — verify the scene isn't rotated "
                            f"90° on import (confirm in the Phase 0 smoke test)."))

    if not stage.HasAuthoredMetadata("metersPerUnit"):
        issues.append(Issue("ERROR", "no_meters_per_unit",
                            "metersPerUnit is unauthored — Isaac assumes 0.01 (cm) by "
                            "default, silently shrinking the scene 100×."))
    else:
        mpu = UsdGeom.GetStageMetersPerUnit(stage)
        if mpu <= 0:
            issues.append(Issue("ERROR", "bad_meters_per_unit",
                                f"metersPerUnit = {mpu} (must be > 0)."))
        elif abs(mpu - 1.0) > 1e-6:
            issues.append(Issue("WARN", "non_metric_units",
                                f"metersPerUnit = {mpu}; Isaac Lab tasks assume metres "
                                f"(1.0). Distances/velocities will be off by {mpu}×."))

    # --- collision mesh -------------------------------------------------------
    coll_meshes = [
        p for p in stage.Traverse()
        if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not coll_meshes:
        issues.append(Issue("ERROR", "no_collision_mesh",
                            "no UsdGeom.Mesh with UsdPhysics.CollisionAPI found — the "
                            "robot would fall through the world (no physics surface)."))
    for prim in coll_meshes:
        _check_collision_mesh(prim, issues, max_extent_m, min_extent_m,
                              UsdGeom, UsdPhysics)

    # --- splat layer (not required for physics, but expected in our export) ---
    splat = stage.GetPrimAtPath("/World/GaussianSplats")
    if not splat.IsValid():
        issues.append(Issue("WARN", "no_splat_prim",
                            "/World/GaussianSplats missing — physics is fine, but the "
                            "render layer won't show. OK for a physics-only smoke test."))

    return issues


def _check_collision_mesh(prim, issues, max_extent_m, min_extent_m, UsdGeom, UsdPhysics):
    """Append issues for one collision-mesh prim (geometry sanity + PhysX setup)."""
    path = prim.GetPath()

    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        issues.append(Issue("WARN", "no_mesh_collision_api",
                            f"{path}: no MeshCollisionAPI — PhysX can't pick a triangle-"
                            f"mesh approximation; add it or collision may be ignored."))
    else:
        approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if not approx:
            issues.append(Issue("WARN", "no_approximation",
                                f"{path}: MeshCollisionAPI approximation unset."))
        elif str(approx) not in _VALID_APPROX:
            issues.append(Issue("WARN", "bad_approximation",
                                f"{path}: approximation '{approx}' not a known PhysX "
                                f"value {sorted(_VALID_APPROX)}."))

    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()

    n_pts = len(pts) if pts else 0
    if n_pts < 4:
        issues.append(Issue("ERROR", "degenerate_mesh",
                            f"{path}: only {n_pts} vertices — not a closed volume."))
        return
    if not idx or not counts:
        issues.append(Issue("ERROR", "no_faces",
                            f"{path}: mesh has no faces (empty indices/counts)."))
        return

    # NaN / Inf verts -> PhysX cooking fails
    bad = [i for i, p in enumerate(pts)
           if not all(math.isfinite(c) for c in (p[0], p[1], p[2]))]
    if bad:
        issues.append(Issue("ERROR", "non_finite_vertices",
                            f"{path}: {len(bad)} vertices are NaN/Inf (e.g. index "
                            f"{bad[0]}) — PhysX mesh cooking will fail."))

    # face indices in range
    oob = [i for i in idx if i < 0 or i >= n_pts]
    if oob:
        issues.append(Issue("ERROR", "face_index_oob",
                            f"{path}: {len(oob)} face indices out of range "
                            f"[0,{n_pts}) — malformed mesh."))

    # counts should be triangles (bridge writes 3s); sum(counts) must equal len(idx)
    if sum(counts) != len(idx):
        issues.append(Issue("ERROR", "facevertex_mismatch",
                            f"{path}: sum(faceVertexCounts)={sum(counts)} != "
                            f"len(faceVertexIndices)={len(idx)}."))

    # scene extent -> metric-scale sanity (only meaningful if verts are finite)
    if not bad:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
        extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        issues.append(Issue("INFO", "scene_extent",
                            f"{path}: largest bbox dimension = {extent:.3f} m"))
        if extent > max_extent_m:
            issues.append(Issue("WARN", "extent_too_large",
                                f"{path}: extent {extent:.1f} m > {max_extent_m} m — "
                                f"likely a metric-scale bug (relative depth not aligned)."))
        elif extent < min_extent_m:
            issues.append(Issue("WARN", "extent_too_small",
                                f"{path}: extent {extent:.3f} m < {min_extent_m} m — "
                                f"scene collapsed; check metersPerUnit / depth scale."))


def summarize(issues: List[Issue]) -> str:
    """One-line count summary, e.g. '2 ERROR, 1 WARN, 3 INFO'."""
    n = {s: sum(1 for i in issues if i.severity == s) for s in ("ERROR", "WARN", "INFO")}
    return ", ".join(f"{n[s]} {s}" for s in ("ERROR", "WARN", "INFO"))


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Validate a USD stage for Isaac Sim import")
    ap.add_argument("usd_path", help="path to the exported .usd or .usdz")
    ap.add_argument("--up-axis", default="Z", help="Isaac world up-axis (default Z)")
    ap.add_argument("--max-extent-m", type=float, default=100.0)
    ap.add_argument("--min-extent-m", type=float, default=0.05)
    args = ap.parse_args(argv)

    issues = validate_isaac_stage(
        args.usd_path, expect_up_axis=args.up_axis,
        max_extent_m=args.max_extent_m, min_extent_m=args.min_extent_m,
    )
    for i in issues:
        print(i)
    print(f"\n{args.usd_path}: {summarize(issues)}")
    n_err = sum(1 for i in issues if i.severity == "ERROR")
    if n_err:
        print(f"NOT Isaac-ready ({n_err} error(s)).")
        return 1
    print("Isaac-ready (no errors).")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
