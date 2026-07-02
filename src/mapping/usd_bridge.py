"""OpenUSD scene bridge for the Gaussian SLAM pipeline.

Writes a USD stage containing two layers:
  1. /World/GaussianSplats  — a ParticleField prim carrying splat geometry and
                               attributes for Omniverse NuRec rendering.
  2. /World/CollisionMesh   — a UsdGeom.Mesh with UsdPhysics.CollisionAPI and
                               MeshCollisionAPI applied, set invisible so it acts
                               as a pure physics obstacle in Isaac Sim PhysX.

ParticleField note
------------------
`ParticleField` is an Omniverse-extended USD schema (not in the base OpenUSD
spec). When this code runs inside an Omniverse Kit environment, the schema is
registered automatically. When running against vanilla `usd-core`, the prim is
created with that typename but treated as a generic (schemaless) prim — its
attributes are still stored and round-trip correctly through USD serialisation.
If you need standard-USD portability, swap the typename to `UsdGeom.Points`
and populate the `points` and `widths` attributes.

Physics approximation
---------------------
`convexDecomposition` lets PhysX run the VHACD decomposition at load time so
the RL physics loop never touches the raw triangle mesh. For very coarse meshes
(<500 tris from our 64³ TSDF), `convexHull` is faster and sufficient.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _require_pxr():
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        raise ImportError(
            "OpenUSD (pxr) is required for USD export.\n"
            "  pip install usd-core\n"
            "  or use the pxr bundled with Isaac Sim / Omniverse Kit."
        )


class UsdBridge:
    """Manages a single USD stage containing Gaussian Splat and collision layers.

    Typical usage::

        bridge = UsdBridge("/tmp/scene.usd")
        bridge.update_collision_mesh(mesh.vertices, mesh.faces)
        bridge.update_gaussian_splats(means, scales, quats, opacities)
        bridge.save()
        bridge.export_usdz("/tmp/scene.usdz")
    """

    def __init__(self, usd_path: str):
        _require_pxr()
        from pxr import Usd, UsdGeom

        self._usd_path = usd_path
        os.makedirs(os.path.dirname(os.path.abspath(usd_path)), exist_ok=True)

        self._stage = Usd.Stage.CreateNew(usd_path)
        self._stage.SetStartTimeCode(0)
        self._stage.SetEndTimeCode(0)

        # Stage-level metadata required by Isaac Sim
        UsdGeom.SetStageUpAxis(self._stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(self._stage, 1.0)   # SI metres

        # Root Xform
        root = UsdGeom.Xform.Define(self._stage, "/World")
        self._stage.SetDefaultPrim(root.GetPrim())

        self._splat_prim = None
        self._mesh_prim = None
        self._collision_prim_path = "/World/CollisionMesh"
        self._splat_prim_path = "/World/GaussianSplats"

    # ------------------------------------------------------------------
    # Gaussian Splat layer
    # ------------------------------------------------------------------

    def update_gaussian_splats(
        self,
        means: np.ndarray,           # (N, 3) float32  world-space centres
        scales: np.ndarray,          # (N, 3) float32  log-scale
        rotations: np.ndarray,       # (N, 4) float32  quaternion (w,x,y,z)
        opacities: np.ndarray,       # (N,)   float32  sigmoid-space opacity
        sh_coeffs: Optional[np.ndarray] = None,   # (N, C) float32 SH coefficients
    ) -> None:
        """Write or overwrite the Gaussian Splat prim."""
        from pxr import Gf, Sdf, Vt

        N = means.shape[0]

        # Create the prim as "ParticleField" — recognised by Omniverse NuRec.
        # In vanilla usd-core this is a schemaless prim with type name set.
        if self._splat_prim is None:
            self._splat_prim = self._stage.DefinePrim(self._splat_prim_path, "ParticleField")

        prim = self._splat_prim

        # ---- Positions (required by NuRec for spatial lookup) ----
        # Vec3fArray is the USD typed array pxr expects for Point3fArray.
        prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(
            Vt.Vec3fArray([Gf.Vec3f(*r) for r in means.tolist()])
        )

        # ---- Gaussian-specific attributes stored as flat FloatArrays ----
        # Flat layout (N*3 for scales/rotations) keeps prim I/O simple and
        # avoids fighting pxr's Vec3f / Vec4f type inference for custom namespaces.
        def _float_attr(name: str, data: np.ndarray) -> None:
            prim.CreateAttribute(f"splat:{name}", Sdf.ValueTypeNames.FloatArray).Set(
                Vt.FloatArray(data.ravel().tolist())
            )

        _float_attr("scales",    scales)          # N*3 floats  (sx, sy, sz per splat)
        _float_attr("rotations", rotations)       # N*4 floats  (w, x, y, z per splat)
        _float_attr("opacities", opacities)       # N floats

        if sh_coeffs is not None:
            _float_attr("sh_coeffs", sh_coeffs)  # N*C floats

        prim.CreateAttribute("splat:count",  Sdf.ValueTypeNames.Int).Set(N)
        prim.CreateAttribute("splat:format", Sdf.ValueTypeNames.Token).Set("DepthAnythingV2")

    # ------------------------------------------------------------------
    # Collision mesh layer
    # ------------------------------------------------------------------

    def update_collision_mesh(
        self,
        vertices: np.ndarray,   # (V, 3) float32 world-space
        faces: np.ndarray,      # (F, 3) int32 triangle indices
        approximation: str = "convexDecomposition",
    ) -> None:
        """Write or overwrite the collision mesh prim.

        Applies UsdPhysics.CollisionAPI + MeshCollisionAPI and hides the mesh
        so it is invisible to the renderer but active in the PhysX simulation.

        Parameters
        ----------
        approximation : str
            PhysX collision shape approximation. Options:
            "none" (raw triangles, expensive), "convexHull", "convexDecomposition"
            (VHACD — best for complex shapes), "meshSimplification".
        """
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

        # Create or re-use the mesh prim
        mesh_geom = UsdGeom.Mesh.Define(self._stage, self._collision_prim_path)
        prim = mesh_geom.GetPrim()
        self._mesh_prim = prim

        # Geometry
        mesh_geom.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*v) for v in vertices.tolist()])
        )
        flat_faces = faces.ravel().tolist()
        mesh_geom.GetFaceVertexIndicesAttr().Set(Vt.IntArray(flat_faces))
        mesh_geom.GetFaceVertexCountsAttr().Set(
            Vt.IntArray([3] * len(faces))
        )
        mesh_geom.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        # Make invisible — the mesh is a pure physics object in Isaac Sim
        UsdGeom.Imageable(prim).MakeInvisible()

        # Physics APIs
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)

        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)

        mesh_coll = UsdPhysics.MeshCollisionAPI(prim)
        mesh_coll.GetApproximationAttr().Set(approximation)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Flush the in-memory stage to the .usd file on disk."""
        self._stage.GetRootLayer().Save()

    def export_usdz(self, usdz_path: str) -> str:
        """Package the stage as a self-contained .usdz (zip) archive.

        Saves the .usd first, then creates the .usdz package alongside it.
        Returns the absolute path of the written .usdz file.
        """
        from pxr import Sdf, UsdUtils

        self.save()

        usdz_path = os.path.abspath(usdz_path)
        os.makedirs(os.path.dirname(usdz_path), exist_ok=True)

        result = UsdUtils.CreateNewUsdzPackage(
            Sdf.AssetPath(self._usd_path), usdz_path
        )
        if not result:
            raise RuntimeError(
                f"UsdUtils.CreateNewUsdzPackage failed for {usdz_path}. "
                "Check that the .usd source exists and is valid."
            )
        return usdz_path

    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass   # Stage stays open until GC; call save() / export_usdz() explicitly
