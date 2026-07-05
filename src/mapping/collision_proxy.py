"""Incremental TSDF collision-proxy builder.

Architecture
------------
TSDFVolume
    Pure-numpy TSDF on a fixed 3D grid. Integrates depth maps frame-by-frame
    using a vectorised volume-to-image projection. Mesh extraction runs
    marching cubes on the grid — at 64³ voxels this takes ~5ms on a modern
    CPU, well under the 100ms budget for a 10Hz update rate.

CollisionProxyExtractor
    Background daemon thread. Consumes (depth_map, intrinsics, pose) tuples
    from a bounded queue, integrates them into the TSDF, and extracts a fresh
    coarse mesh every `update_interval_s` seconds. The main thread reads
    `latest_mesh` at any time without blocking.

Depth scale note
    Depth Anything V2 outputs *relative* (unitless) depth. In the full system,
    a scale factor aligned against a known reference (ARCore/stereo/metric IMU)
    must be applied before calling `push_depth`. The `depth_scale` parameter
    on `push_depth` provides that hook.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Optional, Tuple

import numpy as np

from . import tsdf_cuda


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """Pinhole camera model used for depth-to-volume projection."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, fov_deg: float, width: int, height: int) -> "CameraIntrinsics":
        """Build intrinsics from a symmetric horizontal FOV angle."""
        import math
        fx = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        fy = fx
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height)


@dataclass
class TriangleMesh:
    vertices: np.ndarray   # (N, 3) float32 world-space
    faces: np.ndarray      # (M, 3) int32 vertex indices
    timestamp: float = field(default_factory=time.time)

    @property
    def is_empty(self) -> bool:
        return self.vertices.shape[0] == 0 or self.faces.shape[0] == 0


_IDENTITY_POSE = np.eye(4, dtype=np.float32)

# Default intrinsics for the 518×518 Depth Anything V2 output (~70° FOV)
DEFAULT_INTRINSICS = CameraIntrinsics.from_fov(fov_deg=70.0, width=518, height=518)


# ---------------------------------------------------------------------------
# TSDF Volume
# ---------------------------------------------------------------------------

class TSDFVolume:
    """Truncated Signed Distance Field on a regular 3-D grid.

    Conventions
    -----------
    - World frame: right-handed, +Z points away from camera (depth direction).
    - TSDF value:  +1 in free space, –1 inside surfaces, 0 on the surface.
    - Weighted running average: each observed voxel contributes weight=1.

    Parameters
    ----------
    voxel_size : float
        Edge length of one voxel in metres.
    grid_dim : int
        Number of voxels along each axis (cube grid).
    trunc : float
        Truncation distance in metres. Voxels farther than this from the
        nearest surface hold the clamped value ±1.
    origin : ndarray, optional
        World-space coordinates of voxel [0,0,0]. Defaults to centering the
        volume in front of the camera.
    use_cuda : bool, optional
        Integrate with the custom CUDA kernel, keeping the volume resident on
        the GPU and syncing to host only for mesh/occupancy extraction. ``None``
        (default) auto-detects (`tsdf_cuda.available()`); ``True`` requires it
        and raises if unavailable; ``False`` forces the numpy path.
    """

    def __init__(
        self,
        voxel_size: float = 0.05,
        grid_dim: int = 64,
        trunc: float = 0.10,
        origin: Optional[np.ndarray] = None,
        use_cuda: Optional[bool] = None,
    ):
        self.voxel_size = float(voxel_size)
        self.grid_dim = int(grid_dim)
        self.trunc = float(trunc)
        N = self.grid_dim

        half = N * voxel_size / 2.0
        self.origin = (
            np.array([-half, -half, 0.0], dtype=np.float32)
            if origin is None
            else np.asarray(origin, dtype=np.float32)
        )

        self._tsdf = np.ones((N, N, N), dtype=np.float32)
        self._weight = np.zeros((N, N, N), dtype=np.float32)

        # Pre-compute voxel-centre world coordinates — shape (N³, 3)
        i, j, k = np.meshgrid(
            np.arange(N, dtype=np.float32),
            np.arange(N, dtype=np.float32),
            np.arange(N, dtype=np.float32),
            indexing="ij",
        )
        self._vox_world = (
            np.stack([i, j, k], axis=-1).reshape(-1, 3) * voxel_size
            + self.origin
        )  # (N³, 3) — built once, reused every integrate() call

        # --- Optional GPU-resident volume (custom CUDA integrate kernel) ---
        if use_cuda is None:
            self.use_cuda = tsdf_cuda.available()
        elif use_cuda:
            if not tsdf_cuda.available():
                raise RuntimeError(
                    "use_cuda=True but the CUDA TSDF kernel is unavailable "
                    "(build with `python setup.py build_ext --inplace`)")
            self.use_cuda = True
        else:
            self.use_cuda = False

        self._torch = None
        self._tsdf_t = None          # flat (N³,) CUDA tensors — source of truth
        self._weight_t = None        #   when use_cuda; host arrays are a mirror
        self._host_dirty = False     # host mirror stale vs GPU?
        if self.use_cuda:
            import torch
            self._torch = torch
            dev = torch.device("cuda")
            self._tsdf_t = torch.ones(N * N * N, dtype=torch.float32, device=dev)
            self._weight_t = torch.zeros(N * N * N, dtype=torch.float32, device=dev)

    # ------------------------------------------------------------------

    def _sync_host(self) -> None:
        """Copy the GPU volume down into the numpy mirror if it has drifted.

        A no-op on the numpy path. Called before any host-side read
        (mesh/occupancy) so those methods keep operating on `_tsdf`/`_weight`
        unchanged.
        """
        if self.use_cuda and self._host_dirty:
            N = self.grid_dim
            self._tsdf = self._tsdf_t.cpu().numpy().reshape(N, N, N)
            self._weight = self._weight_t.cpu().numpy().reshape(N, N, N)
            self._host_dirty = False

    def _integrate_cuda(self, depth, K, pose) -> None:
        """Integrate one frame on the GPU; only depth crosses the PCIe bus."""
        if pose is None:
            pose = _IDENTITY_POSE
        torch = self._torch
        dev = self._tsdf_t.device
        depth_t = torch.from_numpy(
            np.ascontiguousarray(depth, dtype=np.float32)).to(dev)
        R_t = torch.from_numpy(
            np.ascontiguousarray(pose[:3, :3], dtype=np.float32)).to(dev)
        t_t = torch.from_numpy(
            np.ascontiguousarray(pose[:3, 3], dtype=np.float32)).to(dev)
        tsdf_cuda.integrate_cuda(
            self._tsdf_t, self._weight_t, depth_t, R_t, t_t,
            self.grid_dim, self.voxel_size, self.origin, K, self.trunc)
        self._host_dirty = True

    # ------------------------------------------------------------------

    def integrate(
        self,
        depth: np.ndarray,
        K: CameraIntrinsics,
        pose: Optional[np.ndarray] = None,
    ) -> None:
        """Fuse one depth map into the TSDF.

        Parameters
        ----------
        depth : ndarray  (H, W) float32, metres
        K : CameraIntrinsics
        pose : (4,4) camera-to-world transform. Identity = fixed camera.
        """
        if self.use_cuda:
            self._integrate_cuda(depth, K, pose)
            return

        if pose is None:
            pose = _IDENTITY_POSE

        R_wc = pose[:3, :3].astype(np.float32)   # world ← camera
        t_wc = pose[:3, 3].astype(np.float32)

        # Transform voxel centres to camera frame  (N³, 3)
        # errstate suppresses spurious float32 BLAS subnormal warnings on macOS.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            vox_cam = (self._vox_world - t_wc) @ R_wc

        z = vox_cam[:, 2]
        valid_z = z > 0.01

        # Project onto image plane — avoid divide-by-zero in masked positions
        safe_z = np.where(valid_z, z, 1.0)
        u = K.fx * vox_cam[:, 0] / safe_z + K.cx
        v = K.fy * vox_cam[:, 1] / safe_z + K.cy

        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)

        in_bounds = (
            valid_z
            & (ui >= 0) & (ui < K.width)
            & (vi >= 0) & (vi < K.height)
        )

        # Sample observed depth at each projected voxel location
        d_obs = np.zeros(len(self._vox_world), dtype=np.float32)
        idx = np.where(in_bounds)[0]
        d_obs[idx] = depth[vi[idx], ui[idx]]

        valid_obs = in_bounds & (d_obs > 0.01)

        # SDF: positive = in front of surface; negative = behind
        tsdf_raw = np.clip((d_obs - z) / self.trunc, -1.0, 1.0)

        # Weighted running average update
        N = self.grid_dim
        flat_w = self._weight.ravel()
        flat_t = self._tsdf.ravel()

        w_new = flat_w + valid_obs.astype(np.float32)
        flat_t = np.where(
            valid_obs,
            (flat_t * flat_w + tsdf_raw) / np.where(w_new > 0, w_new, 1.0),
            flat_t,
        )
        flat_w = w_new

        self._tsdf = flat_t.reshape(N, N, N)
        self._weight = flat_w.reshape(N, N, N)

    # ------------------------------------------------------------------

    def extract_mesh(self) -> Optional[TriangleMesh]:
        """Run marching cubes and return a coarse TriangleMesh in world coords.

        Returns None if the volume has no zero-crossing (no observed surface).
        """
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            raise ImportError("scikit-image required. pip install scikit-image")

        self._sync_host()
        # Treat unobserved voxels as free space so they don't generate spurious surface
        tsdf = np.where(self._weight > 0, self._tsdf, 1.0)

        if tsdf.min() >= 0.0 or tsdf.max() <= 0.0:
            return None   # No zero crossing yet

        verts, faces, _, _ = marching_cubes(tsdf, level=0.0, allow_degenerate=False)
        verts_world = (verts * self.voxel_size + self.origin).astype(np.float32)
        return TriangleMesh(vertices=verts_world, faces=faces.astype(np.int32))

    def occupancy_grid_2d(
        self,
        vertical_axis: int = 1,
        surface_level: float = 0.0,
    ) -> np.ndarray:
        """Collapse the volume into a top-down 2-D occupancy map.

        Projects the observed voxels along the vertical (`Y`) axis to produce a
        floor-plan the robot planner can consume directly. Cells carry three
        states:

            -1  unknown  — no observed voxel in this column
             0  free     — observed, but no surface voxel in this column
             1  occupied — at least one observed voxel is on/behind the surface

        Parameters
        ----------
        vertical_axis : int
            Grid axis to collapse. Defaults to 1 (world +Y), the vertical axis
            for the pipeline's camera convention, leaving an (X, Z) grid.
        surface_level : float
            TSDF threshold below which a voxel counts as solid. 0.0 = on or
            behind the zero-crossing surface.

        Returns
        -------
        ndarray (N, N) int8 indexed [X, Z] (with the default vertical_axis).
        """
        self._sync_host()
        observed = self._weight > 0
        occupied_vox = observed & (self._tsdf <= surface_level)
        col_observed = observed.any(axis=vertical_axis)
        col_occupied = occupied_vox.any(axis=vertical_axis)

        grid = np.full(col_observed.shape, -1, dtype=np.int8)
        grid[col_observed] = 0
        grid[col_occupied] = 1
        return grid

    def reset(self) -> None:
        self._tsdf[:] = 1.0
        self._weight[:] = 0.0
        if self.use_cuda:
            self._tsdf_t.fill_(1.0)
            self._weight_t.fill_(0.0)
            self._host_dirty = False


# ---------------------------------------------------------------------------
# Async extractor
# ---------------------------------------------------------------------------

_DepthItem = Tuple[np.ndarray, CameraIntrinsics, Optional[np.ndarray]]


class CollisionProxyExtractor:
    """Asynchronous TSDF integrator running in a background thread.

    The consumer (main pipeline) calls `push_depth` at frame rate (30+ Hz).
    The background thread integrates depth maps and extracts a fresh collision
    mesh at `update_hz` (default 10 Hz). `get_latest_mesh` is non-blocking.

    Parameters
    ----------
    tsdf : TSDFVolume
    update_hz : float
        How often to extract a new collision mesh.
    queue_size : int
        How many unprocessed depth frames to buffer. Older frames are dropped
        when the buffer is full — freshness beats completeness for real-time
        collision.
    """

    def __init__(
        self,
        tsdf: Optional[TSDFVolume] = None,
        update_hz: float = 10.0,
        queue_size: int = 8,
    ):
        self._tsdf = tsdf or TSDFVolume()
        self._update_interval = 1.0 / update_hz
        self._queue: Queue[_DepthItem] = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._mesh_lock = threading.Lock()
        self._latest_mesh: Optional[TriangleMesh] = None
        self._latest_occupancy: Optional[np.ndarray] = None
        self.mesh_ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames_integrated = 0
        self.meshes_extracted = 0

    # ------------------------------------------------------------------

    def start(self) -> "CollisionProxyExtractor":
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="TSDFWorker"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def push_depth(
        self,
        depth: np.ndarray,
        K: Optional[CameraIntrinsics] = None,
        pose: Optional[np.ndarray] = None,
        depth_scale: float = 1.0,
    ) -> None:
        """Enqueue a depth frame for integration. Non-blocking; drops oldest on overflow."""
        if K is None:
            K = DEFAULT_INTRINSICS
        scaled = depth.astype(np.float32) * depth_scale
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Empty:
                pass
        try:
            self._queue.put_nowait((scaled, K, pose))
        except Exception:
            pass

    def get_latest_mesh(self) -> Optional[TriangleMesh]:
        """Return the most recent collision mesh, or None if none built yet."""
        with self._mesh_lock:
            return self._latest_mesh

    def get_latest_occupancy(self) -> Optional[np.ndarray]:
        """Return the most recent top-down occupancy grid, or None if none yet.

        A fresh copy taken under the mesh lock is returned so the caller can
        render it while the worker keeps integrating.
        """
        with self._mesh_lock:
            return None if self._latest_occupancy is None else self._latest_occupancy.copy()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    # ------------------------------------------------------------------

    def _worker(self) -> None:
        next_extract = time.monotonic() + self._update_interval
        while not self._stop.is_set():
            # Drain all queued frames into the TSDF
            while True:
                try:
                    depth, K, pose = self._queue.get_nowait()
                    self._tsdf.integrate(depth, K, pose)
                    self.frames_integrated += 1
                except Empty:
                    break

            now = time.monotonic()
            if now >= next_extract:
                mesh = self._tsdf.extract_mesh()
                # Occupancy is meaningful even before a zero-crossing exists
                # (free space is information too), so refresh it every tick.
                occupancy = self._tsdf.occupancy_grid_2d()
                have_mesh = mesh is not None and not mesh.is_empty
                with self._mesh_lock:
                    if have_mesh:
                        self._latest_mesh = mesh
                    self._latest_occupancy = occupancy
                if have_mesh:
                    self.mesh_ready.set()
                    self.meshes_extracted += 1
                next_extract = now + self._update_interval

            # Sleep a fraction of the update interval to stay responsive
            # without busy-waiting
            sleep_s = max(0.0, next_extract - time.monotonic()) * 0.5
            if sleep_s > 0.001:
                time.sleep(sleep_s)
