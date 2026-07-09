"""Interactive browser 3-D viewer for a scene snapshot, built on viser.

An upgrade over the stdlib/Three.js `web_viewer` for *looking at a reconstruction*:
viser ([nerfstudio-project/viser]) gives real orbit/pan/zoom, a ground grid, and
handles large point clouds. We feed it the same :class:`SceneSnapshot` the other
viewer uses (means + per-splat colour + occupancy), auto-**upright** the cloud via
``mapping.visualization.orient_upright`` (so it loads level instead of at whatever
tilt the world frame carried), and — when camera poses are supplied — draw the
solved trajectory as camera frustums.

viser is an optional dependency: imported lazily here so the core pipeline never
needs it. `serve_snapshot(..., block=False)` returns the running server (used by
tests); `block=True` keeps it alive until Ctrl-C (the CLI path).

Inspiration for the presentation (upright load, ground plane, hero framing):
donalleniii/lingbot-desktop-mac's viser `presentation` module.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from mapping.visualization import orient_upright


def _auto_point_size(means: np.ndarray) -> float:
    """A point size proportional to the scene extent, so density reads well
    regardless of the reconstruction's metric scale."""
    if means.shape[0] < 2:
        return 0.01
    diag = float(np.linalg.norm(means.max(0) - means.min(0))) or 1.0
    return max(diag / 600.0, 1e-4)


def serve_snapshot(
    snap,
    cam_up: Optional[Sequence[float]] = None,
    camera_poses: Optional[Sequence[np.ndarray]] = None,
    port: int = 8080,
    max_points: int = 300_000,
    point_size: Optional[float] = None,
    block: bool = True,
):
    """Serve a :class:`SceneSnapshot` in a viser scene.

    Parameters
    ----------
    snap : SceneSnapshot
        The cloud to show (``means`` + ``colors``, optionally ``occupancy``).
    cam_up : (3,) optional
        Mean camera-up hint for uprighting (see ``estimate_up``).
    camera_poses : list of (4,4) optional
        Camera-to-world poses to draw as frustums (the solved trajectory).
    port : int
        HTTP/websocket port (0 picks a free one).
    max_points : int
        Decimate the cloud to at most this many points before sending.
    point_size : float, optional
        Override the auto point size.
    block : bool
        Keep the process alive after building the scene (CLI). False returns the
        server immediately (tests / embedding).
    """
    import viser  # lazy: optional dependency

    snap = snap.decimated(max_points)
    means_up, R = orient_upright(snap.means, cam_up=cam_up)
    colors = np.clip(np.asarray(snap.colors, dtype=np.float64), 0.0, 1.0)
    colors_u8 = (colors * 255.0).astype(np.uint8)

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")             # match orient_upright's target
    server.scene.add_grid("/ground", width=10.0, height=10.0)
    server.scene.add_point_cloud(
        "/scene",
        points=means_up,
        colors=colors_u8,
        point_size=point_size or _auto_point_size(means_up),
        point_shape="circle",
    )

    if camera_poses:
        for i, pose in enumerate(camera_poses):
            pose = np.asarray(pose, dtype=np.float64)
            # rotate the pose into the uprighted frame, then draw a small frustum
            t = R @ (pose[:3, 3] - np.median(snap.means, axis=0)) + np.median(snap.means, axis=0)
            Rc = R @ pose[:3, :3]
            wxyz = _mat_to_wxyz(Rc)
            server.scene.add_camera_frustum(
                f"/trajectory/cam_{i:04d}", fov=1.0, aspect=1.3,
                scale=0.08, wxyz=wxyz, position=tuple(t.astype(float)),
                color=(120, 200, 255),
            )

    n = int(means_up.shape[0])
    print(f"viser scene up at http://localhost:{server.get_port()}  "
          f"({n} points{', ' + str(len(camera_poses)) + ' cameras' if camera_poses else ''})")

    if block:
        import time
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nviewer stopped.")
    return server


def _mat_to_wxyz(Rm: np.ndarray) -> tuple:
    """3x3 rotation → (w, x, y, z) quaternion (viser's orientation convention)."""
    m = np.asarray(Rm, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return tuple((q / (np.linalg.norm(q) + 1e-12)).astype(float))
