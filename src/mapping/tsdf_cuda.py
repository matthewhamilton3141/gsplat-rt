"""Python front-end for the custom CUDA TSDF integrate kernel.

`kernels/tsdf_integrate.cu` is compiled into the `gaussian_kernels` extension by
`python setup.py build_ext --inplace` on a CUDA box. This module wraps it with a
capability probe and a numpy reference that mirrors the kernel exactly, so the
pipeline degrades gracefully (and stays testable) on machines without a GPU.

The reference (`integrate_reference`) is a line-for-line transcription of the
kernel's per-voxel arithmetic. It is the oracle the unit tests hold the kernel
to, and doubles as the CPU fallback path.
"""

from __future__ import annotations

import numpy as np

_EXT = None
_PROBED = False


def _load_ext():
    """Import the compiled extension once; cache the result (or None)."""
    global _EXT, _PROBED
    if not _PROBED:
        _PROBED = True
        try:
            import gaussian_kernels  # built by setup.py on a CUDA box
            _EXT = gaussian_kernels
        except Exception:
            _EXT = None
    return _EXT


def available() -> bool:
    """True when the CUDA extension imports and a CUDA device is present."""
    if _load_ext() is None:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def integrate_cuda(tsdf, weight, depth, R_wc, t_wc, N, voxel_size,
                   origin, K, trunc) -> None:
    """Integrate one depth frame in place on the GPU.

    `tsdf`, `weight`, `depth`, `R_wc`, `t_wc` are float32 CUDA tensors. `K` is a
    `CameraIntrinsics`. Raises if the extension is unavailable — callers should
    gate on `available()`.
    """
    ext = _load_ext()
    if ext is None:
        raise RuntimeError("gaussian_kernels extension not built")
    ox, oy, oz = (float(o) for o in origin)
    ext.tsdf_integrate(
        tsdf, weight, depth, R_wc, t_wc, int(N), float(voxel_size),
        ox, oy, oz,
        float(K.fx), float(K.fy), float(K.cx), float(K.cy),
        int(K.width), int(K.height), float(trunc))


def integrate_reference(tsdf, weight, depth, R_wc, t_wc, N, voxel_size,
                        origin, K, trunc) -> None:
    """Per-voxel numpy transcription of the CUDA kernel (in place).

    Deliberately loop-free but scalar-faithful: every step maps onto exactly one
    line of `tsdf_integrate_kernel`, so it validates the kernel's arithmetic
    without a GPU. Operates on flat (N^3,) float32 arrays.
    """
    ox, oy, oz = (np.float32(o) for o in origin)
    R = np.asarray(R_wc, dtype=np.float32).reshape(3, 3)
    t = np.asarray(t_wc, dtype=np.float32).reshape(3)

    idx = np.arange(N * N * N)
    k = idx % N
    j = (idx // N) % N
    i = idx // (N * N)

    wx = ox + i.astype(np.float32) * voxel_size
    wy = oy + j.astype(np.float32) * voxel_size
    wz = oz + k.astype(np.float32) * voxel_size

    dx, dy, dz = wx - t[0], wy - t[1], wz - t[2]
    # vox_cam = (world - t) @ R_wc  (dot with columns of R_wc)
    cxc = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
    cyc = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
    czc = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        ui = np.rint(K.fx * cxc / czc + K.cx).astype(np.int64)
        vi = np.rint(K.fy * cyc / czc + K.cy).astype(np.int64)

    valid = (czc > 0.01) & (ui >= 0) & (ui < K.width) & (vi >= 0) & (vi < K.height)
    d_obs = np.zeros(N * N * N, dtype=np.float32)
    sel = np.where(valid)[0]
    d_obs[sel] = depth.reshape(-1)[vi[sel] * K.width + ui[sel]]
    valid &= d_obs > 0.01

    sdf = np.clip((d_obs - czc) / trunc, -1.0, 1.0).astype(np.float32)
    w_old = weight[valid]
    w_new = w_old + 1.0
    tsdf[valid] = (tsdf[valid] * w_old + sdf[valid]) / w_new
    weight[valid] = w_new
