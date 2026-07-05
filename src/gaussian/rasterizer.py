"""Differentiable EWA splatting rasteriser (M5), pure numpy.

Forward pass follows the 3D Gaussian Splatting pipeline (Kerbl et al. 2023):

  1. transform centres world -> camera, cull those behind the near plane;
  2. project centres to pixels with the pinhole model;
  3. push the 3D covariance through the projection Jacobian to a 2D covariance
     (the "EWA" step), add a small screen-space low-pass filter;
  4. invert to a 2D conic, splat each Gaussian over its pixel footprint;
  5. depth-sort front-to-back and alpha-composite with transmittance.

The analytic backward pass (``rasterize_backward``) is added in the next step;
the forward pass here stashes everything the backward will need in the returned
``cache``. Kept deliberately un-tiled and per-Gaussian-looped: correctness and
gradient-checkability first, the CUDA tiling is the A10G port.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gaussian_model import GaussianModel

# Screen-space low-pass: dilates every splat by ~sqrt(BLUR) px so sub-pixel
# Gaussians stay differentiable and cover at least one pixel (as in 3DGS).
BLUR = 0.3


@dataclass
class Camera:
    """Pinhole camera with a world->camera extrinsic (p_cam = R @ p_world + t)."""

    R: np.ndarray   # (3, 3) world-to-camera rotation
    t: np.ndarray   # (3,)   world-to-camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    near: float = 0.2

    @classmethod
    def look_at(cls, eye, target, fx, fy, width, height, up=(0.0, 1.0, 0.0),
                near=0.2) -> "Camera":
        """Build a camera looking from ``eye`` toward ``target`` (+Z forward)."""
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)
        fwd = target - eye
        fwd /= np.linalg.norm(fwd) + 1e-12
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right) + 1e-12
        true_up = np.cross(fwd, right)
        # Rows map world directions into camera axes (x=right, y=down, z=fwd).
        R = np.stack([right, true_up, fwd], axis=0)
        t = -R @ eye
        return cls(R, t, fx, fy, width / 2.0, height / 2.0, width, height, near)


def rasterize(model: GaussianModel, cam: Camera, bg: float = 0.0):
    """Render ``model`` from ``cam``. Returns (image (H, W, 3), cache).

    The cache holds per-Gaussian intermediates for the backward pass.
    """
    H, W = cam.height, cam.width
    means = model.means
    p_cam = means @ cam.R.T + cam.t          # (N, 3)
    z = p_cam[:, 2]
    visible = z > cam.near
    idx = np.nonzero(visible)[0]

    image = np.full((H, W, 3), bg, dtype=np.float64)
    alpha_accum = np.zeros((H, W), dtype=np.float64)
    if idx.size == 0:
        return image, _empty_cache(cam, bg)

    x, y, zc = p_cam[idx, 0], p_cam[idx, 1], z[idx]
    u = cam.fx * x / zc + cam.cx
    v = cam.fy * y / zc + cam.cy

    # 2D covariance via the projection Jacobian: Σ' = J (R Σ Rᵀ) Jᵀ.
    cov3d = model.covariance3d()[idx]                       # (M, 3, 3)
    cov_cam = cam.R @ cov3d @ cam.R.T                        # (M, 3, 3)
    M = idx.size
    J = np.zeros((M, 2, 3), dtype=np.float64)
    J[:, 0, 0] = cam.fx / zc
    J[:, 0, 2] = -cam.fx * x / (zc * zc)
    J[:, 1, 1] = cam.fy / zc
    J[:, 1, 2] = -cam.fy * y / (zc * zc)
    cov2d = J @ cov_cam @ np.transpose(J, (0, 2, 1))        # (M, 2, 2)
    cov2d[:, 0, 0] += BLUR
    cov2d[:, 1, 1] += BLUR

    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    # conic = inverse of the 2x2 covariance.
    conic = np.empty_like(cov2d)
    conic[:, 0, 0] = cov2d[:, 1, 1] / det
    conic[:, 1, 1] = cov2d[:, 0, 0] / det
    conic[:, 0, 1] = -cov2d[:, 0, 1] / det
    conic[:, 1, 0] = -cov2d[:, 1, 0] / det

    # 3-sigma pixel radius from the larger eigenvalue of cov2d.
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    disc = np.sqrt(np.maximum(mid * mid - det, 0.0))
    lam_max = mid + disc
    radius = np.ceil(3.0 * np.sqrt(np.maximum(lam_max, 1e-6))).astype(np.int64)

    alphas = model.alphas[idx]
    rgb = model.rgb[idx]

    order = np.argsort(zc)   # front-to-back
    # Per-Gaussian records for the backward pass, in composite order.
    recs = []
    for k in order:
        r = radius[k]
        cu, cv = u[k], v[k]
        u0 = max(int(np.floor(cu - r)), 0)
        u1 = min(int(np.ceil(cu + r)) + 1, W)
        v0 = max(int(np.floor(cv - r)), 0)
        v1 = min(int(np.ceil(cv + r)) + 1, H)
        if u0 >= u1 or v0 >= v1:
            continue
        px = np.arange(u0, u1)
        py = np.arange(v0, v1)
        dx = px[None, :] - cu                     # (h, w)
        dy = py[:, None] - cv
        A, Bc, C = conic[k, 0, 0], conic[k, 0, 1], conic[k, 1, 1]
        power = -0.5 * (A * dx * dx + C * dy * dy) - Bc * dx * dy
        g = np.exp(np.minimum(power, 0.0))        # (h, w) gaussian weight
        a_raw = alphas[k] * g
        clamped = a_raw > 0.999
        a = np.minimum(a_raw, 0.999)              # per-pixel opacity

        T = 1.0 - alpha_accum[v0:v1, u0:u1]        # transmittance so far
        contrib = a * T                            # (h, w)
        for c in range(3):
            image[v0:v1, u0:u1, c] += contrib * rgb[k, c]
        alpha_accum[v0:v1, u0:u1] += contrib

        recs.append({
            "g": g, "a": a, "clamped": clamped, "T": T.copy(), "rgb": rgb[k].copy(),
            "gidx": int(idx[k]), "vidx": int(k), "u0": u0, "u1": u1, "v0": v0, "v1": v1,
            "dx": dx, "dy": dy, "conic": conic[k].copy(), "alpha": alphas[k],
            "pcam": p_cam[idx[k]].copy(),
        })

    image += bg * (1.0 - alpha_accum)[:, :, None]
    cache = {
        "recs": recs, "alpha_accum": alpha_accum, "cam": cam,
        "bg": bg, "H": H, "W": W, "model": model, "idx": idx,
        "cov3d": cov3d, "cov_cam": cov_cam, "J": J,   # per-visible-index (M,...)
    }
    return image, cache


def _empty_cache(cam: Camera, bg: float):
    return {"recs": [], "alpha_accum": np.zeros((cam.height, cam.width)),
            "cam": cam, "bg": bg, "H": cam.height, "W": cam.width}


def _drotmat_dquat(q: np.ndarray):
    """Return (dR/dw, dR/dx, dR/dy, dR/dz) for a *unit* quaternion (w,x,y,z)."""
    w, x, y, z = q
    dRdw = 2 * np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    dRdx = 2 * np.array([[0, y, z], [y, -2 * x, -w], [z, w, -2 * x]], dtype=np.float64)
    dRdy = 2 * np.array([[-2 * y, x, w], [x, 0, z], [-w, z, -2 * y]], dtype=np.float64)
    dRdz = 2 * np.array([[-2 * z, -w, x], [w, -2 * z, y], [x, y, 0]], dtype=np.float64)
    return dRdw, dRdx, dRdy, dRdz


def rasterize_backward(grad_image: np.ndarray, cache: dict) -> dict:
    """Backprop dL/d(image) to raw Gaussian parameters (analytic gradients).

    Returns a dict of arrays matching ``GaussianModel`` fields, each (N, ...):
    ``means``, ``log_scales``, ``quats``, ``opacities``, ``colors``.
    """
    model: GaussianModel = cache["model"]
    cam: Camera = cache["cam"]
    N = model.num_gaussians
    g_means = np.zeros((N, 3))
    g_logscales = np.zeros((N, 3))
    g_quats = np.zeros((N, 4))
    g_opacit = np.zeros((N,))
    g_colors = np.zeros((N, 3))
    if not cache["recs"]:
        return {"means": g_means, "log_scales": g_logscales, "quats": g_quats,
                "opacities": g_opacit, "colors": g_colors}

    fx, fy = cam.fx, cam.fy
    scales = model.scales
    alphas_all = model.alphas
    rgb_all = model.rgb
    quats = model.quats
    quat_norm = np.linalg.norm(quats, axis=1) + 1e-12
    qhat = quats / quat_norm[:, None]
    rotmats = model.rotmats

    # Behind-color accumulator B (normalised), starts as the background.
    B = np.full((cache["H"], cache["W"], 3), cache["bg"], dtype=np.float64)

    # Iterate back-to-front so B holds the colour behind each Gaussian.
    for rec in reversed(cache["recs"]):
        u0, u1, v0, v1 = rec["u0"], rec["u1"], rec["v0"], rec["v1"]
        gi = grad_image[v0:v1, u0:u1, :]        # (h, w, 3) dL/dC
        B_win = B[v0:v1, u0:u1, :]
        c = rec["rgb"]                           # (3,)
        T = rec["T"]                             # (h, w) transmittance in front
        a = rec["a"]                             # (h, w) opacity
        g = rec["g"]
        gidx = rec["gidx"]
        vidx = rec["vidx"]
        active = ~rec["clamped"]

        # dL/dc_i = a_i T_i  ->  colour gradient (activated rgb).
        w_ct = (a * T)[:, :, None]               # (h, w, 1)
        grad_rgb = (gi * w_ct).sum(axis=(0, 1))  # (3,)
        # dL/dalpha_pixel = sum_c gi_c (c_c - B_c) T
        dLda = (gi * (c[None, None, :] - B_win)).sum(axis=2) * T   # (h, w)
        dLda = dLda * active
        # a = alpha * g (unclamped)
        grad_alpha = float((dLda * g).sum())     # scalar (activated opacity)
        grad_g = dLda * rec["alpha"]             # (h, w)

        # update B to include this Gaussian (alpha-over) for more-front splats.
        B[v0:v1, u0:u1, :] = a[:, :, None] * c[None, None, :] + (1 - a)[:, :, None] * B_win

        # --- g -> power -> conic & projected mean (u, v) -------------------
        dx, dy = rec["dx"], rec["dy"]
        A0, B0, C0 = rec["conic"][0, 0], rec["conic"][0, 1], rec["conic"][1, 1]
        gp = grad_g * g                          # dL/dpower  (h, w)
        grad_A = float((gp * (-0.5 * dx * dx)).sum())
        grad_Bc = float((gp * (-dx * dy)).sum())
        grad_C = float((gp * (-0.5 * dy * dy)).sum())
        grad_u = float((gp * (A0 * dx + B0 * dy)).sum())
        grad_v = float((gp * (C0 * dy + B0 * dx)).sum())

        # --- conic -> cov2d (a2,b2,c2) via explicit 2x2 inverse partials ---
        J = cache["J"][vidx]                     # (2, 3)
        cov_cam = cache["cov_cam"][vidx]         # (3, 3)
        cov2d = J @ cov_cam @ J.T
        a2 = cov2d[0, 0] + BLUR
        b2 = cov2d[0, 1]
        c2 = cov2d[1, 1] + BLUR
        det = a2 * c2 - b2 * b2
        det = det if abs(det) > 1e-12 else 1e-12
        d2 = det * det
        # partials of (conic00, conic01, conic11) w.r.t (a2, b2, c2)
        dCon00 = np.array([-c2 * c2, 2 * b2 * c2, -b2 * b2]) / d2
        dCon01 = np.array([b2 * c2, -(det + 2 * b2 * b2), a2 * b2]) / d2
        dCon11 = np.array([-b2 * b2, 2 * a2 * b2, -a2 * a2]) / d2
        grad_cov2d_vec = grad_A * dCon00 + grad_Bc * dCon01 + grad_C * dCon11
        ga2, gb2, gc2 = grad_cov2d_vec           # w.r.t (a2, b2, c2)
        # full symmetric 2x2 grad (off-diagonal split, see derivation).
        G2 = np.array([[ga2, gb2 / 2.0], [gb2 / 2.0, gc2]])

        # --- cov2d = J cov_cam J^T -> J and cov_cam ------------------------
        grad_cov_cam = J.T @ G2 @ J              # (3, 3)
        grad_J = 2.0 * G2 @ J @ cov_cam          # (2, 3)

        # --- cov_cam = R cov3d R^T -> cov3d --------------------------------
        grad_cov3d = cam.R.T @ grad_cov_cam @ cam.R   # (3, 3)

        # --- cov3d = M M^T, M = R_g diag(s) -> scales, quats ---------------
        R_g = rotmats[gidx]
        s = scales[gidx]
        M = R_g * s[None, :]
        grad_M = 2.0 * grad_cov3d @ M            # G3 symmetric
        grad_s = (grad_M * R_g).sum(axis=0)      # (3,)
        g_logscales[gidx] += grad_s * s          # s = exp(log_scale)
        grad_Rg = grad_M * s[None, :]            # (3, 3), dL/dR_g
        dRdw, dRdx, dRdy, dRdz = _drotmat_dquat(qhat[gidx])
        gqh = np.array([(grad_Rg * dRdw).sum(), (grad_Rg * dRdx).sum(),
                        (grad_Rg * dRdy).sum(), (grad_Rg * dRdz).sum()])
        # normalisation Jacobian: q_hat = q/|q|
        qn = quat_norm[gidx]
        qh = qhat[gidx]
        g_quats[gidx] += (gqh - (gqh @ qh) * qh) / qn

        # --- projected mean (u, v) & J both depend on p_cam = (x, y, z) ----
        x, y, zc = rec["pcam"]
        z2, z3 = zc * zc, zc * zc * zc
        gx = grad_u * (fx / zc)
        gy = grad_v * (fy / zc)
        gz = grad_u * (-fx * x / z2) + grad_v * (-fy * y / z2)
        # J entries: J00=fx/z, J02=-fx x/z^2, J11=fy/z, J12=-fy y/z^2
        gx += grad_J[0, 2] * (-fx / z2)
        gy += grad_J[1, 2] * (-fy / z2)
        gz += (grad_J[0, 0] * (-fx / z2) + grad_J[0, 2] * (2 * fx * x / z3)
               + grad_J[1, 1] * (-fy / z2) + grad_J[1, 2] * (2 * fy * y / z3))
        grad_pcam = np.array([gx, gy, gz])
        # p_cam = R @ mean + t  ->  dmean = R^T grad_pcam
        g_means[gidx] += cam.R.T @ grad_pcam

        # --- opacity & colour activations ---------------------------------
        al = alphas_all[gidx]
        g_opacit[gidx] += grad_alpha * al * (1 - al)
        rc = rgb_all[gidx]
        g_colors[gidx] += grad_rgb * rc * (1 - rc)

    return {"means": g_means, "log_scales": g_logscales, "quats": g_quats,
            "opacities": g_opacit, "colors": g_colors}
