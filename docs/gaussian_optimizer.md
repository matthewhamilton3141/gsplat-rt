# Gaussian optimizer (3DGS)

*Built + wired into the pipeline.*

A differentiable EWA-splatting rasterizer with hand-derived analytic gradients (verified
against finite differences to <1e-4) and a numpy Adam loop (`src/gaussian/`). Fits posed
views to >60 dB PSNR at ~2 ms/iter on CPU; ports to CUDA/torch on the A10G unchanged.

The pipeline runs it as an **offline finalize stage** (`optimize_on_finalize`): the hot
path stashes RGB keyframes + poses, and `stop()` seeds Gaussians from the fused point
cloud, fits them against the keyframes, and exports optimized splats as a 3DGS `.ply` —
kept off the 30 FPS path since pure-numpy is too slow per frame.

## Loss — full 3DGS photometric loss

`(1−λ)·L1 + λ·(1−SSIM)` (`src/gaussian/ssim.py`): an 11×11 Gaussian-window SSIM with a
hand-derived **analytic D-SSIM gradient** (self-adjoint zero-padded filter → exact adjoint;
matches finite differences to <1e-5), wired via `finalize_ssim_weight` (default 0.2, the
paper value).

## Adaptive Density Control

`src/gaussian/densify.py`: the backward pass surfaces the view-space position gradient +
visibility, and the controller **clones** small under-reconstructed Gaussians (nudged along
−∇means), **splits** large ones into children sampled from their own 3-D covariance (÷1.6),
and **prunes** transparent ones — resizing the Adam moments in lock-step (persisting
Gaussians keep their momentum, children start fresh). Opt-in via `finalize_densify`; a fit
seeded with a single Gaussian grows itself to reconstruct a multi-Gaussian target.

Next: SH view-dependent colour and a CUDA/torch fit fast enough to run online.
