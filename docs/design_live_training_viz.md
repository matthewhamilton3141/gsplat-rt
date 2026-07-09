# Design note — live "watch it train" visualization (Option B)

Status: **deferred / planbook.** Not on the M6 critical path. Do after the M6
end-to-end pose verify is green (see `box_runbook_m6_verify.md`). This is a
demo/portfolio polish item — it makes the Gaussian optimizer's convergence
*visible* in the browser viewer.

## Why it isn't already possible

The optimizer `src/gaussian/optimizer.py::fit()` is a **blocking** loop: it runs
`iters` iterations, mutating the `GaussianModel` in place, prints loss/PSNR every
`log_every`, and only **returns** at the end. The finalize stage sets
`manager.optimized_gaussians` once, on completion. `PipelineSceneSource.snapshot()`
reads that field — so the browser viewer sees a single **jump** from raw
height-colored points → finished anisotropic splats, never the convergence in
between.

## What "training" looks like once streamed (the payoff)

Per-iteration: round isotropic blobs **stretch into surface-aligned ellipsoids**,
colors sharpen toward the captured views, floaters fade (opacity→0), and
densification **clones/splits** Gaussians in high-gradient regions so edges fill
in. Loss curve drops, PSNR climbs. That progression is the thing worth showing.

## Proposed implementation (small, Mac-testable)

1. **Callback hook in `fit()`** — add an optional
   `on_iter: Callable[[int, GaussianModel, float], None] = None` param, called at
   the end of each iteration with `(it, model, losses[-1])`. Zero cost when None;
   preserves the existing `FitResult` return. Unit-test on Mac that it fires
   `iters` times with a monotonic-ish loss (mirror existing optimizer tests).
2. **Publish intermediate snapshots** — finalize passes an `on_iter` that, every
   N iters (throttle, e.g. N=5), sets `manager.optimized_gaussians = model` (it's
   mutated in place, so this is just exposing the reference) and appends the loss
   to a small ring buffer. `PipelineSceneSource` already picks up
   `optimized_gaussians`, so the splats update live with no viewer change.
3. **Loss curve in the browser** — add a `/loss` JSON route to `web_viewer.py`
   (mirrors the existing `/scene` `/occupancy` `/stats` routes) returning the loss
   history; draw a small sparkline/line chart in the page. Optional but it's what
   sells "training."

## Scope guard

- Throttle snapshot publishing (every N iters) — don't stream every iteration or
  the viewer's decimate/serialize cost perturbs the fit's wall-clock.
- Keep the callback strictly optional so the headless bench path is untouched.
- No new deps — reuse the stdlib server + existing scene JSON plumbing.

## First PR

Steps 1 + 2 only (live splats converging in the viewer). Step 3 (loss curve
chart) as a fast follow. Gate: existing optimizer + viewer tests still pass, and
a Mac `--demo`-style run shows the model reference updating mid-fit.
