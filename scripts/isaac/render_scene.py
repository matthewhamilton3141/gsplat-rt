#!/usr/bin/env python3
"""Render the pipeline's reconstructed scene in Isaac Sim 5.1's RTX renderer (headless PNGs).

The visual payoff of the M7 Isaac bridge: load the exported `.usdz` (reconstructed mesh), light
it, place cameras, and capture RGB via Replicator — proving reconstruct → Isaac → photoreal-render
end to end. Runs inside NVIDIA's Isaac Sim container (see below); Isaac Sim is not on the dev Mac.

⚠ Requires a box whose NVIDIA driver matches Isaac Sim 5.1 (validated **580.65**). The R590/595
driver branch crashes the RTX renderer (see scripts/isaac_setup.sh). Run via the container:

    docker run --rm --gpus all --user root \
      -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PYTHONUNBUFFERED=1 \
      --entrypoint /isaac-sim/python.sh \
      -v $HOME/output_scene.usdz:/tmp/scene.usdz:ro \
      -v $PWD/scripts/isaac/render_scene.py:/tmp/render.py:ro \
      -v $HOME/isaac_cache/kit:/isaac-sim/kit/cache:rw \
      -v $HOME/isaac_out:/out:rw \
      nvcr.io/nvidia/isaac-sim:5.1.0 /tmp/render.py

Notes learned the hard way (2026-07-18, verified on the A10G box):
  - The exported CollisionMesh loads with **visibility=invisible** (physics proxy) — make it
    visible before rendering (`UsdGeom.Imageable(mesh).MakeVisible()`), else you render an empty
    lit dome (a flat gray frame).
  - The reference composes over a few frames — `stage.Load()` + several `app.update()` before
    querying bounds, or the world bound comes back empty and the camera aims at nothing.
  - RTX renders a default material, not the mesh's `primvars:displayColor`; a UsdPreviewSurface +
    primvar-reader binding is finicky in RTX (colors may not translate) — geometry renders clean
    in clay/gray regardless, which is enough to prove the bridge.
"""

import sys

OUT = "/out"
SCENE = "/tmp/scene.usdz"


def log(m):
    print("### " + m, flush=True)


def main() -> int:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    rc = 1
    try:
        import numpy as np
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom, Usd, UsdLux

        stage = omni.usd.get_context().get_stage()
        add_reference_to_stage(usd_path=SCENE, prim_path="/World/Scene")
        stage.Load()
        for _ in range(20):
            app.update()

        # The reconstructed mesh is authored invisible (collision proxy) — make it renderable.
        mesh = stage.GetPrimAtPath("/World/Scene/CollisionMesh")
        UsdGeom.Imageable(mesh).MakeVisible()
        for _ in range(5):
            app.update()

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                                  useExtentsHint=True)
        r = cache.ComputeWorldBound(mesh).ComputeAlignedRange()
        mn, mx = r.GetMin(), r.GetMax()
        c = [(mn[i] + mx[i]) / 2.0 for i in range(3)]
        sz = max(mx[i] - mn[i] for i in range(3)) or 3.15
        log("mesh bbox center=%.2f,%.2f,%.2f size=%.2f" % (c[0], c[1], c[2], sz))

        UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1000.0)
        UsdLux.DistantLight.Define(stage, "/World/Sun").CreateIntensityAttr(3000.0)
        for _ in range(5):
            app.update()

        import omni.replicator.core as rep
        # Scene is Y-up: lift camera in +Y, pull back in +Z (and +X for iso), look at the centre.
        views = {
            "front": (c[0], c[1] + sz * 0.15, c[2] + sz * 1.5),
            "iso": (c[0] + sz * 0.8, c[1] + sz * 0.6, c[2] + sz * 0.8),
        }
        for name, pos in views.items():
            cam = rep.create.camera(position=pos, look_at=tuple(c))
            rp = rep.create.render_product(cam, (1600, 900))
            w = rep.WriterRegistry.get("BasicWriter")
            w.initialize(output_dir=OUT + "/" + name, rgb=True)
            w.attach([rp])
            for _ in range(30):
                rep.orchestrator.step(rt_subframes=32)   # let RTX accumulate/denoise
            w.detach()
            log("rendered " + name + " -> " + OUT + "/" + name)
        rc = 0
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log("EXC " + repr(e))
    finally:
        app.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
