# A note on precision (FP16 vs TF32)

TensorRT 11.1 removed the weakly-typed `BuilderFlag.FP16`, so true FP16 requires a
**strongly-typed** network — which honours the ONNX's own dtypes and inserts *no*
auto-casts. `export_onnx.py --fp16` therefore emits a genuinely *uniform* fp16 graph:
it converts every weight, forces the graph I/O to fp16, and (the subtle part)
**retargets the model's internal `Cast(to=fp32)` nodes** — Depth Anything casts its
input to the weight dtype, which otherwise reintroduces an fp32 activation into an fp16
conv and breaks the strongly-typed parse. `DepthEstimator` reads each binding's dtype
and sizes its buffers to match, so one estimator runs either engine.

Measured on the A10G: **TF32 14.2 ms → FP16 6.3 ms (2.24×)** with output fidelity
**corr 0.99996, max|Δ| 0.02** vs TF32 — half precision left the depth map essentially
unchanged (`scripts/bench_depth.py`).

Network-flag logic is unit-tested (`tests/test_compile_trt_flags.py`);
`scripts/inspect_onnx.py` dumps graph dtypes for debugging.
