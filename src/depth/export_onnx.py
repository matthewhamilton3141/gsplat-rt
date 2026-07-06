"""Download Depth Anything V2 Small and export a fixed-shape ONNX model.

Usage:
    python src/depth/export_onnx.py
    python src/depth/export_onnx.py --output models/depth_v2_small.onnx

The model is frozen at 518x518 (14-pixel-patch multiple for ViT-S).
Dynamic axes are intentionally disabled so TensorRT can apply all
layer fusions and kernel selection that require constant tensor shapes.
"""

import argparse
import os

import onnx
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
INPUT_H = 518       # Must be a multiple of patch_size (14) for ViT-S
INPUT_W = 518
OPSET = 17

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DEFAULT_ONNX_PATH = os.path.join(_ROOT, "models", "depth_v2_small.onnx")
FP16_ONNX_PATH = os.path.join(_ROOT, "models", "depth_v2_small_fp16.onnx")


class _DepthWrapper(torch.nn.Module):
    """Thin wrapper that converts HuggingFace model output to a single (B,1,H,W) tensor.

    The DPT head may output predicted_depth at a lower spatial resolution than
    the input; we upsample here with fixed constants so TensorRT sees a
    shape-static graph.
    """

    def __init__(self, model: torch.nn.Module, out_h: int, out_w: int):
        super().__init__()
        self.model = model
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values)
        depth = outputs.predicted_depth       # (B, H_out, W_out)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)        # (B, 1, H_out, W_out)
        if depth.shape[2:] != (self.out_h, self.out_w):
            depth = F.interpolate(
                depth, size=(self.out_h, self.out_w),
                mode="bilinear", align_corners=False,
            )
        return depth                           # (1, 1, 518, 518)


def export(onnx_path: str = DEFAULT_ONNX_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    print(f"[export] Downloading {MODEL_ID} …")
    hf_model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    hf_model.eval()

    wrapper = _DepthWrapper(hf_model, INPUT_H, INPUT_W)
    wrapper.eval()

    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)

    print(f"[export] Tracing model …")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            onnx_path,
            input_names=["pixel_values"],
            output_names=["depth"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,      # Fixed shape: max TRT optimization
            verbose=False,
            # Force the legacy TorchScript exporter. torch 2.x defaults to the
            # dynamo exporter, which on this model emits a weight-stripped graph
            # (opset-18, external data) that compile_trt's in-memory parser.parse()
            # cannot resolve. dynamo=False embeds weights in a single self-
            # contained file (~50 MB) that TensorRT parses directly.
            dynamo=False,
        )

    print(f"[export] Validating ONNX graph …")
    proto = onnx.load(onnx_path)
    onnx.checker.check_model(proto)

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[export] OK — {onnx_path}  ({size_mb:.1f} MB)")
    print(f"[export] Input : pixel_values  (1, 3, {INPUT_H}, {INPUT_W})  float32")
    print(f"[export] Output: depth         (1, 1, {INPUT_H}, {INPUT_W})  float32")


def to_fp16(src_path: str = DEFAULT_ONNX_PATH,
            dst_path: str = FP16_ONNX_PATH,
            keep_io_types: bool = False) -> None:
    """Convert the fp32 ONNX to fp16 for a strongly-typed TensorRT FP16 build.

    A **strongly-typed** network (TRT 10+/11, the only route to true FP16 now
    that the weakly-typed ``BuilderFlag.FP16`` is gone) inserts *no* automatic
    casts — it honours the ONNX dtypes exactly. So the graph must be internally
    type-consistent: ``keep_io_types=False`` produces a **uniformly fp16** graph
    (fp16 I/O + fp16 weights), which strongly-typed parses cleanly.

    (``keep_io_types=True`` keeps fp32 I/O with fp16 weights and relies on the
    runtime to reconcile them — fine under weakly-typed TRT, but under
    strongly-typed the first conv sees an fp32 activation into fp16 weights and
    the parse fails. Hence the fp16 default here.)

    Because the engine's I/O bindings are now fp16, ``DepthEstimator`` reads each
    binding's dtype and sizes its buffers to match (it casts host input to fp16
    and the output back to fp32), so it runs either engine transparently.
    """
    from onnxconverter_common import float16

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"fp32 ONNX not found: {src_path}\nRun: python src/depth/export_onnx.py")

    print(f"[fp16] Loading {src_path}")
    model = onnx.load(src_path)
    model16 = float16.convert_float_to_float16(model, keep_io_types=keep_io_types)
    onnx.checker.check_model(model16)
    onnx.save(model16, dst_path)

    io = "fp32 I/O + fp16 internals" if keep_io_types else "uniformly fp16 (I/O + weights)"
    size_mb = os.path.getsize(dst_path) / 1e6
    print(f"[fp16] OK — {dst_path}  ({size_mb:.1f} MB)  ({io})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Depth Anything V2 Small to ONNX")
    parser.add_argument("--output", default=DEFAULT_ONNX_PATH, help="Destination .onnx path")
    parser.add_argument("--fp16", action="store_true",
                        help="Also emit an fp16 ONNX (for a strongly-typed FP16 engine)")
    args = parser.parse_args()
    export(args.output)
    if args.fp16:
        to_fp16(args.output, FP16_ONNX_PATH)
