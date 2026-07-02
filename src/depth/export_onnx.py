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
        )

    print(f"[export] Validating ONNX graph …")
    proto = onnx.load(onnx_path)
    onnx.checker.check_model(proto)

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[export] OK — {onnx_path}  ({size_mb:.1f} MB)")
    print(f"[export] Input : pixel_values  (1, 3, {INPUT_H}, {INPUT_W})  float32")
    print(f"[export] Output: depth         (1, 1, {INPUT_H}, {INPUT_W})  float32")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Depth Anything V2 Small to ONNX")
    parser.add_argument("--output", default=DEFAULT_ONNX_PATH, help="Destination .onnx path")
    args = parser.parse_args()
    export(args.output)
