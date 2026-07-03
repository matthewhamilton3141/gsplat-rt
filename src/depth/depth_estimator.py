"""TensorRT FP16 depth estimator.

Design constraints:
- All device buffers are pre-allocated at init and reused every frame —
  no per-frame malloc/cudaMalloc on the hot path.
- The host input buffer is pinned memory so H2D transfer uses DMA.
- Inference runs on a private CUDA stream to avoid blocking the capture thread.
- Preprocessing (BGR resize + ImageNet normalize) runs on the GPU to overlap
  with the H2D copy.
- Compatible with TensorRT >= 9.0 (uses IExecutionContext v3 API).

Typical usage:
    estimator = DepthEstimator("models/depth_engine.engine")
    for frame in video_stream:            # frame: (480, 640, 3) uint8 BGR
        depth = estimator.infer(frame)    # returns (518, 518) float32 numpy array
"""

import os
from typing import Optional

import cv2
import numpy as np
import torch

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DEFAULT_ENGINE_PATH = os.path.join(_ROOT, "models", "depth_engine.engine")

INPUT_H = 518
INPUT_W = 518

# ImageNet normalization — kept on GPU for zero-copy preprocessing
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class DepthEstimator:
    """TensorRT FP16 inference wrapper for Depth Anything V2 Small.

    Thread safety: a single DepthEstimator must not be shared across threads;
    each pipeline stage that needs depth should own its own instance (the
    engine weights are shared in GPU memory by TRT automatically).
    """

    def __init__(self, engine_path: str = DEFAULT_ENGINE_PATH):
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "TensorRT is not installed.\n"
                "pip install tensorrt --extra-index-url https://pypi.ngc.nvidia.com"
            )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TensorRT inference.")

        if not os.path.exists(engine_path):
            raise FileNotFoundError(
                f"Engine not found: {engine_path}\n"
                "Run: python src/depth/export_onnx.py && python src/depth/compile_trt.py"
            )

        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, "")

        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

        self._context = self._engine.create_execution_context()
        self._trt = trt      # Hold reference so logger stays alive

        # Private CUDA stream: inference doesn't block other GPU work
        self._stream = torch.cuda.Stream()

        # Pre-allocate device tensors — reused every call (no malloc on hot path).
        # Dtype MUST match the engine's I/O binding types. The FP16 builder flag
        # only enables reduced precision for internal layers; the network's input
        # and output bindings keep the ONNX-declared float32, so these buffers are
        # float32. Binding fp16 buffers here would size them at half the bytes TRT
        # reads/writes → out-of-bounds access.
        self._dev_input = torch.empty(
            (1, 3, INPUT_H, INPUT_W), dtype=torch.float32, device="cuda"
        )
        self._dev_output = torch.empty(
            (1, 1, INPUT_H, INPUT_W), dtype=torch.float32, device="cuda"
        )

        # Pinned host buffer for fast DMA H2D copy
        self._host_input = torch.empty(
            (1, 3, INPUT_H, INPUT_W), dtype=torch.float32
        ).pin_memory()

        # ImageNet stats pinned on GPU — no CPU↔GPU transfer for normalization
        self._mean_gpu = _MEAN.cuda()
        self._std_gpu = _STD.cuda()

        # Bind I/O tensor addresses once; TRT reuses them every execute call
        self._input_name = "pixel_values"
        self._output_name = "depth"
        self._context.set_tensor_address(self._input_name, self._dev_input.data_ptr())
        self._context.set_tensor_address(self._output_name, self._dev_output.data_ptr())

        # Tell TRT the (fixed) input shape for this execution context
        self._context.set_input_shape(self._input_name, tuple(self._dev_input.shape))

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def preprocess_cpu(self, bgr: np.ndarray) -> None:
        """Resize + normalize frame on the CPU and DMA into the pinned buffer.

        Separated so callers can overlap it with other work before calling
        execute(). For single-frame usage, infer() does both steps.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        # HWC uint8 → CHW float32 in [0, 1], then ImageNet-normalize
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        t.sub_(_MEAN).div_(_STD)
        self._host_input[0].copy_(t)

    def execute(self) -> np.ndarray:
        """Run one TRT inference pass. Assumes preprocess_cpu() was called first."""
        with torch.cuda.stream(self._stream):
            # H2D: pinned float32 → device float32, straight DMA into the bound
            # buffer (no per-frame temporaries). TRT casts to fp16 internally.
            self._dev_input.copy_(self._host_input, non_blocking=True)

            # TensorRT async inference on our private stream
            ok = self._context.execute_async_v3(stream_handle=self._stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 returned False")

            # D2H: bring result back on the same stream to preserve ordering
            result_cpu = self._dev_output.squeeze().cpu()

        self._stream.synchronize()
        return result_cpu.numpy()   # (518, 518) float32

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """Full pipeline: BGR frame in, (518 × 518) depth map out.

        Args:
            bgr: uint8 BGR frame from OpenCV/VideoStream — any resolution.
        Returns:
            Float32 numpy array of shape (INPUT_H, INPUT_W). Values are
            relative depth (not metric); larger = farther.
        """
        self.preprocess_cpu(bgr)
        return self.execute()

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def __del__(self):
        # Explicit teardown order matters: context before engine before runtime
        if hasattr(self, "_context"):
            del self._context
        if hasattr(self, "_engine"):
            del self._engine

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.__del__()
