"""SuperPoint + LightGlue learned front-end (the A10G upgrade for RGBDOdometry).

Runs the fused SuperPoint+LightGlue ONNX pipeline (from fabio-sim/LightGlue-ONNX)
via onnxruntime and exposes ``match_pair(rgb0, rgb1) -> (uv0, uv1)`` so it drops
straight into RGBDOdometry's pairwise branch. LightGlue jointly attends over both
images' keypoints, so matching is inherently pairwise (not a splittable
descriptor NN) — hence a fused pipeline rather than a detect/match front-end.

Fused model I/O (exported at a fixed HxW, num_keypoints K):
    input   images    [2, 1, H, W]   grayscale pair, [0, 1]
    outputs keypoints [2, K, 2]       (x, y) per image, in model-pixel coords
            matches   [M, 3]          [batch, idx_in_img0, idx_in_img1]
            mscores   [M]             match confidence

onnxruntime is an optional dependency (GPU box only); importing this module does
not require it — the session is created lazily in the constructor.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


def ort_providers(backend: str, onnx_path: str = "") -> list:
    """onnxruntime execution-provider list for a backend.

    'tensorrt' → FP16 TensorRT EP (with an on-disk engine cache next to the
    ONNX) then CUDA/CPU fallback; 'cuda' → CUDA then CPU; 'cpu' → CPU only.
    """
    if backend == "cpu":
        return ["CPUExecutionProvider"]
    if backend == "tensorrt":
        cache = os.path.join(os.path.dirname(onnx_path) or ".", ".trt_cache")
        return [
            ("TensorrtExecutionProvider",
             {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
              "trt_engine_cache_path": cache}),
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


class SuperPointLightGlueFrontend:
    """Pairwise learned matcher backed by the fused SuperPoint+LightGlue ONNX.

    Parameters
    ----------
    onnx_path : path to the exported fused pipeline.
    height, width : the fixed input size the model was exported with.
    providers : onnxruntime execution providers (defaults to CUDA→CPU).
    min_score : drop matches below this LightGlue confidence (0 = keep all).
    """

    def __init__(self, onnx_path: str, height: int = 480, width: int = 640,
                 providers: Optional[List[str]] = None, min_score: float = 0.0):
        import onnxruntime as ort  # local import: optional GPU-box dependency

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(onnx_path, providers=providers)
        self._in = self._sess.get_inputs()[0].name
        self.h, self.w = height, width
        self.min_score = float(min_score)

    @property
    def providers(self) -> List[str]:
        return self._sess.get_providers()

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """(H0,W0[,3]) uint8 → (1, H, W) float32 in [0,1] at the model size."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.ndim == 3 else rgb
        if gray.shape[:2] != (self.h, self.w):
            gray = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return (gray.astype(np.float32) / 255.0)[None]

    def match_pair(self, rgb0: np.ndarray, rgb1: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
        h0, w0 = rgb0.shape[:2]
        images = np.stack([self._preprocess(rgb0), self._preprocess(rgb1)], axis=0)
        kpts, matches, mscores = self._sess.run(None, {self._in: images})
        kpts = np.asarray(kpts)
        matches = np.asarray(matches)
        if matches.size == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        if self.min_score > 0.0:
            matches = matches[np.asarray(mscores) >= self.min_score]
            if matches.size == 0:
                return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        uv0 = kpts[0][matches[:, 1].astype(np.int64)].astype(np.float32)
        uv1 = kpts[1][matches[:, 2].astype(np.int64)].astype(np.float32)

        # Rescale from model-pixel coords back to the input frame so the pixels
        # agree with the depth map and camera intrinsics.
        if (w0, h0) != (self.w, self.h):
            uv0 *= np.array([w0 / self.w, h0 / self.h], np.float32)
            uv1 *= np.array([w0 / self.w, h0 / self.h], np.float32)
        return uv0, uv1
