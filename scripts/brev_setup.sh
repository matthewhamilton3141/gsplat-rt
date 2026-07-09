#!/usr/bin/env bash
# Bootstrap a fresh NVIDIA Brev GPU instance for gsplat-rt.
#
# Target: A10G (Ampere, sm_86 — matches setup.py's -arch flag). Brev images
# ship with CUDA, Python 3.10+, and pip preinstalled.
#
# Idempotent: safe to re-run after a partial failure; every step skips work
# that is already done (existing clone, built engine, installed packages).
#
# Usage (on the Brev box):
#   bash <(curl -fsSL https://raw.githubusercontent.com/matthewhamilton3141/gsplat-rt/main/scripts/brev_setup.sh)
# or, if the repo is already cloned:
#   bash scripts/brev_setup.sh

set -euo pipefail

REPO_URL="https://github.com/matthewhamilton3141/gsplat-rt.git"
REPO_DIR="${REPO_DIR:-$HOME/gsplat-rt}"

log() { printf '\n[brev_setup] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 0. Sanity: GPU visible?
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null; then
    echo "[brev_setup] ERROR: nvidia-smi not found — is this a GPU instance?" >&2
    exit 1
fi
log "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"

# ---------------------------------------------------------------------------
# 1. Repo
# ---------------------------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
    log "Repo exists — pulling latest main"
    git -C "$REPO_DIR" pull --ff-only origin main
else
    log "Cloning $REPO_URL"
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# 2. Python dependencies
# ---------------------------------------------------------------------------
log "Installing Python dependencies"
# Some Brev VM images ship a venv Python without pip — bootstrap it if missing.
if ! python3 -m pip --version >/dev/null 2>&1; then
    log "pip not found in this Python — bootstrapping via ensurepip"
    python3 -m ensurepip --upgrade
fi
python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt -q

# TensorRT lives on NVIDIA's index, not PyPI's default.
# PIN TO MAJOR 10: onnxruntime-gpu's TensorRT execution provider is compiled
# against a specific libnvinfer SONAME. onnxruntime-gpu 1.2x links libnvinfer.so.10
# (TensorRT 10). An unpinned ">=9.0.0" grabs the newest wheel (TensorRT 11 ->
# libnvinfer.so.11), which the EP cannot dlopen -- it fails with
# "libnvinfer.so.10: cannot open shared object file" and silently drops to CUDA.
# Keep this range aligned with the onnxruntime-gpu pin in requirements.txt.
log "Installing TensorRT (NGC index)"
python3 -m pip install "tensorrt>=10,<11" --extra-index-url https://pypi.ngc.nvidia.com -q

# pxr (OpenUSD) — enables the USD tests instead of mock-only runs
log "Installing usd-core (pxr)"
python3 -m pip install usd-core -q

python3 - <<'EOF'
import torch, tensorrt
from pxr import Usd
assert torch.cuda.is_available(), "torch sees no CUDA device"
print(f"[brev_setup] torch {torch.__version__} | CUDA OK | TensorRT {tensorrt.__version__} | pxr OK")
cc = torch.cuda.get_device_capability()
print(f"[brev_setup] Compute capability: sm_{cc[0]}{cc[1]} (setup.py targets sm_86)")
EOF

# ---------------------------------------------------------------------------
# 3. Depth model: ONNX export → TensorRT engine
# ---------------------------------------------------------------------------
if [ -f models/depth_v2_small.onnx ]; then
    log "ONNX model already exported — skipping"
else
    log "Exporting Depth Anything V2 Small to ONNX + fp16 (downloads ~100 MB from HuggingFace)"
    python3 src/depth/export_onnx.py --fp16
fi

# fp16 ONNX may be missing on boxes provisioned before the --fp16 flag existed.
if [ ! -f models/depth_v2_small_fp16.onnx ]; then
    log "Converting fp32 ONNX → fp16"
    python3 -c "from src.depth.export_onnx import to_fp16; to_fp16()" \
        || python3 -c "import sys; sys.path.insert(0,'src'); from depth.export_onnx import to_fp16; to_fp16()"
fi

# The engine builds are wrapped so a failure warns but never aborts the rest of
# the bootstrap (kernel build, TUM fetch, tests). With `set -e` an unguarded
# failure here would kill the whole script — which is exactly how a broken FP16
# build once left a box with no CUDA kernel and no dataset.
if [ -f models/depth_engine.engine ]; then
    log "TensorRT engine (TF32) already built — skipping (delete to rebuild)"
else
    log "Building TensorRT engine — default (TF32 on Ampere) (2-5 minutes)"
    python3 src/depth/compile_trt.py \
        || log "WARNING: TF32 engine build failed — pipeline falls back to the mock estimator"
fi

if [ -f models/depth_engine_fp16.engine ]; then
    log "TensorRT engine (FP16 strongly-typed) already built — skipping"
else
    log "Building TensorRT engine — true FP16 strongly-typed (2-5 minutes)"
    python3 src/depth/compile_trt.py --fp16 \
        || log "WARNING: FP16 engine build failed — continuing without it (TF32 engine still usable)"
fi

# ---------------------------------------------------------------------------
# 4. Custom CUDA kernels (no-op while kernels/ has no .cu files)
# ---------------------------------------------------------------------------
if ls kernels/*.cu >/dev/null 2>&1; then
    log "Compiling custom CUDA kernels (CUDA TSDF integrate)"
    python3 setup.py build_ext --inplace \
        && python3 -c "import sys; sys.path.insert(0,'src'); from mapping import tsdf_cuda; assert tsdf_cuda.available(); print('[brev_setup] CUDA TSDF kernel: available')" \
        || log "WARNING: CUDA kernel build/import failed — TSDF falls back to numpy (~13 ms vs 0.3 ms)"
else
    log "No .cu files in kernels/ — skipping kernel build"
fi

# ---------------------------------------------------------------------------
# 4b. TUM RGB-D sequence for M6 SLAM (real depth + ground-truth poses)
# ---------------------------------------------------------------------------
# Set FETCH_TUM=0 to skip (e.g. a depth-only benchmark run).
if [ "${FETCH_TUM:-1}" = "1" ]; then
    log "Fetching TUM ${TUM_SEQ:-freiburg1_desk} for SLAM work"
    bash scripts/fetch_tum.sh "${TUM_SEQ:-freiburg1_desk}" \
        || log "WARNING: TUM fetch failed — SLAM/metric-scale eval will skip (re-run scripts/fetch_tum.sh)"
else
    log "FETCH_TUM=0 — skipping TUM dataset download"
fi

# ---------------------------------------------------------------------------
# 5. Full test suite (GPU benchmarks included)
# ---------------------------------------------------------------------------
log "Running full test suite"
python3 -m pytest tests/ -v \
    || log "WARNING: some tests failed — see output above"

log "Done. Next:"
log "  FP16 pipeline bench : python3 scripts/bench_pipeline.py --engine models/depth_engine_fp16.engine --out output/bench_results.json"
log "  depth TF32 vs FP16  : python3 scripts/bench_depth.py --frames 200"
log "  TUM metric scale    : python3 scripts/eval_metric_scale.py --tum data/tum/rgbd_dataset_freiburg1_desk"
