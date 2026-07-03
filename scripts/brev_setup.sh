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
python3 -m pip install --upgrade pip -q
python3 -m pip install -r requirements.txt -q

# TensorRT lives on NVIDIA's index, not PyPI's default
log "Installing TensorRT (NGC index)"
python3 -m pip install "tensorrt>=9.0.0" --extra-index-url https://pypi.ngc.nvidia.com -q

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
    log "Exporting Depth Anything V2 Small to ONNX (downloads ~100 MB from HuggingFace)"
    python3 src/depth/export_onnx.py
fi

if [ -f models/depth_engine.engine ]; then
    log "TensorRT engine already built — skipping (delete models/depth_engine.engine to rebuild)"
else
    log "Building TensorRT FP16 engine (2-5 minutes)"
    python3 src/depth/compile_trt.py
fi

# ---------------------------------------------------------------------------
# 4. Custom CUDA kernels (no-op while kernels/ has no .cu files)
# ---------------------------------------------------------------------------
if ls kernels/*.cu >/dev/null 2>&1; then
    log "Compiling custom CUDA kernels"
    python3 setup.py build_ext --inplace
else
    log "No .cu files in kernels/ — skipping kernel build"
fi

# ---------------------------------------------------------------------------
# 4b. TUM RGB-D sequence for M6 SLAM (real depth + ground-truth poses)
# ---------------------------------------------------------------------------
# Set FETCH_TUM=0 to skip (e.g. a depth-only benchmark run).
if [ "${FETCH_TUM:-1}" = "1" ]; then
    log "Fetching TUM ${TUM_SEQ:-freiburg1_desk} for SLAM work"
    bash scripts/fetch_tum.sh "${TUM_SEQ:-freiburg1_desk}"
else
    log "FETCH_TUM=0 — skipping TUM dataset download"
fi

# ---------------------------------------------------------------------------
# 5. Full test suite (GPU benchmarks included)
# ---------------------------------------------------------------------------
log "Running full test suite"
python3 -m pytest tests/ -v

log "Done. Next: python3 scripts/bench_pipeline.py --out output/bench_results.json"
