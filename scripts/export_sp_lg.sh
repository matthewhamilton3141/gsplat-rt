#!/usr/bin/env bash
# Export the fused SuperPoint+LightGlue ONNX used by the 'superpoint' pose
# tracker (PipelineConfig.pose_tracking="superpoint", scripts/eval_odometry.py
# --frontend superpoint). Idempotent.
#
# Produces: models/sp_lg_tum.onnx  — fused extractor+matcher, fixed [2,1,480,640],
# 1024 keypoints. Exported WITHOUT --fuse-multi-head-attention (that fusion is
# ONNX-Runtime-only) so the graph also compiles to a TensorRT engine.
#
# Uses fabio-sim/LightGlue-ONNX in an isolated uv venv, so it never touches the
# pipeline's system Python env. Run on a machine with the export deps (a GPU box
# is simplest; export itself is CPU-capable).
#
# Usage:
#   bash scripts/export_sp_lg.sh                 # -> <repo>/models/sp_lg_tum.onnx
#   bash scripts/export_sp_lg.sh /path/out.onnx  # custom output path
set -euo pipefail

REPO_DIR="${LIGHTGLUE_ONNX_DIR:-$HOME/LightGlue-ONNX}"
OUT="${1:-$(cd "$(dirname "$0")/.." && pwd)/models/sp_lg_tum.onnx}"

if [ -f "$OUT" ]; then
    echo "[export_sp_lg] $OUT already exists — skipping (delete it to re-export)"
    exit 0
fi

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[export_sp_lg] cloning LightGlue-ONNX into $REPO_DIR"
    git clone https://github.com/fabio-sim/LightGlue-ONNX "$REPO_DIR"
fi

cd "$REPO_DIR"
command -v uv >/dev/null 2>&1 || pip install uv
uv sync --group export

mkdir -p "$(dirname "$OUT")"
uv run lightglue-onnx export superpoint --num-keypoints 1024 -b 2 -h 480 -w 640 -o "$OUT"
echo "[export_sp_lg] wrote $OUT"
