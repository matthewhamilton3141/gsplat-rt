#!/usr/bin/env bash
# Fetch ONE NVIDIA NuRec reconstructed driving clip and drive the shielded car through it.
#
# NuRec ships real drives reconstructed as 3DGS + a surface mesh (+ .xodr map), ~20 s clips as
# USDZ (~2 GB each). We only need the surface mesh: src/isaac/nurec_scene.py projects it to a
# GridWorld and the shielded DWA car drives it — all CPU, no box.
#
# ONE-TIME, INTERACTIVE (cannot be scripted — do these yourself first):
#   1) Accept the dataset terms (click "Agree and access"):
#        https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
#   2) Log in with a token from https://hf.co/settings/tokens :
#        hf auth login
#
# USAGE:
#   scripts/fetch_nurec.sh --list                 # print the dataset file tree, pick a clip
#   scripts/fetch_nurec.sh '<clip_dir>/**'        # download just that clip into $NUREC_DIR
#   NUREC_DIR=~/data/nurec scripts/fetch_nurec.sh '<clip_dir>/**'
set -euo pipefail

REPO="nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
DEST="${NUREC_DIR:-$HOME/nurec}"

command -v hf >/dev/null 2>&1 || { echo "hf CLI missing: pip install -U 'huggingface_hub[cli]'"; exit 1; }

if [ "${1:-}" = "--list" ] || [ -z "${1:-}" ]; then
  echo "# Dataset files (first 80) — pick a clip directory, then re-run with '<clip_dir>/**':"
  python3 - <<'PY'
from huggingface_hub import list_repo_files
try:
    files = list_repo_files("nvidia/PhysicalAI-Autonomous-Vehicles-NuRec", repo_type="dataset")
except Exception as e:
    raise SystemExit(f"could not list (accept terms + `hf auth login` first): {e}")
for f in files[:80]:
    print(" ", f)
print(f"... {len(files)} files total")
PY
  exit 0
fi

CLIP_GLOB="$1"
echo "Downloading '$CLIP_GLOB' from $REPO -> $DEST ..."
hf download "$REPO" --repo-type dataset --include "$CLIP_GLOB" --local-dir "$DEST"

USDZ="$(find "$DEST" -name '*.usdz' | head -1 || true)"
echo
echo "Done -> $DEST"
if [ -n "$USDZ" ]; then
  echo "Drive it (no box):"
  echo "  python3 scripts/nav/drive_scene.py --mesh '$USDZ' --out /tmp/nurec_drive.mp4 --png /tmp/nurec_drive.png"
  echo "  # if the car drives through walls, the mesh is Y-up: add --mesh-up-axis 1"
else
  echo "No .usdz found under $DEST — check the clip path with '--list'."
fi
