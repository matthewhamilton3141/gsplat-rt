#!/usr/bin/env bash
# Download + extract a TUM RGB-D benchmark sequence for M6 SLAM work.
#
# Idempotent: skips the download if the tarball is present and skips extraction
# if the sequence directory already exists.
#
# Usage:
#   bash scripts/fetch_tum.sh                 # default: freiburg1_desk
#   bash scripts/fetch_tum.sh freiburg1_room
#   bash scripts/fetch_tum.sh freiburg2_desk  # (adjust the base URL group below)

set -euo pipefail

SEQ="${1:-freiburg1_desk}"
GROUP="${SEQ%%_*}"                                   # freiburg1 / freiburg2 / freiburg3
BASE="https://cvg.cit.tum.de/rgbd/dataset/${GROUP}"
TARBALL="rgbd_dataset_${SEQ}.tgz"
DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/tum"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -d "rgbd_dataset_${SEQ}" ]; then
    echo "[fetch_tum] rgbd_dataset_${SEQ} already extracted — nothing to do."
    exit 0
fi

if [ ! -f "$TARBALL" ]; then
    echo "[fetch_tum] Downloading ${BASE}/${TARBALL}"
    curl -fL --retry 3 -o "$TARBALL" "${BASE}/${TARBALL}"
else
    echo "[fetch_tum] $TARBALL already present — skipping download."
fi

echo "[fetch_tum] Extracting $TARBALL"
tar -xzf "$TARBALL"
echo "[fetch_tum] Ready: $DATA_DIR/rgbd_dataset_${SEQ}"
