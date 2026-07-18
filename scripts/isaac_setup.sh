#!/usr/bin/env bash
# Provision a headless NVIDIA Brev A10G box for gsplat-rt M7 (Isaac Sim + Isaac Lab).
#
# Companion to scripts/brev_setup.sh (which provisions the depth/SLAM pipeline). This
# installs Isaac Sim via the pip path (cleanest for a headless cloud box — no Omniverse
# Launcher, no display) into its OWN venv, then Isaac Lab on top, and runs a headless
# smoke import so the box is proven before any real work.
#
# ── PARTIALLY VERIFIED (2026-07-18, A10G Brev box) ────────────────────────────────────
# Isaac Sim 5.1.0 pip-installs and Kit BOOTS on the box (extensions load), BUT: the RTX
# scene renderer (librtx.scenedb.plugin / libcarb.scenerenderer-rtx / libomni.hydra.rtx)
# SEGFAULTS at startup on this bare headless A10G — even with a base experience and
# {"renderer":"None"}. It's NOT OOM (13.5 GB free at crash) and NOT our scene (validated
# 0-ERROR by usd_isaac_check). It's a missing/incompatible headless GRAPHICS runtime
# (Vulkan/GL userspace) that pip Isaac doesn't bundle.
#   → RECOMMENDED FIX for the next attempt: use NVIDIA's Isaac Sim **Docker container**
#     (bundles the correct Vulkan/GL/driver userspace), or run under a virtual display
#     (Xvfb) with the Vulkan ICD set up. A pip install on a bare box is the fragile path.
#   → Also: this g5.xlarge has only 16 GB RAM (Isaac recommends 32 GB) — fine for the
#     Phase-0 drop-test, but add a bigger box before RL training with many parallel envs.
# The reconstruct→physics bridge is ALREADY proven via the PyBullet backend (nav_pybullet)
# — Isaac is the eventual GPU-parallel/photoreal target, not a correctness gate.
#
# VERIFIED requirements/steps (2026-07-18): driver 595.71 (>=580.65 ✓), Ubuntu 22.04,
# Python 3.11 (Isaac 5.X needs 3.11 — get it via `uv python install 3.11`, no sudo), and
# these system libs the headless renderer dlopen's (else "libXt.so.6/libGLU.so.1 not found"):
#   sudo apt-get install -y libglu1-mesa libxt6 libgl1 libglib2.0-0 libsm6 libice6 \
#        libxrender1 libxext6 libxrandr2 libxi6 libxcursor1 libxinerama1
# Isaac Sim pip needs the EULA env var at import (not just install): OMNI_KIT_ACCEPT_EULA=YES.
# Everything is idempotent so re-running is safe.
# ─────────────────────────────────────────────────────────────────────────────────────
#
# Usage (on the Brev box):
#   OMNI_KIT_ACCEPT_EULA=YES bash scripts/isaac_setup.sh

set -euo pipefail

ISAAC_VERSION="${ISAAC_VERSION:-5.1.0}"        # latest stable (Jan 2026); needs Python 3.11
ISAAC_VENV="${ISAAC_VENV:-$HOME/isaacsim-venv}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
NV_INDEX="https://pypi.nvidia.com"

log() { printf '\n[isaac_setup] %s\n' "$*"; }
die() { printf '\n[isaac_setup] ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Sanity: GPU, driver, python, EULA
# ---------------------------------------------------------------------------
command -v nvidia-smi >/dev/null || die "nvidia-smi not found — is this a GPU instance?"
log "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"

# System libs the (headless) RTX renderer dlopen's — without these Kit fails with
# "libXt.so.6 / libGLU.so.1: cannot open shared object file". Verified needed 2026-07-18.
if command -v apt-get >/dev/null; then
    log "Installing headless graphics system libs (needs sudo)"
    sudo apt-get update -qq || log "WARN: apt-get update failed (continuing)"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        libglu1-mesa libxt6 libgl1 libglib2.0-0 libsm6 libice6 libxrender1 \
        libxext6 libxrandr2 libxi6 libxcursor1 libxinerama1 || \
        log "WARN: system-lib install failed — the RTX renderer may crash headless"
fi

# Isaac Sim 5.X pip wheels need CPython 3.11 (4.X used 3.10). If python3.11 isn't on PATH,
# `uv python install 3.11` provides it without sudo (the box has uv).
PYBIN="${PYBIN:-python3.11}"
command -v "$PYBIN" >/dev/null || die "$PYBIN not found — Isaac Sim 5.X needs Python 3.11 (try: uv python install 3.11)"

if [ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]; then
    die "Set OMNI_KIT_ACCEPT_EULA=YES to accept the NVIDIA Omniverse license (headless install)."
fi

# ---------------------------------------------------------------------------
# 1. Dedicated venv (Isaac Sim pulls a large, pinned dep tree — keep it isolated)
# ---------------------------------------------------------------------------
if [ ! -d "$ISAAC_VENV" ]; then
    log "Creating Isaac venv at $ISAAC_VENV ($PYBIN)"
    "$PYBIN" -m venv "$ISAAC_VENV"
fi
# shellcheck disable=SC1091
source "$ISAAC_VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null

# ---------------------------------------------------------------------------
# 2. Isaac Sim (pip)
# ---------------------------------------------------------------------------
if python -c "import isaacsim" 2>/dev/null; then
    log "isaacsim already importable — skipping install"
else
    log "Installing Isaac Sim $ISAAC_VERSION from $NV_INDEX (large download, several minutes)"
    # 5.1.0 uses the 'all,extscache' extras (extscache ships the Kit extension cache).
    # Verified command 2026-07-18: pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url ...
    python -m pip install "isaacsim[all,extscache]==${ISAAC_VERSION}" --extra-index-url "$NV_INDEX"
fi

# ---------------------------------------------------------------------------
# 3. Headless smoke: can we boot a SimulationApp with no display?
# ---------------------------------------------------------------------------
log "Headless SimulationApp smoke (first boot compiles shaders — can take a few minutes)"
OMNI_KIT_ACCEPT_EULA=YES python - <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("[isaac_setup] SimulationApp booted headless OK")
app.close()
PY

# ---------------------------------------------------------------------------
# 4. Isaac Lab (RL task framework + rsl_rl) on top of Isaac Sim
# ---------------------------------------------------------------------------
if [ -d "$ISAACLAB_DIR/.git" ]; then
    log "Isaac Lab present — pulling latest"
    git -C "$ISAACLAB_DIR" pull --ff-only || log "WARN: Isaac Lab pull skipped"
else
    log "Cloning Isaac Lab -> $ISAACLAB_DIR"
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
log "Installing Isaac Lab (uses the active Isaac venv's python)"
# isaaclab.sh --install wires Isaac Lab against the pip Isaac Sim in this venv and
# installs an RL library (rsl_rl by default). Confirm the flag/rl-lib for your version.
( cd "$ISAACLAB_DIR" && ./isaaclab.sh --install rsl_rl ) || \
    die "Isaac Lab install failed — see output; check the isaaclab.sh flags for $ISAAC_VERSION"

python -c "import isaaclab; print('[isaac_setup] isaaclab importable OK')" || \
    die "isaaclab not importable after install"

# ---------------------------------------------------------------------------
# 5. Done
# ---------------------------------------------------------------------------
log "Done. Next (activate the venv first: source $ISAAC_VENV/bin/activate):"
log "  Phase 0 smoke : python ~/gsplat-rt/scripts/isaac/phase0_smoke.py --usdz <scene.usdz>"
log "  (validate the .usdz on ANY box first: python -m src.mapping.usd_isaac_check <scene.usdz>)"
