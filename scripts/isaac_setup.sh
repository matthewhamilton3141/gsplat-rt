#!/usr/bin/env bash
# Provision a headless NVIDIA Brev A10G box for gsplat-rt M7 (Isaac Sim + Isaac Lab).
#
# Companion to scripts/brev_setup.sh (which provisions the depth/SLAM pipeline). This
# installs Isaac Sim via the pip path (cleanest for a headless cloud box — no Omniverse
# Launcher, no display) into its OWN venv, then Isaac Lab on top, and runs a headless
# smoke import so the box is proven before any real work.
#
# ── UNVERIFIED SCAFFOLD ──────────────────────────────────────────────────────────────
# This has NOT been run on the box yet (Isaac Sim isn't installed anywhere on the dev
# Mac). Isaac Sim's pip package set + version pins change release-to-release, so treat
# ISAAC_VERSION and the package list as things to confirm against the release notes for
# your pinned version:
#   https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html
#   https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
# Expect to iterate the first time. Everything is idempotent so re-running is safe.
# ─────────────────────────────────────────────────────────────────────────────────────
#
# Requirements the box must already meet (checked below): NVIDIA driver new enough for
# the pinned Isaac Sim (RTX/ray-tracing — the A10G qualifies), Python 3.10, glibc 2.34+.
#
# Usage (on the Brev box):
#   OMNI_KIT_ACCEPT_EULA=YES bash scripts/isaac_setup.sh

set -euo pipefail

ISAAC_VERSION="${ISAAC_VERSION:-4.5.0}"        # confirm against current release notes
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

# Isaac Sim's pip wheels are built for CPython 3.10 specifically.
PYBIN="${PYBIN:-python3.10}"
command -v "$PYBIN" >/dev/null || die "$PYBIN not found — Isaac Sim pip wheels need Python 3.10"

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
    # The 'isaacsim[all]' extra pulls the full runtime (kit, core, replicator, RL, ...).
    # If the extra name differs for your version, install the meta + needed extensions
    # per the release notes (e.g. isaacsim-rl isaacsim-replicator isaacsim-core ...).
    python -m pip install "isaacsim[all]==${ISAAC_VERSION}" --extra-index-url "$NV_INDEX"
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
