#!/usr/bin/env bash
# install_telemetry.sh — Install DCGM Exporter for GPU telemetry
# Checks if already installed, installs NVIDIA DCGM if not, and verifies.
set -euo pipefail

# ── Colour helpers ───────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── 1. Check for NVIDIA driver ──────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found. Install the NVIDIA driver before telemetry."
fi

# ── 2. Check if DCGM is already installed ───────────────────────────────
if command -v dcgmi &>/dev/null; then
    info "DCGM is already installed."
    DCGM_VERSION=$(dcgmi --version 2>/dev/null | head -n1 || echo "unknown")
    info "  Version: ${DCGM_VERSION}"
else
    info "DCGM not found — installing..."

    # ── 3. Add NVIDIA DCGM repository ───────────────────────────────────
    export DEBIAN_FRONTEND=noninteractive

    # Detect Ubuntu version for the correct repo
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        UBUNTU_VERSION="${VERSION_ID:-22.04}"
    else
        UBUNTU_VERSION="22.04"
    fi
    DISTRO="ubuntu${UBUNTU_VERSION//./}"

    info "Adding NVIDIA DCGM repo for ${DISTRO}..."
    KEYRING_DIR="/usr/share/keyrings"
    sudo mkdir -p "${KEYRING_DIR}"

    # Fetch and install the CUDA keyring package (covers DCGM repo)
    CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb"
    TMPFILE=$(mktemp /tmp/cuda-keyring-XXXXXX.deb)
    info "Downloading CUDA keyring from ${CUDA_KEYRING_URL}..."
    wget -q -O "${TMPFILE}" "${CUDA_KEYRING_URL}" || die "Failed to download CUDA keyring."
    sudo dpkg -i "${TMPFILE}"
    rm -f "${TMPFILE}"

    sudo apt-get update -qq

    # ── 4. Install DCGM ─────────────────────────────────────────────────
    info "Installing datacenter-gpu-manager..."
    sudo apt-get install -y -qq datacenter-gpu-manager

    # Enable and start nv-hostengine (DCGM daemon)
    info "Enabling nv-hostengine service..."
    if systemctl list-unit-files | grep -q nv-hostengine; then
        sudo systemctl enable nv-hostengine
        sudo systemctl start nv-hostengine || warn "nv-hostengine failed to start (may need reboot)."
    else
        # Fallback: start the host engine manually
        warn "nv-hostengine systemd unit not found — starting manually."
        sudo nv-hostengine -D 2>/dev/null || warn "nv-hostengine may already be running."
    fi
fi

# ── 5. Install DCGM Exporter (Prometheus metrics) ───────────────────────
if command -v dcgm-exporter &>/dev/null; then
    info "dcgm-exporter binary already present."
else
    info "Installing dcgm-exporter..."
    sudo apt-get install -y -qq dcgm-exporter 2>/dev/null || {
        warn "dcgm-exporter apt package not available; pulling container image instead."
        if command -v docker &>/dev/null; then
            docker pull nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04
            info "dcgm-exporter container pulled. Run with:"
            info "  docker run -d --gpus all -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:3.3.8-3.6.0-ubuntu22.04"
        else
            warn "docker not found — skipping dcgm-exporter container pull."
        fi
    }
fi

# ── 6. Verify DCGM ─────────────────────────────────────────────────────
info "Verifying DCGM installation with 'dcgmi discovery -l'..."
if dcgmi discovery -l; then
    info "DCGM discovery succeeded."
else
    warn "dcgmi discovery returned non-zero. The host engine may not be running."
    warn "Try: sudo nv-hostengine -D && dcgmi discovery -l"
fi

info "────────────────────────────────────────────────"
info "Telemetry setup complete."
info "  dcgmi:          $(command -v dcgmi 2>/dev/null || echo 'not found')"
info "  dcgm-exporter:  $(command -v dcgm-exporter 2>/dev/null || echo 'container')"
info "────────────────────────────────────────────────"
