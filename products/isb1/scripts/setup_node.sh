#!/usr/bin/env bash
# setup_node.sh — GPU node setup for ISB-1 benchmark on Ubuntu 22.04/24.04
# Checks GPU availability, installs system dependencies, creates a Python
# virtual environment, installs vLLM (pinned) and the ISB-1 harness.
set -euo pipefail

# ── Colour helpers ───────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Configuration ────────────────────────────────────────────────────────
VENV_DIR="${ISB1_VENV:-/opt/isb1/venv}"
VLLM_VERSION="0.8.4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 1. Check Ubuntu version ─────────────────────────────────────────────
info "Checking OS..."
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    case "${VERSION_ID:-}" in
        22.04|24.04) info "Detected Ubuntu ${VERSION_ID}" ;;
        *) warn "Unsupported Ubuntu version: ${VERSION_ID:-unknown}. Proceeding anyway." ;;
    esac
else
    warn "/etc/os-release not found — cannot verify OS."
fi

# ── 2. Check GPU availability ───────────────────────────────────────────
info "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found.  Install the NVIDIA driver first."
fi
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -lt 1 ]]; then
    die "No NVIDIA GPUs detected."
fi
info "Found ${GPU_COUNT} NVIDIA GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/       /'

# ── 3. Install system dependencies ──────────────────────────────────────
info "Installing system packages..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-venv python3-pip python3-dev \
    git curl wget jq \
    build-essential \
    numactl \
    libibverbs-dev  # RDMA / NVLink helpers

# ── 4. Create virtual environment ───────────────────────────────────────
info "Creating Python virtual environment at ${VENV_DIR}..."
sudo mkdir -p "$(dirname "${VENV_DIR}")"
sudo chown "$(id -u):$(id -g)" "$(dirname "${VENV_DIR}")"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

info "Upgrading pip / setuptools / wheel..."
pip install --quiet --upgrade pip setuptools wheel

# ── 5. Install vLLM (pinned version) ────────────────────────────────────
info "Installing vLLM==${VLLM_VERSION}..."
pip install --quiet "vllm==${VLLM_VERSION}"

# ── 6. Install ISB-1 harness ────────────────────────────────────────────
info "Installing ISB-1 harness from ${REPO_ROOT}..."
pip install --quiet -e "${REPO_ROOT}"

# ── 7. Install additional Python dependencies ───────────────────────────
info "Installing huggingface-cli for model downloads..."
pip install --quiet "huggingface_hub[cli]>=0.20"

# ── 8. Verify installation ──────────────────────────────────────────────
info "Verifying installation..."

PYTHON="${VENV_DIR}/bin/python"

${PYTHON} -c "import vllm; print(f'  vLLM version: {vllm.__version__}')"
INSTALLED_VLLM=$(${PYTHON} -c "import vllm; print(vllm.__version__)")
if [[ "${INSTALLED_VLLM}" != "${VLLM_VERSION}" ]]; then
    die "Expected vLLM ${VLLM_VERSION}, got ${INSTALLED_VLLM}"
fi

${PYTHON} -c "import harness; print('  ISB-1 harness: OK')"
${PYTHON} -c "import workloads; print('  Workload generators: OK')"
${PYTHON} -c "import yaml; print('  PyYAML: OK')"
${PYTHON} -c "import click; print('  Click: OK')"
${PYTHON} -c "import numpy; print(f'  NumPy: {numpy.__version__}')"

info "────────────────────────────────────────────────"
info "Node setup complete."
info "  Virtual environment : ${VENV_DIR}"
info "  vLLM version        : ${VLLM_VERSION}"
info "  GPU(s)              : ${GPU_COUNT}"
info "Activate with: source ${VENV_DIR}/bin/activate"
info "────────────────────────────────────────────────"
