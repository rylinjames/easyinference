#!/usr/bin/env bash
# download_models.sh — Download HuggingFace models listed in configs/models/*.yaml.
# Usage:
#   ./download_models.sh                  # Download all models
#   ./download_models.sh --model dsr1     # Download a specific model by short name
#   ./download_models.sh --model meta-llama/Llama-3.3-70B-Instruct  # By HF id
set -euo pipefail

# ── Colour helpers ───────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Parse arguments ─────────────────────────────────────────────────────
MODEL_FILTER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL_FILTER="${2:-}"
            [[ -z "${MODEL_FILTER}" ]] && die "--model requires an argument"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--model <short_name|hf_model_id>]"
            echo ""
            echo "Downloads HuggingFace models from configs/models/*.yaml."
            echo "Without --model, downloads all models marked available: true."
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

# ── Locate config directory ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_CONFIG_DIR="${REPO_ROOT}/configs/models"

if [[ ! -d "${MODEL_CONFIG_DIR}" ]]; then
    die "Model config directory not found: ${MODEL_CONFIG_DIR}"
fi

# ── Check prerequisites ─────────────────────────────────────────────────
VENV_DIR="${ISB1_VENV:-/opt/isb1/venv}"
if [[ -f "${VENV_DIR}/bin/huggingface-cli" ]]; then
    HF_CLI="${VENV_DIR}/bin/huggingface-cli"
elif command -v huggingface-cli &>/dev/null; then
    HF_CLI="huggingface-cli"
else
    die "huggingface-cli not found. Run setup_node.sh or: pip install 'huggingface_hub[cli]'"
fi

info "Using: ${HF_CLI}"

# ── Verify we can parse YAML (use Python for portability) ────────────────
PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    if command -v python3 &>/dev/null; then
        PYTHON="python3"
    else
        die "Python not found. Run setup_node.sh first."
    fi
fi

# ── Extract model list from YAML configs ─────────────────────────────────
# Produces lines of: model_short|hf_model_id|available
MODEL_LIST=$("${PYTHON}" -c "
import yaml, pathlib, sys
config_dir = pathlib.Path('${MODEL_CONFIG_DIR}')
for p in sorted(config_dir.glob('*.yaml')):
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        continue
    short = cfg.get('model_short', '')
    hf_id = cfg.get('hf_model_id', '')
    avail = cfg.get('available', False)
    print(f'{short}|{hf_id}|{avail}')
") || die "Failed to parse model configs."

if [[ -z "${MODEL_LIST}" ]]; then
    die "No models found in ${MODEL_CONFIG_DIR}"
fi

# ── Filter and download ─────────────────────────────────────────────────
DOWNLOADED=0
SKIPPED=0
FAILED=0

while IFS='|' read -r model_short hf_model_id available; do
    # Skip unavailable models unless explicitly requested
    if [[ "${available}" != "True" && -z "${MODEL_FILTER}" ]]; then
        warn "Skipping ${model_short} (${hf_model_id}) — not available."
        ((SKIPPED++)) || true
        continue
    fi

    # Apply filter if specified
    if [[ -n "${MODEL_FILTER}" ]]; then
        if [[ "${MODEL_FILTER}" != "${model_short}" && "${MODEL_FILTER}" != "${hf_model_id}" ]]; then
            continue
        fi
    fi

    info "──────────────────────────────────────────────"
    info "Downloading: ${model_short} (${hf_model_id})"

    if "${HF_CLI}" download "${hf_model_id}" --quiet 2>/dev/null; then
        info "Successfully downloaded ${model_short}."
        ((DOWNLOADED++)) || true
    else
        # Retry without --quiet for better diagnostics
        warn "Retrying with verbose output..."
        if "${HF_CLI}" download "${hf_model_id}"; then
            info "Successfully downloaded ${model_short} (retry)."
            ((DOWNLOADED++)) || true
        else
            warn "Failed to download ${model_short} (${hf_model_id})."
            ((FAILED++)) || true
        fi
    fi

done <<< "${MODEL_LIST}"

# ── Check filter matched ────────────────────────────────────────────────
if [[ -n "${MODEL_FILTER}" && "${DOWNLOADED}" -eq 0 && "${FAILED}" -eq 0 ]]; then
    die "Model '${MODEL_FILTER}' not found in configs. Available models:"$'\n'"$(echo "${MODEL_LIST}" | cut -d'|' -f1,2 | tr '|' ' ')"
fi

# ── Summary ──────────────────────────────────────────────────────────────
info "────────────────────────────────────────────────"
info "Model download complete."
info "  Downloaded : ${DOWNLOADED}"
info "  Skipped    : ${SKIPPED}"
info "  Failed     : ${FAILED}"
info "  Cache dir  : $(${HF_CLI} env 2>/dev/null | grep -i 'cache' | head -1 || echo 'default')"
info "────────────────────────────────────────────────"

[[ "${FAILED}" -gt 0 ]] && exit 1
exit 0
