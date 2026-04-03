#!/usr/bin/env bash
# download_datasets.sh — Download ShareGPT V3 dataset and BurstGPT traces.
# Usage: ./download_datasets.sh [DATASET_DIR]
#   DATASET_DIR defaults to /opt/isb1/datasets
set -euo pipefail

# ── Colour helpers ───────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Configuration ────────────────────────────────────────────────────────
DATASET_DIR="${1:-/opt/isb1/datasets}"

SHAREGPT_DIR="${DATASET_DIR}/sharegpt_v3"
SHAREGPT_FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/${SHAREGPT_FILE}"

BURSTGPT_DIR="${DATASET_DIR}/burstgpt"
BURSTGPT_REPO="https://huggingface.co/datasets/HPMLL/BurstGPT"

# ── Pre-flight ───────────────────────────────────────────────────────────
for cmd in wget git; do
    if ! command -v "${cmd}" &>/dev/null; then
        die "'${cmd}' is required but not found. Run setup_node.sh first."
    fi
done

info "Dataset directory: ${DATASET_DIR}"
mkdir -p "${DATASET_DIR}"

# ── 1. ShareGPT V3 ──────────────────────────────────────────────────────
info "──── ShareGPT V3 ────────────────────────────"
mkdir -p "${SHAREGPT_DIR}"

if [[ -f "${SHAREGPT_DIR}/${SHAREGPT_FILE}" ]]; then
    FILE_SIZE=$(stat -c%s "${SHAREGPT_DIR}/${SHAREGPT_FILE}" 2>/dev/null || echo 0)
    if [[ "${FILE_SIZE}" -gt 1000000 ]]; then
        info "ShareGPT already downloaded (${FILE_SIZE} bytes). Skipping."
    else
        warn "ShareGPT file exists but is suspiciously small (${FILE_SIZE} bytes). Re-downloading..."
        rm -f "${SHAREGPT_DIR}/${SHAREGPT_FILE}"
    fi
fi

if [[ ! -f "${SHAREGPT_DIR}/${SHAREGPT_FILE}" ]]; then
    info "Downloading ShareGPT V3 from HuggingFace..."
    wget -q --show-progress -O "${SHAREGPT_DIR}/${SHAREGPT_FILE}" "${SHAREGPT_URL}" || {
        warn "Direct download failed. Trying git-lfs clone..."
        if command -v git-lfs &>/dev/null || git lfs version &>/dev/null; then
            TMPDIR=$(mktemp -d)
            git clone --depth 1 "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered" "${TMPDIR}/sharegpt_repo"
            cp "${TMPDIR}/sharegpt_repo/${SHAREGPT_FILE}" "${SHAREGPT_DIR}/${SHAREGPT_FILE}"
            rm -rf "${TMPDIR}"
        else
            die "git-lfs not available. Install with: sudo apt-get install git-lfs"
        fi
    }
    info "ShareGPT V3 downloaded to ${SHAREGPT_DIR}/${SHAREGPT_FILE}"
fi

# Validate JSON
info "Validating ShareGPT JSON..."
if command -v jq &>/dev/null; then
    ENTRY_COUNT=$(jq 'length' "${SHAREGPT_DIR}/${SHAREGPT_FILE}" 2>/dev/null || echo "error")
    if [[ "${ENTRY_COUNT}" == "error" ]]; then
        warn "ShareGPT JSON validation failed — file may be corrupt."
    else
        info "ShareGPT contains ${ENTRY_COUNT} conversations."
    fi
else
    warn "jq not installed — skipping JSON validation."
fi

# ── 2. BurstGPT Traces ──────────────────────────────────────────────────
info "──── BurstGPT Traces ──────────────────────────"
mkdir -p "${BURSTGPT_DIR}"

if [[ -d "${BURSTGPT_DIR}/.git" ]] || [[ -f "${BURSTGPT_DIR}/README.md" ]]; then
    info "BurstGPT traces already present. Updating..."
    (cd "${BURSTGPT_DIR}" && git pull --ff-only 2>/dev/null) || warn "BurstGPT update failed (may be a partial clone)."
else
    info "Cloning BurstGPT traces from HuggingFace..."
    if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
        GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 "${BURSTGPT_REPO}" "${BURSTGPT_DIR}" || {
            warn "Full clone failed. Trying without LFS..."
            GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${BURSTGPT_REPO}" "${BURSTGPT_DIR}"
            warn "BurstGPT cloned without LFS — large files may be pointers."
        }
    else
        warn "git-lfs not available — cloning without LFS."
        GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "${BURSTGPT_REPO}" "${BURSTGPT_DIR}"
    fi
    info "BurstGPT traces downloaded to ${BURSTGPT_DIR}"
fi

# ── Summary ──────────────────────────────────────────────────────────────
info "────────────────────────────────────────────────"
info "Dataset download complete."
info "  ShareGPT V3 : ${SHAREGPT_DIR}/${SHAREGPT_FILE}"
info "  BurstGPT    : ${BURSTGPT_DIR}"
du -sh "${SHAREGPT_DIR}" "${BURSTGPT_DIR}" 2>/dev/null | sed 's/^/       /'
info "────────────────────────────────────────────────"
