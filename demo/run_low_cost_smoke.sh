#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  demo/run_low_cost_smoke.sh --endpoint https://<endpoint>
  MODAL_WORKSPACE=<workspace> demo/run_low_cost_smoke.sh --deploy-modal

Behavior:
  - optionally deploys demo/modal_vllm.py
  - warms the endpoint via /v1/models
  - runs InferScope coding-smoke benchmark-plan
  - runs InferScope profile-runtime against the same metrics surface
  - runs ISB-1 simple and coding quick benches
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENDPOINT=""
DEPLOY_MODAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --endpoint)
      ENDPOINT="${2:-}"
      shift 2
      ;;
    --deploy-modal)
      DEPLOY_MODAL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$DEPLOY_MODAL" -eq 1 ]]; then
  : "${MODAL_WORKSPACE:?Set MODAL_WORKSPACE when using --deploy-modal.}"
  if ! command -v modal >/dev/null 2>&1; then
    echo "modal CLI not found on PATH" >&2
    exit 1
  fi
  (cd "${REPO_ROOT}" && modal deploy demo/modal_vllm.py)
  ENDPOINT="https://${MODAL_WORKSPACE}--easyinference-demo-serve.modal.run"
fi

if [[ -z "${ENDPOINT}" ]]; then
  echo "Either --endpoint or --deploy-modal is required." >&2
  usage >&2
  exit 1
fi

echo "Using endpoint: ${ENDPOINT}"
curl -fsS "${ENDPOINT%/}/v1/models" >/dev/null
echo "Endpoint warmup check passed."

(
  cd "${REPO_ROOT}/products/inferscope"
  uv sync --dev --no-editable
  uv run inferscope benchmark-plan coding-smoke "${ENDPOINT}" --gpu a10g --num-gpus 1
  uv run inferscope profile-runtime "${ENDPOINT}" --metrics-endpoint "${ENDPOINT}" --scrape-timeout-seconds 90
)

(
  cd "${REPO_ROOT}/products/isb1"
  uv sync --dev --no-editable
  uv run --no-sync isb1 quick-bench "${ENDPOINT}" --model-id Qwen2.5-7B-Instruct --requests 1 --duration 120
  uv run --no-sync isb1 quick-bench "${ENDPOINT}" --model-id Qwen2.5-7B-Instruct --workload coding --requests 1 --duration 120
)
