#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-GLM/configs/runpod.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/train_$(date +%Y%m%d_%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)}"
RESUME_PATH="${RESUME_PATH:-}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export WANDB_DIR="${WANDB_DIR:-/workspace/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${WANDB_DIR}"

echo "[launch] repo_root=${REPO_ROOT}"
echo "[launch] config=${CONFIG_PATH}"
echo "[launch] output_dir=${OUTPUT_DIR}"
echo "[launch] nproc_per_node=${NPROC_PER_NODE}"
echo "[launch] master_port=${MASTER_PORT}"

CMD=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  GLM/scripts/train.py
  --config "${CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${RESUME_PATH}" ]]; then
  CMD+=(--resume "${RESUME_PATH}")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[launch] command: ${CMD[*]}"
"${CMD[@]}"
