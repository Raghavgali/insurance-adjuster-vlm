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

MODEL_ID="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import yaml
config_path = Path("GLM/configs/runpod.yaml")
env_path = Path(__import__("os").environ.get("CONFIG_PATH", "GLM/configs/runpod.yaml"))
path = env_path if env_path.exists() else config_path
config = yaml.safe_load(path.read_text())
model_cfg = config.get("model", {})
model_id = config.get("model_id", model_cfg.get("model_id"))
print(model_id or "")
PY
)"

if [[ -n "${MODEL_ID}" && "${MODEL_ID}" == */* && ! -d "${MODEL_ID}" ]]; then
  echo "[launch] prefetch_model=${MODEL_ID}"
  MODEL_ID="${MODEL_ID}" "${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

snapshot_download(
    repo_id=model_id,
    token=token,
)
PY
fi

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
