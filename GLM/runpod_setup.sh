#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALLER="${INSTALLER:-pip}"
CREATE_VENV="${CREATE_VENV:-0}"
AUTO_DOWNLOAD_DATASET="${AUTO_DOWNLOAD_DATASET:-0}"
CONFIG_PATH="${CONFIG_PATH:-GLM/configs/runpod.yaml}"
DOWNLOAD_CONFIG_PATH="${DOWNLOAD_CONFIG_PATH:-GLM/configs/download_dataset.yaml}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[setup] repo_root=${REPO_ROOT}"
echo "[setup] python=${PYTHON_BIN}"
echo "[setup] installer=${INSTALLER}"

if [[ "${INSTALLER}" != "pip" && "${INSTALLER}" != "uv" ]]; then
  echo "[setup] error: INSTALLER must be 'pip' or 'uv'"
  exit 1
fi

if [[ "${INSTALLER}" == "uv" ]] && ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found, installing via pip"
  "${PYTHON_BIN}" -m pip install --upgrade uv
fi

if [[ "${CREATE_VENV}" == "1" ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    if [[ "${INSTALLER}" == "uv" ]]; then
      uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
    else
      "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
  fi
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup] detected GPUs:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "[setup] warning: nvidia-smi not found"
fi

if [[ "${INSTALLER}" == "uv" ]]; then
  uv pip install --python "${PYTHON_BIN}" -r requirements.txt
else
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export WANDB_DIR="${WANDB_DIR:-/workspace/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p /workspace/data
mkdir -p /workspace/outputs
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${WANDB_DIR}"

if grep -R "REPLACE_WITH_" "${CONFIG_PATH}" >/dev/null 2>&1; then
  echo "[setup] error: unresolved REPLACE_WITH_ placeholders found in ${CONFIG_PATH}"
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"[setup] torch={torch.__version__}")
print(f"[setup] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] gpu_count={torch.cuda.device_count()}")
PY

if [[ "${AUTO_DOWNLOAD_DATASET}" == "1" ]]; then
  if grep -R "REPLACE_WITH_" "${DOWNLOAD_CONFIG_PATH}" >/dev/null 2>&1; then
    echo "[setup] error: unresolved REPLACE_WITH_ placeholders found in ${DOWNLOAD_CONFIG_PATH}"
    exit 1
  fi
  echo "[setup] downloading dataset snapshot"
  "${PYTHON_BIN}" GLM/data/download_dataset.py --config "${DOWNLOAD_CONFIG_PATH}"
fi

echo "[setup] complete"
echo "[setup] next: bash GLM/runpod_ddp.sh"
