#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV_PATH="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${CONDA_ENV_PATH}" || true
fi
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

GPU="${1:-0}"
MODE="${2:-all}"
shift $(( $# >= 2 ? 2 : $# )) || true

exec "${PYTHON_BIN}" scripts/experiments/factorcl_vt_workflow.py \
  --dataset mosi \
  --default_config ./configs/FactorCL/Mosi/default_config_mosi_VT.json \
  --release_dir ./configs/FactorCL/Mosi/release/VT \
  --syn_dir ./configs/FactorCL/Mosi/syn/VT \
  --exclude_methods "${EXCLUDE_METHODS:-AGM}" \
  --gpu "${GPU}" \
  --mode "${MODE}" \
  "$@"
