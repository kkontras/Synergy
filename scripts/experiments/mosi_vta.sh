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
MODE="${2:-all}"   # all | unimodal | methods

DEFAULT_CONFIG="./configs/FactorCL/Mosi/default_config_mosi_VTA.json"
RELEASE_DIR="./configs/FactorCL/Mosi/release/VTA"
UNIMODALS=(
  "${RELEASE_DIR}/unimodal_video.json"
  "${RELEASE_DIR}/unimodal_text.json"
  "${RELEASE_DIR}/unimodal_audio.json"
)
METHODS=(
  "${RELEASE_DIR}/AGM.json"
  "${RELEASE_DIR}/MCR.json"
  "${RELEASE_DIR}/MLB.json"
  "${RELEASE_DIR}/MMPareto.json"
  "${RELEASE_DIR}/ens.json"
  "${RELEASE_DIR}/joint_training.json"
  "${RELEASE_DIR}/multiloss.json"
  "${RELEASE_DIR}/pre_finetuned.json"
  "${RELEASE_DIR}/pre_frozen.json"
)

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV:-0,1,2}"
IFS=',' read -r -a UNIMODAL_LRS <<< "${UNIMODAL_LRS_CSV:-0.001,0.0005,0.0001,0.00005}"
IFS=',' read -r -a UNIMODAL_WDS <<< "${UNIMODAL_WDS_CSV:-0.001,0.0001,0.00001}"
IFS=',' read -r -a METHOD_LRS <<< "${METHOD_LRS_CSV:-0.001,0.0001}"
IFS=',' read -r -a METHOD_WDS <<< "${METHOD_WDS_CSV:-0.001,0.0001}"

run_train() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/train.py "$@"; }

do_unimodal() { [[ "${MODE}" == "all" || "${MODE}" == "unimodal" ]]; }
do_methods() { [[ "${MODE}" == "all" || "${MODE}" == "methods" ]]; }

if do_unimodal; then
  for fold in "${FOLDS[@]}"; do
    for cfg in "${UNIMODALS[@]}"; do
      for lr in "${UNIMODAL_LRS[@]}"; do
        for wd in "${UNIMODAL_WDS[@]}"; do
          run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
        done
      done
    done
  done
fi

if do_methods; then
  for fold in "${FOLDS[@]}"; do
    for cfg in "${METHODS[@]}"; do
      for lr in "${METHOD_LRS[@]}"; do
        for wd in "${METHOD_WDS[@]}"; do
          run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
        done
      done
    done
  done
fi
