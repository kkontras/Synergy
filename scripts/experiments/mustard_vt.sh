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
MODE="${2:-all}"   # all | train | show | ceu

DEFAULT_CONFIG="./configs/FactorCL/Mustard/default_config_mustard_VT.json"
RELEASE_DIR="./configs/FactorCL/Mustard/release/VT"
UNIMODALS=(
  "${RELEASE_DIR}/unimodal_video.json"
  "${RELEASE_DIR}/unimodal_text.json"
)
METHODS=(
  "${RELEASE_DIR}/DnR.json"
  "${RELEASE_DIR}/MCR.json"
  "${RELEASE_DIR}/MMPareto.json"
  "${RELEASE_DIR}/synprom_RMask.json"
  "${RELEASE_DIR}/synprom_RMask_learned.json"
  "${RELEASE_DIR}/synprom_RMask_random.json"
)

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV:-0}"
LR="${LR:-0.0001}"
WD="${WD:-0.0001}"

run_train() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/train.py "$@"; }
run_show()  { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/show.py "$@"; }
run_ceu()   { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/get_ceu_cli.py "$@"; }

do_train() { [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; }
do_show()  { [[ "${MODE}" == "all" || "${MODE}" == "show"  ]]; }
do_ceu()   { [[ "${MODE}" == "all" || "${MODE}" == "ceu"   ]]; }

for fold in "${FOLDS[@]}"; do
  for cfg in "${UNIMODALS[@]}"; do
    if do_train; then
      run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${LR}" --wd "${WD}" --validate_with accuracy
    fi
    if do_show; then
      run_show --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${LR}" --wd "${WD}" --validate_with accuracy
    fi
  done
done

if do_ceu; then
  run_ceu \
    --dataset mustard \
    --default_config "${DEFAULT_CONFIG}" \
    --unimodal_configs "${UNIMODALS[0]}" "${UNIMODALS[1]}" \
    --folds "${FOLDS[@]}" \
    --lr "${LR}" \
    --wd "${WD}" \
    --validate_with accuracy
fi

for fold in "${FOLDS[@]}"; do
  for cfg in "${METHODS[@]}"; do
    if do_train; then
      run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${LR}" --wd "${WD}" --validate_with accuracy
    fi
    if do_show; then
      run_show --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${LR}" --wd "${WD}" --validate_with accuracy
    fi
  done
done
