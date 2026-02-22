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
MODE="${2:-all}"   # all | unimodal | ceu | methods

DEFAULT_CONFIG="./configs/FactorCL/Mustard/default_config_mustard_VT.json"
RELEASE_DIR="./configs/FactorCL/Mustard/release/VT"
UNIMODAL_VIDEO="${RELEASE_DIR}/unimodal_video.json"
UNIMODAL_TEXT="${RELEASE_DIR}/unimodal_text.json"
METHODS=(
  "${RELEASE_DIR}/DnR.json"
  "${RELEASE_DIR}/MCR.json"
  "${RELEASE_DIR}/MMPareto.json"
  "${RELEASE_DIR}/synprom_RMask.json"
  "${RELEASE_DIR}/synprom_RMask_learned.json"
  "${RELEASE_DIR}/synprom_RMask_random.json"
)

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV:-0,1,2}"
IFS=',' read -r -a UNIMODAL_LRS <<< "${UNIMODAL_LRS_CSV:-0.0005}"
IFS=',' read -r -a UNIMODAL_WDS <<< "${UNIMODAL_WDS_CSV:-0.0001}"
IFS=',' read -r -a METHOD_LRS <<< "${METHOD_LRS_CSV:-0.0005}"
IFS=',' read -r -a METHOD_WDS <<< "${METHOD_WDS_CSV:-0.0001}"
IFS=',' read -r -a RMASK_LS <<< "${RMASK_LS_CSV:-0,0.1,1}"

# get_ceu_cli uses one lr/wd suffix for both unimodals
CEU_LR="${CEU_LR:-0.0005}"
CEU_WD="${CEU_WD:-0.0001}"

run_train() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/train.py "$@"; }
run_ceu() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/get_ceu_cli.py "$@"; }

do_unimodal() { [[ "${MODE}" == "all" || "${MODE}" == "unimodal" ]]; }
do_ceu() { [[ "${MODE}" == "all" || "${MODE}" == "ceu" ]]; }
do_methods() { [[ "${MODE}" == "all" || "${MODE}" == "methods" ]]; }

is_pretrained_method() {
  local cfg="$1"
  rg -q '"pretrainedEncoder"[[:space:]]*:[[:space:]]*\\{[^}]*"use"[[:space:]]*:[[:space:]]*true' "${cfg}" || \
  rg -q '"use"[[:space:]]*:[[:space:]]*true' <(awk '/"pretrainedEncoder"[[:space:]]*:/,/\}/ {print}' "${cfg}")
}

# 1) Unimodal runs
if do_unimodal; then
  for fold in "${FOLDS[@]}"; do
    for lr in "${UNIMODAL_LRS[@]}"; do
      for wd in "${UNIMODAL_WDS[@]}"; do
        run_train --config "${UNIMODAL_VIDEO}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
        run_train --config "${UNIMODAL_TEXT}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
      done
    done
  done
fi

# 2) CEU generation (after unimodals)
if do_ceu; then
  run_ceu \
    --dataset mustard \
    --default_config "${DEFAULT_CONFIG}" \
    --unimodal_configs "${UNIMODAL_VIDEO}" "${UNIMODAL_TEXT}" \
    --folds "${FOLDS[@]}" \
    --lr "${CEU_LR}" \
    --wd "${CEU_WD}" \
    --validate_with accuracy
fi

# 3) Remaining methods (after CEU)
if do_methods; then
  NON_PRETRAINED_METHODS=()
  PRETRAINED_METHODS=()
  for cfg in "${METHODS[@]}"; do
    if is_pretrained_method "${cfg}"; then
      PRETRAINED_METHODS+=("${cfg}")
    else
      NON_PRETRAINED_METHODS+=("${cfg}")
    fi
  done

  ORDERED_METHODS=("${NON_PRETRAINED_METHODS[@]}" "${PRETRAINED_METHODS[@]}")
  echo "Method order (non-pretrained first):"
  for cfg in "${ORDERED_METHODS[@]}"; do
    echo "  - ${cfg}"
  done

  for fold in "${FOLDS[@]}"; do
    for cfg in "${ORDERED_METHODS[@]}"; do
      for lr in "${METHOD_LRS[@]}"; do
        for wd in "${METHOD_WDS[@]}"; do
          if [[ "${cfg}" == *"synprom_RMask"* ]]; then
            for l in "${RMASK_LS[@]}"; do
              run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --l "${l}" --cls mlp --validate_with syn_accuracy
            done
          else
            run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
          fi
        done
      done
    done
  done
fi
