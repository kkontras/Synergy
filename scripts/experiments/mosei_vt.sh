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

DEFAULT_CONFIG="./configs/FactorCL/Mosei/default_config_mosei_VT_syn.json"
SYN_DIR="./configs/FactorCL/Mosei/syn/VT"
UNIMODAL_VIDEO="${SYN_DIR}/unimodal_video.json"
UNIMODAL_TEXT="${SYN_DIR}/unimodal_text.json"

IFS=',' read -r -a FOLDS        <<< "${FOLDS_CSV:-0,1,2}"
IFS=',' read -r -a UNIMODAL_LRS <<< "${UNIMODAL_LRS_CSV:-0.001,0.0005,0.0001,0.00005}"
IFS=',' read -r -a UNIMODAL_WDS <<< "${UNIMODAL_WDS_CSV:-0.001,0.0001,0.00001}"

# Fixed lr/wd for methods (override via env if needed)
METHOD_LR="${METHOD_LR:-0.0005}"
METHOD_WD="${METHOD_WD:-0.001}"

# Per-method hyperparameter grids
IFS=',' read -r -a MCR_LS        <<< "${MCR_L_CSV:-0.001,0.01,0.1,1}"
IFS=',' read -r -a MCR_MULTILS   <<< "${MCR_MULTIL_CSV:-0.01,0.1,1}"
IFS=',' read -r -a MMPARETO_ALPHAS <<< "${MMPARETO_ALPHA_CSV:-0.5,1.0,1.5,2.0,3.0,5.0}"

BEST_VIDEO_LR="${BEST_VIDEO_LR:-0.001}"
BEST_VIDEO_WD="${BEST_VIDEO_WD:-0.001}"
BEST_TEXT_LR="${BEST_TEXT_LR:-0.0005}"
BEST_TEXT_WD="${BEST_TEXT_WD:-0.001}"

run_train() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/train.py "$@"; }
run_ceu()   { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/get_ceu_cli.py "$@"; }
run_train_safe() {
  if ! run_train "$@"; then
    echo "Method run failed (continuing): $*"
  fi
}

do_unimodal() { [[ "${MODE}" == "all" || "${MODE}" == "unimodal" ]]; }
do_ceu()      { [[ "${MODE}" == "all" || "${MODE}" == "ceu" ]]; }
do_methods()  { [[ "${MODE}" == "all" || "${MODE}" == "methods" ]]; }

if do_unimodal; then
  echo "Unimodal stage: video lr=${BEST_VIDEO_LR} wd=${BEST_VIDEO_WD} | text lr=${BEST_TEXT_LR} wd=${BEST_TEXT_WD}"
  for fold in "${FOLDS[@]}"; do
    run_train --config "${UNIMODAL_VIDEO}" --default_config "${DEFAULT_CONFIG}" \
      --fold "${fold}" --lr "${BEST_VIDEO_LR}" --wd "${BEST_VIDEO_WD}" --validate_with accuracy
    run_train --config "${UNIMODAL_TEXT}"  --default_config "${DEFAULT_CONFIG}" \
      --fold "${fold}" --lr "${BEST_TEXT_LR}"  --wd "${BEST_TEXT_WD}"  --validate_with accuracy
  done
fi

if [[ "${MODE}" == "all" && ( -z "${BEST_VIDEO_LR}" || -z "${BEST_VIDEO_WD}" || -z "${BEST_TEXT_LR}" || -z "${BEST_TEXT_WD}" ) ]]; then
  echo "Set BEST_VIDEO_LR/BEST_VIDEO_WD and BEST_TEXT_LR/BEST_TEXT_WD, then rerun with MODE=ceu (or MODE=all)."
  exit 0
fi

if do_ceu; then
  if [[ -z "${BEST_VIDEO_LR}" || -z "${BEST_VIDEO_WD}" || -z "${BEST_TEXT_LR}" || -z "${BEST_TEXT_WD}" ]]; then
    echo "Missing BEST_* vars for CEU stage."
    echo "Required: BEST_VIDEO_LR BEST_VIDEO_WD BEST_TEXT_LR BEST_TEXT_WD"
    exit 1
  fi
  if [[ "${BEST_VIDEO_LR}" != "${BEST_TEXT_LR}" || "${BEST_VIDEO_WD}" != "${BEST_TEXT_WD}" ]]; then
    echo "Warning: get_ceu_cli uses one lr/wd suffix for both unimodals."
    echo "Using video settings for CEU: lr=${BEST_VIDEO_LR} wd=${BEST_VIDEO_WD}"
  fi
  run_ceu \
    --dataset mosei \
    --default_config "${DEFAULT_CONFIG}" \
    --unimodal_configs "${UNIMODAL_VIDEO}" "${UNIMODAL_TEXT}" \
    --folds "${FOLDS[@]}" \
    --lr "${BEST_TEXT_LR}" \
    --wd "${BEST_TEXT_WD}" \
    --validate_with accuracy
fi

if do_methods; then
  echo "Methods stage: lr=${METHOD_LR} wd=${METHOD_WD}"

  # MCR (release) — l x multil sweep
  for fold in "${FOLDS[@]}"; do
    for l in "${MCR_LS[@]}"; do
      for multil in "${MCR_MULTILS[@]}"; do
        run_train_safe --config "${SYN_DIR}/MCR.json" --default_config "${DEFAULT_CONFIG}" \
          --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" \
          --l "${l}" --multil "${multil}" --validate_with accuracy
      done
    done
  done

  # MMPareto — alpha sweep
  for fold in "${FOLDS[@]}"; do
    for alpha in "${MMPARETO_ALPHAS[@]}"; do
      run_train_safe --config "${SYN_DIR}/MMPareto.json" --default_config "${DEFAULT_CONFIG}" \
        --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" \
        --alpha "${alpha}" --validate_with accuracy
    done
  done

  # Ensemble — fixed lr/wd
  for fold in "${FOLDS[@]}"; do
    run_train_safe --config "${SYN_DIR}/ens.json" --default_config "${DEFAULT_CONFIG}" \
      --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" --validate_with accuracy
  done

  # Joint Training — fixed lr/wd
  for fold in "${FOLDS[@]}"; do
    run_train_safe --config "${SYN_DIR}/joint_training.json" --default_config "${DEFAULT_CONFIG}" \
      --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" --validate_with accuracy
  done

  # Syn MCR — l x multil sweep
  for fold in "${FOLDS[@]}"; do
    for l in "${MCR_LS[@]}"; do
      for multil in "${MCR_MULTILS[@]}"; do
        run_train_safe --config "${SYN_DIR}/MCR.json" --default_config "${DEFAULT_CONFIG}" \
          --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" \
          --l "${l}" --multil "${multil}" --validate_with accuracy
      done
    done
  done

  # Syn RMask — l=0, fixed lr/wd
  for fold in "${FOLDS[@]}"; do
    run_train_safe --config "${SYN_DIR}/synprom_RMask.json" --default_config "${DEFAULT_CONFIG}" \
      --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" --l 0 --validate_with accuracy
  done

  # Syn RMask nopre — l=0, fixed lr/wd (no pretrained encoders → uses VT_syn save dir)
  for fold in "${FOLDS[@]}"; do
    run_train_safe --config "${SYN_DIR}/synprom_RMask_nopre.json" --default_config "${DEFAULT_CONFIG}" \
      --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" --l 0 --validate_with accuracy
  done
fi
