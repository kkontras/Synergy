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
MODE="${2:-all}"   # all | unimodal | ceu | methods (methods runs CEU first)

DEFAULT_CONFIG="./configs/FactorCL/Mustard/default_config_mustard_VT.json"
RELEASE_DIR="./configs/FactorCL/Mustard/release/VT"
UNIMODAL_VIDEO="${RELEASE_DIR}/unimodal_video.json"
UNIMODAL_TEXT="${RELEASE_DIR}/unimodal_text.json"
METHODS=(
  "${RELEASE_DIR}/DnR.json"
  "${RELEASE_DIR}/MCR.json"
  "${RELEASE_DIR}/MMPareto.json"
  "${RELEASE_DIR}/ReconBoost.json"
  "${RELEASE_DIR}/synprom_RMask.json"
  "${RELEASE_DIR}/synprom_RMask_learned.json"
  "${RELEASE_DIR}/synprom_RMask_random.json"
)

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV:-0,1,2}"
IFS=',' read -r -a UNIMODAL_LRS <<< "${UNIMODAL_LRS_CSV:-0.001,0.0005,0.0001,0.00005}"
IFS=',' read -r -a UNIMODAL_WDS <<< "${UNIMODAL_WDS_CSV:-0.001,0.0001,0.00001}"
IFS=',' read -r -a METHOD_LRS <<< "${METHOD_LRS_CSV:-0.001,0.0001}"
IFS=',' read -r -a METHOD_WDS <<< "${METHOD_WDS_CSV:-0.001,0.0001}"

BEST_VIDEO_LR="${BEST_VIDEO_LR:-0.0005}"
BEST_VIDEO_WD="${BEST_VIDEO_WD:-0.00001}"
BEST_TEXT_LR="${BEST_TEXT_LR:-0.0005}"
BEST_TEXT_WD="${BEST_TEXT_WD:-0.0001}"
METHOD_FIXED_LR="${METHOD_FIXED_LR:-${BEST_TEXT_LR}}"
METHOD_FIXED_WD="${METHOD_FIXED_WD:-${BEST_TEXT_WD}}"

run_train() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/train.py "$@"; }
run_ceu() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/get_ceu_cli.py "$@"; }

do_unimodal() {
  if [[ "${MODE}" == "unimodal" ]]; then
    return 0
  fi
  if [[ "${MODE}" == "all" ]]; then
    [[ -z "${BEST_VIDEO_LR}" || -z "${BEST_VIDEO_WD}" || -z "${BEST_TEXT_LR}" || -z "${BEST_TEXT_WD}" ]]
    return
  fi
  return 1
}
do_ceu() { [[ "${MODE}" == "all" || "${MODE}" == "ceu" || "${MODE}" == "methods" ]]; }
do_methods() { [[ "${MODE}" == "all" || "${MODE}" == "methods" ]]; }

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

if [[ "${MODE}" == "all" && -n "${BEST_VIDEO_LR}" && -n "${BEST_VIDEO_WD}" && -n "${BEST_TEXT_LR}" && -n "${BEST_TEXT_WD}" ]]; then
  echo "BEST_* vars detected. Skipping unimodal sweep and proceeding to CEU + methods."
fi

if [[ "${MODE}" == "all" && ( -z "${BEST_VIDEO_LR}" || -z "${BEST_VIDEO_WD}" || -z "${BEST_TEXT_LR}" || -z "${BEST_TEXT_WD}" ) ]]; then
  echo "Unimodal search finished."
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
    --dataset mustard \
    --default_config "${DEFAULT_CONFIG}" \
    --unimodal_configs "${UNIMODAL_VIDEO}" "${UNIMODAL_TEXT}" \
    --folds "${FOLDS[@]}" \
    --lr "${BEST_VIDEO_LR}" \
    --wd "${BEST_VIDEO_WD}" \
    --validate_with accuracy
fi

if do_methods; then
  echo "Methods stage uses fixed optimizer lr/wd: lr=${METHOD_FIXED_LR} wd=${METHOD_FIXED_WD} (defaulting to text-best)."
  for fold in "${FOLDS[@]}"; do
    for cfg in "${METHODS[@]}"; do
      cfg_name="$(basename "${cfg}")"
      case "${cfg_name}" in
        DnR.json)
          for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
            for kmpe in 1 3 5 10; do
              run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --alpha "${alpha}" --kmepoch "${kmpe}" --validate_with accuracy
            done
          done
          ;;
        MCR.json)
          for l in 0.001 0.01 0.1 1; do
            for multil in 0.01 0.1 1; do
              run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --l "${l}" --multil "${multil}" --validate_with accuracy
            done
          done
          ;;
        MMPareto.json)
          for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
            run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
              --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
              --alpha "${alpha}" --validate_with accuracy
          done
          ;;
        ReconBoost.json)
          for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
            for recon_stages in 1 4 10; do
              for recon_weight1 in 1 3 5 10; do
                run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                  --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                  --alpha "${alpha}" \
                  --recon_weight1 "${recon_weight1}" --recon_weight2 1 \
                  --recon_epochstages "${recon_stages}" --recon_ensemblestages "${recon_stages}" \
                  --validate_with accuracy
              done
            done
          done
          ;;
        synprom_RMask_learned.json)
          for l in 0.001 0.01 0.1 1; do
            for lsparse in 0.001 0.01 0.1 1 3 5 10; do
              run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --l "${l}" --perturb learned --perturb_fill ema --perturb_lsparse "${lsparse}" \
                --validate_with accuracy
            done
          done
          ;;
        synprom_RMask_random.json)
          for l in 0.001 0.01 0.1 1; do
            for pmin in 0.1 0.3 0.5 0.7 0.9; do
              run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --l "${l}" --perturb random --perturb_fill ema --perturb_pmin "${pmin}" \
                --validate_with accuracy
            done
          done
          ;;
        synprom_RMask.json)
          for l in 0.001 0.01 0.1 1; do
            run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
              --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
              --l "${l}" --validate_with accuracy
          done
          ;;
        *)
          for lr in "${METHOD_LRS[@]}"; do
            for wd in "${METHOD_WDS[@]}"; do
              run_train --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
            done
          done
          ;;
      esac
    done
  done
fi
