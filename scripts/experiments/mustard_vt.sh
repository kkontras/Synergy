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
MODE="${2:-all}"   # all | unimodal | ceu | ens | methods (methods runs CEU first)

DEFAULT_CONFIG="./configs/FactorCL/Mustard/default_config_mustard_VT.json"
RELEASE_DIR="./configs/FactorCL/Mustard/release/VT"
UNIMODAL_VIDEO="${RELEASE_DIR}/unimodal_video.json"
UNIMODAL_TEXT="${RELEASE_DIR}/unimodal_text.json"
ENS_CFG="${RELEASE_DIR}/ens.json"
SYNPROM_RMASK_BASE_CFG="${RELEASE_DIR}/synprom_RMask.json"
METHODS=(
  "${ENS_CFG}"
  "${RELEASE_DIR}/MMPareto.json"
  "${RELEASE_DIR}/DnR.json"
  "${RELEASE_DIR}/MCR.json"
  "${RELEASE_DIR}/ReconBoost.json"
  "${RELEASE_DIR}/synprom_RMask.json"
  "${RELEASE_DIR}/synprom_RMask_nonpre.json"
  "${RELEASE_DIR}/synprom_RMask_learned.json"
  "${RELEASE_DIR}/synprom_RMask_random.json"
)

# Method scheduling:
# - METHOD_RUN_MODE=all (default): run every method sweep
# - METHOD_RUN_MODE=single: run one method to inspect convergence
# - METHOD_TARGET can be basename with/without .json (e.g., MMPareto or MMPareto.json)
METHOD_RUN_MODE="${METHOD_RUN_MODE:-all}"
METHOD_TARGET="${METHOD_TARGET:-MMPareto.json}"

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV:-0,1,2}"
IFS=',' read -r -a UNIMODAL_LRS <<< "${UNIMODAL_LRS_CSV:-0.001,0.0005,0.0001,0.00005}"
IFS=',' read -r -a UNIMODAL_WDS <<< "${UNIMODAL_WDS_CSV:-0.001,0.0001,0.00001}"
IFS=',' read -r -a METHOD_LRS <<< "${METHOD_LRS_CSV:-0.001,0.0001}"
IFS=',' read -r -a METHOD_WDS <<< "${METHOD_WDS_CSV:-0.001,0.0001}"

# Default selected unimodal settings (can still be overridden via env).
BEST_VIDEO_LR="${BEST_VIDEO_LR:-0.001}"
BEST_VIDEO_WD="${BEST_VIDEO_WD:-0.001}"
BEST_TEXT_LR="${BEST_TEXT_LR:-0.0005}"
BEST_TEXT_WD="${BEST_TEXT_WD:-0.001}"

# Method sweeps use a fixed optimizer pair by default (override via env if needed).
METHOD_FIXED_LR="${METHOD_FIXED_LR:-0.0005}"
METHOD_FIXED_WD="${METHOD_FIXED_WD:-0.001}"
SYNPROM_RMASK_BASE_L="${SYNPROM_RMASK_BASE_L:-0}"          # synprom_RMask.json (vanilla RMask / uni-pre finetuning path)
SYNPROM_RMASK_SWEEP_LRWD="${SYNPROM_RMASK_SWEEP_LRWD:-0}"  # 1 => sweep METHOD_LRS/WDS for synprom_RMask

run_train() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/train.py "$@"; }
run_ceu() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/get_ceu_cli.py "$@"; }
run_train_method_safe() {
  if ! run_train "$@"; then
    echo "Method run failed (continuing): $*"
    return 0
  fi
}

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
do_ceu() {
  [[ "${SKIP_CEU:-0}" == "1" ]] && return 1
  [[ "${MODE}" == "all" || "${MODE}" == "ceu" || "${MODE}" == "methods" ]]
}
do_ens_only() { [[ "${MODE}" == "ens" ]]; }
do_methods() { [[ "${MODE}" == "all" || "${MODE}" == "methods" ]]; }

run_ens_bundle() {
  echo "ENS bundle stage uses fixed optimizer lr/wd: lr=${METHOD_FIXED_LR} wd=${METHOD_FIXED_WD}"
  echo "  - Ensemble config: ${ENS_CFG}"
  echo "  - Pretrained-unimodal finetuning path: ${SYNPROM_RMASK_BASE_CFG} with --l ${SYNPROM_RMASK_BASE_L}"
  local fold
  for fold in "${FOLDS[@]}"; do
    run_train_method_safe --config "${ENS_CFG}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
      --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --validate_with accuracy
    run_train_method_safe --config "${SYNPROM_RMASK_BASE_CFG}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
      --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --l "${SYNPROM_RMASK_BASE_L}" --validate_with accuracy
  done
}

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

if do_ens_only; then
  run_ens_bundle
  exit 0
fi

if do_methods; then
  echo "Methods stage uses fixed optimizer lr/wd: lr=${METHOD_FIXED_LR} wd=${METHOD_FIXED_WD} (defaulting to text-best)."
  if [[ "${METHOD_RUN_MODE}" == "single" ]]; then
    echo "Methods stage running ONE method for inspection first: ${METHOD_TARGET}"
    echo "Set METHOD_RUN_MODE=all to sweep all methods."
  fi
  for fold in "${FOLDS[@]}"; do
    for cfg in "${METHODS[@]}"; do
      cfg_name="$(basename "${cfg}")"
      if [[ "${METHOD_RUN_MODE}" == "single" ]]; then
        target="${METHOD_TARGET}"
        [[ "${target}" == *.json ]] || target="${target}.json"
        if [[ "${cfg_name}" != "${target}" ]]; then
          continue
        fi
      fi
      case "${cfg_name}" in
        ens.json)
          run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
            --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --validate_with accuracy
          ;;
        DnR.json)
          for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
            for kmpe in 1 3 5 10; do
              run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --alpha "${alpha}" --kmepoch "${kmpe}" --validate_with accuracy
            done
          done
          ;;
        MCR.json)
          for l in 0.001 0.01 0.1 1; do
            for multil in 0.01 0.1 1; do
              run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --l "${l}" --multil "${multil}" --validate_with accuracy
            done
          done
          ;;
        MMPareto.json)
          for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
            run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
              --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
              --alpha "${alpha}" --validate_with accuracy
          done
          ;;
        ReconBoost.json)
          for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
            for recon_stages in 1 4 10; do
              for recon_weight1 in 1 3 5 10; do
                run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
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
              run_train_method_safe --config "${SYNPROM_RMASK_BASE_CFG}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --perturn learned \
                --l "${l}" --perturb learned --perturb_fill ema --perturb_lsparse "${lsparse}" \
                --validate_with accuracy
            done
          done
          ;;
        synprom_RMask_random.json)
          for l in 0.001 0.01 0.1 1; do
            for pmin in 0.1 0.3 0.5 0.7 0.9; do
              run_train_method_safe --config "${SYNPROM_RMASK_BASE_CFG}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --perturn random \
                --l "${l}" --perturb random --perturb_fill ema --perturb_pmin "${pmin}" \
                --validate_with accuracy
            done
          done
          ;;
        synprom_RMask.json)
          # Vanilla RMask (no IB): fixed at l=0.
          # This is the pretrained-unimodal finetuning / ensemble path used by the show/table pipeline.
          if [[ "${SYNPROM_RMASK_SWEEP_LRWD}" == "1" ]]; then
            for lr in "${METHOD_LRS[@]}"; do
              for wd in "${METHOD_WDS[@]}"; do
                run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                  --lr "${lr}" --wd "${wd}" \
                  --l "${SYNPROM_RMASK_BASE_L}" --validate_with accuracy
              done
            done
          else
            run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
              --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
              --l "${SYNPROM_RMASK_BASE_L}" --validate_with accuracy
          fi
          ;;
        *)
          for lr in "${METHOD_LRS[@]}"; do
            for wd in "${METHOD_WDS[@]}"; do
              run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
            done
          done
          ;;
      esac
    done
  done
fi
