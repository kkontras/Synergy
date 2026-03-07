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
MODE="${2:-all}"   # all | unimodal | ceu | methods | rmask

DEFAULT_CONFIG="./configs/FactorCL/Mosei/default_config_mosei_VT_syn.json"
SYN_DIR="./configs/FactorCL/Mosei/syn/VT"
UNIMODAL_VIDEO="${SYN_DIR}/unimodal_video.json"
UNIMODAL_TEXT="${SYN_DIR}/unimodal_text.json"
SAVE_BASE_DIR="${SAVE_BASE_DIR:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/Rmask/MOSEI/VT}"

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
IFS=',' read -r -a DNR_ALPHAS <<< "${DNR_ALPHA_CSV:-0.5,1.0,1.5,2.0,3.0,5.0}"
IFS=',' read -r -a DNR_KMEPOCHS <<< "${DNR_KMEPOCH_CSV:-1,3,5,10}"
IFS=',' read -r -a RECONBOOST_ALPHAS <<< "${RECONBOOST_ALPHA_CSV:-0.5,1.0,1.5,2.0,3.0,5.0}"
IFS=',' read -r -a RECONBOOST_STAGES <<< "${RECONBOOST_STAGES_CSV:-1,4,10}"
IFS=',' read -r -a RECONBOOST_W1S <<< "${RECONBOOST_W1_CSV:-1,3,5,10}"
IFS=',' read -r -a RMASK_BASE_LS <<< "${RMASK_BASE_L_CSV:-0}"
IFS=',' read -r -a RMASK_LEARNED_LS <<< "${RMASK_LEARNED_L_CSV:-0.001,0.01,0.1,1}"
IFS=',' read -r -a RMASK_LEARNED_LSPARSES <<< "${RMASK_LEARNED_LSPARSE_CSV:-0.001,0.01,0.1,1,3,5,10}"
IFS=',' read -r -a RMASK_RANDOM_LS <<< "${RMASK_RANDOM_L_CSV:-0.001,0.01,0.1,1}"
IFS=',' read -r -a RMASK_RANDOM_PMINS <<< "${RMASK_RANDOM_PMIN_CSV:-0.1,0.3,0.5,0.7,0.9}"

BEST_VIDEO_LR="${BEST_VIDEO_LR:-0.001}"
BEST_VIDEO_WD="${BEST_VIDEO_WD:-0.001}"
BEST_TEXT_LR="${BEST_TEXT_LR:-0.0005}"
BEST_TEXT_WD="${BEST_TEXT_WD:-0.001}"
CEU_VIDEO_LR="${CEU_VIDEO_LR:-${BEST_VIDEO_LR}}"
CEU_VIDEO_WD="${CEU_VIDEO_WD:-${BEST_VIDEO_WD}}"
CEU_TEXT_LR="${CEU_TEXT_LR:-${BEST_TEXT_LR}}"
CEU_TEXT_WD="${CEU_TEXT_WD:-${BEST_TEXT_WD}}"

# RMask switches (1=run, 0=skip)
RUN_RMASK_NOPRE="${RUN_RMASK_NOPRE:-1}"
RUN_RMASK_BASE="${RUN_RMASK_BASE:-1}"
RUN_RMASK_LEARNED="${RUN_RMASK_LEARNED:-1}"
RUN_RMASK_RANDOM="${RUN_RMASK_RANDOM:-1}"

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
do_rmask()    { [[ "${MODE}" == "all" || "${MODE}" == "methods" || "${MODE}" == "rmask" ]]; }

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
  if [[ -z "${CEU_VIDEO_LR}" || -z "${CEU_VIDEO_WD}" || -z "${CEU_TEXT_LR}" || -z "${CEU_TEXT_WD}" ]]; then
    echo "Missing CEU_* vars for CEU stage."
    echo "Required: CEU_VIDEO_LR CEU_VIDEO_WD CEU_TEXT_LR CEU_TEXT_WD"
    exit 1
  fi
  echo "CEU stage:"
  echo "  video lr=${CEU_VIDEO_LR} wd=${CEU_VIDEO_WD}"
  echo "  text  lr=${CEU_TEXT_LR} wd=${CEU_TEXT_WD}"

  for fold in "${FOLDS[@]}"; do
    vckpt="${SAVE_BASE_DIR}/unimodal_video_fold${fold}_vldaccuracy_lr${CEU_VIDEO_LR}_wd${CEU_VIDEO_WD}.pth.tar"
    tckpt="${SAVE_BASE_DIR}/unimodal_text_fold${fold}_vldaccuracy_lr${CEU_TEXT_LR}_wd${CEU_TEXT_WD}.pth.tar"
    if [[ ! -f "${vckpt}" ]]; then
      echo "Missing CEU video checkpoint: ${vckpt}"
      exit 1
    fi
    if [[ ! -f "${tckpt}" ]]; then
      echo "Missing CEU text checkpoint: ${tckpt}"
      exit 1
    fi
  done

  echo "Running get_ceu_cli.py for dataset=mosei folds=${FOLDS[*]}"
  run_ceu \
    --dataset mosei \
    --default_config "${DEFAULT_CONFIG}" \
    --unimodal_configs "${UNIMODAL_VIDEO}" "${UNIMODAL_TEXT}" \
    --folds "${FOLDS[@]}" \
    --unimodal_lrs "${CEU_VIDEO_LR}" "${CEU_TEXT_LR}" \
    --unimodal_wds "${CEU_VIDEO_WD}" "${CEU_TEXT_WD}" \
    --lr "${CEU_TEXT_LR}" \
    --wd "${CEU_TEXT_WD}" \
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

  # DnR — alpha x kmepoch sweep
  for fold in "${FOLDS[@]}"; do
    for alpha in "${DNR_ALPHAS[@]}"; do
      for kmpe in "${DNR_KMEPOCHS[@]}"; do
        run_train_safe --config "${SYN_DIR}/DnR.json" --default_config "${DEFAULT_CONFIG}" \
          --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" \
          --alpha "${alpha}" --kmepoch "${kmpe}" --validate_with accuracy
      done
    done
  done

  # ReconBoost — alpha x stages x w1 sweep
  for fold in "${FOLDS[@]}"; do
    for alpha in "${RECONBOOST_ALPHAS[@]}"; do
      for stages in "${RECONBOOST_STAGES[@]}"; do
        for w1 in "${RECONBOOST_W1S[@]}"; do
          run_train_safe --config "${SYN_DIR}/ReconBoost.json" --default_config "${DEFAULT_CONFIG}" \
            --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" \
            --alpha "${alpha}" \
            --recon_weight1 "${w1}" --recon_weight2 1 \
            --recon_epochstages "${stages}" --recon_ensemblestages "${stages}" \
            --validate_with accuracy
        done
      done
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

fi

if do_rmask; then
  echo "RMask stage: lr=${METHOD_LR} wd=${METHOD_WD}"

  if [[ "${RUN_RMASK_NOPRE}" == "1" ]]; then
    echo "  Syn RMask nopre sweep: l=0, lr x wd"
    for fold in "${FOLDS[@]}"; do
      for lr in "${UNIMODAL_LRS[@]}"; do
        for wd in "${UNIMODAL_WDS[@]}"; do
          run_train_safe --config "${SYN_DIR}/synprom_RMask_nopre.json" --default_config "${DEFAULT_CONFIG}" \
            --fold "${fold}" --lr "${lr}" --wd "${wd}" --l 0 --validate_with accuracy
        done
      done
    done
  fi

  if [[ "${RUN_RMASK_BASE}" == "1" ]]; then
    echo "  Syn RMask base sweep: l"
    for fold in "${FOLDS[@]}"; do
      for l in "${RMASK_BASE_LS[@]}"; do
        run_train_safe --config "${SYN_DIR}/synprom_RMask.json" --default_config "${DEFAULT_CONFIG}" \
          --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" --l "${l}" --validate_with accuracy
      done
    done
  fi

  if [[ "${RUN_RMASK_LEARNED}" == "1" ]]; then
    echo "  Syn RMask learned sweep: l x lsparse"
    for fold in "${FOLDS[@]}"; do
      for l in "${RMASK_LEARNED_LS[@]}"; do
        for lsparse in "${RMASK_LEARNED_LSPARSES[@]}"; do
          run_train_safe --config "${SYN_DIR}/synprom_RMask.json" --default_config "${DEFAULT_CONFIG}" \
            --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" \
            --perturn learned \
            --l "${l}" --perturb learned --perturb_fill ema --perturb_lsparse "${lsparse}" \
            --validate_with accuracy
        done
      done
    done
  fi

  if [[ "${RUN_RMASK_RANDOM}" == "1" ]]; then
    echo "  Syn RMask random sweep: l x pmin"
    for fold in "${FOLDS[@]}"; do
      for l in "${RMASK_RANDOM_LS[@]}"; do
        for pmin in "${RMASK_RANDOM_PMINS[@]}"; do
          run_train_safe --config "${SYN_DIR}/synprom_RMask.json" --default_config "${DEFAULT_CONFIG}" \
            --fold "${fold}" --lr "${METHOD_LR}" --wd "${METHOD_WD}" \
            --perturn random \
            --l "${l}" --perturb random --perturb_fill ema --perturb_pmin "${pmin}" \
            --validate_with accuracy
        done
      done
    done
  fi
fi
