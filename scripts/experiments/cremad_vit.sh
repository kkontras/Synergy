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

DEFAULT_CONFIG="./configs/CREMA_D/default_config_cremad_vit_syn.json"
VIT_VARIANT="${VIT_VARIANT:-legacy}"   # legacy | lora
if [[ "${VIT_VARIANT}" == "lora" ]]; then
  RELEASE_DIR="./configs/CREMA_D/synergy/vit_lora"
else
  RELEASE_DIR="./configs/CREMA_D/synergy/vit"
fi

# Ironic-label controls (set IRONIC_RATE=0 for standard 6-class CREMAD)
IRONIC_RATE="${IRONIC_RATE:-1.0}"
if [[ "${IRONIC_RATE}" == "0" || "${IRONIC_RATE}" == "0.0" ]]; then
  NUM_CLASSES="${NUM_CLASSES:-6}"
else
  NUM_CLASSES="${NUM_CLASSES:-7}"
fi

FOLDS=(0 1 2)
BASE_LRS=(0.0001 0.00005)
BASE_WDS=(0.0001 0.00001 0.000005)
METHOD_LRS=(0.0001 0.00005)
METHOD_WDS=(0.0001 0.00001)

COMMON_ARGS=(--num_classes "${NUM_CLASSES}")
if [[ "${IRONIC_RATE}" != "0" && "${IRONIC_RATE}" != "0.0" ]]; then
  COMMON_ARGS+=(--ironic_rate "${IRONIC_RATE}")
fi

run_train() {
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/train.py "$@"
}

run_show() {
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/show.py "$@"
}

run_ceu() {
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/get_ceu_cli.py "$@"
}

do_train() { [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; }
do_show()  { [[ "${MODE}" == "all" || "${MODE}" == "show"  ]]; }
do_ceu()   { [[ "${MODE}" == "all" || "${MODE}" == "ceu"   ]]; }

if [[ "${VIT_VARIANT}" == "lora" ]]; then
  METHODS=(ens)
else
  METHODS=(ens MCR MMPareto DnR ReconBoost AGM OGM PMR MSLR MMCosine MLA MLB joint_training multiloss pre_finetuned pre_frozen)
fi

# Baselines (unimodal)
for fold in "${FOLDS[@]}"; do
  for lr in "${BASE_LRS[@]}"; do
    for wd in "${BASE_WDS[@]}"; do
      if do_train; then
        run_train --config "${RELEASE_DIR}/unimodal_audio.json" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy "${COMMON_ARGS[@]}"
      fi
      if do_show && [[ "${fold}" == "0" ]]; then
        run_show --config "${RELEASE_DIR}/unimodal_audio.json" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy "${COMMON_ARGS[@]}"
      fi

      if do_train; then
        run_train --config "${RELEASE_DIR}/unimodal_video.json" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy "${COMMON_ARGS[@]}"
      fi
      if do_show && [[ "${fold}" == "0" ]]; then
        run_show --config "${RELEASE_DIR}/unimodal_video.json" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy "${COMMON_ARGS[@]}"
      fi
    done
  done
done

if do_ceu; then
  CEU_LR="${CEU_LR:-0.00005}"
  CEU_WD="${CEU_WD:-0.000005}"
  run_ceu \
    --dataset "${CEU_DATASET_TAG:-cremad_vit_${VIT_VARIANT}}" \
    --default_config "${DEFAULT_CONFIG}" \
    --unimodal_configs \
      "${RELEASE_DIR}/unimodal_audio.json" \
      "${RELEASE_DIR}/unimodal_video.json" \
    --folds 0 1 2 \
    --lr "${CEU_LR}" \
    --wd "${CEU_WD}" \
    --validate_with accuracy \
    --num_classes "${NUM_CLASSES}" \
    --ironic_rate "${IRONIC_RATE}"
fi

# Additional methods
for cfg in "${METHODS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    for lr in "${METHOD_LRS[@]}"; do
      for wd in "${METHOD_WDS[@]}"; do
        if do_train; then
          run_train --config "${RELEASE_DIR}/${cfg}.json" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy "${COMMON_ARGS[@]}"
        fi
        if do_show && [[ "${fold}" == "0" ]]; then
          run_show --config "${RELEASE_DIR}/${cfg}.json" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy "${COMMON_ARGS[@]}"
        fi
      done
    done
  done
done
