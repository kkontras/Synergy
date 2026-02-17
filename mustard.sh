#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
MODE="${2:-all}"   # all | train | show

DEFAULT_CONFIG="./configs/FactorCL/Mustard/default_config_mustard_VT.json"

FOLDS=(0 1 2)
BASE_LRS=(0.0001)
BASE_WDS=(0.0001)
METHOD_LRS=(0.0001 0.00005)
METHOD_WDS=(0.0001 0.00001)
RMASK_LS=(0 0.1 1)

run_train() {
  CUDA_VISIBLE_DEVICES="${GPU}" python train.py "$@"
}

run_show() {
  CUDA_VISIBLE_DEVICES="${GPU}" python show.py "$@"
}

do_train() { [[ "${MODE}" == "all" || "${MODE}" == "train" ]]; }
do_show()  { [[ "${MODE}" == "all" || "${MODE}" == "show"  ]]; }

# Baselines (unimodal)
for fold in "${FOLDS[@]}"; do
  for lr in "${BASE_LRS[@]}"; do
    for wd in "${BASE_WDS[@]}"; do
      if do_train; then
        run_train --config ./configs/FactorCL/Mustard/release/VT/unimodal_video.json --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
      fi
      if do_show && [[ "${fold}" == "0" ]]; then
        run_show --config ./configs/FactorCL/Mustard/release/VT/unimodal_video.json --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
      fi

      if do_train; then
        run_train --config ./configs/FactorCL/Mustard/release/VT/unimodal_text.json --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
      fi
      if do_show && [[ "${fold}" == "0" ]]; then
        run_show --config ./configs/FactorCL/Mustard/release/VT/unimodal_text.json --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
      fi
    done
  done
done

# Additional methods: MCR, MMPareto, DnR
for cfg in MCR MMPareto DnR; do
  for fold in "${FOLDS[@]}"; do
    for lr in "${METHOD_LRS[@]}"; do
      for wd in "${METHOD_WDS[@]}"; do
        if do_train; then
          run_train --config "./configs/FactorCL/Mustard/release/VT/${cfg}.json" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
        fi
        if do_show && [[ "${fold}" == "0" ]]; then
          run_show --config "./configs/FactorCL/Mustard/release/VT/${cfg}.json" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
        fi
      done
    done
  done
done

# RMask family: base / learned / random with extra l-search
for cfg in synprom_RMask synprom_RMask_learned synprom_RMask_random; do
  for fold in "${FOLDS[@]}"; do
    for lr in "${METHOD_LRS[@]}"; do
      for wd in "${METHOD_WDS[@]}"; do
        for l in "${RMASK_LS[@]}"; do
          if do_train; then
            run_train --config "./configs/FactorCL/Mustard/release/VT/${cfg}.json" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --l "${l}" --cls mlp --validate_with syn_accuracy
          fi
          if do_show && [[ "${fold}" == "0" ]]; then
            run_show --config "./configs/FactorCL/Mustard/release/VT/${cfg}.json" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --l "${l}" --cls mlp --validate_with syn_accuracy
          fi
        done
      done
    done
  done
done
