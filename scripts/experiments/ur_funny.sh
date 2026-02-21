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
MODE="${2:-train}"    # all | train | show | ceu[

DEFAULT_CONFIG="./configs/FactorCL/URFunny/default_config_ur_funny_VT.json"

FOLDS=(0 1 2)
BASE_LRS=(0.001 0.0001 0.00005 0.00001)
BASE_WDS=(0.001 0.0001 0.00001)
METHOD_LRS=(0.001 0.0001 0.00005 0.00001)
METHOD_WDS=(0.001 0.0001 0.00001)
RMASK_LS=(0 0.1 1)

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
  
for fold in "${FOLDS[@]}"; do
  echo ""
#  pass
#  for lr in "${BASE_LRS[@]}"; do
#    for wd in "${BASE_WDS[@]}"; do
##      if do_train; then run_train --config ./configs/FactorCL/URFunny/release/VT/unimodal_video.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr "${lr}" --wd "${wd}" --validate_with accuracy; fi
##      if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/unimodal_video.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr "${lr}" --wd "${wd}" --validate_with accuracy; fi
##
##      if do_train; then run_train --config ./configs/FactorCL/URFunny/release/VT/unimodal_text.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr "${lr}" --wd "${wd}" --validate_with accuracy;  fi
##      if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/unimodal_text.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr "${lr}" --wd "${wd}" --validate_with accuracy; fi
#
#      if do_train; then  run_train --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask_nopre.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr "${lr}" --wd "${wd}" --validate_with accuracy; fi
#      if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask_nopre.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr "${lr}" --wd "${wd}" --validate_with accuracy; fi
#    done
#  done
#  if do_train; then  run_train --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l 0 --validate_with accuracy; fi
#  if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l 0 --validate_with accuracy; fi

  if do_train; then  run_train --config ./configs/FactorCL/URFunny/release/VT/ens.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l 0 --validate_with accuracy; fi
  if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/ens.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l 0 --validate_with accuracy; fi
 
#  for l in 0.001 0.01 0.1 1; do for lsparse in 0.001 0.01 0.1 1 3 5 10; do
##      echo "--config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l  --perturb learned --perturb_fill ema --perturb_lsparse $lsparse --validate_with accuracy"
#      if do_train; then  run_train --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l  --perturb learned --perturb_fill ema --perturb_lsparse $lsparse --validate_with accuracy; fi
#      if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l --perturb learned --perturb_fill ema --perturb_lsparse $lsparse  --validate_with accuracy; fi
#  done; done
#
#  for l in 0.001 0.01 0.1 1; do for pmin in 0.1 0.3 0.5 0.7 0.9; do
##      echo "--config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l --perturb random --perturb_fill ema --perturb_pmin $pmin --validate_with accuracy"
#      if do_train; then  run_train --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l --perturb random --perturb_fill ema --perturb_pmin $pmin --validate_with accuracy; fi
#      if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --l $l --wd 0.001 --perturb random --perturb_fill ema --perturb_pmin $pmin  --validate_with accuracy; fi
#  done; done
#
#  for l in 0.001 0.01 0.1 1; do for multil in 0.01 0.1 1; do
##      echo "--config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l --multil $multil --validate_with accuracy"
#      if do_train; then  run_train --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l --multil $multil --validate_with accuracy; fi
#      if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l --multil $multil --validate_with accuracy; fi
#  done; done


#  for l in 0.001 0.01 0.1 1; do for multil in 0.01 0.1 1; do
##      echo "--config ./configs/FactorCL/URFunny/release/VT/MCR.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l --multil $multil --validate_with accuracy"
#      if do_train; then  run_train --config ./configs/FactorCL/URFunny/release/VT/MCR.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l --multil $multil --validate_with accuracy; fi
#      if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/MCR.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --l $l --multil $multil --validate_with accuracy; fi
#  done; done
#  for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
##      echo "--config ./configs/FactorCL/URFunny/release/VT/MMPareto.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --alpha $alpha --validate_with accuracy"
#      if do_train; then  run_train --config ./configs/FactorCL/URFunny/release/VT/MMPareto.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --alpha $alpha --validate_with accuracy; fi
#      if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/MMPareto.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --alpha $alpha --validate_with accuracy; fi
#  done
#  for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do for kmpe in 1 3 5 10; do
##      echo "--config ./configs/FactorCL/URFunny/release/VT/DnR.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --alpha $alpha --kmepoch $kmpe --validate_with accuracy"
#      if do_train; then  run_train --config ./configs/FactorCL/URFunny/release/VT/DnR.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --alpha $alpha --kmepoch $kmpe --validate_with accuracy; fi
#      if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/DnR.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --alpha $alpha --kmepoch $kmpe --validate_with accuracy; fi
#  done; done
  for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do for recon_stages in 1 4 10; do for recon_weight1 in 1 3 5 10; do
#      echo "--config ./configs/FactorCL/URFunny/release/VT/ReconBoost.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --alpha $alpha --recon_weight1 $recon_weight1 --recon_weight2 1 --recon_epochstages $recon_stages --recon_ensemblestages $recon_stages --validate_with accuracy"
      if do_train; then  run_train --config ./configs/FactorCL/URFunny/release/VT/ReconBoost.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --alpha $alpha --recon_weight1 $recon_weight1 --recon_weight2 1 --recon_epochstages $recon_stages --recon_ensemblestages $recon_stages --validate_with accuracy; fi
      if do_show && [[ "${fold}" == "0" ]]; then run_show --config ./configs/FactorCL/URFunny/release/VT/ReconBoost.json --default_config "${DEFAULT_CONFIG}" --fold $fold --lr 0.001 --wd 0.001 --alpha $alpha --recon_weight1 $recon_weight1 --recon_weight2 1 --recon_epochstages $recon_stages --recon_ensemblestages $recon_stages --validate_with accuracy; fi
  done;done;done
done


#SynIB_RMask_fold0_l1_vldaccuracy_perturblearned_fillema_lsparse3_lr0.001_wd0.001.pth.tar
#SynIB_RMask_fold2_l0.001_vldaccuracy_perturbrandom_fillema_pmin0.7_lr0.001_wd0.001.pth.tar

#CEU_LR="${CEU_LR:-0.001}"
#CEU_WD="${CEU_WD:-0.001}"
#run_ceu \
#  --dataset ur_funny \
#  --default_config "${DEFAULT_CONFIG}" \
#  --unimodal_configs \
#    ./configs/FactorCL/URFunny/release/VT/unimodal_video.json \
#    ./configs/FactorCL/URFunny/release/VT/unimodal_text.json \
#  --folds 0 1 2 \
#  --lr "${CEU_LR}" \
#  --wd "${CEU_WD}" \
#  --validate_with accuracy
