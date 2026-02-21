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
MODE="${2:-show}"    # all | train | show | ceu[

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


#if do_show; then run_show --config ./configs/FactorCL/URFunny/release/VT/unimodal_video.json --default_config "${DEFAULT_CONFIG}" --fold 0 --lr 0.001 --wd 0.0001 --validate_with accuracy; fi
#if do_show; then run_show --config ./configs/FactorCL/URFunny/release/VT/unimodal_text.json --default_config "${DEFAULT_CONFIG}" --fold 0 --lr 0.001 --wd 0.001 --validate_with accuracy; fi
#if do_show; then run_show --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask_nopre.json --default_config "${DEFAULT_CONFIG}" --fold 0 --lr 0.001 --wd 0.001 --validate_with accuracy; fi
#if do_show; then run_show --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold 0 --lr 0.001 --wd 0.001 --l 0 --validate_with accuracy; fi
if do_show; then run_show --config ./configs/FactorCL/URFunny/release/VT/ens.json --default_config "${DEFAULT_CONFIG}" --fold 0 --lr 0.001 --wd 0.001 --l 0 --validate_with accuracy; fi
if do_show; then run_show --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold 0 --lr 0.001 --l 1 --wd 0.001 --perturb learned --perturb_fill ema --perturb_lsparse 3   --validate_with accuracy; fi
if do_show; then run_show --config ./configs/FactorCL/URFunny/release/VT/synprom_RMask.json --default_config "${DEFAULT_CONFIG}" --fold 0 --lr 0.001 --l 0.001 --wd 0.001 --perturb random --perturb_fill ema --perturb_pmin 0.7   --validate_with accuracy; fi

