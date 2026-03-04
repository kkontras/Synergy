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
MODE="${2:-all}"   # all | unimodal | ceu | patch | methods

DEFAULT_CONFIG="./configs/FactorCL/Mosi/default_config_mosi_VT.json"
SYN_DIR="./configs/FactorCL/Mosi/syn/VT"
UNIMODAL_VIDEO="${SYN_DIR}/unimodal_video.json"
UNIMODAL_TEXT="${SYN_DIR}/unimodal_text.json"
UNIMODAL_AUDIO="${SYN_DIR}/unimodal_audio.json"
SYNPROM_RMASK_BASE_CFG="${SYN_DIR}/synprom_RMask.json"
SYNPROM_RMASK_NOPRE_CFG="${SYN_DIR}/synprom_RMask_nopre.json"

METHODS=(
  "${SYN_DIR}/MMPareto.json"
  "${SYN_DIR}/DnR.json"
  "${SYN_DIR}/MCR.json"
  "${SYN_DIR}/ReconBoost.json"
  "${SYN_DIR}/MLB.json"
  "${SYN_DIR}/ens.json"
  "${SYN_DIR}/joint_training.json"
  "${SYNPROM_RMASK_NOPRE_CFG}"
  "${SYNPROM_RMASK_BASE_CFG}"
  "${SYN_DIR}/synprom_RMask_learned.json"        # virtual label -> train with base cfg + --perturb learned
  "${SYN_DIR}/synprom_RMask_random.json"         # virtual label -> train with base cfg + --perturb random
  "${SYN_DIR}/synprom_RMask_nopre_learned.json"  # virtual label -> train with nopre cfg + --perturb learned
  "${SYN_DIR}/synprom_RMask_nopre_random.json"   # virtual label -> train with nopre cfg + --perturb random
)

# Method scheduling:
# - METHOD_RUN_MODE=all (default): run every method sweep
# - METHOD_RUN_MODE=single: run one method only (basename/stem accepted)
METHOD_RUN_MODE="${METHOD_RUN_MODE:-all}"
METHOD_TARGET="${METHOD_TARGET:-MMPareto.json}"

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV:-0,1,2}"
IFS=',' read -r -a UNIMODAL_LRS <<< "${UNIMODAL_LRS_CSV:-0.001,0.0005,0.0001,0.00005}"
IFS=',' read -r -a UNIMODAL_WDS <<< "${UNIMODAL_WDS_CSV:-0.001,0.0001,0.00001}"

# Best unimodals should be selected first (same workflow as Mustard/URFunny)
BEST_VIDEO_LR="${BEST_VIDEO_LR:-}"
BEST_VIDEO_WD="${BEST_VIDEO_WD:-}"
BEST_TEXT_LR="${BEST_TEXT_LR:-}"
BEST_TEXT_WD="${BEST_TEXT_WD:-}"
BEST_AUDIO_LR="${BEST_AUDIO_LR:-}"
BEST_AUDIO_WD="${BEST_AUDIO_WD:-}"

# Fixed optimizer pair for methods should come from best synprom_RMask_nopre val acc.
METHOD_FIXED_LR="${METHOD_FIXED_LR:-}"
METHOD_FIXED_WD="${METHOD_FIXED_WD:-}"
VT_STATE_PATH="${VT_STATE_PATH:-./artifacts/reports/mosi_vt_workflow_state.json}"

# Hyperparameter grids for method-specific sweeps (lr/wd stays fixed above).
MCR_L_CSV="${MCR_L_CSV:-0.001,0.01,0.1,1}"
MCR_MULTIL_CSV="${MCR_MULTIL_CSV:-0.01,0.1,1}"
MMPARETO_ALPHA_CSV="${MMPARETO_ALPHA_CSV:-0.5,1.0,1.5,2.0,3.0,5.0}"
DNR_ALPHA_CSV="${DNR_ALPHA_CSV:-0.5,1.0,1.5,2.0,3.0,5.0}"
DNR_KMEPOCH_CSV="${DNR_KMEPOCH_CSV:-1,3,5,10}"
RECONBOOST_ALPHA_CSV="${RECONBOOST_ALPHA_CSV:-0.5,1.0,1.5,2.0,3.0,5.0}"
RECONBOOST_STAGES_CSV="${RECONBOOST_STAGES_CSV:-1,4,10}"
RECONBOOST_W1_CSV="${RECONBOOST_W1_CSV:-1,3,5,10}"
RMASK_BASE_L="${RMASK_BASE_L:-0}"                    # Base RMask row uses --l 0
RMASK_LEARNED_L_CSV="${RMASK_LEARNED_L_CSV:-0.001,0.01,0.1,1}"
RMASK_LEARNED_LSPARSE_CSV="${RMASK_LEARNED_LSPARSE_CSV:-0.001,0.01,0.1,1,3,5,10}"
RMASK_RANDOM_L_CSV="${RMASK_RANDOM_L_CSV:-0.001,0.01,0.1,1}"
RMASK_RANDOM_PMIN_CSV="${RMASK_RANDOM_PMIN_CSV:-0.1,0.3,0.5,0.7,0.9}"

IFS=',' read -r -a MCR_LS <<< "${MCR_L_CSV}"
IFS=',' read -r -a MCR_MULTILS <<< "${MCR_MULTIL_CSV}"
IFS=',' read -r -a MMPARETO_ALPHAS <<< "${MMPARETO_ALPHA_CSV}"
IFS=',' read -r -a DNR_ALPHAS <<< "${DNR_ALPHA_CSV}"
IFS=',' read -r -a DNR_KMEPOCHS <<< "${DNR_KMEPOCH_CSV}"
IFS=',' read -r -a RECONBOOST_ALPHAS <<< "${RECONBOOST_ALPHA_CSV}"
IFS=',' read -r -a RECONBOOST_STAGES <<< "${RECONBOOST_STAGES_CSV}"
IFS=',' read -r -a RECONBOOST_W1S <<< "${RECONBOOST_W1_CSV}"
IFS=',' read -r -a RMASK_LEARNED_LS <<< "${RMASK_LEARNED_L_CSV}"
IFS=',' read -r -a RMASK_LEARNED_LSPARSES <<< "${RMASK_LEARNED_LSPARSE_CSV}"
IFS=',' read -r -a RMASK_RANDOM_LS <<< "${RMASK_RANDOM_L_CSV}"
IFS=',' read -r -a RMASK_RANDOM_PMINS <<< "${RMASK_RANDOM_PMIN_CSV}"

run_train() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/train.py "$@"; }
run_ceu() { CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/get_ceu_cli.py "$@"; }
load_bests_from_state() {
  [[ -f "${VT_STATE_PATH}" ]] || return 0
  local state_out
  state_out="$("${PYTHON_BIN}" - "${VT_STATE_PATH}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
except Exception:
    sys.exit(0)
u = data.get("unimodals", {})
def emit(env_name, key, field):
    v = (((u.get(key) or {}).get("best") or {}).get(field))
    if v is not None:
        print(f'{env_name}="{v}"')
emit("BEST_VIDEO_LR", "unimodal_video", "lr")
emit("BEST_VIDEO_WD", "unimodal_video", "wd")
emit("BEST_TEXT_LR", "unimodal_text", "lr")
emit("BEST_TEXT_WD", "unimodal_text", "wd")
emit("BEST_AUDIO_LR", "unimodal_audio", "lr")
emit("BEST_AUDIO_WD", "unimodal_audio", "wd")
r = data.get("best_rmask_nopre") or {}
if r.get("lr") is not None:
    print(f'METHOD_FIXED_LR="{r["lr"]}"')
if r.get("wd") is not None:
    print(f'METHOD_FIXED_WD="{r["wd"]}"')
PY
)" || true
  [[ -n "${state_out}" ]] || return 0
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    case "${line}" in
      BEST_VIDEO_LR=*) [[ -z "${BEST_VIDEO_LR}" ]] && eval "${line}" ;;
      BEST_VIDEO_WD=*) [[ -z "${BEST_VIDEO_WD}" ]] && eval "${line}" ;;
      BEST_TEXT_LR=*) [[ -z "${BEST_TEXT_LR}" ]] && eval "${line}" ;;
      BEST_TEXT_WD=*) [[ -z "${BEST_TEXT_WD}" ]] && eval "${line}" ;;
      BEST_AUDIO_LR=*) [[ -z "${BEST_AUDIO_LR}" ]] && eval "${line}" ;;
      BEST_AUDIO_WD=*) [[ -z "${BEST_AUDIO_WD}" ]] && eval "${line}" ;;
      METHOD_FIXED_LR=*) [[ -z "${METHOD_FIXED_LR}" ]] && eval "${line}" ;;
      METHOD_FIXED_WD=*) [[ -z "${METHOD_FIXED_WD}" ]] && eval "${line}" ;;
    esac
  done <<< "${state_out}"
}
patch_method_ceu_paths() {
  local ceu_val="./artifacts/ceus/mosi/mosi_ceu_val.pkl"
  local ceu_test="./artifacts/ceus/mosi/mosi_ceu_test.pkl"
  "${PYTHON_BIN}" - "${ceu_val}" "${ceu_test}" "${METHODS[@]}" <<'PY'
import json, sys
from pathlib import Path
ceu_val, ceu_test = sys.argv[1], sys.argv[2]
paths = []
seen = set()
for p in sys.argv[3:]:
    if p in seen:
        continue
    seen.add(p)
    paths.append(Path(p))
for p in paths:
    if not p.exists():
        continue
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        print(f"[ceu-patch] skip {p} (json error: {e})")
        continue
    model = data.setdefault("model", {})
    ceu = model.get("ceu")
    changed = False
    if not isinstance(ceu, dict):
        model["ceu"] = {"val": ceu_val, "test": ceu_test}
        changed = True
    else:
        if ceu.get("val") != ceu_val:
            ceu["val"] = ceu_val
            changed = True
        if ceu.get("test") != ceu_test:
            ceu["test"] = ceu_test
            changed = True
    if changed:
        p.write_text(json.dumps(data, indent=2) + "\n")
        print(f"[ceu-patch] updated {p}")
    else:
        print(f"[ceu-patch] ok {p}")
PY
}
run_patch_configs() {
  "${PYTHON_BIN}" scripts/experiments/factorcl_vt_workflow.py \
    --dataset mosi \
    --default_config "${DEFAULT_CONFIG}" \
    --syn_dir "${SYN_DIR}" \
    --gpu "${GPU}" \
    --python_bin "${PYTHON_BIN}" \
    --mode patch
}
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
    [[ -z "${BEST_VIDEO_LR}" || -z "${BEST_VIDEO_WD}" || -z "${BEST_TEXT_LR}" || -z "${BEST_TEXT_WD}" || -z "${BEST_AUDIO_LR}" || -z "${BEST_AUDIO_WD}" ]]
    return
  fi
  return 1
}
do_ceu() {
  [[ "${SKIP_CEU:-0}" == "1" ]] && return 1
  [[ "${MODE}" == "all" || "${MODE}" == "ceu" || "${MODE}" == "methods" ]]
}
do_patch() {
  [[ "${SKIP_PATCH:-0}" == "1" ]] && return 1
  [[ "${MODE}" == "all" || "${MODE}" == "patch" || "${MODE}" == "methods" ]]
}
do_methods() { [[ "${MODE}" == "all" || "${MODE}" == "methods" ]]; }

load_bests_from_state

if do_unimodal; then
  for fold in "${FOLDS[@]}"; do
    for lr in "${UNIMODAL_LRS[@]}"; do
      for wd in "${UNIMODAL_WDS[@]}"; do
        run_train --config "${UNIMODAL_VIDEO}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
        run_train --config "${UNIMODAL_TEXT}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
        run_train --config "${UNIMODAL_AUDIO}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
      done
    done
  done
fi

if [[ "${MODE}" == "all" && ( -z "${BEST_VIDEO_LR}" || -z "${BEST_VIDEO_WD}" || -z "${BEST_TEXT_LR}" || -z "${BEST_TEXT_WD}" || -z "${BEST_AUDIO_LR}" || -z "${BEST_AUDIO_WD}" ) ]]; then
  echo "Unimodal search finished."
  echo "Set BEST_VIDEO_*, BEST_TEXT_*, BEST_AUDIO_* and rerun with MODE=ceu (or MODE=all)."
  exit 0
fi

if do_ceu; then
  if [[ -z "${BEST_VIDEO_LR}" || -z "${BEST_VIDEO_WD}" || -z "${BEST_TEXT_LR}" || -z "${BEST_TEXT_WD}" ]]; then
    echo "Missing BEST_VIDEO_* / BEST_TEXT_* vars for CEU stage."
    exit 1
  fi
  if [[ "${BEST_VIDEO_LR}" != "${BEST_TEXT_LR}" || "${BEST_VIDEO_WD}" != "${BEST_TEXT_WD}" ]]; then
    echo "Warning: get_ceu_cli uses one lr/wd suffix for both CEU unimodals."
    echo "Using video settings for CEU: lr=${BEST_VIDEO_LR} wd=${BEST_VIDEO_WD}"
  fi
  echo "Note: CEU generation in VT uses video+text unimodals."
  run_ceu \
    --dataset mosi \
    --default_config "${DEFAULT_CONFIG}" \
    --unimodal_configs "${UNIMODAL_VIDEO}" "${UNIMODAL_TEXT}" \
    --folds "${FOLDS[@]}" \
    --unimodal_lrs "${BEST_VIDEO_LR}" "${BEST_TEXT_LR}" \
    --unimodal_wds "${BEST_VIDEO_WD}" "${BEST_TEXT_WD}" \
    --lr "${BEST_VIDEO_LR}" \
    --wd "${BEST_VIDEO_WD}" \
    --validate_with accuracy
  patch_method_ceu_paths
fi

if do_patch; then
  if [[ -z "${BEST_VIDEO_LR}" || -z "${BEST_VIDEO_WD}" || -z "${BEST_TEXT_LR}" || -z "${BEST_TEXT_WD}" ]]; then
    echo "Missing BEST_VIDEO_* / BEST_TEXT_* vars for patch stage."
    exit 1
  fi
  # factorcl_vt_workflow patch mode reads best unimodals from its saved state; if absent, patch manually is not possible.
  # Users typically run scripts/experiments/mosi_vt_prepare.sh first to populate state. This call still works once state exists.
  run_patch_configs || {
    echo "Patch stage failed. Run scripts/experiments/mosi_vt_prepare.sh (select_unimodals + ceu + patch) first, or ensure workflow state exists."
    exit 1
  }
fi

if do_methods; then
  patch_method_ceu_paths
  if [[ -z "${METHOD_FIXED_LR}" || -z "${METHOD_FIXED_WD}" ]]; then
    echo "Missing METHOD_FIXED_LR/METHOD_FIXED_WD."
    echo "Run scripts/experiments/mosi_vt_prepare.sh ... select_rmask_nopre first, or set them explicitly."
    exit 1
  fi

  echo "Methods stage uses fixed optimizer lr/wd from synprom_RMask_nopre best val acc: lr=${METHOD_FIXED_LR} wd=${METHOD_FIXED_WD}"
  if [[ "${METHOD_RUN_MODE}" == "single" ]]; then
    echo "Methods stage running ONE method for inspection first: ${METHOD_TARGET}"
    echo "Set METHOD_RUN_MODE=all to run all methods."
  fi

  for fold in "${FOLDS[@]}"; do
    for cfg in "${METHODS[@]}"; do
      cfg_name="$(basename "${cfg}")"
      if [[ "${METHOD_RUN_MODE}" == "single" ]]; then
        target="${METHOD_TARGET}"
        [[ "${target}" == *.json ]] || target="${target}.json"
        if [[ "${cfg_name}" != "${target}" && "${cfg_name%.json}" != "${target%.json}" ]]; then
          continue
        fi
      fi

      case "${cfg_name}" in
        ens.json)
          run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
            --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --l "${RMASK_BASE_L}" --validate_with accuracy
          ;;
        MCR.json)
          for l in "${MCR_LS[@]}"; do
            for multil in "${MCR_MULTILS[@]}"; do
              run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --l "${l}" --multil "${multil}" --validate_with accuracy
            done
          done
          ;;
        MMPareto.json)
          for alpha in "${MMPARETO_ALPHAS[@]}"; do
            run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
              --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
              --alpha "${alpha}" --validate_with accuracy
          done
          ;;
        DnR.json)
          for alpha in "${DNR_ALPHAS[@]}"; do
            for kmpe in "${DNR_KMEPOCHS[@]}"; do
              run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --alpha "${alpha}" --kmepoch "${kmpe}" --validate_with accuracy
            done
          done
          ;;
        ReconBoost.json)
          for alpha in "${RECONBOOST_ALPHAS[@]}"; do
            for stages in "${RECONBOOST_STAGES[@]}"; do
              for w1 in "${RECONBOOST_W1S[@]}"; do
                run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                  --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                  --alpha "${alpha}" \
                  --recon_weight1 "${w1}" --recon_weight2 1 \
                  --recon_epochstages "${stages}" --recon_ensemblestages "${stages}" \
                  --validate_with accuracy
              done
            done
          done
          ;;
        synprom_RMask_learned.json)
          for l in "${RMASK_LEARNED_LS[@]}"; do
            for lsparse in "${RMASK_LEARNED_LSPARSES[@]}"; do
              run_train_method_safe --config "${SYNPROM_RMASK_BASE_CFG}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --perturn learned \
                --l "${l}" --perturb learned --perturb_fill ema --perturb_lsparse "${lsparse}" \
                --validate_with accuracy
            done
          done
          ;;
        synprom_RMask_random.json)
          for l in "${RMASK_RANDOM_LS[@]}"; do
            for pmin in "${RMASK_RANDOM_PMINS[@]}"; do
              run_train_method_safe --config "${SYNPROM_RMASK_BASE_CFG}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --perturn random \
                --l "${l}" --perturb random --perturb_fill ema --perturb_pmin "${pmin}" \
                --validate_with accuracy
            done
          done
          ;;
        synprom_RMask.json)
          run_train_method_safe --config "${SYNPROM_RMASK_BASE_CFG}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
            --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --l "${RMASK_BASE_L}" --validate_with accuracy
          ;;
        synprom_RMask_nopre.json)
          run_train_method_safe --config "${SYNPROM_RMASK_NOPRE_CFG}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
            --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --validate_with accuracy
          ;;
        synprom_RMask_nopre_learned.json)
          for l in "${RMASK_LEARNED_LS[@]}"; do
            for lsparse in "${RMASK_LEARNED_LSPARSES[@]}"; do
              run_train_method_safe --config "${SYNPROM_RMASK_NOPRE_CFG}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --l "${l}" --perturb learned --perturb_fill ema --perturb_lsparse "${lsparse}" \
                --validate_with accuracy
            done
          done
          ;;
        synprom_RMask_nopre_random.json)
          for l in "${RMASK_RANDOM_LS[@]}"; do
            for pmin in "${RMASK_RANDOM_PMINS[@]}"; do
              run_train_method_safe --config "${SYNPROM_RMASK_NOPRE_CFG}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --l "${l}" --perturb random --perturb_fill ema --perturb_pmin "${pmin}" \
                --validate_with accuracy
            done
          done
          ;;
        *)
          run_train_method_safe --config "${cfg}" --default_config "${DEFAULT_CONFIG}" --fold "${fold}" \
            --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --validate_with accuracy
          ;;
      esac
    done
  done
fi
