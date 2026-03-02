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
FOLD="${FOLD:-0}"
VALIDATE_WITH="${VALIDATE_WITH:-accuracy}"
RUN_FULL_SHOW_BEST="${RUN_FULL_SHOW_BEST:-1}"

DEFAULT_CONFIG="./configs/FactorCL/Mustard/default_config_mustard_VT.json"
RELEASE_DIR="./configs/FactorCL/Mustard/release/VT"
UNIMODAL_VIDEO="${RELEASE_DIR}/unimodal_video.json"
UNIMODAL_TEXT="${RELEASE_DIR}/unimodal_text.json"
METHODS=(
  "${UNIMODAL_VIDEO}"
  "${UNIMODAL_TEXT}"
  "${RELEASE_DIR}/DnR.json"
  "${RELEASE_DIR}/MCR.json"
  "${RELEASE_DIR}/MMPareto.json"
  "${RELEASE_DIR}/ReconBoost.json"
  "${RELEASE_DIR}/synprom_RMask_nonpre.json"
  "${RELEASE_DIR}/synprom_RMask.json"
  "${RELEASE_DIR}/synprom_RMask_learned.json"
  "${RELEASE_DIR}/synprom_RMask_random.json"
)

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

declare -A VAL_LISTS
declare -A TEST_LISTS
declare -A BEST_VAL
declare -A BEST_TEST
declare -A BEST_LABEL
declare -A BEST_ARGS
declare -A BEST_SHOW_CFG  # actual show.py --config path (may differ from tracking key)

strip_ansi() {
  sed -E 's/\x1B\[[0-9;]*[[:alpha:]]//g'
}

float_gt() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit (a>b)?0:1 }'
}

run_show_capture() {
  local cfg="$1"
  shift
  local -a extra=( "$@" )
  local -a cmd=(
    "${PYTHON_BIN}" scripts/entrypoints/show.py
    --config "${cfg}"
    --default_config "${DEFAULT_CONFIG}"
    --fold "${FOLD}"
    --validate_with "${VALIDATE_WITH}"
    "${extra[@]}"
  )
  local out
  out="$(CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" 2>&1 || true)"
  printf '%s\n' "$out"
}

extract_accs() {
  local raw="$1"
  local clean val_line test_line val test
  clean="$(printf '%s\n' "$raw" | strip_ansi)"
  val_line="$(printf '%s\n' "$clean" | grep -m1 '^Val  .*acc_combined:' || true)"
  test_line="$(printf '%s\n' "$clean" | grep -m1 '^Test  .*acc_combined:' || true)"

  # If show.py cannot aggregate (e.g., missing checkpoints), treat as unavailable.
  if [[ -z "${val_line}" || -z "${test_line}" || "${clean}" == *"We could not load"* ]]; then
    printf 'NA|NA\n'
    return 0
  fi
  val="$(printf '%s\n' "$val_line" | grep -oE 'acc_combined: [0-9]+([.][0-9]+)?' | head -n1 | awk '{print $2}' || true)"
  test="$(printf '%s\n' "$test_line" | grep -oE 'acc_combined: [0-9]+([.][0-9]+)?' | head -n1 | awk '{print $2}' || true)"
  [[ -n "${val}" ]] || val="NA"
  [[ -n "${test}" ]] || test="NA"
  printf '%s|%s\n' "${val}" "${test}"
}

record_result() {
  local cfg="$1"
  local label="$2"
  local args_string="$3"
  local val="$4"
  local test="$5"

  VAL_LISTS["$cfg"]+="${label} => ${val}"$'\n'
  TEST_LISTS["$cfg"]+="${label} => ${test}"$'\n'

  if [[ "${val}" != "NA" ]]; then
    if [[ -z "${BEST_VAL[$cfg]:-}" || "${BEST_VAL[$cfg]}" == "NA" ]]; then
      BEST_VAL["$cfg"]="${val}"
      BEST_TEST["$cfg"]="${test}"
      BEST_LABEL["$cfg"]="${label}"
      BEST_ARGS["$cfg"]="${args_string}"
    elif float_gt "${val}" "${BEST_VAL[$cfg]}"; then
      BEST_VAL["$cfg"]="${val}"
      BEST_TEST["$cfg"]="${test}"
      BEST_LABEL["$cfg"]="${label}"
      BEST_ARGS["$cfg"]="${args_string}"
    fi
  fi
}

eval_combo() {
  local cfg="$1"
  local label="$2"
  shift 2
  local -a extra=( "$@" )
  local args_string="${extra[*]}"
  local raw parsed val test

  raw="$(run_show_capture "${cfg}" "${extra[@]}")"
  parsed="$(extract_accs "${raw}")"
  val="${parsed%%|*}"
  test="${parsed##*|}"
  record_result "${cfg}" "${label}" "${args_string}" "${val}" "${test}"
}

print_lists_and_best() {
  local cfg="$1"
  local name
  name="$(basename "$cfg")"

  echo "============================================================"
  echo "CONFIG: ${cfg}"
  echo "VAL_ACC_LIST (${name})"
  if [[ -n "${VAL_LISTS[$cfg]:-}" ]]; then
    printf '%s' "${VAL_LISTS[$cfg]}"
  else
    echo "NA"
  fi

  echo "TEST_ACC_LIST (${name})"
  if [[ -n "${TEST_LISTS[$cfg]:-}" ]]; then
    printf '%s' "${TEST_LISTS[$cfg]}"
  else
    echo "NA"
  fi

  if [[ -n "${BEST_VAL[$cfg]:-}" ]]; then
    echo "BEST_BY_VAL (${name}): ${BEST_LABEL[$cfg]} => val=${BEST_VAL[$cfg]} test=${BEST_TEST[$cfg]:-NA}"
  else
    echo "BEST_BY_VAL (${name}): NA"
  fi
}

run_full_show_for_best() {
  local cfg="$1"
  local show_cfg="${BEST_SHOW_CFG[$cfg]:-$cfg}"
  if [[ "${RUN_FULL_SHOW_BEST}" != "1" ]]; then
    return 0
  fi
  if [[ -z "${BEST_ARGS[$cfg]:-}" ]]; then
    echo "FULL_SHOW_BEST $(basename "$cfg"): skipped (no ready checkpoint)"
    return 0
  fi

  echo "FULL_SHOW_BEST $(basename "$cfg") fold=${FOLD}"
  echo "Command: ${PYTHON_BIN} scripts/entrypoints/show.py --config ${show_cfg} --default_config ${DEFAULT_CONFIG} --fold ${FOLD} --validate_with ${VALIDATE_WITH} ${BEST_ARGS[$cfg]}"
  # shellcheck disable=SC2206
  local extra=( ${BEST_ARGS[$cfg]} )
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" scripts/entrypoints/show.py \
    --config "${show_cfg}" \
    --default_config "${DEFAULT_CONFIG}" \
    --fold "${FOLD}" \
    --validate_with "${VALIDATE_WITH}" \
    "${extra[@]}" || true
}

sweep_config() {
  local cfg="$1"
  local cfg_name
  cfg_name="$(basename "$cfg")"

  case "${cfg_name}" in
    unimodal_video.json)
      local lr wd
      for lr in "${UNIMODAL_LRS[@]}"; do
        for wd in "${UNIMODAL_WDS[@]}"; do
          eval_combo "${cfg}" "lr=${lr},wd=${wd}" --lr "${lr}" --wd "${wd}"
        done
      done
      ;;
    unimodal_text.json)
      local lr wd
      for lr in "${UNIMODAL_LRS[@]}"; do
        for wd in "${UNIMODAL_WDS[@]}"; do
          eval_combo "${cfg}" "lr=${lr},wd=${wd}" --lr "${lr}" --wd "${wd}"
        done
      done
      ;;
    DnR.json)
      local alpha kmpe
      for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
        for kmpe in 1 3 5 10; do
          eval_combo "${cfg}" "lr=${METHOD_FIXED_LR},wd=${METHOD_FIXED_WD},alpha=${alpha},kmepoch=${kmpe}" \
            --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --alpha "${alpha}" --kmepoch "${kmpe}"
        done
      done
      ;;
    MCR.json)
      local l multil
      for l in 0.001 0.01 0.1 1; do
        for multil in 0.01 0.1 1; do
          eval_combo "${cfg}" "lr=${METHOD_FIXED_LR},wd=${METHOD_FIXED_WD},l=${l},multil=${multil}" \
            --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --l "${l}" --multil "${multil}"
        done
      done
      ;;
    MMPareto.json)
      local alpha
      for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
        eval_combo "${cfg}" "lr=${METHOD_FIXED_LR},wd=${METHOD_FIXED_WD},alpha=${alpha}" \
          --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --alpha "${alpha}"
      done
      ;;
    ReconBoost.json)
      local alpha recon_stages recon_weight1
      for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
        for recon_stages in 1 4 10; do
          for recon_weight1 in 1 3 5 10; do
            eval_combo "${cfg}" "lr=${METHOD_FIXED_LR},wd=${METHOD_FIXED_WD},alpha=${alpha},w1=${recon_weight1},w2=1,epochstage=${recon_stages},ensstage=${recon_stages}" \
              --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --alpha "${alpha}" \
              --recon_weight1 "${recon_weight1}" --recon_weight2 1 \
              --recon_epochstages "${recon_stages}" --recon_ensemblestages "${recon_stages}"
          done
        done
      done
      ;;
    synprom_RMask_learned.json)
      # Use synprom_RMask.json config with --perturb learned --perturn learned
      local rmask_cfg="${RELEASE_DIR}/synprom_RMask.json"
      BEST_SHOW_CFG["${cfg}"]="${rmask_cfg}"
      local l lsparse label args_string raw parsed val test
      for l in 0.001 0.01 0.1 1; do
        for lsparse in 0.001 0.01 0.1 1 3 5 10; do
          label="lr=${METHOD_FIXED_LR},wd=${METHOD_FIXED_WD},l=${l},perturb=learned,fill=ema,lsparse=${lsparse}"
          args_string="--lr ${METHOD_FIXED_LR} --wd ${METHOD_FIXED_WD} --l ${l} --perturb learned --perturn learned --perturb_fill ema --perturb_lsparse ${lsparse}"
          raw="$(run_show_capture "${rmask_cfg}" --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
            --l "${l}" --perturb learned --perturn learned --perturb_fill ema --perturb_lsparse "${lsparse}")"
          parsed="$(extract_accs "${raw}")"
          val="${parsed%%|*}"; test="${parsed##*|}"
          record_result "${cfg}" "${label}" "${args_string}" "${val}" "${test}"
        done
      done
      ;;
    synprom_RMask_random.json)
      # Use synprom_RMask.json config with --perturb random --perturn random
      local rmask_cfg="${RELEASE_DIR}/synprom_RMask.json"
      BEST_SHOW_CFG["${cfg}"]="${rmask_cfg}"
      local l pmin label args_string raw parsed val test
      for l in 0.001 0.01 0.1 1; do
        for pmin in 0.1 0.3 0.5 0.7 0.9; do
          label="lr=${METHOD_FIXED_LR},wd=${METHOD_FIXED_WD},l=${l},perturb=random,fill=ema,pmin=${pmin}"
          args_string="--lr ${METHOD_FIXED_LR} --wd ${METHOD_FIXED_WD} --l ${l} --perturb random --perturn random --perturb_fill ema --perturb_pmin ${pmin}"
          raw="$(run_show_capture "${rmask_cfg}" --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
            --l "${l}" --perturb random --perturn random --perturb_fill ema --perturb_pmin "${pmin}")"
          parsed="$(extract_accs "${raw}")"
          val="${parsed%%|*}"; test="${parsed##*|}"
          record_result "${cfg}" "${label}" "${args_string}" "${val}" "${test}"
        done
      done
      ;;
    synprom_RMask_nonpre.json)
      # Vanilla fusion (no pretrained encoders): trained without --l, use fixed lr/wd
      eval_combo "${cfg}" "lr=${METHOD_FIXED_LR},wd=${METHOD_FIXED_WD}" \
        --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}"
      ;;
    synprom_RMask.json)
      # Vanilla fusion (RMask architecture, no IB): fixed at l=0, no perturb
      eval_combo "${cfg}" "lr=${METHOD_FIXED_LR},wd=${METHOD_FIXED_WD},l=0" \
        --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --l 0
      ;;
    *)
      local lr wd
      for lr in "${METHOD_LRS[@]}"; do
        for wd in "${METHOD_WDS[@]}"; do
          eval_combo "${cfg}" "lr=${lr},wd=${wd}" --lr "${lr}" --wd "${wd}"
        done
      done
      ;;
  esac
}

main() {
  echo "Mustard VT sweep via show.py (scores use show.py 3-fold averages)"
  echo "GPU=${GPU} trigger_fold_arg=${FOLD} validate_with=${VALIDATE_WITH} full_show_best=${RUN_FULL_SHOW_BEST}"
  echo "Methods fixed lr/wd for method-specific sweeps: lr=${METHOD_FIXED_LR} wd=${METHOD_FIXED_WD}"

  local cfg
  for cfg in "${METHODS[@]}"; do
    sweep_config "${cfg}"
    print_lists_and_best "${cfg}"
    run_full_show_for_best "${cfg}"
  done
}

main "$@"
