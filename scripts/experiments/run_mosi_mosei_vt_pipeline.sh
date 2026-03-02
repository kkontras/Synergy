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

MOSI_GPU="${MOSI_GPU:-0}"
MOSEI_GPU="${MOSEI_GPU:-0}"
MODE="${MODE:-all}"                 # all | any factorcl_vt_workflow.py mode
DATASETS="${DATASETS:-mosi,mosei}"  # mosi,mosei or single dataset
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/run_mosi_mosei_vt_pipeline.sh [options] [-- extra workflow args]

Runs the VT workflow pipeline for MOSI and MOSEI (or a subset), using:
  scripts/experiments/mosi_vt_prepare.sh
  scripts/experiments/mosei_vt_prepare.sh

Default behavior:
  - dataset order: mosi -> mosei
  - mode: all
  - gpu: 0 for both

Options:
  --mode MODE           Workflow mode (default: all)
  --datasets CSV        mosi,mosei or single dataset (default: mosi,mosei)
  --mosi-gpu ID         GPU id for MOSI runs (default: 0)
  --mosei-gpu ID        GPU id for MOSEI runs (default: 0)
  -h, --help            Show this help

Examples:
  bash scripts/experiments/run_mosi_mosei_vt_pipeline.sh
  bash scripts/experiments/run_mosi_mosei_vt_pipeline.sh --mosi-gpu 0 --mosei-gpu 1
  bash scripts/experiments/run_mosi_mosei_vt_pipeline.sh --mode report
  bash scripts/experiments/run_mosi_mosei_vt_pipeline.sh --mode train_unimodals -- --unimodal_lrs 0.001,0.0001
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --datasets)
      DATASETS="${2:-}"
      shift 2
      ;;
    --mosi-gpu)
      MOSI_GPU="${2:-}"
      shift 2
      ;;
    --mosei-gpu)
      MOSEI_GPU="${2:-}"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"

run_one() {
  local ds="$1"
  local gpu="$2"
  local script=""
  case "${ds}" in
    mosi) script="scripts/experiments/mosi_vt_prepare.sh" ;;
    mosei) script="scripts/experiments/mosei_vt_prepare.sh" ;;
    *)
      echo "Unsupported dataset '${ds}'. Use mosi and/or mosei." >&2
      return 1
      ;;
  esac

  echo
  echo "============================================================"
  echo "Dataset: ${ds} | GPU: ${gpu} | Mode: ${MODE}"
  echo "Script:  ${script}"
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    echo "Extra:   ${EXTRA_ARGS[*]}"
  fi
  echo "============================================================"

  bash "${script}" "${gpu}" "${MODE}" "${EXTRA_ARGS[@]}"
}

for ds in "${DATASET_LIST[@]}"; do
  ds="$(echo "${ds}" | xargs)"
  [[ -z "${ds}" ]] && continue
  if [[ "${ds}" == "mosi" ]]; then
    run_one "mosi" "${MOSI_GPU}"
  elif [[ "${ds}" == "mosei" ]]; then
    run_one "mosei" "${MOSEI_GPU}"
  else
    echo "Unsupported dataset '${ds}' in --datasets=${DATASETS}" >&2
    exit 1
  fi
done

echo
echo "Finished ${MODE} for datasets: ${DATASETS}"
