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

GPU="${GPU:-${1:-0}}"
MODE="${MODE:-all}" # all | prep | ceu | methods | table

run_step() {
  echo
  echo "=== $* ==="
  "$@"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/experiments/mosi_vt_full_pipeline.sh [GPU]

Environment:
  MODE=all|prep|ceu|methods|table   (default: all)

What "all" runs:
  1) select_unimodals (workflow state)
  2) select_rmask_nopre (workflow state -> best method lr/wd)
  3) mosi_vt.sh ceu      (uses best unimodal lr/wd from state, writes CEU files)
  4) mosi_vt.sh patch    (patch encoder + CEU paths into configs)
  5) mosi_vt.sh methods  (auto-reads best RMask_nopre lr/wd from state)
  6) mosi_vt_build_method_table.sh

Examples:
  bash scripts/experiments/mosi_vt_full_pipeline.sh 0
  MODE=prep bash scripts/experiments/mosi_vt_full_pipeline.sh 0
  MODE=methods bash scripts/experiments/mosi_vt_full_pipeline.sh 0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "${MODE}" in
  prep)
    run_step bash scripts/experiments/mosi_vt_prepare.sh "${GPU}" select_unimodals
    run_step bash scripts/experiments/mosi_vt_prepare.sh "${GPU}" select_rmask_nopre
    ;;
  ceu)
    run_step bash scripts/experiments/mosi_vt.sh "${GPU}" ceu
    run_step bash scripts/experiments/mosi_vt.sh "${GPU}" patch
    ;;
  methods)
    run_step bash scripts/experiments/mosi_vt.sh "${GPU}" methods
    ;;
  table)
    run_step bash scripts/experiments/mosi_vt_build_method_table.sh
    ;;
  all)
    run_step bash scripts/experiments/mosi_vt_prepare.sh "${GPU}" select_unimodals
    run_step bash scripts/experiments/mosi_vt_prepare.sh "${GPU}" select_rmask_nopre
    run_step bash scripts/experiments/mosi_vt.sh "${GPU}" ceu
    run_step bash scripts/experiments/mosi_vt.sh "${GPU}" patch
    run_step bash scripts/experiments/mosi_vt.sh "${GPU}" methods
    run_step bash scripts/experiments/mosi_vt_build_method_table.sh
    ;;
  *)
    echo "Unsupported MODE=${MODE}" >&2
    usage
    exit 1
    ;;
esac

echo
echo "Finished MODE=${MODE} (GPU=${GPU})"
