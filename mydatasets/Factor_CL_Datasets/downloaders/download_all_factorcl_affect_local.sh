#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${1:-}"
OUTPUT_ROOT="${2:-mydatasets/Factor_CL_Datasets/prepared}"
MODE="${3:-copy}" # copy | symlink
FORCE_FLAG="${4:-}"

if [[ -z "${SOURCE_ROOT}" ]]; then
  echo "Usage: $0 <source_root> [output_root] [copy|symlink] [--force]"
  exit 1
fi

if [[ "${MODE}" != "copy" && "${MODE}" != "symlink" ]]; then
  echo "Invalid mode '${MODE}'. Use 'copy' or 'symlink'."
  exit 1
fi

if [[ "${FORCE_FLAG}" == "--force" ]]; then
  FORCE_OPT="--force"
else
  FORCE_OPT=""
fi

if [[ "${MODE}" == "symlink" ]]; then
  LINK_OPT="--symlink"
else
  LINK_OPT=""
fi

resolve_source() {
  local dataset="$1"
  local filename="$2"
  local a="${SOURCE_ROOT}/${dataset}/${filename}"
  local b="${SOURCE_ROOT}/${filename}"
  if [[ -f "${a}" ]]; then
    echo "${a}"
    return 0
  fi
  if [[ -f "${b}" ]]; then
    echo "${b}"
    return 0
  fi
  echo ""
}

MOSI_SRC="$(resolve_source mosi mosi_data.pkl)"
MOSEI_SRC="$(resolve_source mosei mosei_data.pkl)"
UR_FUNNY_SRC="$(resolve_source ur_funny ur_funny_data.pkl)"
MUSTARD_SRC="$(resolve_source mustard mustard_data.pkl)"

for src in "${MOSI_SRC}" "${MOSEI_SRC}" "${UR_FUNNY_SRC}" "${MUSTARD_SRC}"; do
  if [[ -z "${src}" ]]; then
    echo "Missing one or more source files under ${SOURCE_ROOT}"
    echo "Expected either:"
    echo "  ${SOURCE_ROOT}/<dataset>/<dataset>_data.pkl"
    echo "or"
    echo "  ${SOURCE_ROOT}/<dataset>_data.pkl"
    exit 1
  fi
done

echo "[factorcl-affect-local] source root: ${SOURCE_ROOT}"
echo "[factorcl-affect-local] output root: ${OUTPUT_ROOT}"
echo "[factorcl-affect-local] mode: ${MODE}"
echo "[factorcl-affect-local] force: ${FORCE_OPT:-false}"

python "${SCRIPT_DIR}/download_mosi.py" \
  --output-root "${OUTPUT_ROOT}" --local-file "${MOSI_SRC}" ${LINK_OPT} ${FORCE_OPT}

python "${SCRIPT_DIR}/download_mosei.py" \
  --output-root "${OUTPUT_ROOT}" --local-file "${MOSEI_SRC}" ${LINK_OPT} ${FORCE_OPT}

python "${SCRIPT_DIR}/download_ur_funny.py" \
  --output-root "${OUTPUT_ROOT}" --local-file "${UR_FUNNY_SRC}" ${LINK_OPT} ${FORCE_OPT}

python "${SCRIPT_DIR}/download_mustard.py" \
  --output-root "${OUTPUT_ROOT}" --local-file "${MUSTARD_SRC}" ${LINK_OPT} ${FORCE_OPT}

echo "[factorcl-affect-local] all datasets staged from local sources"

