#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="${1:-mydatasets/Factor_CL_Datasets/prepared}"
FORCE_FLAG="${2:-}"

if [[ "${FORCE_FLAG}" == "--force" ]]; then
  FORCE_OPT="--force"
else
  FORCE_OPT=""
fi

echo "[factorcl-affect] output root: ${OUTPUT_ROOT}"
echo "[factorcl-affect] force: ${FORCE_OPT:-false}"

python "${SCRIPT_DIR}/download_mosi.py" \
  --output-root "${OUTPUT_ROOT}" ${FORCE_OPT}

python "${SCRIPT_DIR}/download_mosei.py" \
  --output-root "${OUTPUT_ROOT}" ${FORCE_OPT}

python "${SCRIPT_DIR}/download_ur_funny.py" \
  --output-root "${OUTPUT_ROOT}" ${FORCE_OPT}

python "${SCRIPT_DIR}/download_mustard.py" \
  --output-root "${OUTPUT_ROOT}" ${FORCE_OPT}

echo "[factorcl-affect] all downloads complete"
