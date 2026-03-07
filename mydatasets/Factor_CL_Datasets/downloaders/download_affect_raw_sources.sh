#!/usr/bin/env bash
set -euo pipefail

# Download raw MultiBench affect datasets from official source folders.
# Requires: python + gdown
#
# Usage:
#   bash download_affect_raw_sources.sh [RAW_ROOT] [--skip-existing]
#
# Example:
#   bash download_affect_raw_sources.sh \
#     /esat/smcdata/users/kkontras/Image_Dataset/no_backup/FactorCL_Raw/raw_sources \
#     --skip-existing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

RAW_ROOT="${1:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/FactorCL_Raw/raw_sources}"
SKIP_EXISTING_FLAG="${2:-}"

if [[ "${SKIP_EXISTING_FLAG}" == "--skip-existing" ]]; then
  SKIP_EXISTING=1
else
  SKIP_EXISTING=0
fi

mkdir -p "${RAW_ROOT}"

python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("gdown") is None:
    print("gdown is not installed. Install it with: python -m pip install gdown")
    sys.exit(2)
PY

download_folder() {
  local name="$1"
  local url="$2"
  local out_dir="$3"

  mkdir -p "${out_dir}"
  if [[ "${SKIP_EXISTING}" -eq 1 ]] && [[ -n "$(ls -A "${out_dir}" 2>/dev/null || true)" ]]; then
    echo "[raw:${name}] skip (already non-empty): ${out_dir}"
    return 0
  fi

  echo "[raw:${name}] downloading -> ${out_dir}"
  python -m gdown --folder "${url}" -O "${out_dir}"
}

# Official links from MultiBench datadownload.rst
download_folder "mosi" "https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq" "${RAW_ROOT}/mosi"
download_folder "mosei" "https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv" "${RAW_ROOT}/mosei"
download_folder "ur_funny" "https://drive.google.com/drive/folders/1Agzm157lciMONHOHemHRSySmjn1ahHX1?usp=sharing" "${RAW_ROOT}/ur_funny"
download_folder "mustard" "https://drive.google.com/drive/folders/1JFcX-NF97zu9ZOZGALGU9kp8dwkP7aJ7" "${RAW_ROOT}/mustard"

echo "[raw] complete: ${RAW_ROOT}"

