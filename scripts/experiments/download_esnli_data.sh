#!/usr/bin/env bash
# download_esnli_data.sh — Download required ESNLI assets (Flickr30k + e-SNLI-VE)
#
# Run once on a Tier1 login/interactive node:
#   bash condor_logs/tier1/download_esnli_data.sh
# Or pass a custom root:
#   bash condor_logs/tier1/download_esnli_data.sh /path/to/ESNLI
#
# What this does:
#   1. Downloads Flickr30k images from HuggingFace (nlphuji/flickr30k) into
#      DATA_ROOT/flickr30k-images/*.jpg  (~9 GB, 31k images)
#   2. Downloads/extracts e-ViL repo zip (contains train/dev/test split files).
#
# The cache builder (jobfile_esnli_cache.pbs) expects:
#   DATA_ROOT/flickr30k-images/<img_id>.jpg

set -euo pipefail

SCRIPT_REV="2026-03-04-hfzip-v2"
DATA_ROOT="${1:-${DATA_ROOT:-/dodrio/scratch/projects/2026_029/kkontras/data/ESNLI}}"
IMG_DIR="${DATA_ROOT}/flickr30k-images"
HF_CACHE="${DATA_ROOT}/hf_cache"
EVIL_ZIP="${DATA_ROOT}/e-ViL-main.zip"
EVIL_DIR="${DATA_ROOT}/e-ViL-main"
EVIL_URL_PRIMARY="https://github.com/maximek3/e-ViL/archive/refs/heads/main.zip"
EVIL_URL_FALLBACK="https://github.com/virginie-do/e-SNLI-VE/archive/refs/heads/master.zip"

mkdir -p "${IMG_DIR}" "${HF_CACHE}"
export IMG_DIR HF_CACHE

echo "[download_esnli_data.sh] revision=${SCRIPT_REV}"
SKIP_IMAGES="${SKIP_IMAGES:-0}"

if [ "${SKIP_IMAGES}" != "1" ]; then
echo "Downloading Flickr30k images to ${IMG_DIR} ..."
echo "(This requires ~9 GB of disk space and may take 20-40 minutes.)"

python - <<'PYEOF'
import os
from pathlib import Path
import zipfile

img_dir  = Path(os.environ["IMG_DIR"])
hf_cache = Path(os.environ["HF_CACHE"])
extracted_dir = img_dir / "flickr30k-images"

img_dir.mkdir(parents=True, exist_ok=True)
hf_cache.mkdir(parents=True, exist_ok=True)

def count_jpgs(root: Path) -> int:
    return len(list(root.rglob("*.jpg")))

if extracted_dir.is_dir() and count_jpgs(extracted_dir) >= 30000:
    print(f"Found extracted Flickr30k at {extracted_dir}, skipping download.")
    raise SystemExit(0)

from huggingface_hub import hf_hub_download

print(f"Downloading flickr30k-images.zip from HF dataset repo (cache: {hf_cache}) ...")
zip_path = Path(
    hf_hub_download(
        repo_id="nlphuji/flickr30k",
        filename="flickr30k-images.zip",
        repo_type="dataset",
        cache_dir=str(hf_cache),
    )
)
print(f"Downloaded zip to: {zip_path}")

print(f"Extracting zip to: {img_dir}")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(img_dir)

if extracted_dir.is_dir():
    final_total = count_jpgs(extracted_dir)
    final_dir = extracted_dir
else:
    final_total = count_jpgs(img_dir)
    final_dir = img_dir

if final_total < 30000:
    raise RuntimeError(
        f"Expected about 31k Flickr30k images, found only {final_total} in {final_dir}. "
        "Check network/access and retry."
    )
print(f"Done. Flickr30k images ready: {final_total} files in {final_dir}.")
PYEOF
else
echo "SKIP_IMAGES=1 -> skipping Flickr30k image download/extract step."
fi

echo ""
echo "Ensuring e-SNLI-VE annotations from e-ViL ..."
if [ ! -d "${EVIL_DIR}" ]; then
    if [ ! -s "${EVIL_ZIP}" ]; then
        echo "Downloading ${EVIL_ZIP} ..."
        if ! wget -O "${EVIL_ZIP}" "${EVIL_URL_PRIMARY}"; then
            echo "Primary e-ViL URL failed, trying fallback e-SNLI-VE repo ..."
            EVIL_ZIP="${DATA_ROOT}/e-SNLI-VE-master.zip"
            EVIL_DIR="${DATA_ROOT}/e-SNLI-VE-master"
            wget -O "${EVIL_ZIP}" "${EVIL_URL_FALLBACK}"
        fi
    else
        echo "Found existing ${EVIL_ZIP}"
    fi
    echo "Extracting ${EVIL_ZIP} ..."
    unzip -q -o "${EVIL_ZIP}" -d "${DATA_ROOT}"
else
    echo "Found extracted repo: ${EVIL_DIR}"
fi

echo "Checking for split files ..."
SPLIT_COUNT=$(find "${EVIL_DIR}" -type f | grep -E 'train|dev|test' | grep -E '\.(csv|tsv|json|jsonl)$' | wc -l || true)
if [ "${SPLIT_COUNT}" -lt 3 ]; then
    echo "ERROR: Could not find expected train/dev/test split files under ${EVIL_DIR}"
    exit 1
fi

echo "Done. ESNLI assets are ready under ${DATA_ROOT}"
