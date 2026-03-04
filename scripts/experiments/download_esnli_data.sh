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

DATA_ROOT="${1:-${DATA_ROOT:-/dodrio/scratch/projects/2026_029/kkontras/data/ESNLI}}"
IMG_DIR="${DATA_ROOT}/flickr30k-images"
HF_CACHE="${DATA_ROOT}/hf_cache"
EVIL_ZIP="${DATA_ROOT}/e-ViL-main.zip"
EVIL_DIR="${DATA_ROOT}/e-ViL-main"

mkdir -p "${IMG_DIR}" "${HF_CACHE}"
export IMG_DIR HF_CACHE

echo "Downloading Flickr30k images to ${IMG_DIR} ..."
echo "(This requires ~9 GB of disk space and may take 20-40 minutes.)"

python - <<'PYEOF'
import os, sys
from pathlib import Path
from datasets import load_dataset

img_dir  = Path(os.environ["IMG_DIR"])
hf_cache = Path(os.environ["HF_CACHE"])

print(f"Loading nlphuji/flickr30k from HuggingFace (cache: {hf_cache}) ...")
ds = load_dataset("nlphuji/flickr30k", split="test", cache_dir=str(hf_cache), trust_remote_code=True)

total = len(ds)
saved = 0
skipped = 0
for i, ex in enumerate(ds):
    img_id = ex.get("img_id") or ex.get("filename") or f"{i}.jpg"
    if not img_id.endswith(".jpg"):
        img_id = img_id + ".jpg"
    out_path = img_dir / img_id
    if out_path.exists():
        skipped += 1
    else:
        ex["image"].save(str(out_path))
        saved += 1
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{total}  saved={saved}  skipped={skipped}", flush=True)

print(f"Done. {saved} images saved, {skipped} already existed.")
PYEOF

echo ""
echo "Ensuring e-SNLI-VE annotations from e-ViL ..."
if [ ! -d "${EVIL_DIR}" ]; then
    if [ ! -s "${EVIL_ZIP}" ]; then
        echo "Downloading ${EVIL_ZIP} ..."
        wget -O "${EVIL_ZIP}" "https://github.com/multimodal-ai-lab/e-ViL/archive/refs/heads/main.zip"
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
