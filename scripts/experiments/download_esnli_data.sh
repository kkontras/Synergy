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
import os
import subprocess
from pathlib import Path
import shutil

img_dir  = Path(os.environ["IMG_DIR"])
hf_cache = Path(os.environ["HF_CACHE"])
hf_repo_dir = hf_cache / "flickr30k_repo"
flickr_tar = hf_cache / "flickr30k.tar.gz"
flickr_url = "http://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.tar.gz"

img_dir.mkdir(parents=True, exist_ok=True)
hf_cache.mkdir(parents=True, exist_ok=True)

def copy_images_from_tree(src_root: Path) -> int:
    copied = 0
    for p in src_root.rglob("*.jpg"):
        dst = img_dir / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
            copied += 1
    return copied

current = len(list(img_dir.glob("*.jpg")))
if current >= 30000:
    print(f"Found {current} images already present in {img_dir}, skipping download.")
    raise SystemExit(0)

hf_ok = False
try:
    from huggingface_hub import snapshot_download
    print(f"Trying HuggingFace snapshot download (nlphuji/flickr30k) into {hf_repo_dir} ...")
    snapshot_download(
        repo_id="nlphuji/flickr30k",
        repo_type="dataset",
        cache_dir=str(hf_cache),
        local_dir=str(hf_repo_dir),
        allow_patterns=["*.jpg", "**/*.jpg"],
    )
    copied = copy_images_from_tree(hf_repo_dir)
    total = len(list(img_dir.glob("*.jpg")))
    print(f"HuggingFace step done. Copied {copied} new images, total now {total}.")
    hf_ok = total >= 30000
except Exception as e:
    print(f"HuggingFace image download path failed: {e}")

if not hf_ok:
    print(f"Falling back to direct Flickr30k tarball: {flickr_url}")
    if not flickr_tar.exists() or flickr_tar.stat().st_size == 0:
        subprocess.check_call(["wget", "-O", str(flickr_tar), flickr_url])
    else:
        print(f"Using existing tarball: {flickr_tar}")

    extract_dir = hf_cache / "flickr30k_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["tar", "-xzf", str(flickr_tar), "-C", str(extract_dir)])
    copied = copy_images_from_tree(extract_dir)
    total = len(list(img_dir.glob("*.jpg")))
    print(f"Tarball step done. Copied {copied} new images, total now {total}.")

final_total = len(list(img_dir.glob("*.jpg")))
if final_total < 30000:
    raise RuntimeError(
        f"Expected about 31k Flickr30k images, found only {final_total} in {img_dir}. "
        "Check network/access and retry."
    )
print(f"Done. Flickr30k images ready: {final_total} files in {img_dir}.")
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
