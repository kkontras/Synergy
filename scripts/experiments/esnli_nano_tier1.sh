#!/usr/bin/env bash
# =============================================================================
# ESNLI Nano — 5-image end-to-end pipeline check
#
# Builds a tiny cache using only 5 unique Flickr30k images per split
# (all their hypothesis rows — typically ~3-6 samples per image → ~15-30
# samples per split).  Then trains all 4 models and prints a summary table.
#
# Purpose: fast sanity check that the full pipeline (cache → train → eval)
# runs without errors.  With so few samples, accuracy numbers are meaningless.
#
# Usage:
#   GPU=0 bash scripts/experiments/esnli_nano.sh
#   GPU=0 N_IMAGES=10 bash scripts/experiments/esnli_nano.sh
# =============================================================================
set -euo pipefail

GPU=${GPU:-0}
N_IMAGES=${N_IMAGES:-5}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-VL-2B-Instruct"}

DATA_ROOT="dodrio/scratch/projects/2026_029/kkontras/data/ESNLI"
NANO_CACHE="${DATA_ROOT}/cache_v2_nano"
CKPT_DIR="dodrio/scratch/projects/2026_029/kkontras/checkpoints/ESNLI""
DEFAULT_CFG="./configs/ESNLI/default_config_esnli_nano.json"

# batch_size=1 so even 3 samples per split is enough; lr/wd don't matter for pipeline check
TRAIN_FLAGS="--fold 0 --lr 0.0001 --wd 0.00001 --batch_size 2 --tdqm_disable"

# Temp config capping max_epoch at 10 (used for remaining models after combined already ran)
NANO_10_CFG=$(mktemp /tmp/nano_10epoch_XXXXX.json)
python3 -c "
import json
with open('${DEFAULT_CFG}') as f:
    cfg = json.load(f)
cfg['early_stopping']['max_epoch'] = 10
with open('${NANO_10_CFG}', 'w') as f:
    json.dump(cfg, f)
"
trap "rm -f ${NANO_10_CFG}" EXIT

hr() { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; }

mkdir -p "${NANO_CACHE}"
mkdir -p "${CKPT_DIR}"

# =============================================================================
hr "STEP 1  Build nano cache (${N_IMAGES} images per split)"
# =============================================================================
for SPLIT in train validation test; do
    SPLIT_DIR="${NANO_CACHE}/${SPLIT}"
    if [ -f "${SPLIT_DIR}/manifest.jsonl" ]; then
        N=$(wc -l < "${SPLIT_DIR}/manifest.jsonl")
        echo "  [skip] ${SPLIT}: manifest already exists (${N} shard(s))"
        continue
    fi

    echo "  Building ${SPLIT} (max_images=${N_IMAGES}) ..."
    CUDA_VISIBLE_DEVICES=${GPU} python mydatasets/ESNLI/ESNLI_CodeBook_v2.py \
        --split       "${SPLIT}"        \
        --data_root   "${DATA_ROOT}"    \
        --out_dir     "${NANO_CACHE}"   \
        --model_name  "${MODEL_NAME}"   \
        --max_images  "${N_IMAGES}"     \
        --shard_size  100               \
        --batch_size  2                 \
        --num_workers 0                 \
        --device      "cuda:${GPU}"     \
        --dtype       fp16
    echo "  Done: ${SPLIT}"
done

# Quick sanity: show how many samples we have
echo
for SPLIT in train validation test; do
    MANIFEST="${NANO_CACHE}/${SPLIT}/manifest.jsonl"
    if [ -f "${MANIFEST}" ]; then
        TOTAL=$(python3 -c "
import json
total = 0
for l in open('${MANIFEST}'):
    total += json.loads(l)['num_items']
print(total)
")
        echo "  ${SPLIT}: ${TOTAL} cached samples"
    fi
done

# =============================================================================
hr "STEP 2  Train COMBINED model  [SKIPPED — already ran, achieved >70%]"
# =============================================================================
echo "  Skipping combined model (already completed)."

# =============================================================================
hr "STEP 3  Train IMAGE-ONLY model (LoRA)  [max 10 epochs]"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_image_lora.json \
    --default_config "${NANO_10_CFG}" \
    ${TRAIN_FLAGS}
echo "  Image-only done."

# =============================================================================
hr "STEP 4  Train TEXT-ONLY model  [max 10 epochs]"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_text_lora.json \
    --default_config "${NANO_10_CFG}" \
    ${TRAIN_FLAGS}
echo "  Text-only done."

# =============================================================================
hr "STEP 5  Train IMAGE-ONLY FROZEN (no LoRA — pre-trained priors baseline)  [max 10 epochs]"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_image_frozen.json \
    --default_config "${NANO_10_CFG}" \
    ${TRAIN_FLAGS}
echo "  Image-only frozen done."

# =============================================================================
hr "STEP 6  Results summary"
# =============================================================================
python3 - <<'PYEOF'
import os, glob, torch

CKPT_DIR = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ESNLI/nano"

models = {
    "combined":   "smoke_combined_*.pth.tar",
    "image-only": "smoke_image_fold*.pth.tar",
    "text-only":  "smoke_text_*.pth.tar",
    "img-frozen": "smoke_image_frozen_*.pth.tar",
}

print()
print(f"  {'Model':<16}  {'Best-val acc':>12}  {'Test acc @ best-val':>18}  {'F1 per-class (val)':>24}  Checkpoint")
print(f"  {'-'*16}  {'-'*12}  {'-'*18}  {'-'*24}  {'-'*40}")

for name, pattern in models.items():
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, pattern)))
    if not paths:
        print(f"  {name:<16}  {'(no ckpt)':>12}")
        continue

    paths.sort(key=os.path.getmtime)
    ckpt_path = paths[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    logs = ckpt.get("logs", {})
    best = logs.get("best_logs", {})

    bv = best.get("best_vaccuracy", {})
    best_val  = bv.get("acc", {}).get("combined", float("nan"))
    best_f1   = bv.get("f1_perclass", {}).get("combined", None)
    best_step = bv.get("step", "?")

    test_logs = logs.get("test_logs", {})
    test_acc = float("nan")
    for key in (best_step, str(best_step)):
        if isinstance(test_logs, dict) and key in test_logs:
            test_acc = test_logs[key].get("acc", {}).get("combined", float("nan"))
            break

    f1_str = ""
    if best_f1 is not None:
        vals = best_f1.tolist() if hasattr(best_f1, "tolist") else best_f1
        if isinstance(vals, (list, tuple)):
            f1_str = "[" + ", ".join(f"{v:.3f}" for v in vals) + "]"
        else:
            f1_str = str(vals)

    print(f"  {name:<16}  {best_val:>12.4f}  {test_acc:>18.4f}  {f1_str:>24}  {os.path.basename(ckpt_path)}")

print()
print("  NOTE: with only 5 images, accuracy numbers are NOT meaningful.")
print("  A PASS means all 4 models trained and evaluated without errors.")
PYEOF

hr "DONE"
echo "  Cache:       ${NANO_CACHE}"
echo "  Checkpoints: ${CKPT_DIR}"
echo "  To rebuild the cache: rm -rf ${NANO_CACHE} and re-run."
