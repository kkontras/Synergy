#!/usr/bin/env bash
# =============================================================================
# ESNLI Smoke — TRAIN ONLY (cache already built)
#
# Assumes cache_v2_smoke already exists at:
#   /esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_v2_smoke/
#
# Trains three models and prints a comparison table:
#   combined | image-only | text-only
#
# Usage:
#   GPU=0 bash scripts/experiments/esnli_smoke_train_only.sh
#   GPU=0 BS=8 bash scripts/experiments/esnli_smoke_train_only.sh
# =============================================================================
set -euo pipefail

GPU=${GPU:-0}
BS=${BS:-4}
LR=${LR:-0.0001}
WD=${WD:-0.00001}

DEFAULT_CFG="./configs/ESNLI/default_config_esnli_smoke.json"
CKPT_DIR="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ESNLI/smoke"
SMOKE_CACHE="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_v2_smoke"

TRAIN_FLAGS="--fold 0 --lr ${LR} --wd ${WD} --batch_size ${BS} --tdqm_disable"

hr() { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; }

# Verify cache exists
for SPLIT in train validation test; do
    MANIFEST="${SMOKE_CACHE}/${SPLIT}/manifest.jsonl"
    if [ ! -f "${MANIFEST}" ]; then
        echo "[ERROR] Missing cache manifest: ${MANIFEST}"
        echo "  Build the cache first with esnli_smoke_test.sh (STEP 1 only)."
        exit 1
    fi
    N=$(wc -l < "${MANIFEST}")
    echo "  [cache OK] ${SPLIT}: ${N} shard(s) in manifest"
done

mkdir -p "${CKPT_DIR}"

# =============================================================================
hr "STEP 1  Train COMBINED model (image + text)"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_lora.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  Combined done."

# =============================================================================
hr "STEP 2  Train IMAGE-ONLY model"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_image_lora.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  Image-only done."

# =============================================================================
hr "STEP 3  Train TEXT-ONLY model"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_text_lora.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  Text-only done."

# =============================================================================
hr "STEP 4  Train IMAGE-ONLY FROZEN (no LoRA — pre-trained priors baseline)"
# Backbone weights are frozen; only the linear classification head is trained.
# If this also gets ~65%, the high accuracy is inherent to Qwen3-VL pre-training,
# not a fine-tuning artifact.  If it drops to ~33%, LoRA is the culprit.
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_image_frozen.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  Image-only frozen done."

# =============================================================================
hr "STEP 5  Results summary"
# =============================================================================
python3 - <<'PYEOF'
import os, glob, torch
from collections import defaultdict

CKPT_DIR = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ESNLI/smoke"

models = {
    "combined":      "smoke_combined_*.pth.tar",
    "image-only":    "smoke_image_fold*.pth.tar",   # LoRA: name contains "fold"
    "text-only":     "smoke_text_*.pth.tar",
    "img-frozen":    "smoke_image_frozen_*.pth.tar",
}

print()
print(f"  {'Model':<16}  {'Best-val acc':>12}  {'Test acc @ best-val':>18}  {'F1 per-class (val)':>24}  Checkpoint")
print(f"  {'-'*16}  {'-'*12}  {'-'*18}  {'-'*24}  {'-'*40}")

for name, pattern in models.items():
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, pattern)))
    if not paths:
        print(f"  {name:<16}  {'(no ckpt)':>12}")
        continue

    # pick the most recently modified checkpoint
    paths.sort(key=os.path.getmtime)
    ckpt_path = paths[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    logs = ckpt.get("logs", {})
    best = logs.get("best_logs", {})

    # Best val accuracy entry
    bv = best.get("best_vaccuracy", {})
    best_val = bv.get("acc", {}).get("combined", float("nan"))
    best_f1  = bv.get("f1_perclass", {}).get("combined", None)
    best_step = bv.get("step", "?")

    # Corresponding test accuracy at that step
    test_logs = logs.get("test_logs", {})
    test_acc = float("nan")
    if isinstance(test_logs, dict) and best_step in test_logs:
        test_acc = test_logs[best_step].get("acc", {}).get("combined", float("nan"))
    elif isinstance(test_logs, dict) and str(best_step) in test_logs:
        test_acc = test_logs[str(best_step)].get("acc", {}).get("combined", float("nan"))

    f1_str = ""
    if best_f1 is not None and hasattr(best_f1, "tolist"):
        f1_str = "[" + ", ".join(f"{v:.3f}" for v in best_f1.tolist()) + "]"
    elif best_f1 is not None:
        f1_str = str(best_f1)

    print(f"  {name:<16}  {best_val:>12.4f}  {test_acc:>18.4f}  {f1_str:>24}  {os.path.basename(ckpt_path)}")

print()
print("  Expected for a well-behaved ESNLI run:")
print("    combined   > 70%")
print("    text-only  > 60%")
print("    image-only ~ 33%  (random — no label info without hypothesis)")
print("    img-frozen ~ 33%  (pre-trained priors baseline, backbone frozen)")
print()
print("  Diagnosis guide:")
print("    img-frozen ~33% and image-only (LoRA) ~33% → masking works, visual features non-predictive")
print("    img-frozen ~33% and image-only (LoRA) >50% → LoRA overfits visual patterns on 1k samples")
print("    img-frozen >50%                           → pre-trained Qwen3-VL priors are the source")
print("  If image-only >> 33% → check hint_mask warnings in training log above.")
PYEOF

hr "DONE"
echo "  Checkpoints: ${CKPT_DIR}"
