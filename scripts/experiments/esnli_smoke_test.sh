#!/usr/bin/env bash
# =============================================================================
# ESNLI Codebook-v2 Smoke Test
#
# Validates the CLS token fix end-to-end:
#   1. Build a small v2 cache (CLS stored as special token ID, masks["hint"] saved)
#   2. Verify cache quality (CLS hit rate, hint_mask coverage)
#   3. Train combined / image-only / text-only models
#   4. Print final accuracy comparison
#
# Usage:
#   GPU=0 bash scripts/experiments/esnli_smoke_test.sh
#   GPU=1 TRAIN_SAMPLES=500 bash scripts/experiments/esnli_smoke_test.sh
# =============================================================================
set -euo pipefail

# ---- Tuneable params --------------------------------------------------------
GPU=${GPU:-0}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-1000}  # number of train items to cache
VAL_SAMPLES=${VAL_SAMPLES:-1000}
TEST_SAMPLES=${TEST_SAMPLES:-1000}
BUILD_BATCH=${BUILD_BATCH:-4}         # batch size for the cache builder
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-VL-2B-Instruct"}

DATA_ROOT="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI"
SMOKE_CACHE="${DATA_ROOT}/cache_v2_smoke"
CKPT_DIR="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ESNLI/smoke"

DEFAULT_CFG="./configs/ESNLI/default_config_esnli_smoke.json"

# Common training CLI flags (fast, no wandb, small LR)
TRAIN_FLAGS="--fold 0 --lr 0.0001 --wd 0.00001 --batch_size 4 --tdqm_disable"

# ---- Helpers ----------------------------------------------------------------
hr() { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; }

# =============================================================================
hr "STEP 1  Build v2 smoke cache (train / validation / test)"
# =============================================================================
mkdir -p "$SMOKE_CACHE"
mkdir -p "$CKPT_DIR"

for SPLIT in train validation test; do
    if   [ "$SPLIT" = "train" ];      then N=$TRAIN_SAMPLES
    elif [ "$SPLIT" = "validation" ]; then N=$VAL_SAMPLES
    else                                   N=$TEST_SAMPLES
    fi

    SPLIT_DIR="${SMOKE_CACHE}/${SPLIT}"
    if [ -f "${SPLIT_DIR}/manifest.jsonl" ]; then
        CACHED=$(wc -l < "${SPLIT_DIR}/manifest.jsonl")
        echo "  [skip] ${SPLIT}: manifest already exists (${CACHED} shards)"
        continue
    fi

    echo "  Building ${SPLIT} (max_samples=${N}) ..."
    CUDA_VISIBLE_DEVICES=${GPU} python mydatasets/ESNLI/ESNLI_CodeBook_v2.py \
        --split       "${SPLIT}"           \
        --data_root   "${DATA_ROOT}"       \
        --out_dir     "${SMOKE_CACHE}"     \
        --model_name  "${MODEL_NAME}"      \
        --max_samples "${N}"               \
        --shard_size  "${N}"               \
        --batch_size  "${BUILD_BATCH}"     \
        --device      "cuda:${GPU}"        \
        --dtype       fp16
    echo "  Done: ${SPLIT}"
done

# =============================================================================
hr "STEP 2  Verify cache quality"
# =============================================================================
python3 - <<'PYEOF'
import os, sys, torch, json

SMOKE_CACHE = os.environ.get(
    "SMOKE_CACHE",
    "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_v2_smoke"
)

# After the model adds <CLS> as a special token to Qwen3-VL-2B (vocab 151669),
# the new token gets ID = original_vocab_size = 151669.
# We verify this directly from the cached input_ids.
CLS_TEXT_IDS = {27, 87716, 397}   # old: '<', 'CLS', '>'

all_ok = True
for split in ("train", "validation", "test"):
    split_dir = os.path.join(SMOKE_CACHE, split)
    manifest  = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.exists(manifest):
        print(f"  [WARN] {split}: manifest missing, skipping verify")
        continue

    with open(manifest) as f:
        shards = [json.loads(l)["shard"] for l in f if l.strip()]

    total = hint_hit = cls_hit = 0
    for shard_name in shards:
        items = torch.load(os.path.join(split_dir, shard_name),
                           map_location="cpu", weights_only=False)
        for it in items:
            total += 1
            # Check masks["hint"] is saved and non-empty
            masks = it.get("masks", {})
            h = masks.get("hint", None)
            if torch.is_tensor(h) and h.any():
                hint_hit += 1

            # Check that CLS appears as a NEW special token (not old text tokens)
            ids = it["input_ids"].reshape(-1).tolist()
            # New special token: any ID >= 151669 that is not an image token
            has_new_cls = any(tid >= 151669 for tid in ids)
            # Old text CLS: all three of <, CLS, > present
            has_old_cls = CLS_TEXT_IDS.issubset(set(ids))
            if has_new_cls and not has_old_cls:
                cls_hit += 1
            elif has_old_cls:
                # Old-style CLS: still text tokens — this is the bug we fixed
                pass

    hint_pct = 100 * hint_hit / total if total else 0
    cls_pct  = 100 * cls_hit  / total if total else 0
    status   = "OK" if hint_pct > 80 and cls_pct > 80 else "WARN"
    print(f"  [{status}] {split:12s}: {total} items | "
          f"hint_mask coverage {hint_pct:5.1f}% | "
          f"CLS-as-special-token {cls_pct:5.1f}%")
    if status == "WARN":
        all_ok = False

if all_ok:
    print("\n  [PASS] Cache verification passed.")
else:
    print("\n  [WARN] Some checks below 80% — investigate before full run.")
PYEOF

# Export for the inline Python to pick up (bash env → python via os.environ)
export SMOKE_CACHE

# =============================================================================
hr "STEP 3  Train COMBINED model"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_lora.json       \
    --default_config "${DEFAULT_CFG}"                       \
    ${TRAIN_FLAGS}
echo "  Combined model training finished."

# =============================================================================
hr "STEP 4  Train IMAGE-ONLY model"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_image_lora.json  \
    --default_config "${DEFAULT_CFG}"                        \
    ${TRAIN_FLAGS}
echo "  Image-only model training finished."

# =============================================================================
hr "STEP 5  Train TEXT-ONLY model"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_text_lora.json   \
    --default_config "${DEFAULT_CFG}"                        \
    ${TRAIN_FLAGS}
echo "  Text-only model training finished."

# =============================================================================
hr "STEP 6  Train IMAGE-ONLY FROZEN (no LoRA — pre-trained priors baseline)"
# Backbone weights are frozen; only the linear classification head is trained.
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/smoke_image_frozen.json \
    --default_config "${DEFAULT_CFG}"                        \
    ${TRAIN_FLAGS}
echo "  Image-only frozen model training finished."

# =============================================================================
hr "STEP 7  Results summary"
# =============================================================================
python3 - <<'PYEOF'
import os, glob, torch

CKPT_DIR = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ESNLI/smoke"

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
print("  Expected for a well-behaved ESNLI run:")
print("    combined   > 70%")
print("    text-only  > 60%")
print("    image-only ~ 33%  (random — no label info without hypothesis)")
print("    img-frozen ~ 33%  (frozen backbone, linear head only)")
print()
print("  Diagnosis guide:")
print("    img-frozen ~33% and image-only (LoRA) ~33% → masking works correctly")
print("    img-frozen ~33% and image-only (LoRA) >50% → LoRA overfits on 1k samples")
print("    img-frozen >50%                           → pre-trained Qwen3-VL priors are predictive")
print("    image-only >> 33% → check for hint_mask warnings in the training log above")
PYEOF

hr "SMOKE TEST COMPLETE"
echo "  Cache: ${SMOKE_CACHE}"
echo "  Checkpoints: ${CKPT_DIR}"
