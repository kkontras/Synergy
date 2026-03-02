#!/usr/bin/env bash
# =============================================================================
# ESNLI Full Pipeline — MIB cluster
#
# Builds a full v2 cache (401k train / 14k val / 14k test) then trains all
# 4 models: combined, image-only (LoRA), text-only (LoRA), image-only (frozen).
#
# Paths assume the MIB scratch filesystem at /scratch/kkontras/.
# The Flickr30k images and e-ViL annotations must already be present under
# /scratch/kkontras/ESNLI/.
#
# Usage (sequential, single GPU):
#   bash scripts/experiments/esnli_full_mib.sh
#
# Usage (cache only, then train separately):
#   MODE=cache bash scripts/experiments/esnli_full_mib.sh
#   MODE=train bash scripts/experiments/esnli_full_mib.sh
#
# Tuneable env vars:
#   GPU=5            which CUDA device to use
#   BUILD_BATCH=4    batch size for the cache builder
#   SHARD_SIZE=4096  items per shard file
#   MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct
# =============================================================================
set -eu

GPU=${GPU:-5}
MODE=${MODE:-all}       # all | cache | train
BUILD_BATCH=${BUILD_BATCH:-4}
SHARD_SIZE=${SHARD_SIZE:-4096}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-VL-2B-Instruct"}

DATA_ROOT="/scratch/kkontras/ESNLI"
FULL_CACHE="${DATA_ROOT}/cache_v2_full"
CKPT_DIR="/scratch/kkontras/checkpoints/synergy/ESNLI/full"
DEFAULT_CFG="./configs/ESNLI/default_config_esnli_cache_mib.json"

TRAIN_FLAGS="--fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 --tdqm_disable"

hr() { echo; echo "================================================================"; echo "  $*"; echo "================================================================"; }

mkdir -p "${FULL_CACHE}" "${CKPT_DIR}"

# =============================================================================
if [ "${MODE}" = "all" -o "${MODE}" = "cache" ]; then

hr "STEP 1  Build full v2 cache"
# Train: ~401k samples | Val: ~14k | Test: ~14k
# With SHARD_SIZE=4096 → ~98 train shards, ~4 val/test shards each.
# Expected time: several hours depending on GPU speed.

for SPLIT in train validation test; do
    SPLIT_DIR="${FULL_CACHE}/${SPLIT}"
    if [ -f "${SPLIT_DIR}/manifest.jsonl" ]; then
        N=$(python3 -c "
import json
total = sum(json.loads(l)['num_items'] for l in open('${SPLIT_DIR}/manifest.jsonl'))
print(total)
")
        echo "  [skip] ${SPLIT}: manifest exists (${N} cached items)"
        continue
    fi

    echo "  Building ${SPLIT} ..."
    CUDA_VISIBLE_DEVICES=${GPU} python mydatasets/ESNLI/ESNLI_CodeBook_v2.py \
        --split       "${SPLIT}"        \
        --data_root   "${DATA_ROOT}"    \
        --out_dir     "${FULL_CACHE}"   \
        --model_name  "${MODEL_NAME}"   \
        --shard_size  "${SHARD_SIZE}"   \
        --batch_size  "${BUILD_BATCH}"  \
        --device      "cuda:${GPU}"     \
        --dtype       fp16
    echo "  Done: ${SPLIT}"
done

# =============================================================================
hr "STEP 2  Verify cache quality"

python3 - <<'PYEOF'
import os, torch, json, glob

FULL_CACHE = "/scratch/kkontras/ESNLI/cache_v2_full"
CLS_TEXT_IDS = {27, 87716, 397}

all_ok = True
for split in ("train", "validation", "test"):
    split_dir = os.path.join(FULL_CACHE, split)
    manifest  = os.path.join(split_dir, "manifest.jsonl")
    if not os.path.exists(manifest):
        print(f"  [WARN] {split}: manifest missing")
        all_ok = False
        continue

    with open(manifest) as f:
        shards = [json.loads(l)["shard"] for l in f if l.strip()]

    # Sample-check first shard only (full check would be slow on 400k items)
    sample_shards = shards[:3]
    total = hint_hit = cls_hit = 0
    for shard_name in sample_shards:
        items = torch.load(os.path.join(split_dir, shard_name),
                           map_location="cpu", weights_only=False)
        for it in items:
            total += 1
            masks = it.get("masks", {})
            h = masks.get("hint", None)
            if torch.is_tensor(h) and h.any():
                hint_hit += 1
            ids = it["input_ids"].reshape(-1).tolist()
            has_new_cls = any(tid >= 151669 for tid in ids)
            has_old_cls = CLS_TEXT_IDS.issubset(set(ids))
            if has_new_cls and not has_old_cls:
                cls_hit += 1

    hint_pct = 100 * hint_hit / total if total else 0
    cls_pct  = 100 * cls_hit  / total if total else 0
    n_shards = len(shards)
    status   = "OK" if hint_pct > 80 and cls_pct > 80 else "WARN"
    print(f"  [{status}] {split:12s}: {n_shards} shards | "
          f"sampled {total} items | "
          f"hint_mask {hint_pct:5.1f}% | "
          f"CLS-special {cls_pct:5.1f}%")
    if status == "WARN":
        all_ok = False

if all_ok:
    print("\n  [PASS] Cache spot-check passed.")
else:
    print("\n  [WARN] Cache issues detected — investigate before training.")
    exit(1)
PYEOF

fi   # end cache block

# =============================================================================
if [ "${MODE}" = "all" -o "${MODE}" = "train" ]; then

# =============================================================================
hr "STEP 3  Train COMBINED model (image + text, LoRA)"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/full_lora.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  Combined done."

# =============================================================================
hr "STEP 4  Train IMAGE-ONLY model (LoRA)"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/full_image_lora.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  Image-only done."

# =============================================================================
hr "STEP 5  Train TEXT-ONLY model (LoRA)"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/full_text_lora.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  Text-only done."

# =============================================================================
hr "STEP 6  Train IMAGE-ONLY FROZEN (no LoRA — pre-trained priors baseline)"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/full_image_frozen.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  Image-only frozen done."

# =============================================================================
hr "STEP 7  Train MCR model (combined, LoRA + MCR bias infusion)"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/full_mcr.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  MCR done."

# =============================================================================
hr "STEP 8  Train MMPareto model (combined, LoRA + Pareto gradient)"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/full_mmpareto.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  MMPareto done."

# =============================================================================
hr "STEP 9  Train DnR model (dual-forward, LoRA + Disagree-and-Reweight)"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/full_dnr.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  DnR done."

# =============================================================================
hr "STEP 10  Train ReconBoost model (dual-forward, LoRA + ReconBoost)"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/full_reconboost.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  ReconBoost done."

# =============================================================================
hr "STEP 11  Train RMask model (combined, LoRA + random modality masking)"
# =============================================================================
CUDA_VISIBLE_DEVICES=${GPU} python scripts/entrypoints/train.py \
    --config         ./configs/ESNLI/full_rmask.json \
    --default_config "${DEFAULT_CFG}" \
    ${TRAIN_FLAGS}
echo "  RMask done."

# =============================================================================
hr "STEP 12  Results summary"
# =============================================================================
python3 - <<'PYEOF'
import os, glob, torch

CKPT_DIR = "/scratch/kkontras/checkpoints/synergy/ESNLI/full"

models = {
    "combined":   "esnli_full_combined_*.pth.tar",
    "image-only": "esnli_full_image_fold*.pth.tar",
    "text-only":  "esnli_full_text_*.pth.tar",
    "img-frozen": "esnli_full_image_frozen_*.pth.tar",
    "mcr":        "esnli_full_mcr_*.pth.tar",
    "mmpareto":   "esnli_full_mmpareto_*.pth.tar",
    "dnr":        "esnli_full_dnr_*.pth.tar",
    "reconboost": "esnli_full_reconboost_*.pth.tar",
    "rmask":      "esnli_full_rmask_*.pth.tar",
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

    print(f"  {name:<16}  {best_val:>12.4f}  {test_acc:>18.4f}  {f1_str:>24}  {os.path.basename(ckpt_path)}")

print()
print("  Expected:")
print("    combined   > 80%  (Qwen3-VL-2B + LoRA on full dataset)")
print("    text-only  > 70%  (hypothesis alone is very informative for NLI)")
print("    image-only ~ 33%  (no label info without hypothesis)")
print("    img-frozen ~ 33%  (frozen backbone, linear head only)")
print("  Bias-infusion methods (dual-forward, trained on full dataset):")
print("    mcr        ~ combined  (MCR degenerate w/o perturbation outputs)")
print("    mmpareto   >= combined (Pareto gradient balancing)")
print("    dnr        >= combined (Disagree-and-Reweight)")
print("    reconboost >= combined (boosting between modalities)")
print("    rmask      >= combined (random modality masking, p=0.15 each)")
PYEOF

fi   # end train block

hr "DONE"
echo "  Cache:       ${FULL_CACHE}"
echo "  Checkpoints: ${CKPT_DIR}"
