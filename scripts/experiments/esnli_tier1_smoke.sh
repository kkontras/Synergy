#!/usr/bin/env bash
# =============================================================================
# ESNLI Tier1 Smoke Test (Dodrio)
#
# Purpose:
#   Fast end-to-end sanity checks on Tier1 for:
#   1) Cache builders (v2 and/or legacy v1)
#   2) Training entrypoint with representative ESNLI model configs
#
# Usage:
#   bash scripts/experiments/esnli_tier1_smoke.sh
#
# Useful env vars:
#   MODE=all|cache|train              (default: all)
#   CACHE_METHOD=v2|legacy|both       (default: both)
#   MODEL_SET=basic|extended          (default: basic)
#   GPU=0
#   PROJECT_ID=2026_029
#   DATA_ROOT=/dodrio/scratch/projects/2026_029/$USER/data/ESNLI
#   MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct
#   N_IMAGES=8                        (v2 cache tiny size)
#   LEGACY_MAX_SAMPLES=64             (v1 cache tiny size)
#   BUILD_BATCH=2
#   TRAIN_BS=4
#   TRAIN_MAX_EPOCH=2
# =============================================================================
set -euo pipefail

MODE="${MODE:-all}"                       # all | cache | train
CACHE_METHOD="${CACHE_METHOD:-both}"      # v2 | legacy | both
MODEL_SET="${MODEL_SET:-basic}"           # basic | extended

GPU="${GPU:-0}"
PROJECT_ID="${PROJECT_ID:-2026_029}"
USER_NAME="${USER:-kkontras}"

BASE_ROOT="${BASE_ROOT:-/dodrio/scratch/projects/${PROJECT_ID}/${USER_NAME}}"
DATA_ROOT="${DATA_ROOT:-${BASE_ROOT}/data/ESNLI}"
CACHE_V2_DIR="${CACHE_V2_DIR:-${DATA_ROOT}/cache_v2_tier1_smoke}"
CACHE_V1_DIR="${CACHE_V1_DIR:-${DATA_ROOT}/cache_v1_tier1_smoke}"
SAVE_BASE_DIR="${SAVE_BASE_DIR:-${BASE_ROOT}/checkpoints/synergy/esnli_smoke}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
N_IMAGES="${N_IMAGES:-8}"
LEGACY_MAX_SAMPLES="${LEGACY_MAX_SAMPLES:-64}"
BUILD_BATCH="${BUILD_BATCH:-2}"
TRAIN_BS="${TRAIN_BS:-4}"
TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-2}"

TIER1_DEFAULT_CFG="./configs/ESNLI/default_config_esnli_tier1.json"
RUNTIME_CFG="$(mktemp /tmp/esnli_tier1_smoke_cfg.XXXXXX.json)"
trap 'rm -f "${RUNTIME_CFG}"' EXIT

hr() {
  echo
  echo "================================================================"
  echo "  $*"
  echo "================================================================"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

need_cmd python
need_cmd bash

case "${MODE}" in
  all|cache|train) ;;
  *) die "MODE must be one of: all|cache|train (got: ${MODE})" ;;
esac

case "${CACHE_METHOD}" in
  v2|legacy|both) ;;
  *) die "CACHE_METHOD must be one of: v2|legacy|both (got: ${CACHE_METHOD})" ;;
esac

case "${MODEL_SET}" in
  basic|extended) ;;
  *) die "MODEL_SET must be one of: basic|extended (got: ${MODEL_SET})" ;;
esac

if [ ! -d "${DATA_ROOT}" ]; then
  die "DATA_ROOT does not exist: ${DATA_ROOT}"
fi

FLICKR_IMAGES_DIR="${FLICKR_IMAGES_DIR:-}"
if [ -z "${FLICKR_IMAGES_DIR}" ]; then
  if [ -d "${DATA_ROOT}/flickr30k-images/flickr30k-images" ]; then
    FLICKR_IMAGES_DIR="${DATA_ROOT}/flickr30k-images/flickr30k-images"
  elif [ -d "${DATA_ROOT}/flickr30k-images" ]; then
    FLICKR_IMAGES_DIR="${DATA_ROOT}/flickr30k-images"
  else
    die "Could not find Flickr30k directory under ${DATA_ROOT}. Expected one of:
  - ${DATA_ROOT}/flickr30k-images
  - ${DATA_ROOT}/flickr30k-images/flickr30k-images"
  fi
fi

mkdir -p "${CACHE_V2_DIR}" "${CACHE_V1_DIR}" "${SAVE_BASE_DIR}"

hr "Tier1 Smoke Config"
echo "MODE=${MODE}"
echo "CACHE_METHOD=${CACHE_METHOD}"
echo "MODEL_SET=${MODEL_SET}"
echo "GPU=${GPU}"
echo "PROJECT_ID=${PROJECT_ID}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "CACHE_V2_DIR=${CACHE_V2_DIR}"
echo "CACHE_V1_DIR=${CACHE_V1_DIR}"
echo "SAVE_BASE_DIR=${SAVE_BASE_DIR}"
echo "FLICKR_IMAGES_DIR=${FLICKR_IMAGES_DIR}"
echo "MODEL_NAME=${MODEL_NAME}"

if [ "${MODE}" = "all" ] || [ "${MODE}" = "cache" ]; then
  if [ "${CACHE_METHOD}" = "v2" ] || [ "${CACHE_METHOD}" = "both" ]; then
    hr "STEP 1A  Build v2 smoke cache (train/validation/test)"
    for SPLIT in train validation test; do
      echo "[v2] Building split=${SPLIT} (max_images=${N_IMAGES})"
      CUDA_VISIBLE_DEVICES="${GPU}" python mydatasets/ESNLI/ESNLI_CodeBook_v2.py \
        --data_root "${DATA_ROOT}" \
        --out_dir "${CACHE_V2_DIR}" \
        --model_name "${MODEL_NAME}" \
        --split "${SPLIT}" \
        --batch_size "${BUILD_BATCH}" \
        --num_workers 0 \
        --shard_size 128 \
        --max_images "${N_IMAGES}" \
        --device "cuda:0" \
        --dtype fp16
    done

    hr "STEP 1B  Verify v2 cache manifests"
    for SPLIT in train validation test; do
      MANIFEST="${CACHE_V2_DIR}/${SPLIT}/manifest.jsonl"
      [ -f "${MANIFEST}" ] || die "[v2] Missing manifest: ${MANIFEST}"
      N_LINES=$(wc -l < "${MANIFEST}")
      [ "${N_LINES}" -gt 0 ] || die "[v2] Empty manifest: ${MANIFEST}"
      echo "[v2] ${SPLIT}: ${N_LINES} shard entries"
    done
  fi

  if [ "${CACHE_METHOD}" = "legacy" ] || [ "${CACHE_METHOD}" = "both" ]; then
    hr "STEP 2A  Build legacy v1 smoke cache (validation only)"
    CUDA_VISIBLE_DEVICES="${GPU}" python mydatasets/ESNLI/ESNLI_CodeBook.py \
      --data_root "${DATA_ROOT}" \
      --flickr_images_dir "${FLICKR_IMAGES_DIR}" \
      --model_name "${MODEL_NAME}" \
      --output_dir "${CACHE_V1_DIR}" \
      --split validation \
      --max_samples "${LEGACY_MAX_SAMPLES}" \
      --batch_size "${BUILD_BATCH}" \
      --num_workers 0 \
      --shard_size 128 \
      --device "cuda:0" \
      --dtype float16 \
      --verify_every_flush 1 \
      --verify_n_show 1

    hr "STEP 2B  Verify legacy v1 cache manifest"
    LEGACY_MANIFEST="${CACHE_V1_DIR}/validation/manifest.jsonl"
    [ -f "${LEGACY_MANIFEST}" ] || die "[legacy] Missing manifest: ${LEGACY_MANIFEST}"
    LEGACY_LINES=$(wc -l < "${LEGACY_MANIFEST}")
    [ "${LEGACY_LINES}" -gt 0 ] || die "[legacy] Empty manifest: ${LEGACY_MANIFEST}"
    echo "[legacy] validation: ${LEGACY_LINES} shard entries"
  fi
fi

if [ "${MODE}" = "all" ] || [ "${MODE}" = "train" ]; then
  [ -f "${CACHE_V2_DIR}/train/manifest.jsonl" ] || die "v2 cache not found. Run MODE=cache with CACHE_METHOD=v2|both first."

  hr "STEP 3  Build runtime training config for smoke"
  python - "${TIER1_DEFAULT_CFG}" "${RUNTIME_CFG}" "${DATA_ROOT}" "${CACHE_V2_DIR}" "${SAVE_BASE_DIR}" "${TRAIN_BS}" "${TRAIN_MAX_EPOCH}" <<'PYEOF'
import json
import sys

template, out_cfg, data_root, cache_root, save_dir, train_bs, max_epoch = sys.argv[1:]
train_bs = int(train_bs)
max_epoch = int(max_epoch)

with open(template, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["exp_name"] = "ESNLI Tier1 Smoke"
cfg["dataset"]["data_roots"] = data_root
cfg["dataset"]["cache_root"] = cache_root
cfg["model"]["save_base_dir"] = save_dir

cfg["training_params"]["batch_size"] = train_bs
cfg["training_params"]["test_batch_size"] = train_bs
cfg["training_params"]["data_loader_workers"] = 0
cfg["training_params"]["async_loading"] = False
cfg["training_params"]["wandb_disable"] = True
cfg["training_params"]["tdqm_disable"] = True
cfg["training_params"]["res"] = True

cfg["early_stopping"]["max_epoch"] = max_epoch
cfg["early_stopping"]["end_of_epoch_check"] = True
cfg["early_stopping"]["validate_every"] = 5
cfg["early_stopping"]["log_interval"] = 5
cfg["early_stopping"]["save_every_step"] = 1000000
cfg["early_stopping"]["save_every_valstep"] = 1000000

with open(out_cfg, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PYEOF
  echo "Runtime config: ${RUNTIME_CFG}"

  hr "STEP 4  Train smoke model set (${MODEL_SET})"
  MODEL_CONFIGS=(
    "./configs/ESNLI/full_lora.json"
    "./configs/ESNLI/full_image_lora.json"
    "./configs/ESNLI/full_text_lora.json"
    "./configs/ESNLI/full_image_frozen.json"
  )
  if [ "${MODEL_SET}" = "extended" ]; then
    MODEL_CONFIGS+=(
      "./configs/ESNLI/full_mcr.json"
      "./configs/ESNLI/full_mmpareto.json"
      "./configs/ESNLI/full_dnr.json"
      "./configs/ESNLI/full_reconboost.json"
      "./configs/ESNLI/full_rmask.json"
    )
  fi

  for CFG in "${MODEL_CONFIGS[@]}"; do
    [ -f "${CFG}" ] || die "Missing model config: ${CFG}"
    echo "[train] ${CFG}"
    CUDA_VISIBLE_DEVICES="${GPU}" python scripts/entrypoints/train.py \
      --config "${CFG}" \
      --default_config "${RUNTIME_CFG}" \
      --fold 0 \
      --lr 0.0001 \
      --wd 0.0001 \
      --batch_size "${TRAIN_BS}" \
      --start_over \
      --tdqm_disable
  done

  hr "STEP 5  Smoke summary"
  python - "${SAVE_BASE_DIR}" <<'PYEOF'
import glob
import os
import sys
import torch

save_base = sys.argv[1]
ckpts = sorted(glob.glob(os.path.join(save_base, "*.pth.tar")), key=os.path.getmtime)
print(f"Checkpoint dir: {save_base}")
print(f"Checkpoint count: {len(ckpts)}")
if not ckpts:
    raise SystemExit("No checkpoints found. Training smoke failed.")

for p in ckpts[-10:]:
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        logs = ckpt.get("logs", {})
        best = logs.get("best_logs", {}).get("best_vaccuracy", {})
        acc = best.get("acc", {}).get("combined", None)
        print(f"  OK  {os.path.basename(p)}  best_val_acc={acc}")
    except Exception as e:
        print(f"  WARN {os.path.basename(p)}  unreadable logs ({e})")
PYEOF
fi

hr "TIER1 SMOKE COMPLETE"
echo "Mode: ${MODE}"
echo "Cache method: ${CACHE_METHOD}"
echo "Model set: ${MODEL_SET}"
