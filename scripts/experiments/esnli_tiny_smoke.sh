#!/usr/bin/env bash
set -euo pipefail

# Tiny ESNLI smoke test:
# 1) builds a small cached dataset
# 2) runs full, image-only, and text-only cached training quickly
#
# Usage:
#   bash scripts/experiments/esnli_tiny_smoke.sh
# Optional env vars:
#   DATA_ROOT=/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI
#   OUT_DIR=/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_qwen3_vl_2b_nocls_vis_tiny
#   TRAIN_SAMPLES=2000 VAL_SAMPLES=300 TEST_SAMPLES=300
#   LR=0.0001 WD=0.01 BS=8

DATA_ROOT="${DATA_ROOT:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI}"
OUT_DIR="${OUT_DIR:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_qwen3_vl_2b_nocls_vis_tiny}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2000}"
VAL_SAMPLES="${VAL_SAMPLES:-300}"
TEST_SAMPLES="${TEST_SAMPLES:-300}"
LR="${LR:-0.0001}"
WD="${WD:-0.01}"
BS="${BS:-8}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"

DEFAULT_CONFIG="./configs/ESNLI/default_config_esnli_cache_tiny.json"

echo "[ESNLI tiny] Data root: ${DATA_ROOT}"
echo "[ESNLI tiny] Cache out:  ${OUT_DIR}"
echo "[ESNLI tiny] Samples: train=${TRAIN_SAMPLES}, val=${VAL_SAMPLES}, test=${TEST_SAMPLES}"

rm -rf "${OUT_DIR}"

python mydatasets/ESNLI/ESNLI_CodeBook_v2.py \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --split train \
  --batch_size "${BS}" \
  --shard_size 512 \
  --max_samples "${TRAIN_SAMPLES}"

python mydatasets/ESNLI/ESNLI_CodeBook_v2.py \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --split validation \
  --batch_size "${BS}" \
  --shard_size 512 \
  --max_samples "${VAL_SAMPLES}"

python mydatasets/ESNLI/ESNLI_CodeBook_v2.py \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --split test \
  --batch_size "${BS}" \
  --shard_size 512 \
  --max_samples "${TEST_SAMPLES}"

python scripts/entrypoints/train.py \
  --config ./configs/ESNLI/cache_lora.json \
  --default_config "${DEFAULT_CONFIG}" \
  --fold 0 --lr "${LR}" --wd "${WD}" --batch_size "${BS}" --start_over

python scripts/entrypoints/train.py \
  --config ./configs/ESNLI/cache_image_lora.json \
  --default_config "${DEFAULT_CONFIG}" \
  --fold 0 --lr "${LR}" --wd "${WD}" --batch_size "${BS}" --start_over

python scripts/entrypoints/train.py \
  --config ./configs/ESNLI/cache_text_lora.json \
  --default_config "${DEFAULT_CONFIG}" \
  --fold 0 --lr "${LR}" --wd "${WD}" --batch_size "${BS}" --start_over

echo "[ESNLI tiny] Done."
