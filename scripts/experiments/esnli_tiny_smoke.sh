#!/usr/bin/env bash
set -euo pipefail

# Tiny ESNLI smoke test:
# 1) builds a small cached dataset
# 2) runs full, image-only, and text-only cached training quickly
#
# Usage:
#   bash scripts/experiments/esnli_tiny_smoke.sh
# Optional env vars:
#   DATA_ROOT=/scratch/$USER/ESNLI
#   OUT_DIR=/scratch/$USER/ESNLI/cache_qwen3_vl_2b_nocls_vis_tiny
#   SAVE_BASE_DIR=/scratch/$USER/checkpoints/synergy/ESNLI
#   TRAIN_SAMPLES=2000 VAL_SAMPLES=300 TEST_SAMPLES=300
#   LR=0.0001 WD=0.01 BS=8

DEFAULT_DATA_ROOT="/scratch/${USER}/ESNLI"
if [[ ! -d "${DEFAULT_DATA_ROOT}" ]]; then
  DEFAULT_DATA_ROOT="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI"
fi

DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/cache_qwen3_vl_2b_nocls_vis_tiny}"
SAVE_BASE_DIR="${SAVE_BASE_DIR:-/scratch/${USER}/checkpoints/synergy/ESNLI}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2000}"
VAL_SAMPLES="${VAL_SAMPLES:-300}"
TEST_SAMPLES="${TEST_SAMPLES:-300}"
LR="${LR:-0.0001}"
WD="${WD:-0.01}"
BS="${BS:-8}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"

DEFAULT_CONFIG_TEMPLATE="${DEFAULT_CONFIG_TEMPLATE:-./configs/ESNLI/default_config_esnli_cache_tiny.json}"
DEFAULT_CONFIG="$(mktemp /tmp/esnli_tiny_smoke_default_config.XXXXXX.json)"
trap 'rm -f "${DEFAULT_CONFIG}"' EXIT

export CUDA_VISIBLE_DEVICES=7

mkdir -p "${SAVE_BASE_DIR}"

python - "${DEFAULT_CONFIG_TEMPLATE}" "${DEFAULT_CONFIG}" "${DATA_ROOT}" "${OUT_DIR}" "${SAVE_BASE_DIR}" <<'PY'
import json
import sys

template, out_path, data_root, cache_root, save_base_dir = sys.argv[1:]
with open(template, "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg["dataset"]["data_roots"] = data_root
cfg["dataset"]["cache_root"] = cache_root
cfg["model"]["save_base_dir"] = save_base_dir
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY

echo "[ESNLI tiny] Data root: ${DATA_ROOT}"
echo "[ESNLI tiny] Cache out:  ${OUT_DIR}"
echo "[ESNLI tiny] Save base:  ${SAVE_BASE_DIR}"
echo "[ESNLI tiny] Samples: train=${TRAIN_SAMPLES}, val=${VAL_SAMPLES}, test=${TEST_SAMPLES}"
echo "[ESNLI tiny] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[ESNLI tiny] Runtime default config: ${DEFAULT_CONFIG}"

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
