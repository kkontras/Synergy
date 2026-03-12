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

DATA_ROOT="/scratch/kkontras/ESNLI"
CACHE_OUT="${DATA_ROOT}/cache_v2_full"
MODEL_NAME="${model_name:-Qwen/Qwen3-VL-2B-Instruct}"
SHARD_SIZE="${shard_size:-4096}"
BUILD_BATCH="${build_batch:-256}"
HF_CACHE_ROOT="${DATA_ROOT}/hf_cache"

mkdir -p "${CACHE_OUT}" "${HF_CACHE_ROOT}" ./condor_logs/logs_vsc
export HF_HOME="${HF_CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
export TORCH_HOME="${HF_CACHE_ROOT}/torch"
echo "HF_HOME=${HF_HOME}"

echo "Preflight: ensuring ESNLI assets under ${DATA_ROOT}"
# bash ./condor_logs/tier1/download_esnli_data.sh "${DATA_ROOT}"

for SPLIT in train validation test; do
    SPLIT_DIR="${CACHE_OUT}/${SPLIT}"
    if [ -f "${SPLIT_DIR}/manifest.jsonl" ]; then
        echo "[skip] ${SPLIT}: manifest exists"
        continue
    fi

    echo "Building ${SPLIT} ..."
    python mydatasets/ESNLI/ESNLI_CodeBook_v3.py \
        --split      "${SPLIT}"      \
        --data_root  "${DATA_ROOT}"  \
        --out_dir    "${CACHE_OUT}"  \
        --model_name "${MODEL_NAME}" \
        --shard_size "${SHARD_SIZE}" \
        --batch_size "${BUILD_BATCH}" \
        --device     "cuda:0"        \
        --dtype      fp16
    echo "Done: ${SPLIT}"
done

echo "All splits done. Cache at ${CACHE_OUT}"



CUDA_VISIBLE_DEVICES=1 python -u scripts/entrypoints/train.py --config ./configs/ESNLI/full_lora.json --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 
CUDA_VISIBLE_DEVICES=2 python -u scripts/entrypoints/train.py --config ./configs/ESNLI/full_image_lora.json --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 
CUDA_VISIBLE_DEVICES=6 python -u scripts/entrypoints/train.py --config ./configs/ESNLI/full_text_lora.json --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 
CUDA_VISIBLE_DEVICES=7 python -u scripts/entrypoints/train.py --config ./configs/ESNLI/full_image_frozen.json --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 
CUDA_VISIBLE_DEVICES=0 python -u scripts/entrypoints/train.py --config ./configs/ESNLI/full_mcr.json --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 
CUDA_VISIBLE_DEVICES=0 python -u scripts/entrypoints/train.py --config ./configs/ESNLI/full_mmpareto.json --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 
CUDA_VISIBLE_DEVICES=0 python -u scripts/entrypoints/train.py --config ./configs/ESNLI/full_dnr.json --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 
CUDA_VISIBLE_DEVICES=0 python -u scripts/entrypoints/train.py --config ./configs/ESNLI/full_reconboost.json --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 
CUDA_VISIBLE_DEVICES=0 python -u scripts/entrypoints/train.py --config ./configs/ESNLI/full_synib.json --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 8 
