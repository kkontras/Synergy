export HF_HOME=/scratch/kkontras/data/huggingface
export HF_HUB_CACHE=/scratch/kkontras/data/huggingface/hub

git pull
conda init
conda activate /scratch/kkontras/miniconda3/envs/synergy_new
#CUDA_VISIBLE_DEVICES=6 python scripts/entrypoints/train.py --config ./configs/ScienceQA/synprom_ib_gen.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --start_over

#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs

gpus=(0 1 3)   # <- edit this list if you want different GPU ids
i=0

for l in 1.0 0.1; do
  for lsparse in 0.1 0.5 1.0 5.0; do
    (( i < ${#gpus[@]} )) || { echo "Not enough GPUs in gpus=() for all jobs" >&2; exit 1; }
    gpu="${gpus[$i]}"

    s="sqa_g${gpu}_l${l//./p}_ls${lsparse//./p}"
    tmux new -d -s "$s" "bash -lc 'cd \"$(pwd)\" && CUDA_VISIBLE_DEVICES=$gpu python scripts/entrypoints/train.py \
      --config ./configs/ScienceQA/cache_synib_lora.json \
      --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json \
      --fold 0 --lr 0.0001 --wd 0.01 --batch_size 6 \
      --l $l --perturb learned --perturb_lsparse $lsparse \
      |& tee logs/$s.log'"

    echo "launched $s (GPU $gpu)"
    i=$((i+1))
  done
done

echo "tmux ls ; tmux attach -t <session>"
