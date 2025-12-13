export HF_HOME=/scratch/kkontras/data/huggingface
export HF_HUB_CACHE=/scratch/kkontras/data/huggingface/hub
export TRANSFORMERS_CACHE=/scratch/kkontras/data/huggingface/models

git pull
conda init
conda activate /scratch/kkontras/miniconda3/envs/synergy_new
CUDA_VISIBLE_DEVICES=7 python train.py --config ./configs/ScienceQA/synprom_ib_gen.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.0001 --wd 0.0001 --start_over
