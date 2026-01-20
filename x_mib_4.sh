export HF_HOME=/scratch/kkontras/data/huggingface
export HF_HUB_CACHE=/scratch/kkontras/data/huggingface/hub

git pull
conda init
conda activate /scratch/kkontras/miniconda3/envs/synergy_new
#CUDA_VISIBLE_DEVICES=6 python train.py --config ./configs/ScienceQA/synprom_ib_gen.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --start_over

