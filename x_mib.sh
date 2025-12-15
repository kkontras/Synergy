export HF_HOME=/scratch/kkontras/data/huggingface
export HF_HUB_CACHE=/scratch/kkontras/data/huggingface/hub

git pull
conda init
conda activate /scratch/kkontras/miniconda3/envs/synergy_new
#CUDA_VISIBLE_DEVICES=6 python train.py --config ./configs/ScienceQA/synprom_ib_gen.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --start_over

for l in 0.01 0.1 0.5 1.0 10; do
  for fill in mean zero; do
    for contrcoeff in 0.0 0.1 1.0; do
      CUDA_VISIBLE_DEVICES=6 python train.py --config ./configs/CREMA_D/synergy/dec/synprom_IB_mask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l $l --lr 0.00001 --wd 0.0001 --cls mlp --perturb mask --perturb_fill $fill --contrcoeff $contrcoeff --num_samples 32
    done
  done
done
