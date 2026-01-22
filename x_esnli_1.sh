srun --gres=gpu:1 --time=60:00:00 -p pi_ppliang --nodelist=node2500 -c 15 --mem=65G --pty bash
conda init
conda activate synergy_new
cd /home/kkontras/orcd/scratch/Synergy
python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_orcd.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
scancel $SLURM_JOB_ID

