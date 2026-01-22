
getgpu 1 80GB 60:00:00
conda activate synergy_new
cd /home/kkontras/orcd/scratch/Synergy
python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_orcd.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
scancel $SLURM_JOB_ID

