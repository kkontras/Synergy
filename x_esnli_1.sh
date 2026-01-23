#srun --gres=gpu:1 --time=60:00:00 -p pi_ppliang --nodelist=node2500 -c 15 --mem=65G --pty bash
conda init
conda activate synergy_new
cd /home/kkontras/orcd/scratch/Synergy/mydatasets/ESNLI

python ESNLI_CodeBook.py --split train --data_root "/home/kkontras/orcd/scratch/data/ESNLI" --flickr_images_dir "/home/kkontras/orcd/scratch/data/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/home/kkontras/orcd/scratch/data/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 32
python ESNLI_CodeBook.py --split test --data_root "/home/kkontras/orcd/scratch/data/ESNLI" --flickr_images_dir "/home/kkontras/orcd/scratch/data/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/home/kkontras/orcd/scratch/data/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 32
python ESNLI_CodeBook.py --split validation --data_root "/home/kkontras/orcd/scratch/data/ESNLI" --flickr_images_dir "/home/kkontras/orcd/scratch/data/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/home/kkontras/orcd/scratch/data/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 32
cd /home/kkontras/orcd/scratch/Synergy/
python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_orcd.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
scancel $SLURM_JOB_ID

python train.py  --config ./configs/ESNLI/synprom_lora.json  --default_config ./configs/ESNLI/default_config_esnli_syn.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5



python train.py  --config ./configs/ESNLI/synprom_lora.json  --default_config ./configs/ESNLI/default_config_esnli_syn.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5




python ESNLI_CodeBook.py --split train --data_root "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI" --flickr_images_dir "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 16
