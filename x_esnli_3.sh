#srun --gres=gpu:1 --time=60:00:00 -p pi_ppliang --nodelist=node2500 -c 15 --mem=80G --pty bash
##conda init
##conda activate synergy_new
#
#cd /scratch/kkontras/Synergy/mydatasets/ESNLI
#rm -rfv /scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis/*/_mm
#CUDA_VISIBLE_DEVICES=5 python ESNLI_CodeBook.py --split train --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 32
#CUDA_VISIBLE_DEVICES=6 python ESNLI_CodeBook.py --split validation --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 32
#CUDA_VISIBLE_DEVICES=7 python ESNLI_CodeBook.py --split test --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 32
#cd /scratch/kkontras/Synergy
#CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 64
#CUDA_VISIBLE_DEVICES=7 python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.001 --wd 0.01 --batch_size 64
#CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/cache_image_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 64
#CUDA_VISIBLE_DEVICES=6 python train.py  --config ./configs/ESNLI/cache_text_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 64
#
#
#CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0 --batch_size 256
#
#CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/synprom_lora.json  --default_config ./configs/ESNLI/default_config_esnli_mib.json --fold 0 --lr 0.0001 --wd 0.0 --batch_size 4
#CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/synprom_lora.json  --default_config ./configs/ESNLI/default_config_esnli_syn.json --fold 0 --lr 0.0001 --wd 0.0 --batch_size 4
#
#CUDA_VISIBLE_DEVICES=0 python ESNLI_CodeBook.py --split validation --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 256
#CUDA_VISIBLE_DEVICES=2 python ESNLI_CodeBook.py --split test --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 256

CUDA_VISIBLE_DEVICES=1 python mydatasets/ESNLI/ESNLI_CodeBook.py --split train --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 1
CUDA_VISIBLE_DEVICES=0 python show.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=1 python show.py  --config ./configs/ESNLI/cache_image_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=4 python train.py  --config ./configs/ESNLI/cache_text_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 32

CUDA_VISIBLE_DEVICES=1 python train.py  --config ./configs/ESNLI/cache_zero_image.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=4 python train.py  --config ./configs/ESNLI/cache_zero_text.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8


CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --l 1.0 --perturb rand --perturb_pmin 0.5
CUDA_VISIBLE_DEVICES=2 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --l 0.1 --perturb rand --perturb_pmin 0.5

CUDA_VISIBLE_DEVICES=2 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --l 0.1 --perturb learned --perturb_lsparse 1.0


CUDA_VISIBLE_DEVICES=1 python train.py  --config ./configs/ESNLI/cache_ens.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8



CUDA_VISIBLE_DEVICES=1 python train.py  --config ./configs/ESNLI/cache_text_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 32



python mydatasets/ESNLI/ESNLI_CodeBook.py --split train --data_root "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI" --flickr_images_dir "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --out_dir "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 1

CUDA_VISIBLE_DEVICES=1 python mydatasets/ESNLI/ESNLI_CodeBook_v2.py --split validation --data_root "/scratch/kkontras/ESNLI" --model_name "Qwen/Qwen3-VL-2B-Instruct" --out_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 128
CUDA_VISIBLE_DEVICES=5 python mydatasets/ESNLI/ESNLI_CodeBook_v2.py --split test --data_root "/scratch/kkontras/ESNLI" --model_name "Qwen/Qwen3-VL-2B-Instruct" --out_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 16

ulimit -v 100000000