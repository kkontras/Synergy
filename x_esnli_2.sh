srun --gres=gpu:1 --time=60:00:00 -p pi_ppliang --nodelist=node2500 -c 15 --mem=80G --pty bash
#conda init
#conda activate synergy_new

cd /scratch/kkontras/Synergy/mydatasets/ESNLI
rm -rfv /scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis/*/_mm
CUDA_VISIBLE_DEVICES=5 python ESNLI_CodeBook.py --split train --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 32
CUDA_VISIBLE_DEVICES=6 python ESNLI_CodeBook.py --split validation --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 32
CUDA_VISIBLE_DEVICES=7 python ESNLI_CodeBook.py --split test --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 32
cd /scratch/kkontras/Synergy
CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 64
CUDA_VISIBLE_DEVICES=7 python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.001 --wd 0.01 --batch_size 64
CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/cache_image_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 64
CUDA_VISIBLE_DEVICES=6 python train.py  --config ./configs/ESNLI/cache_text_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 64


CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.0 --batch_size 256

CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/synprom_lora.json  --default_config ./configs/ESNLI/default_config_esnli_mib.json --fold 0 --lr 0.0001 --wd 0.0 --batch_size 4
CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/synprom_lora.json  --default_config ./configs/ESNLI/default_config_esnli_syn.json --fold 0 --lr 0.0001 --wd 0.0 --batch_size 4

CUDA_VISIBLE_DEVICES=0 python ESNLI_CodeBook.py --split validation --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 1
CUDA_VISIBLE_DEVICES=1 python ESNLI_CodeBook.py --split train --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 1
CUDA_VISIBLE_DEVICES=2 python ESNLI_CodeBook.py --split test --data_root "/scratch/kkontras/ESNLI" --flickr_images_dir "/scratch/kkontras/ESNLI/flickr30k-images/" --model_name "Qwen/Qwen3-VL-2B-Instruct" --output_dir "/scratch/kkontras/ESNLI/cache_qwen3_vl_2b_nocls_vis" --batch_size 1
