



CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/cache_image_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
CUDA_VISIBLE_DEVICES=4 python train.py  --config ./configs/ESNLI/cache_text_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5

