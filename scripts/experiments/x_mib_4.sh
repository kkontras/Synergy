
CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/show.py  --config ./configs/ScienceQA/cache_image_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/show.py  --config ./configs/ScienceQA/cache_text_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/show.py  --config ./configs/ScienceQA/cache_ens.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/show.py  --config ./configs/ScienceQA/cache_ens.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_zero.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8




CUDA_VISIBLE_DEVICES=3 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --start_over --l 0.1 --perturb rand --perturb_pmin 0.5
CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --start_over --l 0.01 --perturb rand --perturb_pmin 0.5

CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_ens.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --start_over
CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_zero.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8

CUDA_VISIBLE_DEVICES=2 python get_ceu.py

CUDA_VISIBLE_DEVICES=0 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --start_over --l 0.1 --perturb rand --perturb_pmin 0.8
CUDA_VISIBLE_DEVICES=1 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --start_over --l 0.01 --perturb rand --perturb_pmin 0.8

CUDA_VISIBLE_DEVICES=3 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8  --l 1 --perturb rand --perturb_pmin 0.5
CUDA_VISIBLE_DEVICES=3 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --start_over --l 1 --perturb rand --perturb_pmin 0.8



