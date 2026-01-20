  CUDA_VISIBLE_DEVICES=2 python train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --l $l --perturb rand --perturb_pmin 0.3
  CUDA_VISIBLE_DEVICES=2 python train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --l $l --perturb rand --perturb_pmin 0.5
  CUDA_VISIBLE_DEVICES=2 python train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --l $l --perturb rand --perturb_pmin 0.7


