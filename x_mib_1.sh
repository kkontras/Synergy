
for l in 0.1 1.0; do
  for lsparse in 0.1 0.5 1.0 5.0 do;
    CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --l $l --perturb learned --perturb_lsparse $lsparse
  done
#  CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --l $l --perturb rand --perturb_pmin 0.3
#  CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --l $l --perturb rand --perturb_pmin 0.5
#  CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --l $l --perturb rand --perturb_pmin 0.7
done


