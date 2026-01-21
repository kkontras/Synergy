
python train.py  --config ./configs/ESNLI/cache_lora_emb_clshead.json  --default_config ./configs/ESNLI/default_config_esnli_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 2
python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 2


CUDA_VISIBLE_DEVICES=6 python train.py  --config ./configs/ESNLI/cache_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
CUDA_VISIBLE_DEVICES=5 python train.py  --config ./configs/ESNLI/cache_image_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
CUDA_VISIBLE_DEVICES=4 python train.py  --config ./configs/ESNLI/cache_text_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5

for l in 0.1 1.0; do

  for lsparse in 0.1 0.5 1.0 5.0; do
    CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb learned --perturb_lsparse $lsparse
  done
#  CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.3
#  CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.5
#  CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.7
done

    CUDA_VISIBLE_DEVICES=7 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1 --perturb learned --perturb_lsparse 1


CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb learned --perturb_lsparse 1.0
CUDA_VISIBLE_DEVICES=1 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb learned --perturb_lsparse 5.0
CUDA_VISIBLE_DEVICES=2 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb learned --perturb_lsparse 5.0
CUDA_VISIBLE_DEVICES=3 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb learned --perturb_lsparse 5.0

CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_ens.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5

CUDA_VISIBLE_DEVICES=0 python main_mcrema_postpred.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb learned --perturb_lsparse 5


for l in 0.1 1.0; do

  for lsparse in 0.1 0.5 1.0 5.0; do
    CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb learned --perturb_lsparse $lsparse
  done
#  CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.3
#  CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.5
#  CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ESNLI/cache_synib_lora.json  --default_config ./configs/ESNLI/default_config_esnli_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.7
done
