
CUDA_VISIBLE_DEVICES=5 python scripts/entrypoints/show.py  --config ./configs/ScienceQA/cache_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
CUDA_VISIBLE_DEVICES=5 python scripts/entrypoints/show.py  --config ./configs/ScienceQA/cache_image_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
CUDA_VISIBLE_DEVICES=5 python scripts/entrypoints/show.py  --config ./configs/ScienceQA/cache_text_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5

python scripts/entrypoints/show.py  --config ./configs/ScienceQA/cache_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
python scripts/entrypoints/show.py  --config ./configs/ScienceQA/cache_image_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
python scripts/entrypoints/show.py  --config ./configs/ScienceQA/cache_text_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5


cp /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/SynIBCache_LoRa_fold0_pmin./configs/ScienceQA/cache_lora.json0_lr0.0001_wd0.01_bs5.pth.tar /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/SynIBCache_LoRa_fold0_lr0.0001_wd0.01_bs5.pth.tar
cp /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/SynIBCache_Image_LoRa_fold0_pmin./configs/ScienceQA/cache_image_lora.json0_lr0.0001_wd0.01_bs5.pth.tar  /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/SynIBCache_Image_LoRa_fold0_lr0.0001_wd0.01_bs5.pth.tar
cp /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/Uni_Text_LoRa_fold0_pmin./configs/ScienceQA/cache_text_lora.json0_lr0.0001_wd0.01_bs5.pth.tar /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/Uni_Text_LoRa_fold0_lr0.0001_wd0.01_bs5.pth.tar


for l in 0.1 1.0; do

  for lsparse in 0.1 0.5 1.0 5.0; do
    CUDA_VISIBLE_DEVICES=0 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb learned --perturb_lsparse $lsparse
  done
#  CUDA_VISIBLE_DEVICES=0 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.3
#  CUDA_VISIBLE_DEVICES=0 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.5
#  CUDA_VISIBLE_DEVICES=0 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.7
done

    CUDA_VISIBLE_DEVICES=7 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1 --perturb learned --perturb_lsparse 1


CUDA_VISIBLE_DEVICES=0 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb learned --perturb_lsparse 1.0
CUDA_VISIBLE_DEVICES=1 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb learned --perturb_lsparse 5.0
CUDA_VISIBLE_DEVICES=2 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb learned --perturb_lsparse 5.0
CUDA_VISIBLE_DEVICES=3 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb learned --perturb_lsparse 5.0

CUDA_VISIBLE_DEVICES=5 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_ens.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5

CUDA_VISIBLE_DEVICES=0 python main_mcrema_postpred.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb learned --perturb_lsparse 5


for l in 0.1 1.0; do

  for lsparse in 0.1 0.5 1.0 5.0; do
    CUDA_VISIBLE_DEVICES=0 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb learned --perturb_lsparse $lsparse
  done
#  CUDA_VISIBLE_DEVICES=0 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.3
#  CUDA_VISIBLE_DEVICES=0 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.5
#  CUDA_VISIBLE_DEVICES=0 python scripts/entrypoints/train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l $l --perturb rand --perturb_pmin 0.7
done
