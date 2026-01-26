
python train.py --config ./configs/ScienceQA/synprom_ib_gen.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --start_over
python train.py --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.0001 --wd 0.01
python train.py --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.001 --wd 0.001
python train.py --config ./configs/ScienceQA/synprom_lora_synib.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.0001 --wd 0.01 --l 1
python train.py --config ./configs/ScienceQA/synprom_lora_synib.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.0001 --wd 0.01 --l 1


CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=1 python train.py --config ./configs/ScienceQA/image_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=3 python train.py --config ./configs/ScienceQA/text_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=3 python train.py --config ./configs/ScienceQA/synprom_lora_synibfaster.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.0001 --wd 0.000001 --batch_size 2 --perturb rand

CUDA_VISIBLE_DEVICES=3 python train.py --config ./configs/ScienceQA/synprom_lora_synibfaster.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8 --perturb learned --perturb_lsparse 5


CUDA_VISIBLE_DEVICES=6 python train.py --config ./configs/ScienceQA/synprom_lora_synib.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.0001 --wd 0.01 --l 1

CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=2 python train.py --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.001 --wd 0.01 --batch_size 8
CUDA_VISIBLE_DEVICES=3 python --mixed_precision bf16 train.py --config ./configs/ScienceQA/synprom_lora_synibfaster.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.001 --wd 0.001 --l 0.1 --batch_size 8
CUDA_VISIBLE_DEVICES=3 python train.py --config ./configs/ScienceQA/synprom_lora_synibfaster.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.001 --wd 0.001  --l 1 --batch_size 8


python show.py  --config ./configs/ScienceQA/synprom_lora_synibfaster.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.0001 --wd 0.01   --l 1
python show.py  --config ./configs/ScienceQA/synprom_lora_synib.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.0001 --wd 0.01   --l 1
python show.py  --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.0001 --wd 0.01

python train.py  --config ./configs/ScienceQA/cache_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 8
python train.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 4
python train.py  --config ./configs/ScienceQA/cache_image_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 4
python train.py  --config ./configs/ScienceQA/cache_text_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 4

scp -rvf /esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA/cache_tokens2B kkontras@mib.media.mit.edu:/scratch/kkontras/ScienceQA/cache_tokens2B

CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
    --mixed_precision bf16 \
    train.py \
    --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.001 --wd 0.001

CUDA_VISIBLE_DEVICES=6,7 accelerate launch train.py --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.001 --wd 0.001 --batch_size 4
CUDA_VISIBLE_DEVICES=4,5 accelerate launch train.py --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.001 --wd 0.001 --batch_size 12

CUDA_VISIBLE_DEVICES=4 python train.py --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.001 --wd 0.001 --batch_size 12

accelerate launch train.py --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.001 --wd 0.001 --batch_size 3
accelerate launch train.py --config ./configs/ScienceQA/synprom_lora_synibfaster.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.001 --wd 0.001 --batch_size 3


python mydatasets/ScienceQA/ScienceQA_Codebook.py --data_root "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA" --out_dir "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA/cache_tokens2B" --model_name "Qwen/Qwen3-VL-2B-Instruct" --split train --batch_size 64  --num_workers 24 --cache_image_embeds



CUDA_VISIBLE_DEVICES=0 python mydatasets/ScienceQA/ScienceQA_Codebook_v2.py --split train --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_qwen3_vl_2b_nocls_vis" --num_workers 16 --batch_size 64
CUDA_VISIBLE_DEVICES=1 python mydatasets/ScienceQA/ScienceQA_Codebook_v2.py --split test --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_qwen3_vl_2b_nocls_vis" --num_workers 16 --batch_size 64
CUDA_VISIBLE_DEVICES=2 python mydatasets/ScienceQA/ScienceQA_Codebook_v2.py --split validation --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_qwen3_vl_2b_nocls_vis" --num_workers 16 --batch_size 64

CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ScienceQA/cache_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 32 --start_over
CUDA_VISIBLE_DEVICES=3 python train.py  --config ./configs/ScienceQA/cache_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 16 --start_over
CUDA_VISIBLE_DEVICES=2 python train.py  --config ./configs/ScienceQA/cache_text_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 32 --start_over

CUDA_VISIBLE_DEVICES=0 python train.py  --config ./configs/ScienceQA/cache_image_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_mib.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 16



python mydatasets/ScienceQA/ScienceQA_Codebook.py --data_root "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA" --out_dir "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA/cache_tokens2B" --model_name "Qwen/Qwen3-VL-2B-Instruct" --split train --batch_size 64  --num_workers 24 --cache_image_embeds



#check tier2 results
srun --partition=interactive --cluster=wice --time=08:00:00 --nodes=1 --cpus-per-task=8 --mem=40G -A lp_biomed_mdv --gpus-per-node=1 --pty bash -l

python show.py  --config ./configs/ScienceQA/cache_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
python show.py  --config ./configs/ScienceQA/cache_ens.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
python show.py  --config ./configs/ScienceQA/cache_image_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5
python show.py  --config ./configs/ScienceQA/cache_text_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5

python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb rand --perturb_pmin 0.3
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb rand --perturb_pmin 0.3
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb rand --perturb_pmin 0.5
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb rand --perturb_pmin 0.5
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb rand --perturb_pmin 0.7
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb rand --perturb_pmin 0.7

python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb learned --perturb_lsparse 0.1
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb learned --perturb_lsparse 0.1
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb learned --perturb_lsparse 0.5
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb learned --perturb_lsparse 0.5
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb learned --perturb_lsparse 1
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb learned --perturb_lsparse 1
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --perturb learned --perturb_lsparse 5
python show.py  --config ./configs/ScienceQA/cache_synib_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1.0 --perturb learned --perturb_lsparse 5


python show.py  --config ./configs/ScienceQA/cache_lora_MMPareto.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 0.5
python show.py  --config ./configs/ScienceQA/cache_lora_MMPareto.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 1.0
python show.py  --config ./configs/ScienceQA/cache_lora_MMPareto.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 1.5
python show.py  --config ./configs/ScienceQA/cache_lora_MMPareto.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 3.0
python show.py  --config ./configs/ScienceQA/cache_lora_MMPareto.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 5.0

python show.py  --config ./configs/ScienceQA/cache_lora_DnR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 1 --kmepoch 1
python show.py  --config ./configs/ScienceQA/cache_lora_DnR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 1 --kmepoch 3
python show.py  --config ./configs/ScienceQA/cache_lora_DnR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 1 --kmepoch 5
python show.py  --config ./configs/ScienceQA/cache_lora_DnR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 3 --kmepoch 1
python show.py  --config ./configs/ScienceQA/cache_lora_DnR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 3 --kmepoch 3
python show.py  --config ./configs/ScienceQA/cache_lora_DnR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 3 --kmepoch 5
python show.py  --config ./configs/ScienceQA/cache_lora_DnR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 5 --kmepoch 1
python show.py  --config ./configs/ScienceQA/cache_lora_DnR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 5 --kmepoch 3
python show.py  --config ./configs/ScienceQA/cache_lora_DnR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --alpha 5 --kmepoch 5

python show.py  --config ./configs/ScienceQA/cache_lora_MCR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1 --multil 1 --num_samples 5 --regby greedy --shuffle_type rand --contrcoeff 1
python show.py  --config ./configs/ScienceQA/cache_lora_MCR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --multil 1 --num_samples 5 --regby greedy --shuffle_type rand --contrcoeff 1
python show.py  --config ./configs/ScienceQA/cache_lora_MCR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.01 --multil 1 --num_samples 5 --regby greedy --shuffle_type rand --contrcoeff 1
python show.py  --config ./configs/ScienceQA/cache_lora_MCR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 1 --multil 0.1 --num_samples 5 --regby greedy --shuffle_type rand --contrcoeff 1
python show.py  --config ./configs/ScienceQA/cache_lora_MCR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.1 --multil 0.1 --num_samples 5 --regby greedy --shuffle_type rand --contrcoeff 1
python show.py  --config ./configs/ScienceQA/cache_lora_MCR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5 --l 0.01 --multil 0.1 --num_samples 5 --regby greedy --shuffle_type rand --contrcoeff 1




cp /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/Cache_LoRaSynIBFaster_fold0_l1.0_perturblearned_pmin./configs/ScienceQA/cache_synib_lora.json0_lsparse0.1_lr0.0001_wd0.01_bs5.pth.tar /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/Cache_LoRaSynIBFaster_fold0_l1.0_perturblearned_lsparse0.1_lr0.0001_wd0.01_bs5.pth.tar


rsync -ahvz --progress  /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/Cache_LoRaSynIBFaster_fold0_l0.1_perturblearned_lsparse0.1_lr0.0001_wd0.01_bs5.pth.tar kkontras@ssh.esat.kuleuven.be:/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ScienceQA/
rsync -ahvz --progress  /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/Cache_LoRaSynIBFaster_fold0_l1.0_perturblearned_lsparse0.1_lr0.0001_wd0.01_bs5.pth.tar kkontras@ssh.esat.kuleuven.be:/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ScienceQA/
rsync -ahvz --progress  /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/SynIBCache_Image_LoRa_fold0_lr0.0001_wd0.01_bs5.pth.tar kkontras@ssh.esat.kuleuven.be:/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ScienceQA/
rsync -ahvz --progress  /scratch/leuven/350/vsc35057/models/Synergy/ScienceQA/SynIBCache_LoRa_fold0_lr0.0001_wd0.01_bs5.pth.tar kkontras@ssh.esat.kuleuven.be:/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ScienceQA/




python train.py  --config ./configs/ScienceQA/cache_lora_MMPareto.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 2 --alpha 5.0
python train.py  --config ./configs/ScienceQA/cache_lora_DnR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 2 --alpha 1 --kmepoch 1
python train.py  --config ./configs/ScienceQA/cache_lora_MCR.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 2 --l 1 --multil 1 --num_samples 2 --regby greedy --shuffle_type rand --contrcoeff 1

--config ./configs/ScienceQA/cache_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_cache.json --fold 0 --lr 0.0001 --wd 0.01 --batch_size 5


