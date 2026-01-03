#
#
#for fold in 0; do
## for fold in 0 1 2; do
#for wd in 0.001 0.0001 0.0; do
##   python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre --frozen --l 1 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
##   echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre  --l 10 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer"
##   echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre  --l 1 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer"
##   echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre  --l 0.1 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer"
##   echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre  --l 0.01 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer"
##   echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre  --l 0 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer"
#
#   python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre  --l 10 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#   python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre  --l 1 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#   python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre  --l 0.1 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#   python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre  --l 0.01 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#   python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre  --l 0 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#
#
#done
#done
#
#for l in 0.01 0.03 0.1 0.3 1 3 10 30 100; do
##  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over
#  echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over"
#  echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 1 --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over"
#  echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 2 --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over"
#done
#
# python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 0.01 --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls conformer --perturb gen --start_over
## python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre 100--l 0.1 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001
## python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 1  --pre --l 0.1 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001
## python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 2  --pre --l 0.1 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001
#
##l 0.1 #[70.6 ± 0.6, 70.5 ± 2.0, 70.1 ± 0.7]
##l 0 #[70.2 ± 1.4, 69.8 ± 0.6, 69.8 ± 1.6]
#for l in 0.001 0.005 0.01 0.05 0.1 0.5 0; do
#  python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --pre --l $l --contrcoeff  --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen
#done
##[ 0.01 0.03 0.1 0.3 1 3 10 30 100 0 ]
##[ 72.6 ± 2.9, 72.7 ± 3.0, 70.2 ± 4.9, 69.4 ± 4.9, 68.4 ± 4.4, 68.4 ± 2.8, 68.3 ± 1.3, 67.6 ± 2.7, 69.6 ± 6.0]
#
#
#[0.001 0.005 0.01 0.05 0.1 0.5 0]
#[72.5 ± 2.3, 72.3 ± 2.2, 72.3 ± 2.4, 72.2 ± 3.0, 72.0 ± 3.6, 71.2 ± 4.9, 72.5 ± 2.3]
## for fold in 2 1 0; do
##     for l in 0.001 0.005 0.01 0.05 0.1 0.5 0; do
##       python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen
##done
##done


# for fold in 2; do
#     for l in 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 0; do
#       python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gennoise
#       python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gennoise
#       python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls conformer --perturb gen
#done
#done

# for fold in 0 1 2; do
#     for l in 0.01 0.03 0.1 0.3 1 3 0; do
#      python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --frozen --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen
#done
#done

# python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib_gen.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 1 --l 1 --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen_v2 --start_over
# python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib_gen.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 2 --l 1 --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen_v2 --start_over

# python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib_gen.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen_v2
# python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib_gen.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 1 --l 1 --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen_v2
# python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib_gen.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 2 --l 1 --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen_v2
#
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_gans.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb gen --start_over
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_gans_hinge.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --perturb gen --start_over
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_gans_wgan.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb gen --start_over
#
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusion.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --perturb gen --start_over
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusion.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb gen --start_over
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusion.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.01 --wd 0.0001 --perturb gen --start_over


#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_Conformer_diffusiontf.json --default_config ./configs/CREMA_D/default_config_cremad_vit_syn.json --fold 0 --lr 0.001 --wd 0.0 --perturb gen --start_over
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusiontf.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0 --perturb gen --start_over
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusiontf_2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb gen --start_over
#
#  python examine_vae_linearprob.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusiontf.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb gen
#
#  python train.py --config ./configs/Flickr/synprom_ib_gen.json --default_config ./configs/Flickr/default_config_flickr_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb gen --start_over
#  python train.py --config ./configs/MSCOCO/synprom_ib_gen.json --default_config ./configs/MSCOCO/default_config_mscoco_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb geninput --start_over
#
#  python train.py --config ./configs/CUB200/synprom_ib_gen.json --default_config ./configs/CUB200/default_config_cub200_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb geninput --start_over
#  python train.py --config ./configs/MMIMDB/synprom_ib_gen.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb geninput --start_over
#
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusiontf.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0 --perturb gen --start_over
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusionmlp.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0 --perturb gen --start_over --cls mlp
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_detmlp.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0 --perturb gen --start_over --cls mlp
#
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generatorVAEOneSide.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0 --perturb gen --start_over --cls mlp --lib 1
#  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_generatorVAEOtherSide.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0 --perturb gen --start_over --cls mlp --lib 1



#!/bin/bash

#!/bin/bash

CONFIG_ONE="./configs/CREMA_D/synergy/nov/synprom_generatorVAEOneSide.json"
CONFIG_OTHER="./configs/CREMA_D/synergy/nov/synprom_generatorVAEOtherSide.json"
DEFAULT_CONFIG="./configs/CREMA_D/default_config_cremad_res_syn.json"

LRS=("0.001" "0.0003" "0.0001")
WDS=("0.0" "1e-5" "1e-4")
LIBS=("1.0" "0.1" "0.01")
FOLDS=("0" "1" "2")


echo "====== Generating Commands for OneSide (z2→z1) ======"
for fold in "${FOLDS[@]}"; do
for lr in "${LRS[@]}"; do
for wd in "${WDS[@]}"; do
for lib in "${LIBS[@]}"; do

    echo "python train.py --config $CONFIG_ONE --default_config $DEFAULT_CONFIG --fold $fold --lr $lr --wd $wd --perturb gen --start_over --cls mlp --lib $lib"

done
done
done
done


echo "====== Generating Commands for OtherSide (z1→z2) ======"
for fold in "${FOLDS[@]}"; do
for lr in "${LRS[@]}"; do
for wd in "${WDS[@]}"; do
for lib in "${LIBS[@]}"; do

    echo "python train.py --config $CONFIG_OTHER --default_config $DEFAULT_CONFIG --fold $fold --lr $lr --wd $wd --perturb gen --start_over --cls mlp --lib $lib"

done
done
done
done


#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/unimodal_audio_fold0_lr0.001_wd0.0001.pth.tar /esat/smcdata/users/kkontras/Rafael/trained_models
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/unimodal_video_fold0_lr0.001_wd0.0001.pth.tar /esat/smcdata/users/kkontras/Rafael/trained_models
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/Balance_Final/CREMA-D/res/MCR_fold0_regbygreedy_l0.1_multil0.01_numsamples32_contrcoeff1_lr0.0001_wd0.0001_bs32_pre.pth.tar /esat/smcdata/users/kkontras/Rafael/trained_models
#
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/unimodal_audio_fold1_lr0.001_wd0.0001.pth.tar /esat/smcdata/users/kkontras/Rafael/trained_models
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/unimodal_video_fold1_lr0.001_wd0.0001.pth.tar /esat/smcdata/users/kkontras/Rafael/trained_models
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/Balance_Final/CREMA-D/res/MCR_fold1_regbygreedy_l0.1_multil0.01_numsamples32_contrcoeff1_lr0.0001_wd0.0001_bs32_pre.pth.tar /esat/smcdata/users/kkontras/Rafael/trained_models
#
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/unimodal_audio_fold2_lr0.001_wd0.0001.pth.tar /esat/smcdata/users/kkontras/Rafael/trained_models
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/unimodal_video_fold2_lr0.001_wd0.0001.pth.tar /esat/smcdata/users/kkontras/Rafael/trained_models
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/Balance_Final/CREMA-D/res/MCR_fold2_regbygreedy_l0.1_multil0.01_numsamples32_contrcoeff1_lr0.0001_wd0.0001_bs32_pre.pth.tar /esat/smcdata/users/kkontras/Rafael/trained_models


#--config ./configs/CREMA_D/synergy/nov/synprom_generatorVAE.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0 --perturb gen --start_over --cls mlp --lib 1

python train.py --config ./configs/CREMA_D/synergy/dec/synprom_IB_Dir_SupRem.json  --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --pre --frozen --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --start_over

python train.py --config ./configs/ScienceQA/synprom_ib_gen.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --start_over
python train.py --config ./configs/ScienceQA/synprom_lora.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.0001 --wd 0.01
python train.py --config ./configs/ScienceQA/synprom_lora_synib.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn.json --fold 0 --lr 0.0001 --wd 0.01 --l 1
python train.py --config ./configs/ScienceQA/synprom_lora_synib.json  --default_config ./configs/ScienceQA/default_config_scienceqa_syn_mib.json --fold 0 --lr 0.0001 --wd 0.01 --l 1

python train.py --config ./configs/CREMA_D/synergy/dec/synprom_IB_mask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb mask

python train.py --config ./configs/CREMA_D/synergy/dec/synprom_IB_mask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --lr 0.00001 --wd 0.0001 --cls mlp --perturb mask --num_samples 32
python train.py --config ./configs/CREMA_D/synergy/dec/synprom_IB_mask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 0 --lr 0.00001 --wd 0.0001 --cls mlp --perturb mask


scp -r /esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA kkontras@mib.media.mit.edu:/scratch/kkontras/data
scp -r /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ScienceQA/Qwen3VL_LHead_fold0.pth.tar kkontras@mib.media.mit.edu:/scratch/kkontras/checkpoints/synergy/ScienceQA/
scp -r /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/ScienceQA/Synprom_LoRaFT_fold0_lr0.0001_wd0.01.pth.tar kkontras@mib.media.mit.edu:/scratch/kkontras/checkpoints/synergy/ScienceQA/

python train.py --config ./configs/CREMA_D/synergy/nov/synprom_IB_Dir_VAE.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --pre --frozen --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over