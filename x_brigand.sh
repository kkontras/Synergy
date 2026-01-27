
# python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 10 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001

# python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --regby z12z1mipd --start_over --tdqm_disable
# python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 1 --l 1 --regby z12z1mipd --start_over --tdqm_disable
# python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 2 --l 1 --regby z12z1mipd --start_over --tdqm_disable
# python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 1  --l 1 --regby z12z1mipd --start_over --tdqm_disable

# python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 10 --start_over --tdqm_disable --lr 0.00001 --cls mlp

#for fold in 0 1 2; do
#  for wd in 0.0; do
#   python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre --l 10 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#   python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre --l 1 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#   python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre --l 0.1 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#   python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre --l 0.01 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#   python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre --l 0 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#done
#done
#
#python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 0.11 --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb cmn
#python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 1 --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb cmn
#python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 1 --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over
#python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 0 --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb cmn
#
#
#python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --frozen --l 1 --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over
#


#for l in 0.01 0.03 0.1 0.3 1 3 10 30 100; do
##  python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over
#  echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over"
#  echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 1 --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over"
#  echo "--config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 2 --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen --start_over"
#done


# for fold in 0 1 2; do
#     for l in 0.001 0.005 0.01 0.05 0.1 0.5 0; do
#       python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen
#done
#done

# for fold in 1; do
#     for l in 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 0; do
#       python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls conformer --perturb gennoise
#       python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls conformer --perturb gennoise
#done
#done
#
# for fold in 0 1 2; do
#     for l in 0.01 0.03 0.1 0.3 1 3 0; do
#      python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --frozen --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen
#done
#done

# for fold in 0 1 2; do
##     for l in 0.01 0.03 0.1 0.3 1 3 0; do
#     for l in 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 0; do
#      python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --frozen --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen
#      python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --frozen --l $l --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gen
#done
#done

#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/synrem_vae/enc_2_fusion_trunk_fold0.pth.tar /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/Lmask/enc_2_fusion_trunk_fold0.pth.tar
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/synrem_vae/enc_3_fusion_head_fold0.pth.tar /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/Lmask/enc_3_fusion_head_fold0.pth.tar
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/synrem_vae/enc_4_unimodal1_fold0.pth.tar /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/Lmask/enc_4_unimodal1_fold0.pth.tar
#cp /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/synrem_vae/enc_5_unimodal2_fold0.pth.tar /esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/Lmask/enc_5_unimodal2_fold0.pth.tar
#
#python examine_vae_linearprob.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusiontf.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb gen
#python examine_vae_linearprob.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusiontf_2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb gen


#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#
#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb diff --perturb_fill ema --perturb_pmin 0.9 --perturb_pmax 1.0 --num_samples 5
#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb diff --perturb_fill ema --perturb_pmin 0.1 --perturb_pmax 0.9 --num_samples 5
#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb diff --perturb_fill ema --perturb_pmin 0.1 --perturb_pmax 0.5 --num_samples 5
#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32
#
#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.1 --perturb_pmax 0.9 --num_samples 5
#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.1 --perturb_pmax 0.5 --num_samples 5
#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32
#
#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb_fill ema --perturb_pmin 1.0 --ending_epoch 15
#python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb_fill ema --perturb_pmin 1.0 --ending_epoch 15
#
#
#python show.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb diff --perturb_fill ema --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python show.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb diff --perturb_fill ema --perturb_pmin 0.1 --perturb_pmax 0.9 --num_samples 5
#python show.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb diff --perturb_fill ema --perturb_pmin 0.1 --perturb_pmax 0.5 --num_samples 5
#python show.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32
#
#python show.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python show.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.1 --perturb_pmax 0.9 --num_samples 5
#python show.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.1 --perturb_pmax 0.5 --num_samples 5
#python show.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32

#for l in 0.001 0.01 0.1 1 10 100; do for lsparse in 0.001 0.01 0.1 1 10 100; do python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb_fill ema --perturb_lsparse $lsparse --start_over; done; done
#for l in 0.01 0.1 1 10 100; do for lsparse in 0.001 0.01 0.1 1 10 100; do python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb_fill ema --perturb_lsparse $lsparse --start_over; done; done
#for l in 0.1; do for lsparse in 0.001 0.01 0.1 1 10 100; do python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb_fill ema --perturb_lsparse $lsparse --start_over; done; done
#
#for l in 0.001 0.01 0.1 1 10 100; do
#  for lsparse in 0.01 0.1 1 10 100; do
#    echo "--config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb_fill ema --perturb_lsparse $lsparse --start_over"
#  done
#done


#for lr in 0.001 0.0001 0.00001; do
#for wd in 0.001 0.0001 0.00001; do
#  echo "--config ./configs/AVE/synergy/dec/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0 --lr $lr --wd $wd --cls mlp --batch_size 64"
#  for l in 0.1 1 10 100; do for lsparse in 0.1 1 5 10; do
#  echo "--config ./configs/AVE/synergy/dec/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l $l --lr $lr --wd $wd --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse"
#  done; done
#done; done



#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5

#for l in 0.1 0.5 1 5 10 100; do for lsparse in 0.1 0.5 1 5 10; do
#python train.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse --start_over
#done; done
#
#for l in 0.1 0.5 1 5 10 100; do for lsparse in 0.1 0.5 1 5 10; do
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse --start_over
#done; done


#for ir in 0.1 0.5 1.0 2.0; do for a in 0.5 1.0 1.5 2.0 3.0 5.0; do
#  python train.py --config ./configs/CREMA_D/synergy/jan/MMPareto.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --alpha $a  --ironic_rate $ir; done; done

#python mydatasets/ScienceQA/ScienceQA_Codebook_v2.py  --split train --data_root "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA" --out_dir "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA/cache_qwen3_vl_2b_nocls_vis" --num_workers 24 --batch_size 4
#python mydatasets/ScienceQA/ScienceQA_Codebook_v2.py  --split test --data_root "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA" --out_dir "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ScienceQA/cache_qwen3_vl_2b_nocls_vis" --num_workers 24 --batch_size 4

for fold in 2; do
  for ir in 1.0 2.0; do
    for l in 1 5 10; do
#          for pmin in 0.2 0.3 0.5 0.7 0.9; do
#            python train.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold $fold --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --ironic_rate $ir --perturb_pmin $pmin
#          done
          for lsparse in 0.1 0.5 1 5 10; do
            python train.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold $fold --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse  --ironic_rate $ir
          done
    done
  done
done


#python train.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb diff --perturb_pmin 0.5 --perturb_pmin 0.8 --num_samples 10

#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 100 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.01 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.01 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16 --perturb None --perturb_fill token --perturb_pmin 0.5

#python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 100 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.01 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 32 --perturb diff --perturb_fill token --perturb_pmin 0.4 --perturb_pmax 0.9 --num_samples 5
#python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.01 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16 --perturb None --perturb_fill token --perturb_pmin 0.5
