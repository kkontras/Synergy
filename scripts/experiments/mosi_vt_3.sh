  PYTHON="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new/bin/python"
  DEFAULT="./configs/FactorCL/Mosi/default_config_mosi_VT.json"
  SYN_DIR="./configs/FactorCL/Mosi/syn/VT"
  NOPRE="${SYN_DIR}/synprom_RMask_nopre.json"
  DNR="${SYN_DIR}/DnR.json"
  RECONBOOST="${SYN_DIR}/ReconBoost.json"
  GPU=0
  LR=0.0005
  WD=0.001

  # # # learned
  # for fold in 2 1 0; do
  #   for l in 0.1 1; do
  #     for lsparse in 0.001 0.01 0.1 1 3 5 10; do
  #       CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/entrypoints/train.py \
  #         --config $NOPRE --default_config $DEFAULT \
  #         --fold $fold --lr $LR --wd $WD \
  #         --l $l --perturb learned --perturb_fill ema --perturb_lsparse $lsparse \
  #         --validate_with accuracy || echo "failed: fold=$fold l=$l lsparse=$lsparse"
  #     done
  #   done
  # done

  # random
  # for fold in 0; do
  #   for l in 0.001 0.01 0.1 1; do
  #     for pmin in 0.1 0.3 0.5 0.7 0.9; do
  #       CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/entrypoints/show.py \
  #         --config $NOPRE --default_config $DEFAULT \
  #         --fold $fold --lr $LR --wd $WD \
  #         --l $l --perturb random --perturb_fill ema --perturb_pmin $pmin \
  #         --validate_with accuracy || echo "failed: fold=$fold l=$l pmin=$pmin"
  #     done
  #   done
  # done

  # DnR
  for fold in 2 1 0; do
    for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
      for kmpe in 1 3 5 10; do
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/entrypoints/train.py \
          --config $DNR --default_config $DEFAULT \
          --fold $fold --lr $LR --wd $WD \
          --alpha $alpha --kmepoch $kmpe \
          --validate_with accuracy || echo "failed: fold=$fold alpha=$alpha kmepoch=$kmpe"
      done
    done
  done

  # ReconBoost
  for fold in 0; do
    for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do  
      for stages in 1 4 10; do
        for w1 in 1 3 5 10; do
          CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/entrypoints/show.py \
            --config $RECONBOOST --default_config $DEFAULT \
            --fold $fold --lr $LR --wd $WD \
            --alpha $alpha \
            --recon_weight1 $w1 --recon_weight2 1 \
            --recon_epochstages $stages --recon_ensemblestages $stages \
            --validate_with accuracy || echo "failed: fold=$fold alpha=$alpha stages=$stages w1=$w1"
        done
      done
    done
  done
