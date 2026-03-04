  PYTHON="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new/bin/python"                                                      
  DEFAULT="./configs/FactorCL/Mosi/default_config_mosi_VT.json"                                                                                  
  NOPRE="./configs/FactorCL/Mosi/syn/VT/synprom_RMask_nopre.json"                                                                                
  GPU=0                                                                                                                                          
  LR=0.0005                                                 
  WD=0.001

  # learned
  for fold in 2; do
    for l in 0.001 0.01 0.1 1; do
      for lsparse in 0.001 0.01 0.1 1 3 5 10; do
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/entrypoints/train.py \
          --config $NOPRE --default_config $DEFAULT \
          --fold $fold --lr $LR --wd $WD \
          --l $l --perturb learned --perturb_fill ema --perturb_lsparse $lsparse \
          --validate_with accuracy || echo "failed: fold=$fold l=$l lsparse=$lsparse"
      done
    done
  done

  # random
  for fold in 2; do
    for l in 0.001 0.01 0.1 1; do
      for pmin in 0.1 0.3 0.5 0.7 0.9; do
        CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/entrypoints/train.py \
          --config $NOPRE --default_config $DEFAULT \
          --fold $fold --lr $LR --wd $WD \
          --l $l --perturb random --perturb_fill ema --perturb_pmin $pmin \
          --validate_with accuracy || echo "failed: fold=$fold l=$l pmin=$pmin"
      done
    done
  done
