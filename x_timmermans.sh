
fold=0
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.0001 --lr 0.01 --start_over
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.0001 --lr 0.001 --start_over
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.000001 --lr 0.001 --start_over
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.000001 --lr 0.01 --start_over
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.0001 --lr 0.01 --start_over
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.0001 --lr 0.1 --start_over
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.0001 --lr 0.001 --start_over
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0 --lr 0.001 --start_over
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0 --lr 0.1 --start_over
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.0001 --lr 0.01 --l 1 --regby z12z1mipd

#for leak_prob in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
##for leak_prob in 0.9; do
##    python test_model.py --leak_prob $leak_prob --val_corr 0.1 --noise_std 0.05 --weight_decay 0.005
##    python test_model_2mod.py --leak_prob $leak_prob --val_corr 0.1 --noise_std 0.05 --weight_decay 0.005 --l 100
#    python test_model_2mod.py --leak_prob $leak_prob --val_corr 0.0 --noise_std 0.1 --weight_decay 0 --l 0 #--verbose
#    python test_model_2mod.py --leak_prob $leak_prob --val_corr 0.0 --noise_std 0.1 --weight_decay 0 --l 1 #--verbose
#done

#for l in 0.001 0.01 0.1 1.0 10; do
#  for fill in mean zero noise; do
##      python train.py --config ./configs/CREMA_D/synergy/dec/synprom_IB_mask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l $l --lr 0.00001 --wd 0.0001 --cls mlp --perturb mask --perturb_fill $fill --contrcoeff $contrcoeff --num_samples 32
#      echo "--config ./configs/CREMA_D/synergy/dec/synprom_IB_mask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l $l --lr 0.00001 --wd 0.0001 --cls mlp --perturb mask --perturb_fill $fill --num_samples 32"
#  done
#done


#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 100 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16 --perturb None --perturb_fill token --perturb_pmin 0.5
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16 --perturb None --perturb_fill token --perturb_pmin 0.5
python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16 --perturb None --perturb_fill token --perturb_pmin 0.5
python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16 --perturb None --perturb_fill token --perturb_pmin 0.2
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16 --perturb None --perturb_fill token --perturb_pmin 0.5
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.01 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16 --perturb None --perturb_fill token --perturb_pmin 0.5
#python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.001 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16 --perturb None --perturb_fill token --perturb_pmin 0.5

python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 100 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16  --perturb_fill token --perturb_pmin 0.5
python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16  --perturb_fill token --perturb_pmin 0.5
python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16  --perturb_fill token --perturb_pmin 0.5
python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16  --perturb_fill token --perturb_pmin 0.5
python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.01 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16  --perturb_fill token --perturb_pmin 0.5
python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0.001 --lr 0.0001 --wd 0.0001 --cls tf --batch_size 16  --perturb_fill token --perturb_pmin 0.5
