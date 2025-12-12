
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

for leak_prob in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#for leak_prob in 0.9; do
#    python test_model.py --leak_prob $leak_prob --val_corr 0.1 --noise_std 0.05 --weight_decay 0.005
#    python test_model_2mod.py --leak_prob $leak_prob --val_corr 0.1 --noise_std 0.05 --weight_decay 0.005 --l 100
    python test_model_2mod.py --leak_prob $leak_prob --val_corr 0.0 --noise_std 0.1 --weight_decay 0 --l 0 #--verbose
    python test_model_2mod.py --leak_prob $leak_prob --val_corr 0.0 --noise_std 0.1 --weight_decay 0 --l 1 #--verbose
done
