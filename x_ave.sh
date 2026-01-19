


python train.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb diff --perturb_pmin 0.5 --perturb_pmin 0.8 --num_samples 10

for l in 1; do for lsparse in 1; do python train.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse; done; done

python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json  --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb_fill ema --perturb_lsparse 5


python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --start_over
python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5 --start_over
python show.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5


python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 128 --start_over
python train.py --config ./configs/AVE/synergy/synprom_RMask.json --default_config ./configs/AVE/default_config_ave_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 128 --start_over
