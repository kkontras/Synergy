
# python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 0.01 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001
# python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 1  --pre --l 0.01 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001
# python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 2  --pre --l 0.01 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001


# for fold in 0 1 2; do
#  for wd in 0.001 0.0001 0.0; do
#   python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre --frozen --l 1 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#   python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold  --pre --frozen --l 0 --tdqm_disable --lr 0.0001 --wd $wd --cls conformer
#done
#done

# for fold in 0; do
##     for l in 0 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 ; do
#       for l in 0.01 0.03 0.1 0.3 1 3 10 30 100; do
#
##       python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gennoise
##       python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gennoise
#       python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls conformer --perturb gen
#done
#done

 for fold in 0; do
#     for l in 0 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 ; do
       for l in 0; do

#       python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gennoise
#       python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gennoise
       python show_v2.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls conformer --perturb gen
done
done

72.6 ± 2.5 72.7 ± 1.5 73.0 ± 1.6

# for fold in 2; do
#     for l in 0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 50 100 0; do
#       python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 0 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gennoise
#       python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls mlp --perturb gennoise
#       python train.py --config ./configs/CREMA_D/synergy/nov/synprom_ib.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold $fold --pre --l $l --contrcoeff 1 --lr 0.0001 --wd 0.0001 --cls conformer --perturb gen
#done
#done