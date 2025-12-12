
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

python examine_vae_linearprob.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusiontf.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb gen
python examine_vae_linearprob.py --config ./configs/CREMA_D/synergy/nov/synprom_generator_diffusiontf_2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --perturb gen