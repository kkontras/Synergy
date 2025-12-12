##python train.py --config ./configs/CREMA_D/synergy/synprom_hpre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp
##python train.py --config ./configs/CREMA_D/synergy/synprom_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 0 --weight2 0
#python train.py --config ./configs/CREMA_D/synergy/synprom_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 1 --weight2 1
#python train.py --config ./configs/CREMA_D/synergy/synprom.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 1 --weight2 1
#python train.py --config ./configs/CREMA_D/synergy/synprom.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 0 --weight2 0
#python train.py --config ./configs/CREMA_D/synergy/synprom_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp  --weight1 0 --weight2 0
##python train.py --config ./configs/CREMA_D/synergy/synprom.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp
#
#
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp  --weight1 1 --weight2 1
#python train.py --config ./configs/CREMA_D/synergy/synprom_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp  --weight1 0 --weight2 0
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls conformer  --weight1 1 --weight2 1
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls conformer  --weight1 0 --weight2 0
#
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 1 --weight2 1
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom.json --[]default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 0 --weight2 0
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls conformer --weight1 1 --weight2 1
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls conformer --weight1 0 --weight2 0
#
#
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom_unireg_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 1 --weight2 1
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom_unireg_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls conformer --weight1 1 --weight2 1
#
#
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom_unireg.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 1 --weight2 1
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom_unireg.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls conformer --weight1 1 --weight2 1
#
#python train.py --config ./configs/CREMA_D/synergy/synprom_unireg_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 0 --weight2 0
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom_unireg_pre.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls conformer --weight1 0 --weight2 0
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom_unireg.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 0 --weight2 0
#python progress_show_synergy.py --config ./configs/CREMA_D/synergy/synprom_unireg.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls conformer --weight1 0 --weight2 0
#
#
#python train.py --config ./configs/CREMA_D/synergy/synprom_unireg.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 1 --weight2 1 --start_over --dmetric cosine --tdqm_disable
#python train.py --config ./configs/CREMA_D/synergy/synprom_unireg.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 1 --weight2 1 --pre --frozen
#
#python train.py --config ./configs/AVE/res/pre_frozen.json --default_config ./configs/AVE/default_config_ave_res.json --fold 0 --lr 0.001 --wd 0.0001 --pre --frozen --tdqm_disable --start_over
#
#
# python train.py --config ./configs/CREMA_D/synergy/synprom_unireg.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 1 --weight2 1 --pre --frozen --tdqm_disable
#
#
#
#
# python train.py --config ./configs/CREMA_D/synergy/synprom_onlyq.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 1 --weight2 10 --start_over --pre
# python train.py --config ./configs/CREMA_D/synergy/synprom_onlyq.json --default_config ./configs/CREMA_D/default_config_cremad.json --fold 0 --cls mlp --weight1 1 --weight2 1 --start_over
#
#
#
#
# python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp --pre --frozen
# python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp
#
#
#python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp --l 0 --lr 0.0001 --pre
#python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp --l 1 --lr 0.0001 --pre
#python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp --l 100 --lr 0.0001 --pre
#
#
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold 0 --cls mlp --l 1
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold 0 --cls mlp --l 1
#
#python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold 0 --start_over --wd 0.001 --lr 0.001 --l 0
#
#for fold in 0 1 2; do
#  python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --wd 0.0001 --lr 0.01 --l 0  --start_over
#  python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --wd 0.0001 --lr 0.01 --l 1 --regby cond  --start_over
#  python train.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --wd 0.0001 --lr 0.01 --l 1 --regby perf  --start_over
#done
fold=0
python show_v2.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.0001 --lr 0.01 --l 0
python show_v2.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.0001 --lr 0.01 --l 1 --regby cond
python show_v2.py --config ./configs/xor/synprom_perf.json --default_config ./configs/xor/default_config_xor.json --fold $fold  --start_over --wd 0.0001 --lr 0.01 --l 1 --regby perf

 python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls conformer --pre --l 0
 python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls conformer --pre --l 1

 python train.py --config ./configs/M3/synprom_perf.json --default_config ./configs/M3/default_config_M3.json --fold 0 --cls mlp --l 1
 python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp --pre --l 0 --start_over
 python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp --pre --l 1 --start_over
 python train.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp --l 1 --start_over


 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --pre --l 1 --regby z12z1_z1 --start_over
 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --pre --l 0 --regby z12z1_z1 --start_over
 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --regby z12z1_z1 --start_over


 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --regby z12z1_z1 --start_over --tdqm_disable
 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 0 --regby z12z1_z1 --start_over --tdqm_disable --lr 0.00001
 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 1 --regby z12z1_z1 --start_over --tdqm_disable --lr 0.00001
 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --regby z12z1_z1 --start_over --tdqm_disable

 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 0 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001
 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 1 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001
 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 0.1 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001
 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 0.01 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001
 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0  --pre --l 10 --regby z12z1mipd --start_over --tdqm_disable --lr 0.00001
 python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --regby z12z1_z1 --start_over --tdqm_disable

#
# python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls conformer --pre --l 0
# python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls conformer --pre --l 1
#
# python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp --pre --l 0 --start_over
# python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp --pre --l 1 --start_over
# python show_v2.py --config ./configs/CREMA_D/synergy/sept/synprom_perf_v2.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --cls mlp --l 1 --start_over
