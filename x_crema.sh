#
#
#
#python train.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb diff --perturb_pmin 0.5 --perturb_pmin 0.8 --num_samples 10
#
#for l in 1; do for lsparse in 1; do python train.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse; done; done
#
#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask_unpre.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb_fill ema --perturb_lsparse 5
#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5
#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5
#
#
#python show.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64
#python show.py --config ./configs/CREMA_D/synergy/jan/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64
#python train.py --config ./configs/CREMA_D/synergy/jan/unimodal_audioTF.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64
#python train.py --config ./configs/CREMA_D/synergy/jan/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64
#
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0 --lr 0.00001 --wd 0.0001 --cls mlp --batch_size 64
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 0.1
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 3
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5

python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64
python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --validate_with syn_accuracy
python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 0.1 --validate_with syn_accuracy
python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --validate_with syn_accuracy
python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --validate_with syn_accuracy
python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 3 --validate_with syn_accuracy
python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5 --validate_with syn_accuracy
python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 10 --validate_with syn_accuracy
python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --validate_with syn_accuracy
python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --validate_with syn_accuracy --start_over



for ir in 0.1 0.5 2.0; do
  echo "--config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate $ir --start_over"
  echo "--config ./configs/CREMA_D/synergy/jan/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64  --cls mlp --ironic_rate $ir --start_over"
done

  python show.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate 2
  python main_mcrema_postpred.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate 2

  python show.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate 2

  for ir in 0.5; do python train.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate $ir; done
  for ir in 0.1 0.5 1.0 2.0; do python get_ceu_post.py --config ./configs/CREMA_D/synergy/jan/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate $ir ; done

for ir in 1.0 ; do
#  python show.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate $ir
#  python show.py --config ./configs/CREMA_D/synergy/jan/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64  --cls mlp --ironic_rate $ir
#  python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --ironic_rate $ir
  for l in 0.1 0.5 1 5 10; do for lsparse in 0.1 0.5 1 5 10; do
    python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse  --ironic_rate $ir
  done; done

done

for fold in 1 2; do
for ir in 0.1 0.5 1.0 2.0; do
  python train.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate $ir
  python train.py --config ./configs/CREMA_D/synergy/jan/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64  --cls mlp --ironic_rate $ir
done
done

for fold in 0 1 2; do
for ir in 0 0.1 0.3 0.5 0.8 1.0 2.0; do
      python get_ceu_post.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold $fold --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate $ir
      python get_ceu_post.py --config ./configs/CREMA_D/synergy/jan/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold $fold --lr 0.0001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate $ir

#  python show.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate $ir
#  python show.py --config ./configs/CREMA_D/synergy/jan/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64  --cls mlp --ironic_rate $ir
#  python train.py --config ./configs/CREMA_D/synergy/jan/ens.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold $fold --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --ironic_rate $ir
#  python train.py --config ./configs/CREMA_D/synergy/jan/ens.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold $fold --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --ironic_rate $ir

done
done


for fold in 0 1 2; do
for ir in 0 0.1 0.3 0.5 0.8 1.0 2.0; do
    python train.py --config ./configs/CREMA_D/synergy/jan/ens.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --ironic_rate $ir
done
done

for ir in 0.1 0.5 1.0 2.0; do
#  python show.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate $ir
#  python show.py --config ./configs/CREMA_D/synergy/jan/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64  --cls mlp --ironic_rate $ir
  python show.py --config ./configs/CREMA_D/synergy/jan/ens.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --ironic_rate $ir
  python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --ironic_rate $ir
#  for l in 0.1 0.5 1 5 10; do
#        for pmin in 0.2 0.3 0.5 0.7 0.9; do
#          python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --ironic_rate $ir --perturb_pmin $pmin
#        done
#        for lsparse in 0.1 0.5 1 5 10; do
#          python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse  --ironic_rate $ir
#        done
#  done
done

python train.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate 0.1 --start_over


#          python train.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --ironic_rate 0.0 --perturb_pmin 0.5

python get_ceu_post.py --config ./configs/CREMA_D/synergy/jan/unimodal_audio.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.001 --wd 0.0001 --batch_size 64 --cls mlp --ironic_rate 0.1
python get_ceu_post.py --config ./configs/CREMA_D/synergy/jan/unimodal_video.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64  --cls mlp --ironic_rate 0.1

python train.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls linear --ironic_rate 0.1  --alpha 3 --kmepoch 3
python train.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls linear --ironic_rate 0.1  --alpha 3 --kmepoch 5
python train.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls linear --ironic_rate 0.1  --alpha 5 --kmepoch 3
python train.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls linear --ironic_rate 0.1  --alpha 5 --kmepoch 5

python train.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls linear --ironic_rate 0.1  --alpha 3 --kmepoch 3 --pre
python train.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls linear --ironic_rate 0.1  --alpha 3 --kmepoch 5 --pre
python train.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls linear --ironic_rate 0.1  --alpha 5 --kmepoch 3 --pre
python train.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls linear --ironic_rate 0.1  --alpha 5 --kmepoch 5 --pre

  python train.py --config ./configs/CREMA_D/synergy/jan/MCR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --l 0.1 --multil 0.01 --num_samples 32 --regby greedy --shuffle_type rand --batch_size 32 --contrcoeff 1 --ironic_rate 0.1
  python train.py --config ./configs/CREMA_D/synergy/jan/MCR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --l 1 --multil 1 --num_samples 32 --regby greedy --shuffle_type rand --batch_size 32 --contrcoeff 1 --ironic_rate 0.1


for ir in 0.1 0.5 1.0 2.0; do for l in 0.001 0.01 0.1 1; do for multil in 0.01 0.1 1; do
  python show.py --config ./configs/CREMA_D/synergy/jan/MCR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --l $l --multil $multil --num_samples 32 --regby greedy --shuffle_type rand --batch_size 32 --contrcoeff 1 --ironic_rate $ir; done; done ; done
for ir in 0.1 0.5 1.0 2.0; do for a in 0.5 1.0 1.5 2.0 3.0 5.0; do
  python show.py --config ./configs/CREMA_D/synergy/jan/MMPareto.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --alpha $a  --ironic_rate $ir; done; done
for ir in 0.1 0.5 1.0 2.0; do for a in 1 3 5; do for kmpe in 1 3 5; do
  python show.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls linear  --alpha $a --kmepoch $kmpe  --ironic_rate $ir --start_over; done; done ; done
for ir in 0.1 0.5 1.0 2.0; do for recon_stages in 1 4; do for recon_weight1 in 1 3 5; do
  python show.py --config ./configs/CREMA_D/synergy/jan/ReconBoost.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --alpha 0.5 --recon_weight1 $recon_weight1 --recon_weight2 1 --recon_epochstages $recon_stages --recon_ensemblestages $recon_stages --ir $ir
done;done;done


for ir in 2.0; do for l in 0.001 0.01 0.1 1; do for multil in 0.01 0.1 1; do
  python show.py --config ./configs/CREMA_D/synergy/jan/MCR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --l $l --multil $multil --num_samples 32 --regby greedy --shuffle_type rand --batch_size 32 --contrcoeff 1 --ironic_rate $ir; done; done ; done

for ir in 0.1 0.5 1.0 2.0; do for a in 1 3 5; do for kmpe in 1 3 5; do
  python train.py --config ./configs/CREMA_D/synergy/jan/DnR.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 64 --cls linear  --alpha $a --kmepoch $kmpe  --ironic_rate $ir --start_over; done; done ; done


srun --partition=interactive --cluster=genius --time=01:00:00 --nodes=1000 -A lp_biomed_mdv --gpus-per-node=1 --pty bash -l

#python train.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.00001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --validate_with syn_accuracy
#python train.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --validate_with syn_accuracy
#python train.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0 --cls tf --batch_size 64 --perturb_fill token --perturb_lsparse 1 --validate_with syn_accuracy

#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask_nopre.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask_nopre.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --validate_with syn_accuracy
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask_nopre.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask_nopre.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1

#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0 --lr 0.00001 --wd 0.0001 --cls mlp --batch_size 64 --validate_with syn_accuracy
#
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.00001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0.1 --lr 0.00001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 1 --lr 0.00001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --validate_with syn_accuracy
#python show.py --config ./configs/CREMA_D/synergy/jan/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremadplus_res_syn.json --fold 0 --l 0.1 --lr 0.00001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --validate_with syn_accuracy





#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5
#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 0.1 --start_over
#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 0.1 --start_over
#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 0.1 --start_over
#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --start_over
#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --start_over
#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 0.1 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 1 --start_over
#python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 10 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5
#
#for lr in 0.0001; do
#for wd in 0.0001 0.00001; do
#  python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 0 --lr $lr --wd $wd --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse 5
#done; done
#for lr in 0.0001; do
#for wd in 0.0001 0.00001; do
#  python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l 0 --lr $lr --wd $wd --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse
#  for l in 0.01 0.1 1; do for lsparse in 0.01 0.1 1; do
#  python show.py --config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l $l --lr $lr --wd $wd --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse
##  echo "--config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l $l --lr $lr --wd $wd --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse"
#  done; done
#done; done
#
#for lr in 1 0.0001; do
#  for wd in 0.0001 0.00001; do for l in 0.01 0.1 1; do for lsparse in 0.01 0.1 1; do echo "--config ./configs/CREMA_D/synergy/dec/synprom_RMask.json --default_config ./configs/CREMA_D/default_config_cremad_res_syn.json --fold 0 --l $l --lr $lr --wd $wd --cls mlp --batch_size 64 --perturb_fill ema --perturb_lsparse $lsparse" done; done done; done

#python train.py --config ./configs/ESNLI/synprom_ib_gen.json --default_config ./configs/ESNLI/default_config_esnli_syn.json

