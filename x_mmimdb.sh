
for l in 0.001 0.01 0.1 1 10 100; do for lsparse in 0.001 0.01 0.1 1 10 100; do python show.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb_fill ema --perturb_lsparse $lsparse --start_over; done; done


for l in 1 10 100; do for lsparse in 10 100; do python train.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l $l --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32 --perturb_fill ema --perturb_lsparse $lsparse; done; done


python show.py --config ./configs/MMIMDB/synprom_SynIB_RMask.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --l 0 --lr 0.0001 --wd 0.0001 --cls mlp --batch_size 32

for fold in 0 1 2; do
for lr in 0.001 0.0001 0.00001; do
  for wd in 0.001 0.0001 0.00001; do
    python show.py --config ./configs/MMIMDB/unimodal_text.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold $fold --lr $lr --wd $wd --batch_size 32
    python show.py --config ./configs/MMIMDB/unimodal_vision.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold $fold --lr $lr --wd $wd --batch_size 32
done
done
done

python show.py --config ./configs/MMIMDB/unimodal_text.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 32
python show.py --config ./configs/MMIMDB/unimodal_vision.json --default_config ./configs/MMIMDB/default_config_mmimdb_syn.json --fold 0 --lr 0.0001 --wd 0.0001 --batch_size 32
