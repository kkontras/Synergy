#!/usr/bin/env bash

#SBATCH --time=24:00:00
#SBATCH --cluster=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000m
#SBATCH --partition gpu_p100
#SBATCH --open-mode=appendj
#SBATCH --output=/scratch/leuven/350/vsc35057/projects/Balance/logs_vsc/slurmout_%J.txt
#SBATCH --error=/scratch/leuven/350/vsc35057/projects/Balance/logs_vsc/slurmerr_%J.txt


cd /scratch/leuven/350/vsc35057/projects/Balance

git pull origin vsc

source /data/leuven/350/vsc35057/anaconda3/etc/profile.d/conda.sh
conda activate /data/leuven/350/vsc35057/balance_env
which python
python -V
echo $PWD

num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ $num_gpus -eq 1 ]; then
    echo "We have 1 GPU"
else
    echo "We have $num_gpus GPUs"
fi

export config=$1
export default_config=$2
export fold=$3
export lr=$4
export wd=$5
export alpha=$6
export kmepoch=$7
export pow=$8
export num_samples=$9
export batch_size=${10}
export recon_weight1=${11}
export recon_weight2=${12}
export recon_epochstages=${13}
export recon_ensemblestages=${14}
export kde_coeff=${15}
export load_ongoing=${16}
export shuffle_type=${17}
export contr_type=${18}
export l_diffsq=${19}
export lib=${20}
export multil=${21}
export ceb=${22}

variables=("config" "default_config" "fold" "lr" "wd" "alpha" "kmepoch" "recon_weight1" "recon_weight2" "recon_epochstages" "recon_ensemblestages" "reg_by" "cls" "clip" "load_ongoing" "num_samples" "reg_by" "batch_size" "pow" "contr_coeff" "temperature" "kde_coeff" "shuffle_type" "contr_type" "l_diffsq" "lib" "multil" "ceb" "l")

for var_name in "${variables[@]}"; do
    var_value="${!var_name}"
    if [ -z "$var_value" ]; then
        declare "$var_name"="None"
    fi
done


accelerate launch /scratch/leuven/350/vsc35057/projects/Balance/main_mcrema_MSLR.py --config $config --default_config $default_config --fold $fold --lr $lr --wd $wd --alpha $alpha --kmepoch $kmepoch  --recon_weight1 $recon_weight1 --recon_weight2 $recon_weight2 --recon_epochstages $recon_epochstages --recon_ensemblestages $recon_ensemblestages --cls $cls --load_ongoing $load_ongoing --clip $clip --l $l --reg_by $reg_by --num_samples $num_samples --batch_size $batch_size --pow $pow --contr_coeff $contr_coeff --temperature $temperature --kde_coeff $kde_coeff --shuffle_type $shuffle_type --contr_type $contr_type --l_diffsq $l_diffsq --lib $lib --multil $multil --ceb $ceb
