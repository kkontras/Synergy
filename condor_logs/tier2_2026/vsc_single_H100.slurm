#!/usr/bin/env bash

#SBATCH --time=55:00:00
#SBATCH --cluster=wice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --open-mode=appendj
#SBATCH --output=/scratch/leuven/350/vsc35057/projects/Balance/logs_vsc/slurmout_%J.txt
#SBATCH --error=/scratch/leuven/350/vsc35057/projects/Balance/logs_vsc/slurmerr_%J.txt


cd $VSC_SCRATCH/projects/Synergy

git pull origin vsc

source /data/leuven/350/vsc35057/anaconda3/etc/profile.d/conda.sh
conda activate /data/leuven/350/vsc35057/synergy_new
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
export l=$6

variables=("config" "default_config" "fold" "lr" "wd" "l" )

for var_name in "${variables[@]}"; do
    var_value="${!var_name}"
    if [ -z "$var_value" ]; then
        declare "$var_name"="None"
    fi
done


accelerate launch /scratch/leuven/350/vsc35057/projects/Balance/main_mcrema_MSLR.py --config $config --default_config $default_config --fold $fold --lr $lr --wd $wd --l $l
