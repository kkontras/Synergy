#!/usr/bin/env bash

#SBATCH --time=24:00:00
#SBATCH --cluster=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=34
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4g
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

export train_file=$1
export default_files="./configs/SthSth/default_config_vsc_sthsth.json"

accelerate launch --config_file /data/leuven/350/vsc35057/.cache/huggingface/accelerate/multigpu_config.yaml /scratch/leuven/350/vsc35057/projects/Balance/main.py $train_file $default_files
