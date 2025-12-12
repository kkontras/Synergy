#!/usr/bin/bash
echo "Starting Job"

IFS=',' read -r -a args <<< $1
echo $args

export PATH="/users/sista/kkontras/anaconda3/bin:$PATH"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new

which python
python -V
cd /esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy
echo $PWD
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ $num_gpus -eq 1 ]; then
    echo "We have 1 GPU"
else
    echo "We have $num_gpus GPUs"
fi

echo "${args[0]}"
echo "${args[1]}"
command="accelerate launch "
command+="${args[0]}"


for index in "${!args[@]}"; do
    if [[ $index -eq 0 ]]; then
        continue  # Skip the first element
    fi
    arg="${args[index]}"
#    this_name="${arg%%-*}"  # Extract name by splitting with "-"
#    arg_value="${arg#*-}"
    this_name="${arg%%-*}"; arg_value="${arg#*-}"; [[ $arg == *-* ]] || arg_value=""
    if [[ -n $arg ]]; then  # Check if the argument is non-empty
        command+=" --$this_name $arg_value"  # Include the non-empty argument in the command
    fi
done

echo "Executing command: $command"
$command

#accelerate launch --config_file  ~/.cache/huggingface/accelerate/singlegpu_config.yaml  /users/sista/kkontras/Documents/Balance/main_mcrema.py --config $config --default_config $default_config --fold $fold --alpha $alpha --ending_epoch $ending_epoch --cls $cls --lr $lr
echo "Job finished"



