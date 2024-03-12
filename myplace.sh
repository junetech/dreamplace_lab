#!/bin/bash
echo "Serial batch of placement with multiple random seeds"
ins_dir="test/ispd2005/"
ext=".json"
declare -a ins_names=("adaptec1" "adaptec2")
declare -a rand_seeds=(1001 1002)
key="random_seed"
output_dir_prefix="result_rs"

for ins_name in "${ins_names[@]}"; do
    original_path=${ins_dir}${ins_name}${ext}
    copied_path=${ins_dir}${ins_name}_copy${ext}
    for rand_seed in "${rand_seeds[@]}"; do
        echo "${copied_path} <- Random seed ${rand_seed}"
        result_dir=${output_dir_prefix}${rand_seed}
        jq ". + {${key}:${rand_seed}, result_dir:\"${result_dir}\"}" ${original_path} > ${copied_path}
        python dreamplace/Placer.py ${copied_path}
    done
    rm ${copied_path}
done
