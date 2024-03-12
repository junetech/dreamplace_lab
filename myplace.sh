#!/bin/bash
echo "Serial batch of placement with multiple random seeds"
declare -a rand_seeds=($(seq 1001 1 1030))
key="random_seed"

ins_dir="test/ispd2005/"
ext=".json"
declare -a ins_names=("adaptec1" "adaptec2" "adaptec3" "adaptec4" "bigblue1" "bigblue2" "bigblue3" "bigblue4")
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

ins_dir="test/ispd2006/"
declare -a ins_names=("adaptec5" "newblue1" "newblue2" "newblue3" "newblue4" "newblue5" "newblue6" "newblue7")

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
