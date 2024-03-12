#!/bin/bash
bench_dir="ispd2006"
echo "Downloads & extracts ISPD 2006 instances to ${bench_dir}"
declare -a ins_names=(
adaptec5
newblue1
newblue2
newblue3
newblue4
newblue5
newblue6
newblue7
)

mkdir $bench_dir
cd $bench_dir
for ins_name in "${ins_names[@]}" ; do
    wget https://www.ispd.cc/contests/06/contest/${ins_name}.tar.gz
    mkdir ${ins_name}
    tar -zxvf ${ins_name}.tar.gz -C ${ins_name}
    cd ${ins_name}
    gunzip *.gz
    cd ..
    rm ${ins_name}.tar.gz
done

cd ..
