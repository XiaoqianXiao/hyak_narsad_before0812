#!/bin/bash
# File: submit_all.sh

# Base directory for scripts
scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_group_searchlight"

# Methods to process
methods=("flameo" "randomise")

for method in "${methods[@]}"; do
    final_scripts_dir="${scripts_dir}/${method}"

    sh_files=("${final_scripts_dir}"/*.sh)
    for script in "${sh_files[@]}"; do
        echo "Submitting $script"
        sbatch "$script"
    done
done
