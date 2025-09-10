#!/bin/bash

# Directory containing your phase subfolders
scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/firstLevel"
#scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss"
#scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step3/phase3/searchlight"


for phaseID in 2 3; do
  PHASE_DIR="$scripts_dir/phase$phaseID"
  final_scripts_dir="$PHASE_DIR/roi"
  #final_scripts_dir="$PHASE_DIR/searchlight"
  for script in "$final_scripts_dir"/*.sh; do
    sbatch "$script"
  done
done
