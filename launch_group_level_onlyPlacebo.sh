#!/bin/bash

scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/groupLevel/whole_brain/Placebo"

for phaseID in 2 3; do
#for phaseID in 3; do
  #for script in "$scripts_dir"/*_phase$phaseID*randomise.sh; do
  for script in "$scripts_dir"/*_phase$phaseID*flameo.sh; do
    echo "Submitting $script"
    sbatch "$script"
  done
done


