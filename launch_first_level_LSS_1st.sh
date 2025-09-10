#!/bin/bash

# Directory containing your phase subfolders
# scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/firstLevel"
scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss"
#scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step3/phase3/searchlight"


PHASE_DIR="$scripts_dir/phase3"
echo "Submitting jobs in: $PHASE_DIR"
#for script in "$PHASE_DIR"/sub_N260_trial_*_slurm.sh; do
#    [ -e "$script" ] || continue   # skip if no file matches
#    echo "Submitting $script"
#    sbatch "$script"
#done
for subID in 120 202 249 255; do
    for script in "$PHASE_DIR"/sub_N${subID}_trial_*_slurm.sh; do
        [ -e "$script" ] || continue   # skip if no file matches
        echo "Submitting $script"
        sbatch "$script"
    done
done

