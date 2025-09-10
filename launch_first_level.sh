#!/bin/bash

# Directory containing your phase subfolders
# scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/firstLevel"
#scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss"
scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step3"
#scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step3/phase3/searchlight"

# Total CPUs you’re willing to consume at once
TOTAL_CPU_LIMIT=28

# CPUs requested by *each* job (must match --cpus-per-task in your .sh scripts)
CPUS_PER_JOB=4

# Maximum concurrent jobs allowed
MAX_JOBS=$(( TOTAL_CPU_LIMIT / CPUS_PER_JOB ))

echo "Allowing up to $MAX_JOBS concurrent jobs (≈${TOTAL_CPU_LIMIT} CPUs at $CPUS_PER_JOB CPUs/job)."

#for phaseID in 2 3; do
#  PHASE_DIR="$scripts_dir/phase$phaseID"
#  for script in "$PHASE_DIR"/*.sh; do
#    sbatch "$script"
#  done
#done
for phaseID in 2 3; do
  PHASE_DIR="$scripts_dir/phase$phaseID"
  final_scripts_dir="$PHASE_DIR/roi"
  #final_scripts_dir="$PHASE_DIR/searchlight"
  for script in "$final_scripts_dir"/*.sh; do
    sbatch "$script"
  done
done
#for phaseID in 2 3; do
#  PHASE_DIR="$scripts_dir/phase$phaseID"
#  for analysis_type in "searchlight"; do
#    final_scripts_dir="$PHASE_DIR/$analysis_type"
#    for script in "$final_scripts_dir"/*.sh; do
#      sbatch "$script"
#    done
#  done
#done
