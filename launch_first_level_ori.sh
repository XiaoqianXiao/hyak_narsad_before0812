#!/bin/bash

# Directory containing your phase subfolders
# scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/firstLevel"
scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step2"
#scripts_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step3/phase3/searchlight"

# Total CPUs you’re willing to consume at once
TOTAL_CPU_LIMIT=28

# CPUs requested by *each* job (must match --cpus-per-task in your .sh scripts)
CPUS_PER_JOB=4

# Maximum concurrent jobs allowed
MAX_JOBS=$(( TOTAL_CPU_LIMIT / CPUS_PER_JOB ))

echo "Allowing up to $MAX_JOBS concurrent jobs (≈${TOTAL_CPU_LIMIT} CPUs at $CPUS_PER_JOB CPUs/job)."

for phaseID in 2 3; do
  PHASE_DIR="$scripts_dir/phase$phaseID"
  echo "Submitting jobs in: $PHASE_DIR"

  for script in "$PHASE_DIR"/*.sh; do
    # Wait here if we’re already at max concurrent jobs
    while true; do
      # Count your running jobs (change -u user if you want to track a different account)
      RUNNING_JOBS=$(squeue -h -u "$USER" -t R | wc -l)
      if [ "$RUNNING_JOBS" -lt "$MAX_JOBS" ]; then
        break
      fi
      echo "  ↳ $RUNNING_JOBS jobs running (limit $MAX_JOBS). Sleeping 30s…"
      sleep 30
    done

    echo "  ↳ Submitting $script"
    sbatch "$script"
  done
done
