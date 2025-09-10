#!/bin/bash

# Environment settings
container_path="/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_group_level_placebo_1.0.sif"
base_dir="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows"
script_base_dir="${base_dir}/groupLevel/whole_brain/Placebo"

tasks=("phase2" "phase3")
contrasts=$(seq 1 31)
analysis_types=("randomise" "flameo")
#analysis_types=("randomise")

mkdir -p "$script_base_dir"

for task in "${tasks[@]}"; do
    for contrast in $contrasts; do
        for analysis_type in "${analysis_types[@]}";do
            job_name="group_${task}_cope${contrast}_${analysis_type}"
            script_path="${script_base_dir}/${job_name}.sh"
            out_path="${script_base_dir}/${job_name}_%j.out"
            err_path="${script_base_dir}/${job_name}_%j.err"

            cat <<EOF > "$script_path"
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=fang
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=${out_path}
#SBATCH --error=${err_path}

module load apptainer
apptainer exec -B /gscratch/fang:/data -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir ${container_path} \\
    python3 /app/run_group_level_onlyPlacebo.py \\
    --task ${task} \\
    --contrast ${contrast} \\
    --analysis_type ${analysis_type} \\
    --base_dir ${base_dir}
EOF

            chmod +x "$script_path"
            echo "Created Slurm script: $script_path"

        done
    done
done
