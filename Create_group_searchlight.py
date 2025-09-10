import os
import argparse
from itertools import combinations
import logging

# Configure logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logging()

def generate_slurm_scripts(method, work_dir, slurm_dir):
    # Trial types
    trial_types = ['SHOCK', 'FIXATION', 'CS-', 'CSS', 'CSR']
    map_types = [f'within-{ttype}' for ttype in trial_types] + [f'between-{t1}-{t2}' for t1, t2 in combinations(trial_types, 2)]
    tasks = ['phase2', 'phase3']
    logger.info(f"Generating Slurm scripts for {len(map_types)} map types and {len(tasks)} tasks with method {method}: {map_types}")

    # Slurm template
    slurm_template = """#!/bin/bash
#SBATCH --account=fang                                                                                            
#SBATCH --partition=ckpt-all  
#SBATCH --job-name=group_searchlight_{map_type}_{task}_{method}
#SBATCH --output=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_group_searchlight/{task}_group_searchlight_{map_type}_{method}_%j.out
#SBATCH --error=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_group_searchlight/{task}_group_searchlight_{map_type}_{method}_%j.err
#SBATCH --time={time}
#SBATCH --mem=20G
#SBATCH --cpus-per-task=16

module load apptainer
export OMP_NUM_THREADS=4
apptainer exec \
    -B /gscratch/fang:/data \
    -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir \
    -B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad/group_searchlight.py:/app/group_searchlight.py \
    /gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif python3 /app/group_searchlight.py --map_type {map_type} --method {method}
"""

    # Set time limit based on method
    time_limit = '02:00:00' if method == 'flameo' else '4:00:00'  # 1 hour for Randomise

    # Generate Slurm script for each map type and task
    script_paths = []
    for map_type in map_types:
        for task in tasks:
            script_path = os.path.join(slurm_dir, f'group_searchlight_{map_type}_{task}_slurm.sh')
            with open(script_path, 'w') as f:
                f.write(slurm_template.format(map_type=map_type, task=task, method=method, time=time_limit))
            os.chmod(script_path, 0o755)  # Make executable
            script_paths.append(script_path)
            logger.info(f"Generated Slurm script: {script_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate Slurm scripts for group-level searchlight analysis.')
    parser.add_argument('--method', choices=['flameo', 'randomise'], default='flameo', help='Analysis method: flameo or randomise')
    args = parser.parse_args()
    method = args.method

    # Create work_dir and slurm_dir
    scrubbed_dir = '/scrubbed_dir'
    project_name = 'NARSAD'
    work_dir = os.path.join(scrubbed_dir, project_name, 'work_flows', 'Lss_group_searchlight')
    slurm_dir = os.path.join(work_dir, method)
    os.makedirs(slurm_dir, exist_ok=True)
    logger.info(f"Slurm scripts directory: {slurm_dir}")

    # Generate Slurm scripts
    generate_slurm_scripts(method, work_dir, slurm_dir)

if __name__ == '__main__':
    main()