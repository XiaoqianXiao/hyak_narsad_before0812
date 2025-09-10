# Script to generate SLURM scripts for LSS-based first-level GLM, auto-detecting all trials
import os
import json
import pandas as pd
from bids.layout import BIDSLayout
from first_level_workflows import first_level_wf_LSS
from nipype import config, logging

# Set FSL environment
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Plugin settings
plugin_settings = {
    'plugin': 'MultiProc',
    'plugin_args': {'n_procs': 4, 'raise_insufficient': False, 'maxtasksperchild': 1}
}

config.set('execution', 'remove_unnecessary_outputs', 'false')
logging.update_logging(config)

# Paths
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
derivatives_dir = os.path.join(data_dir, 'derivatives')
behav_dir = os.path.join(data_dir, 'source_data', 'behav')
output_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS')
LSS_dir = os.path.join(output_dir, 'firstLevel')
results_dir = os.path.join(LSS_dir, 'all_subjects')
os.makedirs(results_dir, exist_ok=True)

scrubbed_dir = '/scrubbed_dir'
container_path = "/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif"

space = ['MNI152NLin2009cAsym']
layout = BIDSLayout(str(data_dir), validate=False, derivatives=str(derivatives_dir))

def create_slurm_script(sub, task, work_dir):
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=LSS_2_{sub}
#SBATCH --account=fang
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step2/{task}_sub_{sub}_%j.out
#SBATCH --error=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step2/{task}_sub_{sub}_%j.err

module load apptainer
apptainer exec -B /gscratch/fang:/data -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir {container_path} \\
    python3 /app/LSS_2_cate_all_trials.py --subject {sub} --task {task}
"""
    script_path = os.path.join(work_dir, f'sub_{sub}_slurm.sh')
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    return script_path


if __name__ == '__main__':
    subjects = layout.get_subjects()
    tasks = layout.get_tasks()

    for sub in subjects:
        for task in tasks:
            query = {
                'desc': 'preproc', 'suffix': 'bold', 'extension': ['.nii', '.nii.gz'],
                'subject': sub, 'task': task, 'space': space[0]
            }
            bold_files = layout.get(**query)
            if not bold_files:
                continue

            part = bold_files[0]
            entities = part.entities
            subquery = {k: v for k, v in entities.items() if k in ['subject', 'task', 'run']}

            work_dir = os.path.join(scrubbed_dir, project_name, f'work_flows/Lss_step2/{task}')
            os.makedirs(work_dir, exist_ok=True)

            script_path = create_slurm_script(sub, task, work_dir)
            print(f"Slurm script created: {script_path}")
