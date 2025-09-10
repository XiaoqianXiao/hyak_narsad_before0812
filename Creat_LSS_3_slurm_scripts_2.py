import os
import json
import pandas as pd
from bids.layout import BIDSLayout
from nipype import config, logging as nipype_logging
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set FSL environment
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

config.set('execution', 'remove_unnecessary_outputs', 'false')
nipype_logging.update_logging(config)

# Paths
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
derivatives_dir = os.path.join(data_dir, 'derivatives')
behav_dir = os.path.join(data_dir, 'source_data', 'behav')
output_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS')
LSS_dir = os.path.join(output_dir, 'firstLevel')
results_dir = os.path.join(LSS_dir, 'all_subjects')
try:
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Created results directory: {results_dir}")
except Exception as e:
    logger.error(f"Error creating results directory {results_dir}: {e}")

scrubbed_dir = '/scrubbed_dir'
container_path = "/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_1st_level_1.0.sif"
combined_atlas_path = ('/scrubbed_dir/parcellation/Tian/3T/'
                      'Cortex-Subcortex/MNIvolumetric/Schaefer2018_100Parcels_7Networks_order_'
                      'Tian_Subcortex_S1_3T_MNI152NLin2009cAsym_2mm.nii.gz')
roi_names_file = ('/scrubbed_dir/parcellation/Tian/3T/'
                 'Cortex-Subcortex/Schaefer2018_100Parcels_7Networks_order_'
                 'Tian_Subcortex_S1_label.txt')

# Verify input paths
logger.info(f"Data directory: {data_dir}, exists: {os.path.exists(data_dir)}")
logger.info(f"Derivatives directory: {derivatives_dir}, exists: {os.path.exists(derivatives_dir)}")
logger.info(f"Behavioral directory: {behav_dir}, exists: {os.path.exists(behav_dir)}")
logger.info(f"Atlas path: {combined_atlas_path}, exists: {os.path.exists(combined_atlas_path)}")
logger.info(f"ROI names file: {roi_names_file}, exists: {os.path.exists(roi_names_file)}")

space = ['MNI152NLin2009cAsym']
layout = BIDSLayout(str(data_dir), validate=False, derivatives=str(derivatives_dir))
logger.info(f"Initialized BIDSLayout with {len(layout.get_subjects())} subjects and {len(layout.get_tasks())} tasks")

def create_slurm_script(sub, task, work_dir, mask_img_path, combined_atlas_path, roi_names_file):
    logger.info(f"Creating SLURM script for sub-{sub}, task-{task}")
    logger.info(f"Work directory: {work_dir}, mask path: {mask_img_path}")
    log_file = f"/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step3/{task}_sub_{sub}_%j_progress.log"
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=LSS_3_{sub}
#SBATCH --account=fang
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step3/{task}_sub_{sub}_%j.out
#SBATCH --error=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step3/{task}_sub_{sub}_%j.err

module load apptainer
apptainer exec \
    -B /gscratch/fang:/data \
    -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir \
    -B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad/LSS_3_similarity.py:/app/LSS_3_similarity.py \
    -B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad/similarity.py:/app/similarity.py \
    {container_path} \
    python3 /app/LSS_3_similarity.py \
    --subject {sub} \
    --task {task} \
    --mask_img_path {mask_img_path} \
    --combined_atlas_path {combined_atlas_path} \
    --roi_names_file {roi_names_file} \
    --log_file {log_file} \
    > {log_file} 2>&1
"""
    script_path = os.path.join(work_dir, f'sub_{sub}_slurm.sh')
    try:
        with open(script_path, 'w') as f:
            f.write(slurm_script)
        logger.info(f"SLURM script created: {script_path}")
    except Exception as e:
        logger.error(f"Error writing SLURM script {script_path}: {e}")
    return script_path

if __name__ == '__main__':
    subjects = layout.get_subjects()
    tasks = layout.get_tasks()
    logger.info(f"Processing {len(subjects)} subjects: {subjects}")
    logger.info(f"Tasks: {tasks}")

    for sub in subjects:
        for task in tasks:
            query = {
                'desc': 'preproc', 'suffix': 'bold', 'extension': ['.nii', '.nii.gz'],
                'subject': sub, 'task': task, 'space': space[0]
            }
            logger.info(f"Querying BOLD files for sub-{sub}, task-{task}, query={query}")
            bold_files = layout.get(**query)
            if not bold_files:
                logger.warning(f"No BOLD files found for sub-{sub}, task-{task}")
                continue
            logger.info(f"Found {len(bold_files)} BOLD files for sub-{sub}, task-{task}")

            part = bold_files[0]
            entities = part.entities
            subquery = {k: v for k, v in entities.items() if k in ['subject', 'task', 'run']}

            work_dir = os.path.join(scrubbed_dir, project_name, f'work_flows/Lss_step3/{task}')
            try:
                os.makedirs(work_dir, exist_ok=True)
                logger.info(f"Created work directory: {work_dir}")
            except Exception as e:
                logger.error(f"Error creating work directory {work_dir}: {e}")
                continue

            try:
                mask_img_path = layout.get(suffix='mask', return_type='file',
                                           extension=['.nii', '.nii.gz'],
                                           space=query['space'], **subquery)[0]
                logger.info(f"Mask image path: {mask_img_path}, exists: {os.path.exists(mask_img_path)}")
            except IndexError:
                logger.warning(f"No mask file found for sub-{sub}, task-{task}, subquery={subquery}")
                continue

            script_path = create_slurm_script(sub, task, work_dir, mask_img_path, combined_atlas_path, roi_names_file)