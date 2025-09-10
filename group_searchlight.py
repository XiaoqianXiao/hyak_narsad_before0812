import os
import argparse
import numpy as np
import pandas as pd
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataSink
from nipype.interfaces.fsl.model import FLAMEO, SmoothEstimate, Cluster, Randomise
from nipype.interfaces.fsl.utils import Merge, ImageMaths
import logging
from glob import glob

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

def create_design_matrix(subjects):
    """Create a two-group design matrix in VEST format for testing group effects.
    
    Args:
        subjects: List of subject IDs where IDs starting with '1' are patients 
                 and IDs starting with '2' are controls
    """
    n_subjects = len(subjects)
    
    # Create group indicator: 1 for patients (subjects starting with '1'), 0 for controls (subjects starting with '2')
    group_indicator = np.array([1 if sub.startswith('1') else 0 for sub in subjects])
    
    # Create design matrix: [ones, group_indicator]
    # First column: intercept (ones)
    # Second column: group indicator (1=patient, 0=control)
    design = np.column_stack([np.ones(n_subjects), group_indicator])
    
    # Create contrasts for the 4 tests:
    # con1: [0, 1] - patients > controls (patients - controls)
    # con2: [0, -1] - patients < controls (controls - patients) 
    # con3: [1, 1] - mean effect in patients (intercept + group effect)
    # con4: [1, 0] - mean effect in controls (intercept only)
    con1 = np.array([0, 1])   # patients > controls
    con2 = np.array([0, -1])  # patients < controls
    con3 = np.array([1, 1])   # mean effect in patients
    con4 = np.array([1, 0])   # mean effect in controls
    
    return design, [con1, con2, con3, con4]

def save_vest_file(data, filename):
    """Save data in VEST format for FSL."""
    with open(filename, 'w') as f:
        f.write(f"/NumWaves\t{data.shape[1]}\n")
        f.write(f"/NumPoints\t{data.shape[0]}\n")
        f.write("/Matrix\n")
        for row in data:
            f.write("\t".join([str(val) for val in row]) + "\n")

def wf_flameo(output_dir, name="wf_flameo"):
    """Workflow for group-level analysis with FLAMEO and GRF clustering."""
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node
    inputnode = Node(
        IdentityInterface(fields=[
            'cope_files', 'var_cope_files', 'mask_file', 'design_file', 'con_file', 'result_dir'
        ]),
        name='inputnode'
    )

    # Merge cope files across subjects
    merge_copes = Node(
        Merge(dimension='t'),
        name='merge_copes'
    )

    # FLAMEO node
    flameo = Node(
        FLAMEO(run_mode='flame1'),
        name='flameo'
    )

    # Smoothness estimation node
    smoothness = Node(
        SmoothEstimate(),
        name='smoothness'
    )

    # Clustering node with GRF
    clustering = Node(
        Cluster(
            threshold=2.3,
            connectivity=26,
            out_threshold_file=True,
            out_index_file=True,
            out_localmax_txt_file=True,
            pthreshold=0.05
        ),
        name='clustering'
    )

    # Output node
    outputnode = Node(
        IdentityInterface(fields=[
            'zstats', 'cluster_thresh', 'cluster_index', 'cluster_peaks'
        ]),
        name='outputnode'
    )

    # DataSink
    datasink = Node(
        DataSink(base_directory=output_dir),
        name='datasink'
    )

    # Workflow connections
    wf.connect([
        # Merge copes
        (inputnode, merge_copes, [('cope_files', 'in_files')]),
        # FLAMEO inputs
        (merge_copes, flameo, [('merged_file', 'cope_file')]),
        (inputnode, flameo, [
            ('var_cope_files', 'var_cope_file'),
            ('mask_file', 'mask_file'),
            ('design_file', 'design_mat'),
            ('con_file', 't_con_file')
        ]),
        # Smoothness estimation
        (flameo, smoothness, [('zstats', 'zstat_file')]),
        (inputnode, smoothness, [('mask_file', 'mask_file')]),
        # Clustering
        (flameo, clustering, [('zstats', 'in_file')]),
        (smoothness, clustering, [('volume', 'volume'), ('dlh', 'dlh')]),
        # Collect outputs
        (flameo, outputnode, [('zstats', 'zstats')]),
        (clustering, outputnode, [
            ('threshold_file', 'cluster_thresh'),
            ('index_file', 'cluster_index'),
            ('localmax_txt_file', 'cluster_peaks')
        ]),
        # Send to DataSink
        (outputnode, datasink, [
            ('zstats', 'stats.@zstats'),
            ('cluster_thresh', 'cluster_results.@thresh'),
            ('cluster_index', 'cluster_results.@index'),
            ('cluster_peaks', 'cluster_results.@peaks')
        ])
    ])

    return wf

def wf_randomise(output_dir, name="wf_randomise"):
    """Workflow for group-level analysis with Randomise + TFCE."""
    wf = Workflow(name=name, base_dir=output_dir)

    # Input node
    inputnode = Node(
        IdentityInterface(fields=[
            'cope_files', 'mask_file', 'design_file', 'con_file'
        ]),
        name='inputnode'
    )

    # Merge cope files across subjects
    merge_copes = Node(
        Merge(dimension='t'),
        name='merge_copes'
    )

    # Randomise node
    randomise = Node(
        Randomise(
            num_perm=10000,
            tfce=True,
            vox_p_values=True
        ),
        name='randomise'
    )

    # Convert TFCE-corrected p’s to z-scores
    fdr_ztop = Node(
        ImageMaths(op_string='-ztop', suffix='_zstat'),
        name='fdr_ztop'
    )

    # Output node
    outputnode = Node(
        IdentityInterface(fields=[
            'tstat_files', 'tfce_corr_p_files', 'z_thresh_files'
        ]),
        name='outputnode'
    )

    # DataSink
    datasink = Node(
        DataSink(base_directory=output_dir),
        name='datasink'
    )

    # Connections
    wf.connect([
        # Merge copes
        (inputnode, merge_copes, [('cope_files', 'in_files')]),
        # Randomise inputs
        (merge_copes, randomise, [('merged_file', 'in_file')]),
        (inputnode, randomise, [
            ('mask_file', 'mask'),
            ('design_file', 'design_mat'),
            ('con_file', 'tcon')
        ]),
        # Convert p-values to z-scores
        (randomise, fdr_ztop, [('t_corrected_p_files', 'in_file')]),
        # Collect outputs
        (randomise, outputnode, [
            ('tstat_files', 'tstat_files'),
            ('t_corrected_p_files', 'tfce_corr_p_files')
        ]),
        (fdr_ztop, outputnode, [('out_file', 'z_thresh_files')]),
        # Send to DataSink
        (outputnode, datasink, [
            ('tstat_files', 'stats.@tstats'),
            ('tfce_corr_p_files', 'stats.@tfce_p'),
            ('z_thresh_files', 'stats.@zscores')
        ])
    ])

    return wf

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run group-level searchlight analysis for a single map type.')
    parser.add_argument('--map_type', required=True, help='Map type to process (e.g., within-FIXATION, between-FIXATION-CS-)')
    parser.add_argument('--method', choices=['flameo', 'randomise'], default='flameo', help='Analysis method: flameo or randomise')
    args = parser.parse_args()
    map_type = args.map_type
    method = args.method
    logger.info(f"Processing group-level analysis for map type: {map_type}, method: {method}")

    # Paths
    root_dir = os.getenv('DATA_DIR', '/data')
    project_name = 'NARSAD'
    derivatives_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives')
    data_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'firstLevel', 'all_subjects', 'similarity', 'searchlight')

    # Process both tasks
    tasks = ['phase2', 'phase3']
    for task in tasks:
        logger.info(f"Processing task: {task}")
        output_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'groupLevel', 'searchlight', method, task)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory for {task}: {output_dir}")

        # Collect subjects
        subject_files = glob(os.path.join(data_dir, f'sub-*_task-{task}_within-FIXATION.nii.gz'))
        subjects = sorted([os.path.basename(f).split('_')[0].replace('sub-', '') for f in subject_files])
        logger.info(f"Found {len(subjects)} subjects for {task}: {subjects}")

        # Common mask - find the correct session directory
        subject_fmriprep_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives', 'fmriprep', f'sub-{subjects[0]}')
        if not os.path.exists(subject_fmriprep_dir):
            logger.error(f"Subject fmriprep directory not found: {subject_fmriprep_dir}")
            continue
            
        # Find session directories
        session_dirs = [d for d in os.listdir(subject_fmriprep_dir) if d.startswith('ses-')]
        if not session_dirs:
            logger.error(f"No session directories found in {subject_fmriprep_dir}")
            continue
            
        # Use the first session found (you can modify this logic if needed)
        session_name = session_dirs[0]
        mask_file = os.path.join(
            subject_fmriprep_dir, session_name, 'func',
            f'sub-{subjects[0]}_{session_name}_task-{task}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
        )
        logger.info(f"Using mask for {task}: {mask_file}, exists: {os.path.exists(mask_file)}")
        if not os.path.exists(mask_file):
            logger.error(f"Mask file not found for {task}: {mask_file}")
            continue

        # Collect cope files for the specified map type
        cope_files = []
        subjects_with_data = []
        for sub in subjects:
            cope_file = os.path.join(data_dir, f'sub-{sub}_task-{task}_{map_type}.nii.gz')
            if os.path.exists(cope_file):
                cope_files.append(cope_file)
                subjects_with_data.append(sub)
            else:
                logger.warning(f"Cope file missing for sub-{sub}, {map_type}, {task}: {cope_file}")
        if len(cope_files) < 2:
            logger.error(f"Insufficient cope files for {map_type}, {task}: {len(cope_files)} found")
            continue
        logger.info(f"Found {len(cope_files)} cope files for {map_type}, {task}")
        
        # Collect var_cope files only if using FLAMEO
        var_cope_files = []
        if method == 'flameo':
            for sub in subjects_with_data:
                var_cope_file = os.path.join(data_dir, f'sub-{sub}_task-{task}_{map_type}_var.nii.gz')
                if os.path.exists(var_cope_file):
                    var_cope_files.append(var_cope_file)
                else:
                    logger.warning(f"Var_cope file missing for sub-{sub}, {map_type}, {task}: {var_cope_file}")
            if len(var_cope_files) != len(cope_files):
                logger.error(f"Var_cope files missing for FLAMEO: {len(var_cope_files)} vs {len(cope_files)} cope files")
                continue
            logger.info(f"Found {len(var_cope_files)} var_cope files for FLAMEO")

        # Create design matrix and contrasts using only subjects with data
        design, contrasts = create_design_matrix(subjects_with_data)
        design_file = os.path.join(output_dir, f'design_{map_type}.mat')
        con_file = os.path.join(output_dir, f'contrast_{map_type}.con')
        save_vest_file(design, design_file)
        
        # Save all contrasts in VEST format
        contrast_matrix = np.array(contrasts)
        save_vest_file(contrast_matrix, con_file)
        logger.info(f"Created design for {task}: {design_file}, contrasts: {con_file}")
        logger.info(f"Contrasts: 1) patients>controls, 2) patients<controls, 3) mean_effect_patients, 4) mean_effect_controls")
        
        # Log group information
        patients = [s for s in subjects_with_data if s.startswith('1')]
        controls = [s for s in subjects_with_data if s.startswith('2')]
        logger.info(f"Group analysis: {len(patients)} patients vs {len(controls)} controls")
        
        # Check group balance
        if len(patients) < 1 or len(controls) < 1:
            logger.error(f"Insufficient subjects in one or both groups for {map_type}, {task}: {len(patients)} patients, {len(controls)} controls")
            continue

        # Select workflow
        wf_name = f'wf_{method}_{map_type}_{task}'
        if method == 'flameo':
            wf = wf_flameo(output_dir=output_dir, name=wf_name)
        else:  # randomise
            wf = wf_randomise(output_dir=output_dir, name=wf_name)

        # Set inputs
        wf.inputs.inputnode.cope_files = cope_files
        wf.inputs.inputnode.mask_file = mask_file
        wf.inputs.inputnode.design_file = design_file
        wf.inputs.inputnode.con_file = con_file
        if method == 'flameo':
            wf.inputs.inputnode.var_cope_files = var_cope_files
            wf.inputs.inputnode.result_dir = os.path.join(output_dir, map_type)

        # Run workflow
        try:
            logger.info(f"Running {method} workflow for {map_type}, {task}")
            wf.run()
            logger.info(f"Completed group-level analysis for {map_type}, {task} with {method}")
        except Exception as e:
            logger.error(f"Error running {method} workflow for {map_type}, {task}: {e}")

if __name__ == '__main__':
    main()