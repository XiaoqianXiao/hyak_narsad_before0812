# run_single_trial.py
import os
import argparse
import pandas as pd
from bids.layout import BIDSLayout
from first_level_workflows import first_level_wf_LSS
from nipype import config, logging

# Set FSL environment
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Nipype plugin settings
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
scrubbed_dir = '/scrubbed_dir'
space = 'MNI152NLin2009cAsym'

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--trial', required=True, type=int)
    args = parser.parse_args()

    sub = args.subject
    task = args.task
    trial_ID = args.trial

    layout = BIDSLayout(str(data_dir), validate=False, derivatives=str(derivatives_dir))

    query = {
        'desc': 'preproc', 'suffix': 'bold', 'extension': ['.nii', '.nii.gz'],
        'subject': sub, 'task': task, 'space': space
    }
    bold_files = layout.get(**query)
    if not bold_files:
        raise FileNotFoundError(f"No BOLD file found for subject {sub}, task {task}")

    part = bold_files[0]
    entities = part.entities
    subquery = {k: v for k, v in entities.items() if k in ['subject', 'task', 'run']}

    bold_file = part.path
    mask_file = layout.get(suffix='mask', extension=['.nii', '.nii.gz'], space=space, **subquery)[0].path
    regressors_file = layout.get(desc='confounds', extension=['.tsv'], **subquery)[0].path
    tr = entities.get('RepetitionTime', 1.5)
    if sub == 'N202' and task == 'phase3':
        events_file = os.path.join(behav_dir, 'single_trial_task-NARSAD_phase-3_sub-202_half_events.csv')
    else:
        events_file = os.path.join(behav_dir, f'single_trial_task-Narsad_{task}_half_events.csv')

    inputs = {
        sub: {
            'bold': bold_file,
            'mask': mask_file,
            'regressors': regressors_file,
            'tr': tr,
            'events': events_file,
            'trial_ID': trial_ID
        }
    }

    work_dir = os.path.join(scrubbed_dir, project_name, f'work_flows/Lss/{task}', f'sub-{sub}', f'trial-{trial_ID}')
    os.makedirs(work_dir, exist_ok=True)

    wf = first_level_wf_LSS(inputs, output_dir, trial_ID)
    wf.base_dir = work_dir
    wf.run(**plugin_settings)
