import os
import re
import numpy as np
import nibabel as nib
import argparse
from bids.layout import BIDSLayout
from nipype import config, logging

# =======================
# FSL and Nipype settings
# =======================
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

plugin_settings = {
    'plugin': 'MultiProc',
    'plugin_args': {
        'n_procs': 4,
        'raise_insufficient': False,
        'maxtasksperchild': 1
    }
}

config.set('execution', 'remove_unnecessary_outputs', 'false')
logging.update_logging(config)

# ===============
# Project settings
# ===============
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
derivatives_dir = os.path.join(data_dir, 'derivatives')
behav_dir = os.path.join(data_dir, 'source_data', 'behav')
output_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS')
LSS_dir = os.path.join(output_dir, 'firstLevel')
results_dir = os.path.join(LSS_dir, 'all_subjects')

scrubbed_dir = '/scrubbed_dir'
space = 'MNI152NLin2009cAsym'
cope = '_cope1'

# =========================
# Core processing function
# =========================
def generate_4d_bold(sub, task):
    layout = BIDSLayout(str(output_dir), validate=False, derivatives=str(LSS_dir))

    query = {
        'extension': ['.nii', '.nii.gz'],
        'desc': r'trial.*',
        'suffix': 'bold',
        'subject': sub,
        'task': task,
        'space': space
    }

    all_bold_files = layout.get(**query, regex_search=True)
    filtered_files = [f for f in all_bold_files if cope in f.filename]
    bold_files = filtered_files

    if not bold_files:
        raise FileNotFoundError(f"No BOLD file found for subject {sub}, task {task}")

    def extract_trial_num(f):
        filename = f.filename if hasattr(f, 'filename') else str(f)
        m = re.search(r'desc-trial(\d+)', filename)
        return int(m.group(1)) if m else float('inf')

    sorted_files = sorted(bold_files, key=extract_trial_num)

    imgs = [nib.load(f.path) for f in sorted_files]
    data_arrays = [img.get_fdata() for img in imgs]
    shapes = [data.shape for data in data_arrays]

    if len(set(shapes)) > 1:
        raise ValueError(f"Input volumes have different shapes: {shapes}")

    data_4d = np.stack(data_arrays, axis=-1)
    affine = imgs[0].affine
    header = imgs[0].header.copy()
    img_4d = nib.Nifti1Image(data_4d, affine, header)

    out_fname = f"sub-{sub}_task-{task}.nii"
    out_path = os.path.join(results_dir, out_fname)
    nib.save(img_4d, out_path)
    print(f"Saved 4D NIfTI for subject {sub}, task {task} → {out_path}")

# =========================
# Command-line interface
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', required=True)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    generate_4d_bold(args.subject, args.task)