import os
import shutil
from bids.layout import BIDSLayout
import pandas as pd
from nipype import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataSink
from group_level_workflows import data_prepare_wf
from templateflow.api import get as tpl_get, templates as get_tpl_list

# Set FSL environment
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'  # Matches the Docker image
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Define directories
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
derivatives_dir = os.path.join(data_dir, 'derivatives')
results_dir = os.path.join(derivatives_dir, 'fMRI_analysis/groupLevel')
scrubbed_dir = '/scrubbed_dir'
workflow_dir = os.path.join(scrubbed_dir, project_name, f'work_flows/groupLevel')
container_path = "/gscratch/scrubbed/fanglab/xiaoqian/images/narsad-fmri_pre_group_level_1.0.sif"

for d in [workflow_dir, results_dir]:
    os.makedirs(d, exist_ok=True)

# Define standard reference image (e.g., MNI152 template from FSL)
group_mask = str(tpl_get('MNI152NLin2009cAsym', resolution=2, desc='brain', suffix='mask'))

sub_no_MRI_phase2 = ['N102', 'N208']
sub_no_MRI_phase3 = ['N102', 'N208', 'N120']

SCR_dir = os.path.join(root_dir, project_name, 'EDR')
drug_file = os.path.join(SCR_dir, 'drug_order.csv')
ECR_file = os.path.join(SCR_dir, 'ECR.csv')

# Load behavioral data
df_drug = pd.read_csv(drug_file)
df_drug['group'] = df_drug['subID'].apply(lambda x: 'Patients' if x.startswith('N1') else 'Controls')
df_ECR = pd.read_csv(ECR_file)
df_behav = df_drug.merge(df_ECR, how='left', left_on='subID', right_on='subID')

# Map groups and drugs
group_levels = df_behav['group'].unique()
drug_levels = df_behav['drug_condition'].unique()
guess_levels = df_behav['guess'].unique()
group_map = {level: idx + 1 for idx, level in enumerate(group_levels)}
drug_map = {level: idx + 1 for idx, level in enumerate(drug_levels)}
guess_map = {level: idx + 1 for idx, level in enumerate(guess_levels)}
df_behav['group_id'] = df_behav['group'].map(group_map)
df_behav['drug_id'] = df_behav['drug_condition'].map(drug_map)
#df_behav['guess_id'] = df_behav['guess'].map(guess_map)

# Load first-level data
firstlevel_dir = os.path.join(derivatives_dir, 'fMRI_analysis/firstLevel')
glayout = BIDSLayout(firstlevel_dir, validate=False, config=['bids', 'derivatives'])
sub_list = sorted(glayout.get_subjects())

contr_list = list(range(1, 32))
tasks = ['phase2', 'phase3']


def collect_task_data(task, contrast, sub_list):
    copes, varcopes = [], []
    for sub in sub_list:
        cope_file = glayout.get(subject=sub, task=task, desc=f'cope{contrast}',
                                extension=['.nii', '.nii.gz'], return_type='file')
        varcope_file = glayout.get(subject=sub, task=task, desc=f'varcope{contrast}',
                                   extension=['.nii', '.nii.gz'], return_type='file')
        if cope_file and varcope_file:
            copes.append(cope_file[0])
            varcopes.append(varcope_file[0])
        else:
            print(f"Missing files for task-{task}, sub-{sub}, cope{contrast}")
    return copes, varcopes


use_guess = False
if __name__ == "__main__":
    for task in tasks:
        if task == 'phase2':
            sub_no_MRI = sub_no_MRI_phase2
        else:
            sub_no_MRI = sub_no_MRI_phase3
        group_info_df = df_behav.loc[df_behav['subID'].isin(sub_list) & ~df_behav['subID'].isin(sub_no_MRI)]
        group_info = list(group_info_df[['subID', 'group_id', 'drug_id']].itertuples(index=False, name=None))
        expected_subjects = len(group_info)
        task_results_dir = os.path.join(results_dir, f'task-{task}')
        task_workflow_dir = os.path.join(workflow_dir, f'task-{task}')
        os.makedirs(task_results_dir, exist_ok=True)
        if os.path.exists(task_workflow_dir):
            shutil.rmtree(task_workflow_dir)  # Clear previous workflow directory to avoid caching
        os.makedirs(task_workflow_dir, exist_ok=True)

        for contrast in contr_list:
            contrast_results_dir = os.path.join(task_results_dir, f'cope{contrast}')
            contrast_workflow_dir = os.path.join(task_workflow_dir, f'cope{contrast}')
            os.makedirs(contrast_results_dir, exist_ok=True)
            os.makedirs(contrast_workflow_dir, exist_ok=True)

            copes, varcopes = collect_task_data(
                task, contrast,[info[0] for info in group_info])

            if len(copes) != expected_subjects or len(varcopes) != expected_subjects:
                print(f"Skipping contrast {contrast}: Expected {expected_subjects} subjects, "
                      f"got copes={len(copes)}, varcopes={len(varcopes)}")
                continue

            # Run Data Preparation Workflow
            prepare_wf = data_prepare_wf(output_dir=contrast_results_dir,
                                         contrast=contrast,
                                         name=f"data_prepare_{task}_cope{contrast}")
            prepare_wf.base_dir = contrast_workflow_dir
            prepare_wf.inputs.inputnode.in_copes = copes
            prepare_wf.inputs.inputnode.in_varcopes = varcopes
            prepare_wf.inputs.inputnode.group_info = group_info
            prepare_wf.inputs.inputnode.result_dir = contrast_results_dir
            prepare_wf.inputs.inputnode.group_mask = group_mask

            print(f"Running Data Preparation for task-{task}, contrast-{contrast}")
            prepare_wf.run(plugin='MultiProc', plugin_args={'n_procs': 4})
            print(f"Completed Data Preparation for task-{task}, contrast-{contrast}")

            # Clean up intermediate directories
            intermediate_dirs = [os.path.join(contrast_workflow_dir, node) for node in
                                 ['merge_copes', 'merge_varcopes', 'resample_copes', 'resample_varcopes']]
            for d in intermediate_dirs:
                if os.path.exists(d):
                    shutil.rmtree(d)
