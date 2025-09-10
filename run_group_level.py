#%%
import os
import argparse
from group_level_workflows import wf_randomise, wf_flameo
from nipype import config, logging
from templateflow.api import get as tpl_get

# Nipype plugin settings
plugin_settings = {
    'plugin': 'MultiProc',
    'plugin_args': {
        'n_procs': 4,
        'raise_insufficient': False,
        'maxtasksperchild': 1,
    }
}

config.set('execution', 'remove_unnecessary_outputs', 'false')
logging.update_logging(config)

# Define directories
root_dir = os.getenv('DATA_DIR', '/data')
project_name = 'NARSAD'
data_dir = os.path.join(root_dir, project_name, 'MRI')
derivatives_dir = os.path.join(data_dir, 'derivatives')
results_dir = os.path.join(derivatives_dir, 'fMRI_analysis/groupLevel')
scrubbed_dir = '/scrubbed_dir'
workflows_dir = os.path.join(scrubbed_dir, project_name, f'work_flows/groupLevel')
def run_group_level_wf(task, contrast, analysis_type, paths):
    wf_func = wf_randomise if analysis_type == 'randomise' else wf_flameo
    wf_name = f"wf_{analysis_type}_{task}_cope{contrast}"

    wf = wf_func(output_dir=paths['result_dir'], name=wf_name)
    wf.base_dir = paths['workflow_dir']
    wf.inputs.inputnode.cope_file = paths['cope_file']
    wf.inputs.inputnode.mask_file = paths['mask_file']
    wf.inputs.inputnode.design_file = paths['design_file']
    wf.inputs.inputnode.con_file = paths['con_file']
    wf.inputs.inputnode.result_dir = paths['result_dir']

    if analysis_type == 'flameo':
        wf.inputs.inputnode.var_cope_file = paths['varcope_file']
        wf.inputs.inputnode.grp_file = paths['grp_file']

    wf.run(**plugin_settings)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--contrast', required=True, type=int)
    parser.add_argument('--analysis_type', default='randomise', choices=['randomise', 'flameo'])
    parser.add_argument('--base_dir', required=True)

    args = parser.parse_args()
    task = args.task
    contrast = args.contrast
    analysis_type = args.analysis_type

    # Use TemplateFlow to get group mask path
    group_mask = str(tpl_get('MNI152NLin2009cAsym', resolution=2, desc='brain', suffix='mask'))

    result_dir = os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'whole_brain')
    workflow_dir = os.path.join(workflows_dir, f'task-{task}', f'cope{contrast}', 'whole_brain')

    paths = {
        'result_dir': result_dir,
        'workflow_dir': workflow_dir,
        'cope_file': os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'merged_cope.nii.gz'),
        'varcope_file': os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'merged_varcope.nii.gz'),
        'design_file': os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'design_files', 'design.mat'),
        'con_file': os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'design_files', 'contrast.con'),
        'grp_file': os.path.join(results_dir, f'task-{task}', f'cope{contrast}', 'design_files', 'design.grp'),
        'mask_file': group_mask
    }

    os.makedirs(paths['result_dir'], exist_ok=True)
    os.makedirs(paths['workflow_dir'], exist_ok=True)

    run_group_level_wf(task, contrast, analysis_type, paths)
