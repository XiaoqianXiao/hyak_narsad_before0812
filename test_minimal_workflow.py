"""
Minimal test script for first_level_workflows.py

This script tests the basic functionality of the first_level_workflows module
using a small synthetic dataset.
"""
import os
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from first_level_workflows import first_level_wf, first_level_wf_LSS

# Set up test directories
test_dir = Path('test_output')
test_dir.mkdir(exist_ok=True)

# Create a minimal BIDS-like directory structure
bids_root = test_dir / 'bids'
bids_root.mkdir(exist_ok=True)

# Create a test subject
test_sub = 'test01'
sub_dir = bids_root / f'sub-{test_sub}'
func_dir = sub_dir / 'func'
func_dir.mkdir(parents=True, exist_ok=True)

# Create a minimal NIfTI file (4D with small dimensions)
def create_mini_nifti(filename, shape=(10, 10, 10, 5)):
    """Create a minimal NIfTI file with random data."""
    data = np.random.random(shape) * 1000  # Scale to typical BOLD range
    img = nib.Nifti1Image(data, np.eye(4))
    img.to_filename(filename)
    return filename

# Create test files
test_bold = func_dir / 'sub-test01_task-test_bold.nii.gz'
test_mask = func_dir / 'sub-test01_task-test_mask.nii.gz'

# Create minimal files
create_mini_nifti(test_bold)
create_mini_nifti(test_mask, shape=(10, 10, 10))  # 3D mask

# Create minimal events file
events_data = {
    'onset': [1, 5, 9, 13, 17],
    'duration': [2, 2, 2, 2, 2],
    'trial_type': ['CS+_safe', 'CS+_reinf', 'CS-', 'FIXATION', 'CS+_safe'],
    'trial_ID': [1, 2, 3, 4, 5]
}
events_df = pd.DataFrame(events_data)
events_file = func_dir / 'sub-test01_task-test_events.tsv'
events_df.to_csv(events_file, sep='\t', index=False)

# Create minimal confounds file
confounds_data = {
    'trans_x': np.random.randn(50) * 0.1,
    'trans_y': np.random.randn(50) * 0.1,
    'trans_z': np.random.randn(50) * 0.1,
    'rot_x': np.random.randn(50) * 0.01,
    'rot_y': np.random.randn(50) * 0.01,
    'rot_z': np.random.randn(50) * 0.01,
    'dvars': np.random.rand(50) * 1.2,
    'framewise_displacement': np.random.rand(50) * 0.5,
    **{f'a_comp_cor_{i:02d}': np.random.randn(50) for i in range(6)},
    **{f'cosine{i:02d}': np.random.randn(50) for i in range(4)}
}
confounds_df = pd.DataFrame(confounds_data)
confounds_file = func_dir / 'sub-test01_task-test_desc-confounds_regressors.tsv'
confounds_df.to_csv(confounds_file, sep='\\t', index=False)

def test_standard_workflow():
    """Test the standard first-level workflow."""
    print("Testing standard first-level workflow...")
    
    # Set up input dictionary
    in_files = {
        test_sub: {
            'bold': str(test_bold),
            'mask': str(test_mask),
            'events': str(events_file),
            'regressors': str(confounds_file),
            'tr': 2.0
        }
    }
    
    # Create output directory
    output_dir = test_dir / 'output_standard'
    output_dir.mkdir(exist_ok=True)
    
    # Run workflow
    workflow = first_level_wf(
        in_files=in_files,
        output_dir=output_dir,
        fwhm=6.0
    )
    
    # Configure and run the workflow
    workflow.base_dir = str(test_dir / 'work_standard')
    crash_dir = test_dir / 'crash_files_standard'
    crash_dir.mkdir(exist_ok=True)
    workflow.config['execution'] = {
        'use_relative_paths': True,
        'remove_unnecessary_outputs': False,
        'crashdump_dir': str(crash_dir)
    }
    
    # Run with a single process for testing
    workflow.run(plugin='Linear')
    
    print(f"Standard workflow completed. Outputs in: {output_dir}")

def test_lss_workflow():
    """Test the LSS (Least Squares - Single Trial) workflow."""
    print("Testing LSS workflow...")
    
    # Use the first trial for testing
    trial_id = events_df['trial_ID'].iloc[0]
    
    # Set up input dictionary
    in_files = {
        test_sub: {
            'bold': str(test_bold),
            'mask': str(test_mask),
            'events': str(events_file),
            'regressors': str(confounds_file),
            'tr': 2.0,
            'trial_ID': trial_id
        }
    }
    
    # Create output directory
    output_dir = test_dir / 'output_lss'
    output_dir.mkdir(exist_ok=True)
    
    # Run workflow
    workflow = first_level_wf_LSS(
        in_files=in_files,
        output_dir=output_dir,
        trial_ID=trial_id,
        fwhm=6.0
    )
    
    # Configure and run the workflow
    workflow.base_dir = str(test_dir / 'work_lss')
    crash_dir = test_dir / 'crash_files_lss'
    crash_dir.mkdir(exist_ok=True)
    workflow.config['execution'] = {
        'use_relative_paths': True,
        'remove_unnecessary_outputs': False,
        'crashdump_dir': str(crash_dir)
    }
    
    # Run with a single process for testing
    workflow.run(plugin='Linear')
    
    print(f"LSS workflow completed. Outputs in: {output_dir}")

if __name__ == "__main__":
    # Test standard workflow
    test_standard_workflow()
    
    # Test LSS workflow
    test_lss_workflow()
    
    print("All tests completed!")
