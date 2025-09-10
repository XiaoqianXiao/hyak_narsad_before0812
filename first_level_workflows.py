#Nipype v1.10.0.
from nipype.pipeline import engine as pe
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces import fsl, utility as niu, io as nio
from niworkflows.interfaces.bids import DerivativesDataSink as BIDSDerivatives
from utils import _dict_ds
from utils import _dict_ds_lss
from utils import _bids2nipypeinfo
from utils import _bids2nipypeinfo_lss
from nipype.interfaces.fsl import SUSAN, ApplyMask, FLIRT, FILMGLS, Level1Design, FEATModel


class DerivativesDataSink(BIDSDerivatives):
    out_path_base = 'firstLevel'


DATA_ITEMS = ['bold', 'mask', 'events', 'regressors', 'tr']


def first_level_wf(in_files, output_dir, fwhm=6.0, brightness_threshold=1000):
    workflow = pe.Workflow(name='wf_1st_level')
    workflow.config['execution']['use_relative_paths'] = True
    workflow.config['execution']['remove_unnecessary_outputs'] = False

    datasource = pe.Node(niu.Function(function=_dict_ds, output_names=DATA_ITEMS),
                         name='datasource')
    datasource.inputs.in_dict = in_files
    datasource.iterables = ('sub', sorted(in_files.keys()))

    # Extract motion parameters from regressors file
    runinfo = pe.Node(niu.Function(
        input_names=['in_file', 'events_file', 'regressors_file', 'regressors_names'],
        function=_bids2nipypeinfo, output_names=['info', 'realign_file']),
        name='runinfo')

    # Set the column names to be used from the confounds file
    runinfo.inputs.regressors_names = ['dvars', 'framewise_displacement'] + \
                                      ['a_comp_cor_%02d' % i for i in range(6)] + \
                                      ['cosine%02d' % i for i in range(4)]

    # Mask
    apply_mask = pe.Node(ApplyMask(), name='apply_mask')
    # SUSAN smoothing
    susan = pe.Node(SUSAN(), name='susan')
    susan.inputs.fwhm = fwhm
    susan.inputs.brightness_threshold = brightness_threshold

    l1_spec = pe.Node(SpecifyModel(
        parameter_source='FSL',
        input_units='secs',
        high_pass_filter_cutoff=100
    ), name='l1_spec')

    # l1_model creates a first-level model design
    l1_model = pe.Node(Level1Design(
        bases={'dgamma': {'derivs': True}},
        model_serial_correlations=True,
        contrasts=[('CS+_safe>CS-', 'T', ['CSS_first_half', 'CSS_second_half', 'CS-_first_half', 'CS-_second_half'],
                    [0.5, 0.5, -0.5, -0.5]),
                   ('CS+_safe<CS-', 'T', ['CSS_first_half', 'CSS_second_half', 'CS-_first_half', 'CS-_second_half'],
                    [-0.5, -0.5, 0.5, 0.5]),
                   ('CS+_reinf>CS-', 'T', ['CSR_first_half', 'CSR_second_half', 'CS-_first_half', 'CS-_second_half'],
                    [0.5, 0.5, -0.5, -0.5]),
                   ('CS+_reinf<CS-', 'T', ['CSR_first_half', 'CSR_second_half', 'CS-_first_half', 'CS-_second_half'],
                    [-0.5, -0.5, 0.5, 0.5]),
                   ('CS+_safe>CS+_reinf', 'T',
                    ['CSS_first_half', 'CSS_second_half', 'CSR_first_half', 'CSR_second_half'], [0.5, 0.5, -0.5, -0.5]),
                   ('CS+_safe<CS+_reinf', 'T',
                    ['CSS_first_half', 'CSS_second_half', 'CSR_first_half', 'CSR_second_half'], [-0.5, -0.5, 0.5, 0.5]),
                   ('CS->FIXATION', 'T',
                    ['CS-_first_half', 'CS-_second_half', 'FIXATION_first_half', 'FIXATION_second_half'],
                    [0.5, 0.5, -0.5, -0.5]),
                   ('CS+_safe>FIXATION', 'T',
                    ['CSS_first_half', 'CSS_second_half', 'FIXATION_first_half', 'FIXATION_second_half'],
                    [0.5, 0.5, -0.5, -0.5]),
                   ('CS+_reinf>FIXATION', 'T',
                    ['CSR_first_half', 'CSR_second_half', 'FIXATION_first_half', 'FIXATION_second_half'],
                    [0.5, 0.5, -0.5, -0.5]),
                   ('first_half_CS+_safe>first_half_CS+_reinf', 'T', ['CSS_first_half', 'CSR_first_half'], [1, -1]),
                   ('first_half_CS+_safe<first_half_CS+_reinf', 'T', ['CSS_first_half', 'CSR_first_half'], [-1, 1]),
                   ('first_half_CS+_safe>CS-', 'T', ['CSS_first_half', 'CS-_first_half'], [1, -1]),
                   ('first_half_CS+_safe<CS-', 'T', ['CSS_first_half', 'CS-_first_half'], [-1, 1]),
                   ('first_half_CS+_reinf>CS-', 'T', ['CSR_first_half', 'CS-_first_half'], [1, -1]),
                   ('first_half_CS+_reinf<CS-', 'T', ['CSR_first_half', 'CS-_first_half'], [-1, 1]),
                   ('first_half_CS+_safe>FIXATION', 'T', ['CSS_first_half', 'FIXATION_first_half'], [1, -1]),
                   ('first_half_CS+_reinf>FIXATION', 'T', ['CSR_first_half', 'FIXATION_first_half'], [1, -1]),
                   ('second_half_CS+_safe>second_half_CS+_reinf', 'T', ['CSS_second_half', 'CSR_second_half'], [1, -1]),
                   ('second_half_CS+_safe<second_half_CS+_reinf', 'T', ['CSS_second_half', 'CSR_second_half'], [-1, 1]),
                   ('second_half_CS+_safe>CS-', 'T', ['CSS_second_half', 'CS-_second_half'], [1, -1]),
                   ('second_half_CS+_safe<CS-', 'T', ['CSS_second_half', 'CS-_second_half'], [-1, 1]),
                   ('second_half_CS+_reinf>CS-', 'T', ['CSR_second_half', 'CS-_second_half'], [1, -1]),
                   ('second_half_CS+_reinf<CS-', 'T', ['CSR_second_half', 'CS-_second_half'], [-1, 1]),
                   ('second_half_CS+_safe>FIXATION', 'T', ['CSS_second_half', 'FIXATION_second_half'], [1, -1]),
                   ('second_half_CS+_reinf>FIXATION', 'T', ['CSR_second_half', 'FIXATION_second_half'], [1, -1]),
                   ('first_half_CS+_safe>second_half_CS+_safe', 'T', ['CSS_first_half', 'CSS_second_half'], [1, -1]),
                   ('first_half_CS+_safe<second_half_CS+_safe', 'T', ['CSS_first_half', 'CSS_second_half'], [-1, 1]),
                   ('first_half_CS+_reinf>second_half_CS+_reinf', 'T', ['CSR_first_half', 'CSR_second_half'], [1, -1]),
                   ('first_half_CS+_reinf<second_half_CS+_reinf', 'T', ['CSR_first_half', 'CSR_second_half'], [-1, 1]),
                   ('early>later_CS+_safe>CS+_reinf', 'T',
                    ['CSS_first_half', 'CSS_second_half', 'CSR_first_half', 'CSR_second_half'], [0.5, -0.5, -0.5, 0.5]),
                   ('early<later_CS+_safe>CS+_reinf', 'T',
                    ['CSS_first_half', 'CSS_second_half', 'CSR_first_half', 'CSR_second_half'], [-0.5, 0.5, 0.5, -0.5])
                   ],
    ), name='l1_model')

    # feat_spec generates an fsf model specification file
    feat_spec = pe.Node(FEATModel(), name='feat_spec')
    # feat_fit actually runs FEAT
    feat_fit = pe.Node(FILMGLS(smooth_autocorr=True, mask_size=5), name='feat_fit', mem_gb=12)
    feat_select = pe.Node(nio.SelectFiles({
        **{f'cope{i}': f'cope{i}.nii.gz' for i in range(1, 32)},
        **{f'varcope{i}': f'varcope{i}.nii.gz' for i in range(1, 32)}
    }), name='feat_select')

    ds_copes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False, desc=f'cope{i}'),
            name=f'ds_cope{i}', run_without_submitting=True)
        for i in range(1, 32)
    ]

    ds_varcopes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False, desc=f'varcope{i}'),
            name=f'ds_varcope{i}', run_without_submitting=True)
        for i in range(1, 32)
    ]

    workflow.connect([
        (datasource, apply_mask, [('bold', 'in_file'),
                                  ('mask', 'mask_file')]),
        (apply_mask, susan, [('out_file', 'in_file')]),
        (datasource, runinfo, [
            ('events', 'events_file'),
            ('regressors', 'regressors_file')]),
        *[
            (datasource, ds_copes[i - 1], [('bold', 'source_file')])
            for i in range(1, 32)
        ],
        *[
            (datasource, ds_varcopes[i - 1], [('bold', 'source_file')])
            for i in range(1, 32)
        ],
        (susan, l1_spec, [('smoothed_file', 'functional_runs')]),
        (datasource, l1_spec, [('tr', 'time_repetition')]),
        (datasource, l1_model, [('tr', 'interscan_interval')]),
        (susan, runinfo, [('smoothed_file', 'in_file')]),
        (runinfo, l1_spec, [
            ('info', 'subject_info'),
            ('realign_file', 'realignment_parameters')]),
        (l1_spec, l1_model, [('session_info', 'session_info')]),
        (l1_model, feat_spec, [
            ('fsf_files', 'fsf_file'),
            ('ev_files', 'ev_files')]),
        # --- Corrected connections for FILMGLS ---
        (feat_spec, feat_fit, [
            ('design_file', 'design_file'),
            ('con_file', 'tcon_file')]),
        (susan, feat_fit, [('smoothed_file', 'in_file')]),
        (feat_fit, feat_select, [('results_dir', 'base_directory')]),
        *[
            (feat_select, ds_copes[i - 1], [(f'cope{i}', 'in_file')])
            for i in range(1, 32)
        ],
        *[
            (feat_select, ds_varcopes[i - 1], [(f'varcope{i}', 'in_file')])
            for i in range(1, 32)
        ],
    ])
    return workflow


DATA_ITEMS_LSS = ['bold', 'mask', 'events', 'regressors', 'tr', 'trial_ID']



def first_level_wf_LSS(in_files, output_dir, trial_ID, fwhm=6.0, brightness_threshold=1000):
    workflow = pe.Workflow(name='wf_1st_level_LSS')
    workflow.config['execution']['use_relative_paths'] = True
    workflow.config['execution']['remove_unnecessary_outputs'] = False
    import numpy as np

    datasource = pe.Node(niu.Function(function=_dict_ds_lss, output_names=DATA_ITEMS_LSS),
                         name='datasource')
    datasource.inputs.in_dict = in_files
    datasource.iterables = ('sub', sorted(in_files.keys()))


    # Extract motion parameters from regressors file
    runinfo = pe.Node(niu.Function(
        input_names=['in_file', 'events_file', 'regressors_file',
                     'trial_ID', 'regressors_names', 'motion_columns',
                     'decimals', 'amplitude'],
        output_names=['info', 'realign_file'],
        function=_bids2nipypeinfo_lss),
        name='runinfo')

    # Set the column names to be used from the confounds file
    runinfo.inputs.regressors_names = ['dvars', 'framewise_displacement'] + \
                                      ['a_comp_cor_%02d' % i for i in range(6)] + \
                                      ['cosine%02d' % i for i in range(4)]

    # Mask
    apply_mask = pe.Node(ApplyMask(), name='apply_mask')

    l1_spec = pe.Node(SpecifyModel(
        parameter_source='FSL',
        input_units='secs',
        high_pass_filter_cutoff=100
    ), name='l1_spec')

    # l1_model creates a first-level model design
    l1_model = pe.Node(Level1Design(
        bases={'dgamma': {'derivs': True}},  # Or {'dgamma': {'derivs': False}} if no derivatives
        model_serial_correlations=True,
        contrasts=[
            ('trial>others', 'T', ['trial', 'others'], [1, -1]),
            ('trial>baseline', 'T', ['trial'], [1])
        ]
    ), name='l1_model')

    # feat_spec generates an fsf model specification file
    feat_spec = pe.Node(FEATModel(), name='feat_spec')

    # feat_fit actually runs FEAT
    feat_fit = pe.Node(FILMGLS(smooth_autocorr=True, mask_size=5), name='feat_fit', mem_gb=12)
    feat_select = pe.Node(nio.SelectFiles({
        **{f'cope{i}': f'cope{i}.nii.gz' for i in range(1, 3)},
        **{f'varcope{i}': f'varcope{i}.nii.gz' for i in range(1, 3)}
    }), name='feat_select')

    ds_copes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False,
            desc=f'trial{int(trial_ID)}_cope{i}'),
            name=f'ds_cope{i}',
            run_without_submitting=True)
        for i in range(1, 3)
    ]

    ds_varcopes = [
        pe.Node(DerivativesDataSink(
            base_directory=str(output_dir), keep_dtype=False,
            desc=f'trial{int(trial_ID)}_varcope{i}'),
            name=f'ds_varcope{i}',
            run_without_submitting=True)
        for i in range(1, 3)
    ]

    workflow.connect([
        (datasource, apply_mask, [('bold', 'in_file'),
                                  ('mask', 'mask_file')]),
        (datasource, runinfo, [
            ('events', 'events_file'),
            ('trial_ID', 'trial_ID'),
            ('regressors', 'regressors_file')]),
        *[
            (datasource, ds_copes[i - 1], [('bold', 'source_file')])
            for i in range(1, 3)
        ],
        *[
            (datasource, ds_varcopes[i - 1], [('bold', 'source_file')])
            for i in range(1, 3)
        ],
        (apply_mask, l1_spec, [('out_file', 'functional_runs')]),
        (datasource, l1_spec, [('tr', 'time_repetition')]),
        (datasource, l1_model, [('tr', 'interscan_interval')]),
        (apply_mask, runinfo, [('out_file', 'in_file')]),
        (runinfo, l1_spec, [
            ('info', 'subject_info'),
            ('realign_file', 'realignment_parameters')]),
        (l1_spec, l1_model, [('session_info', 'session_info')]),
        (l1_model, feat_spec, [
            ('fsf_files', 'fsf_file'),
            ('ev_files', 'ev_files')]),
        # --- Corrected connections for FILMGLS ---
        (feat_spec, feat_fit, [
            ('design_file', 'design_file'),
            ('con_file', 'tcon_file')]),
        (apply_mask, feat_fit, [('out_file', 'in_file')]),
        (feat_fit, feat_select, [('results_dir', 'base_directory')]),
        *[
            (feat_select, ds_copes[i - 1], [(f'cope{i}', 'in_file')])
            for i in range(1, 3)
        ],
        *[
            (feat_select, ds_varcopes[i - 1], [(f'varcope{i}', 'in_file')])
            for i in range(1, 3)
        ]
    ])
    return workflow
