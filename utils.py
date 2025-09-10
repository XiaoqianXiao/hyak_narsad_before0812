def _get_tr(in_dict):
    return in_dict.get('RepetitionTime')


def _len(inlist):
    return len(inlist)


def _dof(inlist):
    return len(inlist) - 1


def _neg(val):
    return -val

def _dict_ds(in_dict, sub, order=['bold', 'mask', 'events', 'regressors', 'tr']):
    return tuple([in_dict[sub][k] for k in order])

def _dict_ds_lss(in_dict, sub, order=['bold', 'mask', 'events', 'regressors', 'tr', 'trial_ID']):
    return tuple([in_dict[sub][k] for k in order])

def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch

    # Process the events file with tab separator (fix for BIDS TSV files)
    events = pd.read_csv(events_file, sep='\t')  # Changed from sep=',' to sep='\t'
    print("=== DEBUG: loaded event columns ===")
    print(events.columns.tolist())
    print(events.head())

    bunch_fields = ['onsets', 'durations', 'amplitudes']

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    out_motion = Path('motion.par').resolve()

    regress_data = pd.read_csv(regressors_file, sep='\t')
    np.savetxt(out_motion, regress_data[motion_columns].values, '%g')
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    runinfo = Bunch(
        scans=in_file,
        conditions=list(set(events.trial_type.values)),
        **{k: [] for k in bunch_fields})

    for condition in runinfo.conditions:
        event = events[events.trial_type.str.match(condition)]

        runinfo.onsets.append(np.round(event.onset.values, 3).tolist())
        runinfo.durations.append(np.round(event.duration.values, 3).tolist())
        if 'amplitudes' in events.columns:
            runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
        else:
            runinfo.amplitudes.append([amplitude] * len(event))

    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        try:
            runinfo.regressors = regress_data[regressors_names]
        except KeyError:
            regressors_names = list(set(regressors_names).intersection(
                set(regress_data.columns)))
            runinfo.regressors = regress_data[regressors_names]
        runinfo.regressors = runinfo.regressors.fillna(0.0).values.T.tolist()

    return [runinfo], str(out_motion)


def _bids2nipypeinfo_lss(in_file, events_file, regressors_file,
                          trial_ID,
                          regressors_names=None,
                          motion_columns=None,
                          decimals=3,
                          amplitude=1.0):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    # Load events and regressors
    events = pd.read_csv(events_file)
    print("LOADED EVENTS COLUMNS:", events.columns.tolist())
    print(events.head())
    regress_data = pd.read_csv(regressors_file, sep='\t')

    # Locate the trial of interest by ID
    trial = events[events['trial_ID'] == trial_ID]
    if trial.empty:
        raise ValueError(f"Trial ID {trial_ID} not found in events file.")
    if len(trial) > 1:
        raise ValueError(f"Trial ID {trial_ID} is not unique in events file.")

    other_trials = events[events['trial_ID'] != trial_ID]

    out_motion = Path('motion.par').resolve()
    np.savetxt(out_motion, regress_data[motion_columns].values, '%g')

    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    # Build the subject_info Bunch
    conditions = ['trial', 'others']
    onsets = [
        np.round(trial['onset'].values.tolist(), decimals),
        np.round(other_trials['onset'].values.tolist(), decimals)
    ]
    durations = [
        np.round(trial['duration'].values.tolist(), decimals),
        np.round(other_trials['duration'].values.tolist(), decimals)
    ]
    amplitudes = [
        [amplitude] * len(onsets[0]),
        [amplitude] * len(onsets[1])
    ]

    runinfo = Bunch(
        scans=in_file,
        conditions=conditions,
        onsets=onsets,
        durations=durations,
        amplitudes=amplitudes
    )

    if regressors_names:
        runinfo.regressor_names = regressors_names
        regress_subset = regress_data[regressors_names].fillna(0.0)
        runinfo.regressors = regress_subset.values.T.tolist()

    return [runinfo], str(out_motion)


def print_input_traits(interface_class):
    """
    Print all input traits of a Nipype interface class, with mandatory inputs listed first,
    and then extract any mutually‐exclusive input groups from the interface’s help().

    Parameters:
    - interface_class: A Nipype interface class (e.g., SpecifyModel)
    """
    import io, sys

    # 1) List all traits
    spec = interface_class().inputs
    traits = spec.traits().items()
    sorted_traits = sorted(traits, key=lambda item: not item[1].mandatory)

    print("Name                           | mandatory")
    print("-------------------------------|----------")
    for name, trait in sorted_traits:
        print(f"{name:30} | {trait.mandatory}")

    # 2) Capture help() output to find the "Mutually exclusive inputs" line
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        interface_class().help()
    finally:
        sys.stdout = old_stdout

    help_text = buf.getvalue().splitlines()
    mux_line = next((line for line in help_text if 'mutually_exclusive' in line), None)

    # 3) Parse and print mutually‐exclusive inputs if present
    if mux_line:
        # e.g. "Mutually exclusive inputs: subject_info, event_files, bids_event_file"
        _, fields = mux_line.split(':', 1)
        names = [n.strip() for n in fields.split(',')]
        print("\nMutually exclusive inputs:")
        for n in names:
            print(f"  - {n}")
    else:
        print("\nNo mutually exclusive inputs found in help().")


def print_output_traits(interface_class):
    """
    Print all input traits of a Nipype interface class, with mandatory inputs listed first,
    and then extract any mutually‐exclusive input groups from the interface’s help().

    Parameters:
    - interface_class: A Nipype interface class (e.g., SpecifyModel)
    """
    import io, sys

    # 1) List all traits
    spec = interface_class().output_spec()  # same as inst.output_spec(), but bound
    traits = spec.traits().items()
    sorted_traits = sorted(traits, key=lambda item: not item[1].mandatory)

    print("Name                           | mandatory")
    print("-------------------------------|----------")
    for name, trait in sorted_traits:
        print(f"{name:30} | {trait.mandatory}")





