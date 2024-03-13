"""
How to use:

1. Scroll down to __main__:
2. Modify the slurm_path to the path that contains folders with names '<params_id>_grow_parity_' and
   files 'rnn_1_N<max_depth>' and 'configs.json'
3. Run the script

Results will be saved under the rnn_timescale_public/results folder with a name that contains the slurm_id and the time
analysis was run to avoid overwriting previous analyses.

Results are written to files named: grow_parity__network_<params_id>_N<max_depth>_acs_taus.pkl

Pickled dictionaries have the following content:
{
    'net_acs_taus': {'ac_pop': list of a single list of auto-correlation values,
                     'effective': list of a single dict:
                                 {'ac_all_single': [list of single-neuron auto-correlation values],
                                  'selected_model_all': [list of 1 or 2, depending on which model fit better],
                                  'tau_net_all'}: [list of network-mediated timescales for each neuron]
                                  }
    'mod_acs_taus': {'ac_pop': list of lists, the i'th element being list of auto-correlation values for module `i`,
                     'effective': list of dicts, i'th dict contains results for each module:
                                 {'ac_all_single': [list of single-neuron auto-correlation values],
                                  'selected_model_all': [list of 1 or 2, depending on which model fit better],
                                  'tau_net_all'}: [list of network-mediated timescales for each neuron]
                                  }
    'trained_taus': trained_taus,
    'configs': {
        'max_fit_lag': fit_lag,
        'duration': T - burn_T,
        'trials': num_trials
    }
}

where:
"""
import pickle
from datetime import datetime

from analysis.timescales import compute_taus_growing, timescales_utils
import os
import re


def analyse_model(full_path, save_path):
    rnn, configs = compute_taus_growing.load_and_hydrate_hierarchical_model(full_path)
    print(configs)

    task = 'parity'
    network_number = subdir.split('_')[0]  # the parameter id in the slurm run
    burn_T = 500  # Burn-in time at the beginning of each simulation to reach stationary state
    T = 10 ** 4 + 500 + burn_T  # number of time steps for simulations
    num_trials = 12  # number of simulated trials
    max_lag = 200  # maximum time lag for saving ACs
    fit_lag = 30  # maximum time-lag for fitting ACs (we choose a small number to
    device = 'cpu'

    trained_taus = [rnn.taus[f'{N}'].detach().cpu().numpy() for N in range(rnn.current_depth)]

    print("Generating data...")
    returned_data = timescales_utils.make_binary_data_growing(rnn, M=T,
                                                              BATCH_SIZE=num_trials,
                                                              device=device)

    print("Running FFT...")
    data_all = returned_data[:, :, :, burn_T:]
    # returns: acs, ac_all_single, selected_model_all, tau_net_all
    net_acs_taus = timescales_utils.effective_taus(data_all, 'network', max_lag, fit_lag)
    mod_acs_taus = timescales_utils.effective_taus(data_all, 'module', max_lag, fit_lag)
    ## making a dictionay and saving as a pickle object
    save_data = {'net_acs_taus': net_acs_taus, 'mod_acs_taus': mod_acs_taus,
                 'trained_taus': trained_taus,
                 'configs': {'max_fit_lag': fit_lag, 'duration': T - burn_T, 'trials': num_trials,
                             'T': T, 'burn_T': burn_T, 'max_lag': max_lag, 'network_number': network_number,
                             'task': task, 'N_max_max': N_max_max, 'device': device, 'full_path': full_path}}


    os.makedirs(save_path, exist_ok=True)
    file_name = f'grow_{task}__network_{network_number}_N{N_max_max}_acs_taus.pkl'

    with open(os.path.join(save_path, file_name), 'wb') as f:
        pickle.dump(save_data, f)


def get_runs_sorted_by_max_val(slurm_path):
    paths = compute_taus_growing.resolve_paths(slurm_path)
    print(paths)
    max_vals = {}
    pattern = r'N(\d*)'
    for k, l in paths.items():
        g = [re.search(pattern, v) for v in l]
        max_val = max([int(gg.group(1)) for gg in g if gg is not None])
        max_vals[k] = max_val
    paths_sorted = sorted(max_vals.items(), key=lambda x: x[1], reverse=True)
    return paths_sorted


def extract_slurm_id(text):
    match = re.search(r'SLURM_ARRAY_JOB_ID=(\d+)_', text)

    # Extracting the number
    if match:
        s = match.group(1)
        return s
    else:
        return 'unknown_slurm_id'


def build_save_path(slurm_path):
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(here, '../../')
    slurm_tag = extract_slurm_id(os.path.basename(slurm_path))
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_path = os.path.join(project_root, 'results', slurm_tag + '_' + now)
    return save_path


if __name__ == '__main__':
    slurm_path = r"C:\Users\MAni\PycharmProjects\rnn_timescale_public\trained_models\SLURM_ARRAY_JOB_ID=7681088_Mar-11-2024-16_09_54"

    save_path = build_save_path(slurm_path)
    paths_sorted = get_runs_sorted_by_max_val(slurm_path)
    # print("Running: ", paths_sorted)
    for subdir, N_max_max in paths_sorted:  # todo: allow for running on intermediate snapshots of the model rather than just the max (N_max_max)
        print("Analysing: ", subdir, f'rnn_1_N{N_max_max}')
        full_path = os.path.join(slurm_path, subdir, f'rnn_1_N{N_max_max}')
        analyse_model(full_path, save_path)