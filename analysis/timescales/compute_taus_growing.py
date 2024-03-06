'''
Script for computing autocorrelations and estimating timescales.

It loads the network for each N, simulates it for T timesteps, computes single-neuron and population activity autocorrelations,
estimates timescales and saves the results in a pickle file.

The saved pickle file includes a dictionary contaning:
    'ac_pop': AC of population activty
    'ac_all': AC of individual neurons
    'taus_net': estimated network-mediated timescale for each neuron (they can be NaN if no model could fit AC)
    'taus_trained': trianed single-neuron timescale for each neuron
    'selected_models': whether the single (1) or double (2) exponential fit better explained the neuron ACs
    'max_fit_lag': maxumum time lag used for fitting ACs
    'duration': duration of simulated time-series
    'trials': number of simulated trials

'''
import json
from collections import defaultdict

import torch
import numpy as np
import os
from tqdm import tqdm

from .timescales_utils import comp_acs
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models import RNN_hier



def get_all_configs_and_stats(slurm_directory):
    file_paths = resolve_paths(slurm_directory)
    all_configs = []
    all_stats = []
    for subdir, files in tqdm(file_paths.items()):
        with open(os.path.abspath(os.path.join(slurm_directory, subdir, 'configs.json')), 'r') as f:
            config_dict = json.load(f)
            config_dict['subdir'] = subdir
            all_configs.append(config_dict)
        stats_path = os.path.abspath(os.path.join(slurm_directory, subdir, 'stats.npy'))
        stats_dict = np.load(stats_path, allow_pickle=True).item()
        stats_dict['subdir'] = subdir
        all_stats.append(stats_dict)

    return all_configs, all_stats

def resolve_paths(slurm_directory):
    data_path = os.path.join(project_root, 'trained_models', slurm_directory)
    # sub directories:
    subdirs = [d for d in tqdm(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, d))]
    file_paths = {subdir: os.listdir(os.path.join(data_path, subdir)) for subdir in subdirs}
    return file_paths


def load_and_hydrate_hierarchical_model(full_path, configs_file_name='configs.json'):
    parent_dir = os.path.dirname(full_path)
    file_path = os.path.join(parent_dir, configs_file_name)
    with open(file_path, 'r') as f:
        configs = json.load(f)
    rnn = init_hierarchical_model(configs)
    rnn.load_state_dict(torch.load(full_path, map_location=torch.device('cpu'))['state_dict'], strict=False)
    return rnn, configs


def init_hierarchical_model(CONFIGS):
    # init new model

    if type(CONFIGS['NET_SIZE']) == int:  # todo: fix this
        NET_SIZE = [CONFIGS['NET_SIZE']] * CONFIGS['MAX_DEPTH']
    else:
        NET_SIZE = CONFIGS['NET_SIZE']

    rnn = RNN_hier.RNN_Hierarchical(max_depth=CONFIGS['MAX_DEPTH'],
                                 input_size=CONFIGS['INPUT_SIZE'],
                                 net_size=NET_SIZE,  # todo: fix
                                 device=CONFIGS['DEVICE'],
                                 num_classes=CONFIGS['NUM_CLASSES'],
                                 bias=CONFIGS['BIAS'],
                                 num_readout_heads_per_mod=CONFIGS['NUM_READOUT_HEADS_PER_MOD'],
                                 fixed_tau_val=1.,
                                 train_tau=CONFIGS['TRAIN_TAU']
                                 )
    return rnn

if __name__ == '__main__':

    # ---------------- setting the parallel threads on CPU for numpy and torch
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    torch.set_num_threads(10)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- data and network params

    data_path = '../../trained_models/'  # path to trained networks
    save_path = '../../results/'  # path for saving the results

    curriculum_type = 'cumulative'
    task = 'parity'
    network_number = 1
    N_max_max = 100
    N_max_range = np.arange(2, N_max_max + 1, 1)  # range of maximum Ns

    num_neurons = 500

    # -------------- AC computation params
    burn_T = 500  # Burn-in time at the beginning of each simulation to reach stationary state
    T = 10 ** 5 + 500 + burn_T  # number of time steps for simulations
    num_trials = 10  # number of simulated trials
    max_lag = 200  # maximum time lag for saving ACs
    fit_lag = 30  # maximum time-lag for fitting ACs (we choose a small number to avoid AC bias)

    # ------------- AC computation and timescale estimation (saves the results in save_path)
    comp_acs(data_path, save_path, curriculum_type, task, network_number, N_max_range, \
             T, num_neurons, num_trials, max_lag, fit_lag, burn_T)
