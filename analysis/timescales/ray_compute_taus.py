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

import torch
import numpy as np

from timescales_utils import comp_acs

#---------------- setting the parallel threads on CPU for numpy and torch
import os
os.environ["OMP_NUM_THREADS"] = "2" 
os.environ["OPENBLAS_NUM_THREADS"] = "2" 
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2"
torch.set_num_threads(10)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#---------------- data and network params

data_path = '../../trained_models/ALIFE2024' # path to trained networks
save_path = '../../results/ALIFE2024' # path for saving the results

curriculum_type = 'cumulative'
task = 'parity'
num_classes = 2

device = 'cpu'

# -------------- AC computation params
burn_T = 500 # Burn-in time at the beginning of each simulation to reach stationary state
T = 10**5 + 500 + burn_T # number of time steps for simulations
num_trials = 10 # number of simulated trials
max_lag = 200 # maximum time lag for saving ACs
fit_lag = 30  # maximum time-lag for fitting ACs (we choose a small number to avoid AC bias)


import ray
ray.init()

@ray.remote
def ray_comp_acs(*args, **kwargs):
    return comp_acs(*args, **kwargs)

network_numbers = [1, 2, 3, 4, 5]
num_neurons_list = [20, 54, 91, 128]
N_max_maxs = [11, 26, 42, 52]

results = []
for i_n, num_neurons in enumerate(num_neurons_list):
    # Create a new directory for each num_neurons
    new_save_path = os.path.join(save_path, 'size_' + str(num_neurons) + '/')
    os.makedirs(new_save_path, exist_ok=True)

    N_max_max = N_max_maxs[i_n]
    N_max_range = np.arange(2, N_max_max + 1, 1)  # range of maximum Ns

    for network_number in network_numbers:
        affixes = []
        if num_neurons != 500:
            affixes += ['size', str(num_neurons)]
        result = ray_comp_acs.remote(device, data_path, new_save_path, curriculum_type, task, network_number, N_max_range,
                                 T, num_neurons, num_trials, max_lag, fit_lag, burn_T, num_classes, affixes=affixes)
        results.append(result)

# Retrieve the results with ray.get()
results = ray.get(results)