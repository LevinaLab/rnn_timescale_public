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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#---------------- data and network params

data_path = '../../trained_models/' # path to trained networks
save_path = '../../results/' # path for saving the results

curriculum_type = 'cumulative'
task = 'parity'
network_number = 1
N_max_max = 100
N_max_range = np.arange(2, N_max_max+1, 1) # range of maximum Ns

num_neurons = 500

# -------------- AC computation params
burn_T = 500 # Burn-in time at the beginning of each simulation to reach stationary state
T = 10**5 + 500 + burn_T # number of time steps for simulations
num_trials = 10 # number of simulated trials
max_lag = 200 # maximum time lag for saving ACs
fit_lag = 30  # maximum time-lag for fitting ACs (we choose a small number to avoid AC bias)


# ------------- AC computation and timescale estimation (saves the results in save_path)
comp_acs(data_path, save_path, curriculum_type, task, network_number, N_max_range,\
         T, num_neurons, num_trials, max_lag, fit_lag, burn_T)
