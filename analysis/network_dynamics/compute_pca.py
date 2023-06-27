'''
Script for computing PCA and dimensionality.

It loads the network for each N, simulates it for T time-steps, computes PC components, 
    returns explained variance for each components and dimensionality of RNN dynamics.

'''

import torch
import numpy as np

from dynamics_utils import comp_pca

# setting the parallel threads on CPU for numpy and torch
import os
os.environ["OMP_NUM_THREADS"] = "2" 
os.environ["OPENBLAS_NUM_THREADS"] = "2" 
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2"
torch.set_num_threads(10)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#---------------- data and network params

data_path = '../../trained_models/' # path to trained networks
# save_path = '../../results/' # path for saving the results

curriculum_type = 'cumulative'
task = 'parity'
network_number = 1
N_max_max = 100
N_max_range = np.arange(2, N_max_max+1, 1) # range of maximum Ns

num_neurons = 500

# -------------- Simulation params
burn_T = 500 # Burn-in time at the beginning of each simulation to reach stationary state
T = 10**5 + 500 + burn_T # number of time steps for simulations
num_trials = 10 # number of simulated trials
max_explained_variance = .9


# ------------- PCA computation
explained_variance_all, dimensionality_all = comp_pca(device, data_path, save_path, curriculum_type, task, network_number, N_max_range, T, num_neurons, num_trials, max_explained_variance, burn_T)
