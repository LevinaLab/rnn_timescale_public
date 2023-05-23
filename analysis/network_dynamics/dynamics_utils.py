import numpy as np
import itertools

import scipy.stats as stats
from scipy.optimize import curve_fit
import os

from sklearn.decomposition import PCA

import pickle

import torch
from tqdm import tqdm

import sys
sys.path.append('../../')
from src.utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_binary_sequence(M):
    """Generate a binary sequence with Bernoulli probability distribution.

    Parameters
    -----------
    M : int
        
    Returns
    -------
    seq : 1d array
        binary sequence
    """
    seq = (torch.rand(M) < torch.rand(1)) * 1.

    return seq

def make_batch_Nbit_parity(M, bs):
     
    '''
    Generate a binary sequences in a batch with Bernoulli probability distribution.
    
    Parameters
    -----------
    M: int
      seq length
    bs: int
        batch_size
    
    Returns
    -------
    sequences : 2d array
        binary sequences
    '''
    
    with torch.no_grad():
        sequences = [generate_binary_sequence(M).unsqueeze(-1) for i in range(bs)]
    sequences = torch.stack(sequences)
        
    return sequences


def make_binary_data(model, M, BATCH_SIZE, NET_SIZE):
    '''
    Simulate the model with binary inputs and return the activity of RNN units.
    
    Parameters
    -----------
    model: object
       RNN model
    M: int
      seq length
    BATCH_SIZE: int
        batch_size
    NET_SIZE: list
     [num_neurons]
    
    Returns
    -------
    save_dict : dict
        activity of neurons across layers
    '''
    
    # preparing training data and labels
    sequences = make_batch_Nbit_parity(M, BATCH_SIZE)
    sequences = sequences.permute(1, 0, 2).to(device)

    with torch.no_grad():
        h_ns, out_class = model.forward(sequences, savetime=True)

    # dict of {layer_i: array(timerseries)} where timeseries is shape [timesteps, batch_size, num_neurons]
    save_dict = {f'l{str(i_l).zfill(2)}': np.array([h_n[i_l].cpu().numpy() for h_n in h_ns]) for i_l in range(len(NET_SIZE))}

    return save_dict


        
def comp_pca(data_path, save_path, curriculum_type, task, network_number, N_max_range, T, num_neurons, num_trials, max_explained_variance, burn_T):
    """ Loads the network for each N, simulates it for T time-steps, computes PC components, 
    returns explained variance for each components and dimensionality of RNN dynamics.
       

    Parameters
    -----------
    data_path : str
        path to network data with different N.
    save_path : string
        path to save the results (acs, taus, etc)
    curriculum_type: str
        'cumulative', f'sliding_{n_heads}_{n_forget}', 'single'
    task: str
        'parity' or 'dms'
    network_number: int
        ID of the trained network (1, 2, 3,...)
    N_max_range: 1d array
        range of maximum N (N-parity and N-DMS tasks)
    T : int
        number of time steps for simulations. 
    num_neurons : int
       number of neurons in each network.
    num_trials: int
        number of simulated trials of the same network.
    max_explained_variance: float between 0 and 1
        maximum explained variance for computing dimensionality
    burn_T: int
        burn-in time at the beginning of each simulation to reach stationary state.
        
        
    Returns
    -------
    explained_variance_all: 2d-array
        computed explained variance for different PC components (num_trials * num_neurons)
    dimensionality_all: 1d-array
        dimensionality of RNN dynamics given max_explained_variance (num_trials)
    
    """


    
    for i, N in enumerate(N_max_range):
  
        explained_variance_all = np.zeros((num_trials, num_neurons))
        dimensionality_all = np.zeros((num_trials))
        
        # setting N_min
        if curriculum_type == 'cumulative':
            N_min = 2
        elif curriculum_type == 'single':
            N_min = N
        elif 'sliding_' in curriculum_type:
            N_min = N - 10 + 1 # 10 is the number of heads
        else:
            raise ValueError('Unknown curriculum_type.')
               
    
        # loading the model
        print('N = ', N)
        rnn = load_model(curriculum_type = curriculum_type, task = task, network_number = network_number, N_max = N, N_min = N_min, base_path = data_path)
        trained_taus = rnn.taus[0].detach().numpy() # trained taus
            
        # simulating the model activity using random binary inputs
        save_dict = make_binary_data(rnn, T, num_trials, [num_neurons])
        data_all = save_dict['l00'][burn_T:,:,:] # time * trials * neurons   
        
       
        # compute PCA of population activity and determine dimensionality
        max_eigen = [] # collecting maximum of an eigen vector over all trials
        cc_0_sum = 0
        ac_pop_sum = 0
        for t in range(num_trials):
            data = data_all[:, t, :]            
            
            # PCA computations
            pca = PCA()
            data_reduced = pca.fit_transform(data)
            explained_variance_all[t, :] = pca.explained_variance_ratio_
            dimensionality_all[t] = (np.where(np.cumsum(explained_variance_all[t,:])>max_explained_variance)[0][0])
    
    return explained_variance_all, dimensionality_all
            
      
            