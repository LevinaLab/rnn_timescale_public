import numpy as np
import itertools

import scipy.stats as stats
from scipy.optimize import curve_fit
import os

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

def comp_ac_fft(data):
    """Compute auto-correlations from binned data (without normalization).
    Uses FFT after zero-padding the time-series in the right side.

    Parameters
    -----------
    data : nd array
        time-series of activity (numTrials * #timesteps).
        
    Returns
    -------
    ac : 1d array
        average non-normalized auto-correlation across all trials.
    """
    n = np.shape(data)[1]
    xp = data - data.mean(1)[:,None]
    xp = np.concatenate((xp,  np.zeros_like(xp)), axis = 1)
    f = np.fft.fft(xp)
    p = np.absolute(f)**2
    pi = np.fft.ifft(p)
    ac_all = np.real(pi)[:, :n-1]/np.arange(1,n)[::-1]
    ac = np.mean(ac_all, axis = 0)  
    return ac

def double_exp(time, a, tau1, tau2, coeff):
    """a double expoenetial decay function.

    Parameters
    -----------
    time : 1d array
        time points.
    a : float
        amplitude of autocorrelation at lag 0. 
    tau1 : float
       first timescale.
    tau2 : float
       second timescale.
    coeff: float
        weight of the first timescale between [0,1]
    
    
    Returns
    -------
    exp_func : 1d array
        double expoenetial decay function.
    """
    exp_func = a * (coeff) * np.exp(-time/tau1) + a * (1-coeff) * np.exp(-time/tau2)
    return  exp_func

def single_exp(time, a, tau):
    """a single expoenetial decay function.

    Parameters
    -----------
    time : 1d array
        time points.
    a : float
        amplitude of autocorrelation at lag 0. 
    tau : float
       timescale.
    
    
    Returns
    -------
    exp_func : 1d array
        single expoenetial decay function.
    """
    exp_func = a * np.exp(-time/tau) 
    return exp_func


def model_comp(ac, lags, min_lag, max_lag):
    """
    Doing model comparison between
    1) single exponential decay, 2) double expoenential decay and returning the slowest timescale from the best model
    
    
    Parameters
    -----------
    ac : 1d array
        autocorrelation.
    lags : 1d array
        time lags.
    min_lag : int
       minimum time lag index.
    max_lag: int
        maximum time lag index
    
    
    Returns
    -------
    selected_model : int
        id of selected model, [1,2].
    selected_tau: float
        slowest timescale from the selected model.
        
    """
    
    xdata = lags[min_lag:max_lag+1]
    ydata = ac[min_lag:max_lag+1]/ac[0]
    
    # 1) fitting single exponential
    try:
        popt_1, pcov = curve_fit(single_exp, xdata, ydata, maxfev = 2000, bounds=((0,0),(1., 100)))
        yfit = single_exp(xdata, *popt_1)

        RSS =((yfit - ydata)**2).sum()
        n = len(xdata) # number of samples
        k = 2 # number of parameters
        AIC_1 = 2*k + n*np.log(RSS)        
    except Exception:
                AIC_1 = 10**5
                pass
    
    # 2) fitting double exponential
    try:
        popt_2, pcov = curve_fit(double_exp, xdata, ydata, maxfev = 2000, bounds=((0,0,0,0),(1., 100, 100, 1.0)))
        yfit = double_exp(xdata, *popt_2)

        RSS =((yfit - ydata)**2).sum()
        n = len(xdata) # number of samples
        k = 4 # number of parameters
        AIC_2 = 2*k + n*np.log(RSS)
    except Exception:
                AIC_2 = 10**5
                pass

    
    # compare models using AIC
    if (AIC_1 < AIC_2):
        selected_tau = popt_1[1]
        selected_model = 1
    elif (AIC_2 < AIC_1):
        selected_tau = np.sort([popt_2[1], popt_2[2]])
        selected_model = 2
    else:
        selected_tau = np.nan
        selected_model = np.nan
    return selected_model, selected_tau




        
def comp_acs(data_path, save_path, curriculum_type, task, network_number, N_max_range, T, num_neurons, num_trials, max_lag, fit_lag, burn_T):
    """ Loads the network for each N, 
    simulates it for T time-steps, computes single-neuron and population activity autocorrelations,
    estimates timescales and saves the results in a pickle file. 
    
    The saved file includes a dictionary contaning:
    'ac_pop': AC of population activty 
    'ac_all': AC of individual neurons
    'taus_net': estimated network-mediated timescale for each neuron (they can be NaN if no model could fit AC)
    'taus_trained': trianed single-neuron timescale for each neuron 
    'selected_models': whether the single (1) or double (2) exponential fit better explained the neuron ACs
    'max_fit_lag': maxumum time lag used for fitting ACs
    'duration': duration of simulated time-series
    'trials': number of simulated trials
    


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
    max_lag: int
         maximum time lag for saving ACs.
    fit_lag: int
        maximum time-lag for fitting ACs (we choose a small number to avoid AC bias)
    burn_T: int
        burn-in time at the beginning of each simulation to reach stationary state.
    
    """


    
    min_lag = 0
    lags = np.arange(0, max_lag + 1)
    
    for i, N in enumerate(N_max_range):
        
        ac_all_single = np.zeros((num_neurons, max_lag))
        selected_model_all = np.zeros(num_neurons)
        tau_net_all = np.zeros(num_neurons)
        
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
        
       
        # compute population AC
        print('Computing population AC')
        pop_activity = np.transpose(np.sum(data_all, axis = 2))
        pop_ac =  comp_ac_fft(pop_activity)
        ac_pop = pop_ac[0:max_lag]
        
        # compute single-neuron AC and estimate network-mediated timescales
        print('Computing single-neuron AC')
        for j in tqdm(range(num_neurons)):
            ac_sum = 0
            data = np.transpose(data_all[:,:, j])
            ac = comp_ac_fft(data)
            ac_sum = ac_sum + ac[0:max_lag]
            ac_all_single[j,:] = ac_sum/(num_trials)
            
            ac_fit = ac_all_single[j,:]
            xdata = lags[min_lag:fit_lag+1]
            ydata = ac_fit[min_lag:fit_lag+1]/ac_fit[0]

            # estimating AC timescales
            selected_model_all[j], selected_tau = model_comp(ac_fit, lags, min_lag, fit_lag)
            if selected_model_all[j] == 1:
                tau_net_all[j] = selected_tau
            elif selected_model_all[j] == 2:
                tau_net_all[j] = selected_tau[1]
            else:
                tau_net_all[j] = np.nan
            
            
        ## making a dictionay and saving as a pickle object
        model_name = os.path.join(
        f'{curriculum_type}_{task}_network_{network_number}')
        save_data = {'ac_pop': ac_pop, 'ac_all': ac_all_single, 'taus_net': tau_net_all,'selected_models':selected_model_all, 'taus_trained': trained_taus, 'max_fit_lag': fit_lag, 'duration': T-burn_T, 'trials': num_trials}
        with open(save_path + model_name +'_N'+str(N) + '_acs_taus.pkl', 'wb') as f:
            pickle.dump(save_data, f)

        print('------------')
            