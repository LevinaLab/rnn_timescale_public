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


def make_binary_data_growing(model, M, BATCH_SIZE, NET_SIZE, device):
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
    # np.array([h_n.cpu().numpy() for h_n in h_ns])
    data = []  # shape: [time, module, batch, neurons]
    # save_dict = {}  # for when/if sina get it wrong
    for h_n_t in h_ns:  # h_ns[ h_n_t=0, h_n_t=1, ...]
        data.append([])
        for d, h_n_d in enumerate(h_n_t):  # h_n_t[ h_n_depth=0, h_n_depth=1, ...]
            data[-1].append(h_n_d.cpu().numpy())  # todo: includes all depths even those that haven't been trained yet
            # save_dict[str(d).zfill(2)] = h_n_d
    # data = np.array(data).reshape(M, BATCH_SIZE, -1)  # todo: what does this do exactly?
    data = np.array(data)
    return data


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
    # np.array([h_n.cpu().numpy() for h_n in h_ns])
    data = []
    save_dict = {}  # for when/if sina get it wrong
    for h_n_t in h_ns: # h_ns[ h_n_t=0, h_n_t=1, ...]
        data.append([])
        for d, h_n_d in enumerate(h_n_t):  # h_n_t[ h_n_depth=0, h_n_depth=1, ...]
            data[-1].append(h_n_d.cpu().numpy())
            save_dict[str(d).zfill(2)] = h_n_d
    data = np.array(data).reshape(M, BATCH_SIZE, -1)
    return data

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


def comp_effective_autocorr(data, level):
    """

    Parameters
    ----------
    data
    level
    time_steps
    num_modules
    num_neurons
    batch_size

    Returns
    -------

    """
    # Assumes data is of shape [0:time, 1:modules, 2:batch, 3:neurons]
    # want: (batch, num_modules, num_neurons, time_steps)
    # collapse: (batch x num_modules x num_neurons, time_steps)
    time_steps = data.shape[0]
    batch_size = data.shape[2]
    data_t = data.transpose(2, 1, 3, 0)  # (batch_size, num_modules,  num_neurons, time_steps)
    if level == 'network':
        data_ready = np.sum(data_t, axis=[1, 2])  # (batch_size, time_steps)
        acs = comp_ac_fft(data_ready)
    elif level == 'single_neuron':
        # data_ready = data_t.reshape(-1, time_steps)  # (batch_size x num_modules x num_neurons, time_steps)
        data_ready = data_t.reshape(batch_size, -1, time_steps)  # (batch_size x num_modules x num_neurons, time_steps)
        acs = [comp_ac_fft(data_ready[:, j, :] for j in range(data_ready.shape[1]))]
    elif level == 'module':
        data_t_s = np.sum(data_t, axis=2)  # ( batch_size, num_modules, time_steps)
        # data_ready = data_t_s.reshape(-1, time_steps)  # (batch_size x num_modules, time_steps)
        acs = [comp_ac_fft(data_t_s[:, j, :]) for j in range(data_t_s.shape[1])]

    return acs


def comp_acs_growing(rnn, save_path, N_max_max,
                     T, num_neurons, num_trials, max_lag,
                     fit_lag, burn_T, affixes=[]):
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
    device: str
        'cuda' or 'cpu'
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
    strict: boolean
        aurgument for loading models (depends on python version)
    mod_model: boolean
        'modified model or default model', if true, tau is outside non-linearity.
    mod_afunc:
        type of non-linearity

    """

    min_lag = 0
    lags = np.arange(0, max_lag + 1)

    # for i, N in enumerate(N_max_range):  # iterates through the modules in the hierarchy
    ac_all_single = np.zeros((num_neurons * (N_max_max - 1), max_lag))
    selected_model_all = np.zeros(num_neurons * (N_max_max - 1))
    tau_net_all = np.zeros(num_neurons * (N_max_max - 1))

    N_min = 2

    # rnn.current_depth = N - 1  # todo: in future this parameter will be loaded when we hydrate the object from torch dumps.
    # trained_taus = rnn.taus[0].detach().numpy() # trained taus
    # trained_taus = [rnn.taus[f'{k}'].detach().cpu().numpy() for k in range(i + 1)]
    trained_taus = [rnn.taus[f'{N}'].detach().cpu().numpy() for N in range(rnn.current_depth)]

    # simulating the model activity using random binary inputs
    returned_data = make_binary_data_growing(rnn, T, num_trials, [num_neurons], device='cpu')
    # todo: for me before the `reshape(M, BATCH_SIZE, -1)` I have [time, module, batch, neurons]
    data_all = returned_data[burn_T:, :, :, :]  # time * depth * trials * neurons ,
    time_steps = data_all.shape[0]
    data_reshaped = data_all.reshape(time_steps, num_trials, -1)
    # todo: note how the reshape will be different for when we do size scheduling.
    # compute population AC
    print('Computing population AC')
    pop_activity = np.transpose(np.sum(data_reshaped, axis=2))
    pop_ac = comp_ac_fft(pop_activity)
    ac_pop = pop_ac[0:max_lag]

    # compute module-based population AC
    # print('Computing module-based population AC')
    # pop_activity_module = np.transpose(np.sum(data_all, axis=3).reshape(time_steps, -1))  # todo: understand why it's not reshape(T, num_trials, -1)
    # pop_ac_module = comp_ac_fft(pop_activity_module)
    # # compute single-neuron AC and estimate network-mediated timescales
    # print('Computing single-neuron AC')
    # todo: currently iterates through the neurons in ALL modules, should just iterate through
    #  the neurons in the current module
    for j in tqdm(range(num_neurons * (N - 1))):
        ac_sum = 0
        data = np.transpose(data_all[:, :, j])
        ac = comp_ac_fft(data)
        ac_sum = ac_sum + ac[0:max_lag]
        ac_all_single[j, :] = ac_sum / (num_trials)

        ac_fit = ac_all_single[j, :]
        xdata = lags[min_lag:fit_lag + 1]
        ydata = ac_fit[min_lag:fit_lag + 1] / ac_fit[0]

        # estimating AC timescales
        selected_model_all[j], selected_tau = model_comp(ac_fit, lags, min_lag, fit_lag)
        if selected_model_all[j] == 1:
            tau_net_all[j] = selected_tau
        elif selected_model_all[j] == 2:
            tau_net_all[j] = selected_tau[1]
        else:
            tau_net_all[j] = np.nan

    affix_str = '_'
    if len(affixes) > 0:
        affix_str += '_'.join(affixes) + '_'

    # ## making a dictionay and saving as a pickle object
    # model_name = os.path.join(
    #     f'{curriculum_type}_{task}{affix_str}network_{network_number}')
    # save_data = {'ac_pop': ac_pop, 'ac_all': ac_all_single, 'taus_net': tau_net_all,
    #              'selected_models': selected_model_all, 'taus_trained': trained_taus, 'max_fit_lag': fit_lag,
    #              'duration': T - burn_T, 'trials': num_trials}
    # with open(save_path + model_name + '_N' + str(N) + '_acs_taus.pkl', 'wb') as f:
    #     pickle.dump(save_data, f)

    print('------------')


def comp_acs(load_function, load_func_kwargs, save_path, curriculum_type, task, network_number,
             N_max_range, T, num_neurons, num_trials, max_lag, fit_lag, burn_T, affixes=[]):
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
    device: str
        'cuda' or 'cpu'
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
    strict: boolean
        aurgument for loading models (depends on python version)
    mod_model: boolean
        'modified model or default model', if true, tau is outside non-linearity.
    mod_afunc: 
        type of non-linearity
    
    """

    
    min_lag = 0
    lags = np.arange(0, max_lag + 1)
    
    for i, N in enumerate(N_max_range):
        ac_all_single = np.zeros((num_neurons * (N - 1), max_lag))
        selected_model_all = np.zeros(num_neurons * (N - 1))
        tau_net_all = np.zeros(num_neurons * (N - 1))
        
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
        load_func_kwargs['N'] = N
        # rnn = load(curriculum_type = curriculum_type, task = task, network_number = network_number, N_max = N, N_min = N_min, device=device, base_path = data_path, strict = strict, mod_model = mod_model, mod_afunc = mod_afunc, affixes = affixes)
        rnn = load_function(**load_func_kwargs)
        # rnn.current_depth = N - 1  # todo: in future this parameter will be loaded when we hydrate the object from torch dumps.
        # trained_taus = rnn.taus[0].detach().numpy() # trained taus
        trained_taus = [rnn.taus[f'{k}'].detach().cpu().numpy() for k in range(i + 1)]

        # simulating the model activity using random binary inputs
        returned_data = make_binary_data(rnn, T, num_trials, [num_neurons])
        data_all = returned_data[burn_T:,:,:]  # time * trials * neurons
        
       
        # compute population AC
        print('Computing population AC')
        pop_activity = np.transpose(np.sum(data_all, axis = 2))
        pop_ac =  comp_ac_fft(pop_activity)
        ac_pop = pop_ac[0:max_lag]
        
        # compute single-neuron AC and estimate network-mediated timescales
        print('Computing single-neuron AC')
        for j in tqdm(range(num_neurons * (N - 1))):
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
            
        
        affix_str = '_'
        if len(affixes) > 0:
            affix_str += '_'.join(affixes) + '_'
        
        ## making a dictionay and saving as a pickle object
        model_name = os.path.join(
        f'{curriculum_type}_{task}{affix_str}network_{network_number}')
        save_data = {'ac_pop': ac_pop, 'ac_all': ac_all_single, 'taus_net': tau_net_all,'selected_models':selected_model_all, 'taus_trained': trained_taus, 'max_fit_lag': fit_lag, 'duration': T-burn_T, 'trials': num_trials}
        with open(save_path + model_name +'_N'+str(N) + '_acs_taus.pkl', 'wb') as f:
            pickle.dump(save_data, f)

        print('------------')
