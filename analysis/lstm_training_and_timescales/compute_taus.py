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
import pickle

import torch
import numpy as np

from tqdm import tqdm

#---------------- setting the parallel threads on CPU for numpy and torch
import os

from analysis.lstm_training_and_timescales.lstm_utils import load_lstm, simulate_lstm_binary
from analysis.timescales.timescales_utils import make_binary_data, comp_ac_fft, model_comp

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2" 
os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2"
torch.set_num_threads(10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#---------------- data and network params

# data_path = '../../trained_models/single_nocurr/N5' # path to trained networks
save_path = '../../results/' # path for saving the results

# curriculum_type = 'single'
# task = 'nocurr_parity_mod_relu'
# network_number = 0
# N_max_max = 5
# N_max_range = np.arange(N_max_max, N_max_max+1, 1) # range of maximum Ns

num_neurons = 500

# -------------- AC computation params
burn_T = 500 # Burn-in time at the beginning of each simulation to reach stationary state
T = 5*10**4 + 500 + burn_T # number of time steps for simulations
num_trials = 10 # number of simulated trials
max_lag = 200 # maximum time lag for saving ACs
fit_lag = 30  # maximum time-lag for fitting ACs (we choose a small number to avoid AC bias)


def _comp_acs_lstm(base_path, N_max_range, network_number, curriculum_type):
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

        ac_all_single = np.zeros((num_neurons, max_lag))
        selected_model_all = np.zeros(num_neurons)
        tau_net_all = np.zeros(num_neurons)

        # loading the model
        print('N = ', N)
        lstm = load_lstm(base_path, N, network_number, curriculum_type, n_min=2)

        # trained_taus = rnn.taus[0].detach().numpy()  # trained taus

        # simulating the model activity using random binary inputs, time x trials x neurons
        print('Simulating LSTM on binary inputs')
        save_dict = simulate_lstm_binary(
            lstm,
            T,
            num_trials,
            device='cpu',
        )
        # print('Done simulating LSTM on binary inputs')
        # data_all = save_dict['l00'][burn_T:, :, :]  # time * trials * neurons
        # print("save_dict['forget_gates']", save_dict['forget_gates'].shape)
        # print("save_dict['cell_states']", save_dict['cell_states'].shape)
        forget_gates = save_dict['forget_gates'][:, burn_T:, :]  # trials * time * neurons
        tau_from_forget_gate = forget_gates.mean(axis=1)  # trials * neurons
        del forget_gates
        cell_states = save_dict['cell_states'][:, burn_T:, :]  # trials * time * neurons
        del save_dict
        # print("tau_from_forget_gate", tau_from_forget_gate.shape)
        # print("cell_states", cell_states.shape)

        # compute population AC
        print('Computing population AC')
        # pop_activity = np.transpose(np.sum(cell_states, axis=2))
        pop_activity = np.sum(cell_states, axis=2)
        pop_ac = comp_ac_fft(pop_activity)
        ac_pop = pop_ac[0:max_lag]
        del pop_activity, pop_ac

        # compute single-neuron AC and estimate network-mediated timescales
        print('Computing single-neuron AC')
        for j in tqdm(range(num_neurons)):
            ac_sum = 0
            # data = np.transpose(cell_states[:, :, j])
            data = cell_states[:, :, j]
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
        del cell_states
        del ac, ac_sum, ac_fit, xdata, ydata

        # making a dictionay and saving as a pickle object
        model_name = os.path.join(
            f'lstm_{curriculum_type}_network_{network_number}')
        save_data = {
            'ac_pop': ac_pop,
            'ac_all': ac_all_single,
            'taus_net': tau_net_all,
            'selected_models': selected_model_all,
            # 'taus_trained': trained_taus,
            'forget_gates_mean': tau_from_forget_gate,
            'max_fit_lag': fit_lag,
            'duration': T - burn_T,
            'trials': num_trials,
        }
        del ac_pop, ac_all_single, tau_net_all, selected_model_all, tau_from_forget_gate
        with open(save_path + model_name + '_N' + str(N) + '_acs_taus.pkl', 'wb') as f:
            pickle.dump(save_data, f)

        print('------------')


do_test = False
do_cumulative = False

curriculum_type = 'cumulative' if do_cumulative else 'single'

if do_test:
    _comp_acs_lstm(
        base_path='../../trained_models',
        N_max_range=np.arange(20, 21),
        network_number=1,
        curriculum_type='cumulative',
    )
else:
    N_max_range = np.arange(2, 100, 5) # range of maximum Ns
    for network_number in range(1, 5):
        print('network number = ', network_number)
        data_path = '../../trained_models'
        try:
            _comp_acs_lstm(
                base_path=data_path,
                N_max_range=N_max_range,
                network_number=network_number,
                curriculum_type=curriculum_type,
            )
        except FileNotFoundError:
            continue
