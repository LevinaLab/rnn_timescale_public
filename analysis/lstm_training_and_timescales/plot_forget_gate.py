import os
import pickle

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from analysis.lstm_training_and_timescales.lstm_utils import load_lstm


def _load_saved_tau(base_path, network_number, n_max, curriculum_type):
    """
    Load the saved taus for the given network number and N_max.
    """
    # Load the saved taus
    with open(
        os.path.join(
            base_path,
            f'lstm_{curriculum_type}_network_{network_number}_N{n_max}_acs_taus.pkl',
        ),
        'rb',
    ) as f:
        data = pickle.load(f)
    taus_forget = data['forget_gates_mean']
    return taus_forget


def _get_forget_gate_bias(base_path, network_number, n_max, curriculum_type):
    """
    Load the saved taus for the given network number and n_max_range.
    """
    lstm = load_lstm(
        base_path,
        n_max,
        network_number,
        curriculum_type,
        n_min=2,
    )
    bias_forget = lstm.lstm.bias_ih_l0[1 * lstm.hidden_size:2 * lstm.hidden_size].detach().numpy()
    bias_forget += lstm.lstm.bias_hh_l0[1 * lstm.hidden_size:2 * lstm.hidden_size].detach().numpy()
    return bias_forget


def _forget_gate_to_tau(forget_gate):
    return 1 / (1 - forget_gate)


def _forget_gate_bias_to_tau(forget_gate_bias):
    return _forget_gate_to_tau(
        1 / (1 + np.exp(-forget_gate_bias))
    )


for curriculum_type in ['single', 'cumulative']:
    fig, axs = plt.subplots(1, 2, figsize=(6, 4), constrained_layout=True)
    sns.despine()
    fig.suptitle(f'Curriculum type: {curriculum_type}')
    for network_number in range(1, 5):
        n_max_range = np.arange(2, 100)  #, 5)
        tau_forget_all = []
        n_max_tau_forget = []
        for n_max in n_max_range:
            try:
                tau_forget_all += [_load_saved_tau(
                    base_path='../../results',
                    network_number=network_number,
                    n_max=n_max,
                    curriculum_type=curriculum_type,
                ).mean()]
                n_max_tau_forget += [n_max]
            except FileNotFoundError as e:
                continue
        tau_forget_all = [_forget_gate_to_tau(b) for b in tau_forget_all]

        bias_forget_all = []
        n_max_bias_forget = []
        for n_max in n_max_range:
            try:
                bias_forget_all += [_get_forget_gate_bias(
                    base_path='../../trained_models',
                    network_number=network_number,
                    n_max=n_max,
                    curriculum_type=curriculum_type,
                ).mean()]
                n_max_bias_forget += [n_max]
            except FileNotFoundError as e:
                continue
        bias_forget_all = [_forget_gate_bias_to_tau(b) for b in bias_forget_all]

        axs[0].plot(n_max_tau_forget, tau_forget_all, color=f'C{network_number}', label=f'{network_number}')
        axs[1].plot(n_max_bias_forget, bias_forget_all, color=f'C{network_number}', label=f'{network_number}')
    axs[0].set_xlabel('N_max')
    axs[0].set_ylabel('Tau from simulation')
    axs[1].set_xlabel('N_max')
    axs[1].set_ylabel('Tau from bias')
    axs[1].legend(title="Network", frameon=False, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.show()

