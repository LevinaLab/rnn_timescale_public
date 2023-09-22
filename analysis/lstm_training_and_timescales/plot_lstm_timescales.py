"""Figure for ICLR"""
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from src.lstm_utils import load_lstm
from src.utils import set_plot_params

###############################################################################
# SETTINGS                                                                    #
###############################################################################
networks = [51, 52, 53, 54]
###############################################################################

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
    taus_net = data['taus_net']
    taus_forget = data['forget_gates_mean']
    return taus_net, taus_forget


def _load_saved_tau_for_n_max_range(base_path, network_number, n_max_range, curriculum_type):
    """
    Load the saved taus for the given network number and n_max_range.
    """
    taus_net_all = []
    taus_forget_all = []
    for n_max in n_max_range:
        try:
            taus_net, taus_forget = _load_saved_tau(
                base_path=base_path,
                network_number=network_number,
                n_max=n_max,
                curriculum_type=curriculum_type,
            )
        except FileNotFoundError as e:
            break
        taus_net_all.append(taus_net)
        taus_forget_all.append(taus_forget)
    n_max_range_net = n_max_range[:len(taus_net_all)]
    return n_max_range_net, np.array(taus_net_all), np.array(taus_forget_all)


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


def _get_forget_gate_bias_for_n_max_range(base_path, network_number, n_max_range, curriculum_type):
    """
    Load the saved taus for the given network number and n_max_range.
    """
    bias_forget_all = []
    for n_max in n_max_range:
        try:
            bias_forget = _get_forget_gate_bias(
                base_path=base_path,
                network_number=network_number,
                n_max=n_max,
                curriculum_type=curriculum_type,
            )
        except FileNotFoundError as e:
            break
        bias_forget_all.append(bias_forget)
    n_max_range_net = n_max_range[:len(bias_forget_all)]
    return n_max_range_net, np.array(bias_forget_all)


def _forget_gate_to_tau(forget_gate):
    return 1 / (1 - forget_gate)


def _forget_gate_bias_to_tau(forget_gate_bias):
    return _forget_gate_to_tau(
        1 / (1 + np.exp(-forget_gate_bias))
    )


set_plot_params()
cm = 1 / 2.54
fig, axs = plt.subplots(1, 3, figsize=(18 * cm, 6 * cm), constrained_layout=True)
sns.despine()

for curriculum_type in ['single', 'cumulative']:
    color = '#D48A6A' if curriculum_type == 'single' else '#489174'
    for network_number in networks:
        n_max_range, taus_net_all, taus_forget_all = _load_saved_tau_for_n_max_range(
            base_path='../../results',
            network_number=network_number,
            n_max_range=np.arange(2, 100, 5),
            curriculum_type=curriculum_type,
        )
        n_max_range_bias, bias_forget_all = _get_forget_gate_bias_for_n_max_range(
            base_path='../../trained_models',
            network_number=network_number,
            n_max_range=np.arange(2, 100),
            curriculum_type=curriculum_type,
        )
        if len(n_max_range) == 0:
            continue
        taus_net = np.nanmean(taus_net_all, axis=1)
        taus_forget = np.nanmean(taus_forget_all, axis=(1, 2))
        taus_forget = _forget_gate_to_tau(taus_forget)
        axs[0].plot(n_max_range, taus_net, color=color, label=f'{network_number}')
        axs[1].plot(n_max_range, taus_forget, color=color, label=f'{network_number}')
        axs[2].plot(n_max_range_bias, _forget_gate_bias_to_tau(bias_forget_all).mean(axis=1), color=color, label=f'{network_number}')
axs[0].set_xlabel('N')
axs[0].set_ylabel(r'$\tau_{\mathrm{net}}$')
axs[1].set_xlabel('N')
axs[1].set_ylabel(r'$\tau_{\mathrm{forget}}$ simulation')
axs[2].set_xlabel('N')
axs[2].set_ylabel(r'$\tau_{\mathrm{forget}}$ bias')

custom_legend_entries = [
    Line2D([0], [0], color='#D48A6A', lw=2, label='Single-head'),
    Line2D([0], [0], color='#489174', lw=2, label='Multi-head')
]

axs[2].legend(
    handles=custom_legend_entries,
    loc='upper right',
    bbox_to_anchor=(1.1, 1.0),
    frameon=False,
    handlelength=1,
)

fig.show()
