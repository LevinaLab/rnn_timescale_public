import os
import pickle

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


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


def _forget_gate_to_tau(forget_gate):
    return 1 / (1 - forget_gate)


for curriculum_type in ['single', 'cumulative']:
    fig, axs = plt.subplots(1, 2, figsize=(6, 4), constrained_layout=True)
    sns.despine()
    fig.suptitle(f'Curriculum type: {curriculum_type}')
    for network_number in range(1, 5):
        n_max_range, taus_net_all, taus_forget_all = _load_saved_tau_for_n_max_range(
            base_path='../../results',
            network_number=network_number,
            n_max_range=np.arange(2, 100, 5),
            curriculum_type=curriculum_type,
        )
        if len(n_max_range) == 0:
            continue
        taus_net = np.nanmean(taus_net_all, axis=1)
        taus_forget = np.nanmean(taus_forget_all, axis=(1, 2))
        taus_forget = _forget_gate_to_tau(taus_forget)
        axs[0].plot(n_max_range, taus_net, color=f'C{network_number}', label=f'{network_number}')
        axs[1].plot(n_max_range, taus_forget, color=f'C{network_number}', label=f'{network_number}')
    axs[0].set_xlabel('N')
    axs[0].set_ylabel(r'$\tau_{\mathrm{net}}$')
    # axs[0].legend(title="Network", frameon=False)
    axs[1].set_xlabel('N')
    axs[1].set_ylabel(r'$\tau_{\mathrm{forget}}$')
    axs[1].legend(title="Network", frameon=False, loc='upper left', bbox_to_anchor=(1.05, 1.0))
    fig.show()
