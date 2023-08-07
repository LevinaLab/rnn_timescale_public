import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.utils import load_model


model_path = '../../trained_models'


def _get_mean_std_tau(
    N_max, duplicate=1, curriculum_type='cumulative', network_number=1, tau=None
):
    affixes = [f'duplicate{duplicate}'] if duplicate > 1 else []
    if tau is not None:
        affixes.append(f'tau{float(tau)}')
    rnn = load_model(
        curriculum_type=curriculum_type,
        task='parity',
        network_number=network_number,
        N_max=N_max,
        N_min=N_max if curriculum_type == 'single' else 2,
        device='cpu',
        base_path=model_path,
        affixes=affixes,
    )
    taus = rnn.taus[0].detach().numpy()

    return np.mean(taus), np.std(taus)


duplicate_list = [1, 2, 3, 5, 10]
N_list = list(range(2, 101))
for curriculum_type in ['cumulative', 'single']:
    fig, axs = plt.subplots(ncols=2, constrained_layout=True)
    fig.suptitle(f'{curriculum_type}')
    for i_duplicate, duplicate in enumerate(duplicate_list):
        for network_number in [1, 2]:
            for tau in [None, duplicate]:
                tau_mean = np.zeros(len(N_list))
                tau_std = np.zeros(len(N_list))
                for i_N, N in enumerate(N_list):
                    try:
                        tau_mean[i_N], tau_std[i_N] = _get_mean_std_tau(
                            N,
                            duplicate=duplicate,
                            curriculum_type=curriculum_type,
                            network_number=network_number,
                            tau=tau,
                        )
                    except FileNotFoundError:
                        break
                axs[0].plot(
                    N_list[:i_N],
                    tau_mean[:i_N],
                    color=f'C{i_duplicate}',
                    label=f'duplicate = {duplicate}' if (network_number == 1 and tau is None) else None,
                    linestyle='-' if tau is None else '--',
                )
                axs[1].plot(
                    N_list[:i_N],
                    tau_std[:i_N],
                    color=f'C{i_duplicate}',
                    #label=f'duplicate = {duplicate}' if (network_number == 1 and tau is None) else None,
                    linestyle='-' if tau is None else '--',
                )
    for ax in axs:
        ax.set_xlabel('N')
        ax.set_ylabel('tau')
    axs[0].legend(frameon=False, loc='upper right')
    axs[0].set_title('mean tau')
    axs[1].set_title('std tau')

    # Create the legend for tau
    lines = [Line2D([0], [0], linestyle=ls, color='black') for ls in ['-', '--']]
    fig.legend(
        [Line2D([0], [0], linestyle=ls, color='black') for ls in ['-', '--']],
        ['1.5', '=duplicate'],
        title='Initial tau',
        frameon=False,
        loc='lower right',
        bbox_to_anchor=(0.45, 0.5),
    )

    fig.show()

"""duplicate_list = [1, 2, 3, 5, 10]
N_list = list(range(2, 101))
tau_means = np.zeros((len(duplicate_list), len(N_list)))
tau_stds = np.zeros((len(duplicate_list), len(N_list)))
for curriculum_type in ['cumulative', 'single']:
    fig, axs = plt.subplots(ncols=2, constrained_layout=True)
    fig.suptitle(f'{curriculum_type}')
    for i_duplicate, duplicate in enumerate(duplicate_list):
        for i_N, N in enumerate(N_list):
            try:
                tau_means[i_duplicate, i_N], tau_stds[i_duplicate, i_N] = _get_mean_std_tau(
                    N,
                    duplicate=duplicate,
                    curriculum_type=curriculum_type,
                    tau=duplicate if duplicate > 1 else None,
                )
            except FileNotFoundError:
                break
        axs[0].plot(N_list[:i_N], tau_means[i_duplicate, :i_N], label=f'duplicate = {duplicate}')
        axs[1].plot(N_list[:i_N], tau_stds[i_duplicate, :i_N], label=f'duplicate = {duplicate}')
    for ax in axs:
        ax.set_xlabel('N')
        ax.set_ylabel('tau')
        ax.legend()
    axs[0].set_title('mean tau')
    axs[1].set_title('std tau')
    fig.show()"""
