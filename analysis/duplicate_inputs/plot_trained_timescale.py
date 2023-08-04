import numpy as np
import torch
from matplotlib import pyplot as plt

from src.utils import load_model


model_path = '../../trained_models'


def _get_mean_std_tau(
    N_max, duplicate=1,
):
    rnn = load_model(
        curriculum_type='cumulative',
        task='parity',
        network_number=1,
        N_max=N_max,
        N_min=2,
        device='cpu',
        base_path=model_path,
        affixes=[f'duplicate{duplicate}'] if duplicate > 1 else [],
    )
    taus = rnn.taus[0].detach().numpy()

    return np.mean(taus), np.std(taus)

duplicate_list = [1, 2, 3, 5, 10]
N_list = list(range(2, 101))
tau_means = np.zeros((len(duplicate_list), len(N_list)))
tau_stds = np.zeros((len(duplicate_list), len(N_list)))
fig, axs = plt.subplots(ncols=2, constrained_layout=True)
for i_duplicate, duplicate in enumerate(duplicate_list):
    for i_N, N in enumerate(N_list):
        try:
            tau_means[i_duplicate, i_N], tau_stds[i_duplicate, i_N] = _get_mean_std_tau(N, duplicate=duplicate)
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
fig.show()
