import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.utils import load_model
from src.utils.plot import set_plot_params

set_plot_params()
color_single = '#D48A6A'
color_multi = '#489174'
model_path = '../../trained_models/continuous'
fig_path = '../../fig'


cm = 1 / 2.54
fig = plt.figure(figsize=(18*cm, 9*cm), layout='constrained')
subfigs = fig.subfigures(2, 1)
fig_top = subfigs[0]
fig_bottom = subfigs[1]
axs_top = fig_top.subplots(1, 2)
axs_bottom = fig_bottom.subplots(1, 3, sharey='row')
for ax in axs_top:
    sns.despine(ax=ax)
for ax in axs_bottom:
    sns.despine(ax=ax)
# letters on each plot
for i, ax in enumerate(np.concatenate([axs_top, axs_bottom])):
    ax.text(-0.1, 1.08, f'{chr(97 + i)}', color='k', fontsize=11, weight='bold', transform=ax.transAxes)


###############################################################################
# plot mean and std of tau
def _get_mean_std_tau(
    N_max, duplicates, network_number, task='parity', curriculum_type='cumulative', nonlinearity='leakyrelu', tau=None, all_tau=False
):
    affixes = [f'duplicates{duplicates}']
    if tau is not None:
        affixes += [f'tau{tau}']
    affixes += ['mod', nonlinearity]
    rnn = load_model(
        curriculum_type=curriculum_type,
        task=task,
        network_number=network_number,
        N_max=N_max,
        N_min=N_max if curriculum_type == 'single' else 2,
        device='cpu',
        base_path=model_path,
        continuous_model=True,
        affixes=affixes,
    )
    taus = rnn.taus[0].detach().numpy()
    if all_tau:
        return np.mean(taus), np.std(taus), taus
    else:
        return np.mean(taus), np.std(taus)


def _get_mean_std_tau_list(
    duplicates, network_number, task='parity', curriculum_type='cumulative', nonlinearity='leakyrelu', tau=None
):
    N_list = list(range(2, 101))
    tau_mean = np.zeros(len(N_list))
    tau_std = np.zeros(len(N_list))
    for i_N, N in enumerate(N_list):
        try:
            tau_mean[i_N], tau_std[i_N] = _get_mean_std_tau(
                N_max=N, duplicates=duplicates, network_number=network_number, task=task, tau=tau, curriculum_type=curriculum_type, nonlinearity=nonlinearity
            )
        except FileNotFoundError:
            break
    return N_list[:i_N], tau_mean[:i_N], tau_std[:i_N]


ax = axs_top[0]
N_list_multi, tau_mean_multi, tau_std_multi = _get_mean_std_tau_list(
    duplicates=[7, 8, 9, 10], network_number=99, curriculum_type='cumulative', nonlinearity='leakyrelu'
)
N_list_single, tau_mean_single, tau_std_single = _get_mean_std_tau_list(
    duplicates=[7, 8, 9, 10], network_number=99, curriculum_type='single', nonlinearity='leakyrelu'
)
ax.plot(N_list_single, tau_mean_single, label='single-head', color=color_single)
ax.plot(N_list_multi, tau_mean_multi, label='multi-head', color=color_multi)
ax.axhline(1, color='grey', linestyle='--')
ax.set_yticks([1, 2])
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'Mean($\tau$)')

ax = axs_top[1]
ax.plot(N_list_single, tau_std_single, label='single-head', color=color_single)
ax.plot(N_list_multi, tau_std_multi, label='multi-head', color=color_multi)
ax.legend(frameon=False, loc='upper left')
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'STD($\tau$)')


##############################################################################
# plot accuracy vs Delta t
def _get_max_N(basepath, network_name):
    # list all files in folder
    files = os.listdir(os.path.join(basepath, network_name))
    # only those that contain two 'N'
    files = [f for f in files if f.count('N') == 2]
    # extract Ns, splitting on second 'N'
    Ns = [int(f.split('N')[2]) for f in files]
    # return max N
    return max(Ns)


result_path = '../../results'
network_list = [
    "single_parity_duplicates[7, 8, 9, 10]_mod_leakyrelu_network_99",
    "cumulative_parity_duplicates[7, 8, 9, 10]_mod_leakyrelu_network_99",
    "cumulative_parity_duplicates[10]_mod_leakyrelu_network_1",
]
N_max_all = max([_get_max_N(model_path, network) for network in network_list])
viridis = plt.cm.get_cmap('viridis', N_max_all - 3).reversed()
for ax, network in zip(axs_bottom, network_list):
    with open(os.path.join(result_path, f'accuracy_{network}.pkl'), 'rb') as f:
        accuracies = pickle.load(f)
    N_max = _get_max_N(model_path, network)
    Ns = np.arange(2, N_max + 1) if 'cumulative' in network else np.arange(N_max, N_max + 1)

    for i_N, N in enumerate(Ns):
        # color continuously based on N
        ax.plot(
            [1/k for k in list(accuracies.keys())],
            [accuracies[k][i_N] for k in accuracies.keys()],
            # label=f'N = {N}',
            color=viridis(int(N - 2)),
        )
    ax.set_xscale('log')
    ax.set_xlabel(r'$\Delta t$')

# colorbar based on viridis with N
ax = axs_bottom[2]
sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=2, vmax=N_max))
sm._A = []
cax = inset_axes(ax, width="5%", height="100%", loc='lower right', bbox_to_anchor=(0.1, 0.45, 1, 1.7), bbox_transform=ax.transAxes)
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
cbar.set_label('N')


ax = axs_bottom[0]
ax.set_ylabel(r'Accuracy (%)')
ax.axvspan(
    1/10,
    1/7,
    alpha=0.5,
    color='red',
    label='training range',
)
ax = axs_bottom[1]
ax.axvspan(
    1/10,
    1/7,
    alpha=0.5,
    color='red',
    label='training range',
)
ax = axs_bottom[2]
ax.axvspan(
    1/10,
    1/10,
    alpha=0.5,
    color='red',
    label='training range',
)
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.6, 1.15))

fig.show()
fig.savefig(os.path.join(fig_path, 'continuous_RNN.pdf'))
