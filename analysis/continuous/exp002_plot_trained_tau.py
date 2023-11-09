import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

from src.utils import load_model


model_path = '../../trained_models/continuous'

duplicates = [7, 8, 9, 10]
task = 'parity'
curriculum_type = 'cumulative'
nonlinearity = 'leakyrelu'
network_number = 99


def _get_mean_std_tau(
    N_max, duplicates, curriculum_type='cumulative', network_number=1, nonlinearity='leakyrelu', all_tau=False
):
    affixes = [f'duplicates{duplicates}']
    affixes += ['mod', nonlinearity]
    rnn = load_model(
        curriculum_type=curriculum_type,
        task='parity',
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


fig, axs = plt.subplots(1, 2, constrained_layout=True)
sns.despine()
N_list = list(range(2, 101))
tau_mean = np.zeros(len(N_list))
tau_std = np.zeros(len(N_list))
taus = np.zeros((len(N_list), 500))
for i_N, N in enumerate(N_list):
    try:
        tau_mean[i_N], tau_std[i_N], taus[i_N] = _get_mean_std_tau(
            N,
            duplicates=duplicates,
            curriculum_type=curriculum_type,
            network_number=network_number,
            nonlinearity=nonlinearity,
            all_tau=True
        )
    except FileNotFoundError:
        break

segments = [
    np.column_stack([N_list[:i_N], taus[:i_N, i]])
    for i in range(500)
]
lc = LineCollection(
    segments,
    color='k',
    alpha=0.1,
    label='individual tau',
)
axs[0].add_collection(lc)

axs[0].plot(
    N_list[:i_N],
    tau_mean[:i_N],
    color='red',
    label=f'duplicates = {duplicates}',
    linewidth=2,
)
axs[1].plot(
    N_list[:i_N],
    tau_std[:i_N],
    color='red',
    linewidth=2,
)
for ax in axs:
    ax.set_xlabel('N')
    ax.set_ylabel('tau')
axs[0].set_title('mean tau')
axs[1].set_title('std tau')
leg = fig.legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 0.1))
for lh in leg.legendHandles:
    lh.set_alpha(1)


fig.show()
