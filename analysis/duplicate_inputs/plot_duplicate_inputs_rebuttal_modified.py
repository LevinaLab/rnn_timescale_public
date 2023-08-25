import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import seaborn as sns

from src.utils import load_model, set_plot_params

model_path = '../../trained_models'
set_plot_params()


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

taus_from_duplicate = {
    1: [None],
    2: [None, 2],
    3: [None, 3],
    5: [None, 5],
    10: [None, 10],
}
duplicate_list = [1, 2, 3, 5, 10]
N_list = list(range(2, 101))
cm = 1/2.54
fig, ax = plt.subplots(constrained_layout=True, figsize=(9*cm, 5*cm))
sns.despine(fig)
for curriculum_type in ['cumulative', 'single']:
    for i_duplicate, duplicate in enumerate([2, 3, 5, 10]):  # enumerate(duplicate_list):
        for network_number in [1, 2]:
            for tau in [duplicate]:  #  taus_from_duplicate[duplicate]:  # [None, duplicate, duplicate + 0.25, duplicate - 0.25]:
                tau_mean = np.zeros(len(N_list))
                for i_N, N in enumerate(N_list):
                    try:
                        tau_mean[i_N], _ = _get_mean_std_tau(
                            N,
                            duplicate=duplicate,
                            curriculum_type=curriculum_type,
                            network_number=network_number,
                            tau=tau,
                        )
                    except FileNotFoundError:
                        break
                ax.plot(
                    N_list[:i_N],
                    tau_mean[:i_N] / duplicate,
                    color=f'C{i_duplicate}',
                    label=f'{duplicate}' if (network_number == 1 and tau==duplicate and curriculum_type=='cumulative') else None,
                    linestyle='-' if curriculum_type=='cumulative' else '--',
                )
ax.set_xlabel(r'$N$')
ax.set_ylabel(r'$<\tau>$ / duplicate')
ax.legend(title="duplicate", frameon=False, loc='upper left', bbox_to_anchor=(0.05, 1), labelspacing=0.1)

# Create the legend for tau
lines = [Line2D([0], [0], linestyle=ls, color='black') for ls in ['-', '--']]
fig.legend(
    [Line2D([0], [0], linestyle=ls, color='black') for ls in ['-', '--']],
    ['multi-head', 'single-head'],
    title='Curriculum',
    frameon=False,
    loc='upper right',
    bbox_to_anchor=(1, 1),
)

fig.show()
fig.savefig('duplicate_inputs_rebuttal_modified.pdf')
