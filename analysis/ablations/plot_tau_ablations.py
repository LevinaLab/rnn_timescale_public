"""Plot the results of the tau ablation experiment. (Figure 7 in the paper)"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import sys
sys.path.append('../../')
from src.utils import set_plot_params


np.random.seed(42000)

df_all = pd.read_pickle('../../results/df_ablate_tau.pkl')

Ns = [5, 30, 35]
network_types = ['single-head', 'sliding', 'cumulative']
fontsize = 10
color_fast = '#AA9C39'
color_slow = '#8C7AAE'  # '#675091'  # '#472E74'

set_plot_params()

df_all['accuracies_relative'] = df_all.apply(
    lambda row: (row['accuracies'] - 0.5) / (np.max(row['baseline_accuracy']) - 0.5),
    axis=1,
)

cm = 1/2.54  # centimeters in inches
fig, axs = plt.subplots(ncols=4, figsize=(18*cm, 5*cm), sharey='row', constrained_layout=True)
for i_N, N in enumerate([5, 30]):
    for i_network_type, network_type in enumerate(['single-head', 'cumulative']):
        # ax = axs[i_N, i_network_type]
        ax = axs[i_N * 2 + i_network_type]
        df = df_all.loc[network_type, [f'network_{i}' for i in range(1, 5)], N]
        if len(df) == 0:
            continue
        accuracies_fast = df['accuracies'].apply(lambda x: x[:20].mean(axis=None))
        accuracies_slow = df['accuracies'].apply(lambda x: x[20:].mean(axis=None))
        if np.sum(accuracies_slow < 0.):
            print(accuracies_slow[accuracies_slow < 0.])
        tau_average_fast = df['tau_ablated'].apply(lambda x: x[:20].mean())
        tau_average_slow = df['tau_ablated'].apply(lambda x: x[20:].mean())
        ax.bar(
            [0, 1],
            [np.mean(accuracies_fast), np.mean(accuracies_slow)],
            yerr=[np.std(accuracies_fast), np.std(accuracies_slow)],
            facecolor='white',
            edgecolor=[color_fast, color_slow],
            linewidth=2,
        )
        sns.stripplot(
            data=[accuracies_fast, accuracies_slow],
            ax=ax,
            palette=[color_fast, color_slow],
            size=6,
            jitter=0.25,
            zorder=1,
            label=['Short', 'Long'],
            linewidth=0.8,
        )
        ax.set_xticks([0, 1], [str(np.round(tau_average_fast.mean(), 2)), str(np.round(tau_average_slow.mean(), 2))])
        ax.set_ylim(0.5, 1.05)
        ax.set_title(f"$N={N}$, {['Single-head', 'Multi-head'][i_network_type]}", fontsize=fontsize)
axs[2].legend(
    labels=['Short', 'Long'], loc='upper right', bbox_to_anchor=(1, 1.1), frameon=False, fontsize=fontsize, handletextpad=-0.3
)
for ax in axs.flatten():
    sns.despine(ax=ax)
axs[0].set_ylabel('Relative accuracy')
# for ax, N in zip(axs[:, 0], [5, 30]):
#    ax.set_ylabel(f"N={N}\n" + ax.get_ylabel())
# for ax, network_type in zip(axs[0, :], ['Single-head', 'Multi-head']):
#    ax.set_title(network_type, fontsize=fontsize)  # fontdict={'fontsize': 10, 'fontweight': 'normal'})
for ax in axs:
    ax.set_xlabel(r'Average $\tau$ of' + '\nablated neurons')
for i_ax, ax in enumerate(axs.flatten()):
    ax.text(-0.10, 1.05, ['a', 'b', 'c', 'd'][i_ax], color='k', fontsize=11, weight='bold', transform=ax.transAxes)
fig.show()
# fig.savefig('fig/exp110_plot_tau_ablations.pdf')
