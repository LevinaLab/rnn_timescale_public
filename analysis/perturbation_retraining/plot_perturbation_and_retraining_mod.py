"""Plot perturbation and retraining results. (Figure 8 in the paper)"""
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

import sys
sys.path.append('../../')
from src.utils import set_plot_params


fontsize = 10
set_plot_params()

color_single = '#D48A6A'
color_multi = '#489174'

network_numbers = list(range(0, 4))

df_rnn = pd.read_pickle('../../results/df_perturb_rnn_normalized_mod.pkl')
df_tau = pd.read_pickle('../../results/df_perturb_tau_normalized_abs_mod.pkl')
for df in [df_rnn, df_tau]:
    # relative accuracy for each row compared to the row where perturbation = 0
    df['accuracy_baseline'] = df.apply(
        lambda row: np.max(df.at[(row.name[0], row.name[1], row.name[2], float(0)), 'accuracy']),
        axis=1,
    )
    df['accuracy_rel'] = df.apply(
        lambda row: (np.array(row['accuracy']) - 0.5) / (row['accuracy_baseline'] - 0.5),
        axis=1,
    )
    # mean and std relative accuracy for each row
    df['accuracy_rel_mean'] = df['accuracy_rel'].apply(np.mean)
    df['accuracy_rel_std'] = df['accuracy_rel'].apply(np.std)
    df.sort_index(inplace=True)
del df

N = 30

cm = 1 / 2.54
fig, axs = plt.subplots(ncols=3, figsize=(18*cm, 5*cm), sharey='all', constrained_layout=True)
for ax, df in zip(axs[:2], [df_rnn, df_tau]):
    for i_network, network in enumerate(network_numbers):
        for i_type, network_type in enumerate(['single', 'cumulative']):
            if network_type == 'single':
                color = color_single
            else:
                color = color_multi
            try:
                df_plot = df.loc[(network_type, network, N), :]
            except KeyError:
                print(f'no results for {network_type}, {network}, {N}')
                continue
            ax.plot(
                df_plot.index.get_level_values('perturbation'),
                df_plot['accuracy_rel_mean'],
                label=['Single-head', 'Multi-head'][i_type] if i_network == 0 else None,
                color=color,
            )
            ax.fill_between(
                df_plot.index.get_level_values('perturbation'),
                df_plot['accuracy_rel_mean'] - df_plot['accuracy_rel_std'],
                df_plot['accuracy_rel_mean'] + df_plot['accuracy_rel_std'],
                color=color,
                alpha=0.2,
            )
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)
    sns.despine(ax=ax)
# axs[1].legend(loc='lower left', bbox_to_anchor=(0.5, 0.7), fontsize=fontsize, frameon=False, handlelength=1, handletextpad=0.2)
for i_ax, ax in enumerate(axs):
    ax.text(-0.05, 1.05, ['a', 'b', 'c'][i_ax], color='k', fontsize=11, weight='bold', transform=ax.transAxes)
axs[0].set_ylabel('Relative accuracy')
axs[0].set_xlabel(r'Perturbation $\varepsilon$ of $W^R$')
axs[1].set_xlabel(r'Perturbation $\varepsilon$ of $\tau$')



# ---------------------------------------------------------------------------
try:
    filename_multi = './fig08_retraining_accuracy/slurm_code/retrain/retrain_as_singlehead/train_curr_cumulative_1_heads_2023/retrained_N016_N036'
    filename_single = './fig08_retraining_accuracy/slurm_code/retrain/retrain_as_singlehead/1_at_a_time_with_forgetting_1_2023/retrained_N016_N036'

    data_multi = pickle.load(open(filename_multi, 'rb'))
    data_single = pickle.load(open(filename_single, 'rb'))


    def get_accuracy(stats):
        num_experiments = len(stats)
        num_Ns = len(stats[0]['stats'])

        retrained_accuracy = np.zeros((num_experiments, num_Ns))  # [experiments, N]
        N_retrained = np.zeros((num_experiments, num_Ns))
        for exp_idx in range(num_experiments):
            for N_idx in range(num_Ns):
                retrained_accuracy[exp_idx, N_idx] = stats[exp_idx]['stats'][N_idx]['accuracy'][-1]
                N_retrained[exp_idx, N_idx] = stats[exp_idx]['N_retrained'][N_idx]

        return N_retrained, retrained_accuracy


    Ns_multi, acc_multi = get_accuracy(data_multi)
    Ns_single, acc_single = get_accuracy(data_single)

    # normalize accuracy
    acc_multi /= 100
    acc_single /= 100
    acc_multi = (acc_multi - 0.5) / (acc_multi[:, 0:1] - 0.5)
    acc_single = (acc_single - 0.5) / (acc_single[:, 0:1] - 0.5)


    ax = axs[2]
    sns.despine(fig, ax)
    # plot accuracies as LineCollection
    line_collection_multi = LineCollection(
        segments=[np.column_stack((Ns_multi[0], acc_multi[i])) for i in range(acc_multi.shape[0])],
        linewidths=1,
        colors=color_multi,
        label='Multi-head',
    )
    line_collection_single = LineCollection(
        segments=[np.column_stack((Ns_single[0], acc_single[i])) for i in range(acc_single.shape[0])],
        linewidths=1,
        colors=color_single,
        label='Single-head',
    )
    ax.add_collection(line_collection_multi)
    ax.add_collection(line_collection_single)

    # ax.plot(Ns_multi[0], acc_multi.T, color=color_multi, label='Multi-head')
    # ax.plot(Ns_single[0], acc_single.T, color=color_single, label='Single-head')
    ax.set_xlabel(r'$N$ of single-head re-training')
    ax.set_xlim(15, 35)
    ax.set_xticks(np.arange(16, 37, 5))
except FileNotFoundError:
    print('no retraining results found')
    pass

# ---------------------------------------------------------------------------
axs[2].legend(loc='lower left', bbox_to_anchor=(0.5, 0.7), fontsize=fontsize, frameon=False, handlelength=1, handletextpad=0.2)
fig.show()
# fig.savefig('../../fig/exp121_perturbations_and_retraining_mod.pdf')
