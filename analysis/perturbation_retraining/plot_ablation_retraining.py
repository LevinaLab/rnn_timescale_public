"""Plot perturbation and retraining results. (Figure 8 in the paper)"""
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

import sys

from scipy.stats import ttest_ind, ranksums

sys.path.append('../../')
from src.utils import set_plot_params


fontsize = 10
set_plot_params()

color_single = '#D48A6A'
color_multi = '#489174'

cm = 1 / 2.54
fig, axs = plt.subplots(
    ncols=5,
    figsize=(18*cm, 4.7*cm),
    # sharey='all',
    constrained_layout=True
)
axs[0].set_ylabel('Relative accuracy')
# axs[2].set_ylabel('Relative accuracy')
# share y-axis between axs[0] and axs[1]
axs[1].get_shared_y_axes().join(axs[1], axs[0])
# share y-axis between 2, 3 and 4
axs[3].get_shared_y_axes().join(axs[3], axs[2], axs[4])

# ---------------------------------------------------------------------------
# plot ablation results
color_fast = '#AA9C39'
color_slow = '#8C7AAE'
N = 30
network_numbers = [f"network_{i}" for i in [1, 2, 3, 4]]
np.random.seed(42008)
df_ablation = pd.read_pickle('../../results/df_ablate_tau.pkl')
df_ablation['accuracies_relative'] = df_ablation.apply(
    lambda row: (row['accuracies'] - 0.5) / (np.max(row['baseline_accuracy']) - 0.5),
    axis=1,
)
for ax, network_type, title in zip(axs[:2], ['single-head', 'cumulative'], ['Single-head', 'Multi-head']):
    df = df_ablation.loc[network_type, network_numbers, N]
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
    ax.set_xlabel(r'Average $\tau$ of' + '\nablated neurons')
    ax.set_ylim(0.6, 1.05)
    ax.set_title(title, fontsize=fontsize)
axs[0].legend(
    labels=['Short', 'Long'], loc='upper right', bbox_to_anchor=(1.2, 1.1), frameon=False, fontsize=fontsize, handletextpad=-0.3
)
# ---------------------------------------------------------------------------
# plot perturbation results
N = 30
network_list = [f"network_{i}" for i in [1, 2, 3, 4]]

df_rnn = pd.read_pickle('../../results/df_perturb_rnn_normalized.pkl')
df_tau = pd.read_pickle('../../results/df_perturb_tau_normalized_abs.pkl')
for df in [df_rnn, df_tau]:
    # relative accuracy for each row compared to the row where perturbation = 0
    df['accuracy_baseline'] = df.apply(
        lambda row: np.mean(df.at[(row.name[0], row.name[1], row.name[2], float(0)), 'accuracy']),
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


for ax, df in zip(axs[2:4], [df_rnn, df_tau]):
    for i_network, network in enumerate(network_list):
        for i_type, network_type in enumerate(['single-head', 'cumulative']):
            if network_type == 'single-head':
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
    ax.text(-0.05, 1.05, ['a', 'b', 'c', 'd', 'e'][i_ax], color='k', fontsize=11, weight='bold', transform=ax.transAxes)
# axs[2].set_ylabel('Relative accuracy')
axs[3].set_xticks([1e-2, 1e0])
axs[2].set_xlabel(r'Perturbation $\varepsilon$ of $W^R$')
axs[3].set_xlabel(r'Perturbation $\varepsilon$ of $\tau$')



# ---------------------------------------------------------------------------
# filename_multi = './fig08_retraining_accuracy/slurm_code/retrain/retrain_as_singlehead/train_curr_cumulative_1_heads_2023/retrained_N016_N036'
# filename_single = './fig08_retraining_accuracy/slurm_code/retrain/retrain_as_singlehead/1_at_a_time_with_forgetting_1_2023/retrained_N016_N036'

epochs_retrained = 20  # 50
filename_multi = f'../../results/{epochs_retrained}_epochs/train_curr_cumulative_1_heads_2023/retrained_N016_N046'
filename_single = f'../../results/{epochs_retrained}_epochs/1_at_a_time_with_forgetting_1_2023/retrained_N016_N046'

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


ax = axs[4]
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
ax.set_xlabel(r'$N$ of single-head' + '\nretraining')
ax.set_xlim(15, 45)
ax.set_xticks(np.arange(16, 47, 10))

# ---------------------------------------------------------------------------
for ax in axs:
    ax.axhline(0, color='grey', linestyle='--', label='Chance level')
axs[4].legend(loc='lower left', bbox_to_anchor=(0.1, 0.5), fontsize=fontsize, frameon=False, handlelength=1, handletextpad=0.2)
fig.show()
fig.savefig(f'../../fig/plot_ablation_perturbation_retraining{epochs_retrained}.pdf')

# ---------------------------------------------------------------------------
# p-values to text file
# test_function = ttest_ind
test_function = ttest_ind
p_levels = np.array([0.05, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
with open(f'../../results/p_values_retraining_{epochs_retrained}.txt', 'w') as f:
    f.write('p-values computed as two-sided t-test (unpaired) between single-head and multi-head (4 networks each)\n')
    f.write('\np-values for each perturbation of reccurent weights\n')
    for perturbation in df_rnn.index.get_level_values('perturbation').unique():
        t, p = test_function(
            np.concatenate(df_rnn.loc[('single-head', slice(None), N, perturbation), 'accuracy_rel']),
            np.concatenate(df_rnn.loc[('cumulative', slice(None), N, perturbation), 'accuracy_rel']),
        )
        # compute number of stars
        stars = np.sum(p_levels[:, None] > p) * '*'
        f.write(f'perturbation={perturbation:.1e}:\tp={p:.1e}\t{stars}\n')

    f.write('\np-values for each perturbation of tau\n')
    for perturbation in df_tau.index.get_level_values('perturbation').unique():
        t, p = test_function(
            np.concatenate(df_tau.loc[('single-head', slice(None), N, perturbation), 'accuracy_rel']),
            np.concatenate(df_tau.loc[('cumulative', slice(None), N, perturbation), 'accuracy_rel']),
        )
        # compute number of stars
        stars = np.sum(p_levels[:, None] > p) * '*'
        f.write(f'perturbation={perturbation:.1e}:\tp={p:.1e}\t{stars}\n')
    f.write('\np-values for each N retraining for {epochs_retrained} epochs\n')
    for N_idx in range(1, acc_multi.shape[1]):
        t, p = test_function(acc_multi[:, N_idx], acc_single[:, N_idx])
        # compute number of stars
        stars = np.sum(p_levels[:, None] > p) * '*'
        f.write(f'N={int(Ns_multi[0, N_idx])}:\tp={p:.1e}\t{stars}\n')

