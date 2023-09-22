import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

model_path = '../../trained_models'
fig_path = '../../fig'


def _load_stats(network_number=1, curriculum_type='cumulative'):
    rnn_subdir = f'lstm_{curriculum_type}_network_{network_number}'
    return np.load(os.path.join(model_path, rnn_subdir, 'stats.npy'), allow_pickle=True).item()


def _get_epoch_max_N(network_number=1, curriculum_type='cumulative'):
    stats_training: dict = _load_stats(
        network_number=network_number,
        curriculum_type=curriculum_type,
    )
    epochs = len(stats_training['loss'])
    if curriculum_type == 'cumulative':
        max_N = [len(stats_training['accuracy'][i]) for i in range(epochs)]
    elif curriculum_type == 'single':
        max_N = np.cumsum(np.array([acc[0] for acc in stats_training['accuracy']]) > 98) + 1
    else:
        raise ValueError(f'Unknown curriculum_type: {curriculum_type}')
    return epochs, max_N


# stats_test: dict = _load_stats(duplicate=1, network_number=1)
# print(len(stats_test['loss']))
# print(len(stats_test['accuracy']))
# max_N = len(stats_test['accuracy'][100])
# print(max_N)

for curriculum_type in ['cumulative', 'single']:
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle(f'Curriculum: {curriculum_type}')
    sns.despine()
    for network_number in np.arange(51, 55):
        try:
            epochs, max_N = _get_epoch_max_N(
                network_number=network_number,
                curriculum_type=curriculum_type,
            )
        except FileNotFoundError:
            continue
        ax.plot(
            range(epochs),
            max_N,
            color=f'C{network_number}',
            label=network_number,
        )

    ax.set_xlabel('epoch')
    ax.set_ylabel('max_N')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.05, 1.0))

    fig.show()


cm = 1 / 2.54
fig, ax = plt.subplots(figsize=[9 * cm, 4.5 * cm], constrained_layout=True)
sns.despine()
for curriculum_type in ['cumulative', 'single']:
    color = '#D48A6A' if curriculum_type == 'single' else '#489174'
    for network_number in np.arange(51, 55):
        try:
            epochs, max_N = _get_epoch_max_N(
                network_number=network_number,
                curriculum_type=curriculum_type,
            )
        except FileNotFoundError:
            continue
        ax.plot(
            range(epochs),
            max_N,
            color=color,
            label=network_number,
        )

ax.set_xlabel('epoch')
ax.set_ylabel(r'$N_{\mathrm{max}}$')

custom_legend_entries = [
    Line2D([0], [0], color='#D48A6A', lw=2, label='Single-head'),
    Line2D([0], [0], color='#489174', lw=2, label='Multi-head')
]

ax.legend(
    handles=custom_legend_entries,
    loc='upper left',
    # bbox_to_anchor=(1.1, 1.0),
    frameon=False,
    handlelength=1,
)

fig.show()
fig.savefig(os.path.join(fig_path, 'lstm_training_progress.pdf'))
