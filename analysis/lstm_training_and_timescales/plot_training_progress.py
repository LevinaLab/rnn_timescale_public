import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

model_path = '../../trained_models'


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

duplicate_list = [1, 2, 3, 5, 10]
for curriculum_type in ['cumulative', 'single']:
    fig, ax = plt.subplots(constrained_layout=True)
    fig.suptitle(f'Curriculum: {curriculum_type}')
    sns.despine()
    for network_number in np.arange(1, 5):
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
