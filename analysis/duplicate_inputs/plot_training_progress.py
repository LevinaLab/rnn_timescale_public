import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

model_path = '../../trained_models'

task = 'parity'


def _load_stats(duplicate=1, network_number=1, curriculum_type='cumulative', tau=None):
    affixes = [f'duplicate{duplicate}'] if duplicate > 1 else []
    if tau is not None:
        affixes.append(f'tau{float(tau)}')
    affix_str = '_'
    if len(affixes) > 0:
        affix_str += '_'.join(affixes) + '_'
    rnn_subdir = f'{curriculum_type}_{task}{affix_str}network_{network_number}'
    return np.load(os.path.join(model_path, rnn_subdir, 'stats.npy'), allow_pickle=True).item()


def _get_epoch_max_N(duplicate=1, network_number=1, curriculum_type='cumulative', tau=None):
    stats_training: dict = _load_stats(
        duplicate=duplicate,
        network_number=network_number,
        curriculum_type=curriculum_type,
        tau=tau,
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
    for i_duplicate, duplicate in enumerate(duplicate_list):
        for network_number in [1, 2]:
            for tau in [None, duplicate]:
                try:
                    epochs, max_N = _get_epoch_max_N(
                        duplicate=duplicate,
                        network_number=network_number,
                        curriculum_type=curriculum_type,
                        tau=tau,
                    )
                except FileNotFoundError:
                    continue
                ax.plot(
                    range(epochs),
                    max_N,
                    color=f'C{i_duplicate}',
                    label=f'duplicate = {duplicate}' if (network_number == 1 and tau is None) else None,
                    linestyle='-' if tau is None else '--',
                )

    ax.set_xlabel('epoch')
    ax.set_ylabel('max_N')
    ax.legend(frameon=False, loc='upper left')

    # Create the legend for tau
    lines = [Line2D([0], [0], linestyle=ls, color='black') for ls in ['-', '--']]
    fig.legend(
        [Line2D([0], [0], linestyle=ls, color='black') for ls in ['-', '--']],
        ['1.5', '=duplicate'],
        title='Initial tau',
        frameon=False,
        loc='lower right',
        bbox_to_anchor=(0.95, 0.2),
    )

    fig.suptitle(f'{curriculum_type} {task}')
    fig.show()
