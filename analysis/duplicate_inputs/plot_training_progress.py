import os

import numpy as np
from matplotlib import pyplot as plt

model_path = '../../trained_models'

task = 'parity'


def _load_stats(duplicate=1, network_number=1, curriculum_type='cumulative'):
    affixes = [f'duplicate{duplicate}'] if duplicate > 1 else []
    affix_str = '_'
    if len(affixes) > 0:
        affix_str += '_'.join(affixes) + '_'
    rnn_subdir = f'{curriculum_type}_{task}{affix_str}network_{network_number}'
    return np.load(os.path.join(model_path, rnn_subdir, 'stats.npy'), allow_pickle=True).item()


# stats_test: dict = _load_stats(duplicate=1, network_number=1)
# print(len(stats_test['loss']))
# print(len(stats_test['accuracy']))
# max_N = len(stats_test['accuracy'][100])
# print(max_N)

duplicate_list = [1, 2, 3, 5, 10]
for curriculum_type in ['cumulative', 'single']:
    fig, ax = plt.subplots(constrained_layout=True)
    for i_duplicate, duplicate in enumerate(duplicate_list):
        stats_training: dict = _load_stats(
            duplicate=duplicate, network_number=1, curriculum_type=curriculum_type
        )
        epochs = len(stats_training['loss'])
        max_N = [len(stats_training['accuracy'][i]) for i in range(epochs)]
        ax.plot(range(epochs), max_N, label=f'duplicate = {duplicate}')
    ax.set_xlabel('epoch')
    ax.set_ylabel('max_N')
    ax.legend()
    fig.suptitle(f'{curriculum_type} {task}')
    fig.show()
