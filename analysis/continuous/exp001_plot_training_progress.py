import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

model_path = '../../trained_models/continuous'

duplicates = [7, 8, 9, 10]
task = 'parity'
curriculum_type = 'cumulative'
nonlinearity = 'leakyrelu'
network_number = 99


def _load_stats(duplicates, network_number, curriculum_type='cumulative', nonlinearity='leakyrelu'):
    affixes = [f'duplicates{duplicates}']
    affixes += ['mod', nonlinearity]
    affix_str = '_'
    if len(affixes) > 0:
        affix_str += '_'.join(affixes) + '_'
    rnn_subdir = f'{curriculum_type}_{task}{affix_str}network_{network_number}'
    return np.load(os.path.join(model_path, rnn_subdir, 'stats.npy'), allow_pickle=True).item()


def _get_epoch_max_N(duplicates, network_number, curriculum_type='cumulative', nonlinearity='leakyrelu'):
    stats_training: dict = _load_stats(
        duplicates=duplicates,
        network_number=network_number,
        curriculum_type=curriculum_type,
        nonlinearity=nonlinearity,
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

fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
epochs, max_N = _get_epoch_max_N(
    duplicates=duplicates,
    network_number=network_number,
    curriculum_type=curriculum_type,
    nonlinearity=nonlinearity,
)
ax.plot(
    range(epochs),
    max_N,
    color=f'C0',
    label=f'duplicates = {duplicates}',
)

ax.set_xlabel('epoch')
ax.set_ylabel('max_N')
ax.legend(frameon=False, loc='upper left')
fig.show()
