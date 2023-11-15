import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

model_path = '../../trained_models/continuous'

# select network
networks = [
    "cumulative_parity_duplicates[7, 8, 9, 10]_mod_leakyrelu_network_99",
    "cumulative_parity_duplicates[7, 8, 9, 10]_tau0.5_mod_leakyrelu_network_1",
    "cumulative_parity_duplicates[7, 8, 9, 10]_tau0.75_mod_leakyrelu_network_1",
    "cumulative_parity_duplicates[7, 8, 9, 10]_tau1.0_mod_leakyrelu_network_1",
]

# default network args
network_args_list = []
for network in networks:
    network_args = {
        "task": "parity",
        "curriculum_type": "cumulative",
        "duplicates": [7, 8, 9, 10],
        "nonlinearity": "leakyrelu",
        "network_number": 1,
        "tau": None,
    }
    # update network args to match network:
    match network:
        case "cumulative_parity_duplicates[7, 8, 9, 10]_mod_leakyrelu_network_99":
            network_args['network_number'] = 99
        case "single_parity_duplicates[7, 8, 9, 10]_mod_leakyrelu_network_99":
            network_args['curriculum_type'] = 'single'
            network_args['network_number'] = 99
        case "cumulative_parity_duplicates[7, 8, 9, 10]_tau0.5_mod_leakyrelu_network_1":
            network_args['tau'] = 0.5
        case "cumulative_parity_duplicates[7, 8, 9, 10]_tau0.75_mod_leakyrelu_network_1":
            network_args['tau'] = 0.75
        case "cumulative_parity_duplicates[7, 8, 9, 10]_tau1.0_mod_leakyrelu_network_1":
            network_args['tau'] = 1.0
        case "cumulative_parity_duplicates[5]_mod_leakyrelu_network_1":
            network_args['duplicates'] = [5]
        case "cumulative_parity_duplicates[7]_mod_leakyrelu_network_1":
            network_args['duplicates'] = [7]
        case "cumulative_parity_duplicates[10]_mod_leakyrelu_network_1":
            network_args['duplicates'] = [10]
        case "cumulative_parity_duplicates[7, 8]_mod_leakyrelu_network_1":
            network_args['duplicates'] = [7, 8]
        case "cumulative_parity_duplicates[7, 10]_mod_leakyrelu_network_1":
            network_args['duplicates'] = [7, 10]
        case _: raise ValueError(f'Unknown network: {network}')
    network_args_list.append(network_args)


def _load_stats(duplicates, network_number, task='parity', curriculum_type='cumulative', nonlinearity='leakyrelu', tau=None):
    affixes = [f'duplicates{duplicates}']
    if tau is not None:
        affixes += [f'tau{tau}']
    affixes += ['mod', nonlinearity]
    affix_str = '_'
    if len(affixes) > 0:
        affix_str += '_'.join(affixes) + '_'
    rnn_subdir = f'{curriculum_type}_{task}{affix_str}network_{network_number}'
    return np.load(os.path.join(model_path, rnn_subdir, 'stats.npy'), allow_pickle=True).item()


def _get_epoch_max_N(duplicates, network_number, task='parity', curriculum_type='cumulative', nonlinearity='leakyrelu', tau=None):
    stats_training: dict = _load_stats(
        duplicates=duplicates,
        network_number=network_number,
        task=task,
        curriculum_type=curriculum_type,
        nonlinearity=nonlinearity,
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

fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
for network_args, network in zip(network_args_list, networks):
    epochs, max_N = _get_epoch_max_N(**network_args)
    ax.plot(
        range(epochs),
        max_N,
        label=network,
    )

ax.set_xlabel('epoch')
ax.set_ylabel('max_N')
ax.legend(frameon=False, loc='lower right')
fig.show()
