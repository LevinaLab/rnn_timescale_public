import os

import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns

from src.tasks import make_batch_Nbit_pair_parity
from src.utils import load_model


model_path = '../../trained_models/continuous'
result_path = '../../results'
device = 'cpu'
max_total_length = 300_000  # RAM restriction: Ns[-1] * duplicate * batch_size
save_results = False

# parameters for testing accuracy
# N_max = 25
batch_size = 256  # 64
test_range = (
        list(range(1, 21)) + [25, 30, 35, 40, 45, 50]
        + [60, 70, 80, 90, 100] + [150, 200]
        # + [300, 500, 750, 1000, 1500, 2000]
)[::-1]

# select network
networks = [
    "cumulative_parity_duplicates[7, 8, 9, 10]_mod_leakyrelu_network_99",
    "single_parity_duplicates[7, 8, 9, 10]_mod_leakyrelu_network_99",
    # "cumulative_parity_duplicates[7, 8, 9, 10]_tau0.5_mod_leakyrelu_network_1",
    # "cumulative_parity_duplicates[7, 8, 9, 10]_tau0.75_mod_leakyrelu_network_1",
    # "cumulative_parity_duplicates[7, 8, 9, 10]_tau1.0_mod_leakyrelu_network_1",
    # "cumulative_parity_duplicates[5]_mod_leakyrelu_network_1",
    # "cumulative_parity_duplicates[7]_mod_leakyrelu_network_1",
    "cumulative_parity_duplicates[10]_mod_leakyrelu_network_1",
    # "cumulative_parity_duplicates[7, 8]_mod_leakyrelu_network_1",
    # "cumulative_parity_duplicates[7, 10]_mod_leakyrelu_network_1",
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



def _load_model(
    N_max, duplicates, task='parity', curriculum_type='cumulative', network_number=1, nonlinearity='leakyrelu', tau=None
):
    affixes = [f'duplicates{duplicates}']
    if tau is not None:
        affixes += [f'tau{tau}']
    affixes += ['mod', nonlinearity]
    rnn = load_model(
        curriculum_type=curriculum_type,
        task='parity',
        network_number=network_number,
        N_max=N_max,
        N_min=N_max if curriculum_type == 'single' else 2,
        device='cpu',
        base_path=model_path,
        continuous_model=True,
        affixes=affixes,
    )

    return rnn


def _get_max_N(basepath, network_name):
    # list all files in folder
    files = os.listdir(os.path.join(basepath, network_name))
    # only those that contain two 'N'
    files = [f for f in files if f.count('N') == 2]
    # extract Ns, splitting on second 'N'
    Ns = [int(f.split('N')[2]) for f in files]
    # return max N
    return max(Ns)


def _get_accuracy(duplicate, Ns, batch_size):
    with torch.no_grad():
        correct_N = np.zeros_like(Ns)
        total = 0

        # compute max_batch_sizes such that Ns[-1] * duplicate * batch_size <= max_total_length
        max_batch_size = max_total_length // (Ns[-1] * duplicate)
        # create list of batch_sizes such that the sum is batch_size
        if max_batch_size < batch_size:
            batch_sizes = [max_batch_size] * (batch_size // max_batch_size)
            if batch_size % max_batch_size > 0:
                batch_sizes += [batch_size % max_batch_size]
        else:
            batch_sizes = [batch_size]

        for batch_size in batch_sizes:
            sequences, labels = make_batch_Nbit_pair_parity(Ns, batch_size, duplicate=duplicate)
            sequences = sequences.permute(1, 0, 2).to(device)
            labels = [l.to(device) for l in labels]

            out, out_class = rnn(sequences, k_data=duplicate)

            for N_i in range(len(Ns)):
                predicted = torch.max(out_class[N_i], 1)[1]

                correct_N[N_i] += (predicted == labels[N_i]).sum()
                total += labels[N_i].size(0)

        accuracy = 100 * correct_N / float(total) * len(Ns)
    return accuracy


for network_args, network in zip(network_args_list, networks):
    # test correctness
    N_max = _get_max_N(model_path, network)
    Ns = torch.arange(2, N_max + 1) if network_args['curriculum_type'] == 'cumulative' else torch.arange(N_max, N_max + 1)
    rnn = _load_model(**network_args, N_max=N_max)
    accuracies = {}
    for duplicate in test_range:  # duplicates:
        accuracy = _get_accuracy(duplicate, Ns, batch_size)
        accuracies[duplicate] = accuracy
        print(f'k = {duplicate}, accuracy = {accuracy}')

    # save accuracies to file
    if save_results:
        with open(os.path.join(result_path, f'accuracy_{network}.pkl'), 'wb') as f:
            pickle.dump(accuracies, f)

    # plot accuracies
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    sns.despine()
    # fill in between duplicates[0] and duplicates[-1]
    ax.axvspan(
        network_args['duplicates'][0],
        network_args['duplicates'][-1],
        alpha=0.5,
        color='red',
        label='training range',
    )

    viridis = plt.cm.get_cmap('viridis', N_max - 3).reversed()
    for i_N, N in enumerate(Ns):
        # color continuously based on N
        ax.plot(
            list(accuracies.keys()),
            [accuracies[k][i_N] for k in accuracies.keys()],
            # label=f'N = {N}',
            color=viridis(int(N - 2)),
        )
    ax.set_xlabel('k')
    ax.set_ylabel('accuracy (%)')
    ax.legend(frameon=False, loc='lower right')
    # colorbar based on viridis with N
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=2, vmax=N_max))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('N')
    ax.set_xscale('log')
    fig.suptitle(network)
    fig.show()
