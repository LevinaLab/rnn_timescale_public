import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

from src.utils import load_model


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


def _get_mean_std_tau(
    N_max, duplicates, network_number, task='parity', curriculum_type='cumulative', nonlinearity='leakyrelu', tau=None, all_tau=False
):
    affixes = [f'duplicates{duplicates}']
    if tau is not None:
        affixes += [f'tau{tau}']
    affixes += ['mod', nonlinearity]
    rnn = load_model(
        curriculum_type=curriculum_type,
        task=task,
        network_number=network_number,
        N_max=N_max,
        N_min=N_max if curriculum_type == 'single' else 2,
        device='cpu',
        base_path=model_path,
        continuous_model=True,
        affixes=affixes,
    )
    taus = rnn.taus[0].detach().numpy()
    if all_tau:
        return np.mean(taus), np.std(taus), taus
    else:
        return np.mean(taus), np.std(taus)


fig, axs = plt.subplots(1, len(networks), constrained_layout=True, figsize=(len(networks) * 4, 4), sharey="row")
sns.despine()
for ax, network, network_args in zip(axs, networks, network_args_list):
    N_list = list(range(2, 101))
    tau_mean = np.zeros(len(N_list))
    tau_std = np.zeros(len(N_list))
    taus = np.zeros((len(N_list), 500))
    for i_N, N in enumerate(N_list):
        try:
            tau_mean[i_N], tau_std[i_N], taus[i_N] = _get_mean_std_tau(
                N,
                **network_args,
                all_tau=True
            )
        except FileNotFoundError:
            break

    segments = [
        np.column_stack([N_list[:i_N], taus[:i_N, i]])
        for i in range(500)
    ]
    lc = LineCollection(
        segments,
        color='k',
        alpha=0.1,
        label='individual tau',
    )
    ax.add_collection(lc)

    ax.plot(
        N_list[:i_N],
        tau_mean[:i_N],
        color='red',
        # label=f'duplicates = {duplicates}',
        linewidth=2,
    )
    ax.set_title(network, fontsize=8)
    ax.set_xlabel('N')
axs[0].set_ylabel('tau')
# leg = fig.legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 0.1))
# for lh in leg.legendHandles:
#     lh.set_alpha(1)


fig.show()
