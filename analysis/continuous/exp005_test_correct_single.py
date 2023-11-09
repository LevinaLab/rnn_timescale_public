import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns

from src.tasks import make_batch_Nbit_pair_parity
from src.utils import load_model


model_path = '../../trained_models/continuous'
device = 'cpu'

duplicates = [7, 8, 9, 10]
task = 'parity'
curriculum_type = 'single'#'cumulative'
nonlinearity = 'leakyrelu'
network_number = 99

# parameters for testing accuracy
N_max = 19#25
max_total_length = 500_000  # RAM restriction: Ns[-1] * duplicate * batch_size


def _load_model(
    N_max, duplicates, curriculum_type='cumulative', network_number=1, nonlinearity='leakyrelu'
):
    affixes = [f'duplicates{duplicates}']
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


# test correctness
Ns = torch.arange(2, N_max + 1) if curriculum_type == 'cumulative' else torch.arange(N_max, N_max + 1)
batch_size = 64
rnn = _load_model(
    N_max=N_max,
    duplicates=duplicates,
    curriculum_type=curriculum_type,
    network_number=network_number,
    nonlinearity=nonlinearity,
)
accuracies = {}
for duplicate in list(range(1, 21)) + [25, 30, 35, 40, 45, 50] + [60, 70, 80, 90, 100] + [150, 200]:  # duplicates:
    accuracy = _get_accuracy(duplicate, Ns, batch_size)
    accuracies[duplicate] = accuracy
    print(f'k = {duplicate}, accuracy = {accuracy}')

# %% more numbers
for duplicate in [300, 500, 750, 1000, 1500, 2000]:
    accuracy = _get_accuracy(duplicate, Ns, batch_size)
    accuracies[duplicate] = accuracy
    print(f'k = {duplicate}, accuracy = {accuracy}')

# %% plot accuracies
fig, ax = plt.subplots(1, 1, constrained_layout=True)
sns.despine()
# fill in between duplicates[0] and duplicates[-1]
ax.axvspan(
    duplicates[0],
    duplicates[-1],
    alpha=0.5,
    color='red',
    label='training range',
)

viridis = plt.cm.get_cmap('viridis', N_max - 2).reversed()
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
ax.legend(frameon=False, loc='center right')
# colorbar based on viridis with N
sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=2, vmax=N_max))
sm._A = []
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('N')
ax.set_xscale('log')
fig.show()


# TODO brainstorm how to do k=8.5
