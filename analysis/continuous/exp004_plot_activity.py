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
curriculum_type = 'cumulative'
nonlinearity = 'leakyrelu'
network_number = 99
Ns = np.arange(2, 25)

rnn = load_model(
    curriculum_type=curriculum_type,
    task='parity',
    network_number=network_number,
    N_max=Ns[-1],
    N_min=2,
    device='cpu',
    base_path=model_path,
    continuous_model=True,
    affixes=[f'duplicates{duplicates}', 'mod', nonlinearity],
)


def _plot_activity(fig, rnn, sequence, duplicate, hidden_idx):
    with torch.no_grad():
        out, out_class = rnn(
            sequence,
            k_data=duplicate,
            classify_in_time=True,
            savetime=True,
        )

        time = np.arange(sequence.shape[0]) / duplicate
    out = torch.stack([o[0][0] for o in out]).to('cpu').numpy()  # [time, 500]
    sequences = sequence[:, 0, 0].to('cpu').numpy()  # [time]
    out_class = torch.stack([torch.stack(o) for o in out_class]).to('cpu').numpy()  # [time, Ns, 1, num_classes]

    axs = fig.subplots(3, 1, sharex='col')
    sns.despine()

    # plot sequence
    ax = axs[0]
    ax.set_title(f'k={duplicate}')
    ax.plot(time, sequences, color='C0', label='input')
    ax.set_ylabel('input')

    # plot hidden layer
    ax = axs[1]
    ax.plot(time, out[1:, hidden_idx], color='k', alpha=0.5)
    ax.set_ylabel('hidden')

    # plot out_class
    ax = axs[2]
    viridis = plt.cm.get_cmap('viridis', len(Ns) - 2).reversed()
    for i_N, N in enumerate(Ns):
        ax.plot(
            time,
            out_class[:, i_N, 0, 0],
            color=viridis(N - 2),
            alpha=1,
        )
    # colorbar based on viridis with N
    sm = plt.cm.ScalarMappable(cmap=viridis, norm=plt.Normalize(vmin=2, vmax=Ns[-1]))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('N')
    ax.set_ylabel('output')

    axs[-1].set_xlabel('time')


k_list = [7, 10, 50]
sequence_base, _ = make_batch_Nbit_pair_parity(Ns, 1, duplicate=1)
fig = plt.figure(layout='constrained', figsize=(12, 4))
subfigs = fig.subfigures(1, len(k_list))
for i, duplicate in enumerate(k_list):
    sequence = torch.repeat_interleave(sequence_base, duplicate, dim=1)
    sequence = sequence.permute(1, 0, 2).to(device)

    # draw 10 random indices between 0 and 500 avoiding same numbers
    hidden_idx = np.random.choice(np.arange(500), size=10, replace=False)

    _plot_activity(subfigs[i], rnn, sequence, duplicate, hidden_idx)
fig.show()

