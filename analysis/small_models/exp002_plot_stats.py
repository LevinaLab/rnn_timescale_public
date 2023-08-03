"""Visualize the training of small models.

Plots:
    - training loss
    - training and test accuracy
    - N solved after each epoch
    - max N solved for each net size
"""
import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.models import init_model

net_sizes = np.arange(1, 11)
net_numbers = [1, 2]


def _get_path(net_size, net_number=1):
    return f"../../trained_models/small_models/cumulative_parity_net_size_{net_size}_network_{net_number}"


def _load_network(net_size, net_number=1, N_max=2):
    rnn_path = os.path.join(
        _get_path(net_size, net_number=net_number),
        f"rnn_N2_N{N_max}",
    )
    device = 'cpu'
    strict = False

    rnn = init_model(DEVICE=device, NET_SIZE=[net_size])
    rnn.load_state_dict(torch.load(rnn_path, map_location=device)['state_dict'], strict=strict)
    return rnn


fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for net_size in net_sizes:
    stats: dict = np.load(os.path.join(_get_path(net_size), "stats.npy"), allow_pickle=True).item()
    ax.plot(stats['loss'], alpha=0.7, label=net_size)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.legend(title='Net size', bbox_to_anchor=(1.05, 1), loc='upper left')
fig.show()


fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for net_size in net_sizes:
    stats: dict = np.load(os.path.join(_get_path(net_size), "stats.npy"), allow_pickle=True).item()
    ax.plot(
        np.array([accuracies[-1] for accuracies in stats['accuracy']]),
        alpha=0.7,
        label=net_size,
    )
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy on last N')
ax.legend(title='Net size', bbox_to_anchor=(1.05, 1), loc='upper left')
fig.show()


fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for net_size in net_sizes:
    stats: dict = np.load(os.path.join(_get_path(net_size), "stats.npy"), allow_pickle=True).item()
    ax.plot(np.array([len(accuracies) for accuracies in stats['accuracy']]), label=net_size)
ax.set_xlabel('Epoch')
ax.set_ylabel('N solved')
ax.legend(title='Net size', bbox_to_anchor=(1.05, 1), loc='upper left')
fig.show()


x_scatter = []
y_scatter = []
max_timescales = []
for net_size in net_sizes:
    for net_number in net_numbers:
        try:
            stats: dict = np.load(os.path.join(_get_path(net_size, net_number), "stats.npy"), allow_pickle=True).item()
        except FileNotFoundError:
            continue
        x_scatter.append(net_size + net_number / 5)
        N_max = len(stats['accuracy'][-1])
        y_scatter.append(N_max)
        if N_max == 1:
            max_timescales.append(1)
        else:
            rnn = _load_network(net_size, net_number, N_max)
            max_timescales.append(np.max(rnn.taus[0].detach().numpy()))
fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
norm = plt.Normalize(vmin=1, vmax=max(max_timescales))
sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=norm)
sm.set_array([])
ax.scatter(
    x_scatter,
    y_scatter,
    color='C1',
    marker='o',
    edgecolors='k',
    facecolors=plt.cm.cool(norm(max_timescales)),
    label='networks',
)
ax.axline((2, 1), (3, 2), color='black', linestyle='--', label='slope=1')
ax.axline((2, 1), (4, 2), color='blue', linestyle='--', label='slope=0.5')
ax.set_xlabel('Net size')
ax.set_ylabel('Max N solved')
ax.legend(frameon=False, loc='upper left')
# colorbar
fig.colorbar(sm, ax=ax, label='Maximum timescale')
fig.show()

