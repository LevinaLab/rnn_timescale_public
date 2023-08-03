"""Test whether networks are feedforward by testing Nilpotency of connectivity matrix."""
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.models import init_model


def _get_path(net_size, net_number, N_max):
    return f"../../trained_models/small_models/cumulative_parity_net_size_{net_size}_network_{net_number}/rnn_N2_N{N_max}"


def _get_N_max(net_size, net_number):
    stats: dict = np.load(
        f"../../trained_models/small_models/cumulative_parity_net_size_{net_size}_network_1/stats.npy",
        allow_pickle=True,
    ).item()
    N_max = len(stats['accuracy'][-1])
    return N_max

def _load_network(net_size, net_number, N_max):
    rnn_path = _get_path(net_size, net_number, N_max)
    device = 'cpu'
    strict = False

    rnn = init_model(DEVICE=device, NET_SIZE=[net_size])
    rnn.load_state_dict(torch.load(rnn_path, map_location=device)['state_dict'], strict=strict)
    return rnn


exponents = np.arange(1, 10)
norms_list = []
for net_size in range(1, 11):
    for net_number in [1, 2]:
        try:
            N_max = _get_N_max(net_size, net_number)
            rnn = _load_network(net_size, net_number, N_max)
        except FileNotFoundError:
            continue
        recurrent_weights = rnn.w_hh[0].weight.data.detach().numpy()

        norms = []
        for exp in exponents:
            if exp == 1:
                weights_mult = recurrent_weights
                # weights_mult[np.abs(weights_mult) < (10 * np.linalg.norm(weights_mult) / np.size(weights_mult))] = 0
            else:
                weights_mult = weights_mult @ recurrent_weights
            norms.append(np.linalg.norm(weights_mult))
        norms_list.append(norms)
norms_list = np.array(norms_list).T


# plot
fig, ax = plt.subplots()
ax.plot(exponents, norms_list)
ax.set_xlabel(r'Exponent $k$')
ax.set_ylabel(r'Frobenius Norm of $W^k$')
ax.set_yscale('log')
fig.show()
