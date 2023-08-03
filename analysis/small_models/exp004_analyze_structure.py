"""Analyze structure size=3, network=2, N_max=2,3.

This network is interesting because it performs the best.
"""
import numpy as np
import pandas as pd
import torch

from src.models import init_model


def _get_path(net_size, N_max, network=1):
    return f"../../trained_models/small_models/cumulative_parity_net_size_{net_size}_network_{network}/rnn_N2_N{N_max}"


def _load_network(net_size, N_max, network=1):
    rnn_path = _get_path(net_size, N_max, network)
    device = 'cpu'
    strict = False

    rnn = init_model(DEVICE=device, NET_SIZE=[net_size])
    rnn.load_state_dict(torch.load(rnn_path, map_location=device)['state_dict'], strict=strict)
    return rnn


# print all weights for net size 3
N_max = 3
print(f"net size 3, N_max {N_max}")
rnn = _load_network(3, N_max, network=2)
# print taus
print('Taus')
print(rnn.taus[0].data)
# print input weights
print('Input weights')
print(rnn.input_layers[0].weight.data)
# print recurrent weights
print('Recurrent weights')
print(rnn.w_hh[0].weight.data)
# rnn bias
print('RNN bias')
print(rnn.w_hh[0].bias.data + rnn.input_layers[0].bias.data)
# print readout weights
print(f'Readout weights + bias N={N_max}')
print(rnn.fc[N_max - 2].weight.data)
print(rnn.fc[N_max - 2].bias.data)
# print readout weights
print(f'Readout weights + bias N={N_max - 1}')
print(rnn.fc[N_max - 3].weight.data)
print(rnn.fc[N_max - 3].bias.data)

# %% test run with some random input
inputs = torch.bernoulli(0.5 * torch.ones((20, 1)))
with torch.no_grad():
    h_t, outputs = rnn(inputs, classify_in_time=True, savetime=True)
inputs = inputs.detach().numpy()[:, 0]
outputs = torch.vstack([o[N_max - 2] for o in outputs]).detach().numpy()
h_t = torch.vstack([h[0] for h in h_t]).detach().numpy()
inputs_cumsum = np.cumsum(inputs)
partial_sums = inputs_cumsum[N_max:] - inputs_cumsum[:-N_max]
outputs_correct = (partial_sums % 2)
test = pd.DataFrame({
    'input': [None] + list(inputs),
    'hidden1': h_t[:, 0],
    'hidden2': h_t[:, 1],
    'hidden3': h_t[:, 2],
    'output1': [None] + list(outputs[:, 0]),
    'output2': [None] + list(outputs[:, 1]),
    'correct': (N_max + 1) * [None] + list(outputs_correct),
})
print(test)
# accuracy
predictions = outputs[N_max:, 0] < outputs[N_max:, 1]
print(f"accuracy: {(outputs_correct == predictions).mean():.3f}")
