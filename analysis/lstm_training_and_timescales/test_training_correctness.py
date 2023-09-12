import numpy as np
import torch

from src.lstm_utils import load_lstm, make_batch_Nbit_pair_parity


base_path = '../../trained_models'
N = 60
network_number = 1
curriculum_type = 'cumulative'

simulation_time = 1000
plot_start = 100
plot_end = 300
plot_slice = slice(plot_start, plot_end)
plot_time = np.arange(plot_start, plot_end)

plot_cells = np.arange(0, 10)

lstm = load_lstm(base_path, N, network_number, curriculum_type, n_min=2)

inputs, labels = make_batch_Nbit_pair_parity(np.arange(2, 102), simulation_time, 1)
with torch.no_grad():
    output, readout = lstm(
        inputs,
        save_readouts=True,
    )

# readout:  num_readout_heads x time x batch_size x 2
# given_answer: num_readout_heads x time
given_answer = np.stack([r[:, 0, 0] > r[:, 0, 1] for r in readout])

print(given_answer[:10, -1])
print(labels[:10])
