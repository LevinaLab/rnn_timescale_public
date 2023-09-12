import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from src.lstm_utils import load_lstm, make_batch_Nbit_pair_parity

base_path = '../../trained_models'
N = 60
network_number = 1
curriculum_type = 'cumulative'

simulation_time = 1000
plot_start = 250
plot_end = 300
plot_slice = slice(plot_start, plot_end)
plot_time = np.arange(plot_start, plot_end)

plot_cells = np.arange(0, 10)

lstm = load_lstm(base_path, N, network_number, curriculum_type, n_min=2)
# plot_cells = np.argpartition(lstm.fc[20].weight.detach().numpy()[0, :], -10)[-10:]

inputs, labels = make_batch_Nbit_pair_parity(np.arange(2, 102), simulation_time, 1)
with torch.no_grad():
    output, readout, hidden_states, cell_states, forget_gates, output_gates = lstm(
        inputs,
        save_hidden_states=True,
        save_cell_states=True,
        save_forget_gates=True,
        save_output_gates=True,
    )
# [batch_size, timesteps, num_neurons]
# cell_states = np.stack([c_n.cpu().numpy() for c_n in cell_states], axis=1)
# [batch_size, timesteps, num_neurons]
# forget_gates = np.stack([f_n.cpu().numpy() for f_n in forget_gates], axis=1)


fig, axs = plt.subplots(5, 1, figsize=(6, 6), sharex='col', constrained_layout=True)
fig.suptitle(f'Curriculum: {curriculum_type}, N = {N}, network {network_number}')
sns.despine()

ax = axs[0]
ax.plot(plot_time, inputs[0, plot_slice, 0])
ax.set_ylabel('Input')

ax = axs[1]
ax.plot(plot_time, cell_states[0, plot_slice, plot_cells].T)
ax.set_ylabel('Cell states')

ax = axs[2]
ax.plot(plot_time, forget_gates[0, plot_slice, plot_cells].T)
ax.set_ylabel('Forget gates')

ax = axs[3]
ax.plot(plot_time, output_gates[0, plot_slice, plot_cells].T)
ax.set_ylabel('Output gates')

ax = axs[4]
ax.plot(plot_time, hidden_states[0, plot_slice, plot_cells].T)
# ax.plot(plot_time, np.sum(hidden_states[0, plot_slice, :], axis=1), 'k--')
ax.set_ylabel('Hidden state')

axs[-1].set_xlabel('Time')
fig.show()

