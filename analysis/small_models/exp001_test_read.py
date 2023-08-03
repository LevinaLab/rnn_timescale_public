import torch

from src.models import init_model
from src.utils.accuracy_perturbation_ablation import calculate_accuracy_

N = 3
rnn_path = '../../trained_models/small_models/cumulative_parity_net_size_5_network_1/rnn_N2_N3'
device = 'cpu'
strict = False

rnn = init_model(DEVICE=device, NET_SIZE=[5])
rnn.load_state_dict(torch.load(rnn_path, map_location=device)['state_dict'], strict=strict)

# print taus
print(rnn.taus[0])
# print input weights
print(rnn.input_layers[0].weight)
# print recurrent weights
print(rnn.w_hh[0].weight)
# print readout weights
for N_readout in range(2, N + 1):
    print(f"readout weights for N={N_readout}")
    print(rnn.fc[N_readout - 2].weight)

# test accuracy on continuous parity task
for N_test in range(2, N + 1):
    acc = calculate_accuracy_(
        rnn,
        network_type='cumulative',
        input_length=100,
        warmup_length=100,
        N=N_test,
        index_in_head=N_test - 2,
    )
    print(f"accuracy on N={N_test} parity task: {acc:.3f}")

