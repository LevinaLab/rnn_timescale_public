import argparse
import os

import torch
import torch.nn as nn

import numpy as np


def generate_binary_sequence(M):
    # return (torch.rand(M) < torch.rand(1)) * 1.
    return ((torch.rand(M) < 0.5) * 1.)*2 - 1


def make_batch_Nbit_pair_parity(Ns, M, bs):
    with torch.no_grad():
        sequences = [generate_binary_sequence(M).unsqueeze(-1) for i in range(bs)]

        labels = [torch.stack([get_parity(s, N) for s in sequences]) for N in Ns]

    return torch.stack(sequences), labels


def get_parity(vec, N):
    return  (((vec + 1)/2)[-N:].sum() % 2).long()
    # return (vec[-N:].sum() % 2).long()


device = 'cpu'  # 'cuda'

class LSTM_custom(nn.Module):
    '''
    Custom LSTM class so that we can make RNNs with layers with different sizes,
    and also to save hidden_layer states through time.

     Parameters
    -----------
    input_size: int
        size of input (it has been always 1)
    net_size: list
        list of number of neurons per layer (size larger than 1 it is for a multi layer network)
    num_classes: int
        number of classes for classification
    bias: boolean, default True
        if we include bias terms
    num_readout_heads: int
        number of outputs
    '''

    def __init__(self, input_size=1,
                 hidden_size=100,
                 num_layers=1,
                 num_classes=2,
                 bias=True,
                 num_readout_heads=1,
                 ):
        super(LSTM_custom, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bias = bias
        self.num_readout_heads = num_readout_heads

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first=True)

        self.fc = [nn.Linear(hidden_size, num_classes, bias=bias) for i in range(num_readout_heads)]

        self.modulelist = nn.ModuleList(self.fc)


    def forward(self, data, h0=None, c0=None):
        '''
        input: data [batch_size, sequence length, input_features]
        output:
                h0:
                readout: readout from the fc layers at the end,
                    shape=[batch_size, self.num_classes],
        '''
        if h0 is None:
            h0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(device)
        output, (h_n, c_n) = self.lstm(data, (h0, c0))

        # readout = [self.fc[i](h_n[-1]) for i in range(self.num_readout_heads)]
        readout = [self.fc[i](output[:, -1, :]) for i in range(self.num_readout_heads)]


        return output, readout


def _save_lstm(lstm, base_path, n_max, network_number, n_min=2, init=False):
    lstm_subdir = os.path.join(base_path, f'lstm_network_{network_number}')
    if init:
        rnn_name = f'lstm_init'
    else:
        rnn_name = f'lstm_N{n_min:d}_N{n_max:d}'

    if not os.path.exists(lstm_subdir):
        os.makedirs(lstm_subdir)
    torch.save(
        {'state_dict': lstm.state_dict()},
        os.path.join(lstm_subdir, rnn_name),
    )
    return lstm_subdir

EPOCHS = 1000
INPUT_SIZE = 1

NET_SIZE = 500
NUM_LAYERS = 1
NUM_CLASSES = 2
BIAS = True

NUM_READOUT_HEADS = 100

BATCH_SIZE = 64
TRAINING_STEPS = 300
TEST_STEPS = 50

lstm = LSTM_custom(
    input_size=INPUT_SIZE,
    hidden_size=NET_SIZE,
    num_layers=NUM_LAYERS,
    bias=BIAS,
    num_readout_heads=NUM_READOUT_HEADS
).to(device)

learning_rate = 1e-3
# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate, momentum=0.1, nesterov=True)
optimizer =  torch.optim.Adam(lstm.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# set the forget gate params to 1. to NOT forget stuff at first
lstm.lstm.weight_ih_l0.data[NET_SIZE:NET_SIZE*2, :] = torch.ones_like(lstm.lstm.weight_ih_l0.data[NET_SIZE:NET_SIZE*2, :])
lstm.lstm.weight_hh_l0.data[NET_SIZE:NET_SIZE*2, :] = torch.ones_like(lstm.lstm.weight_hh_l0.data[NET_SIZE:NET_SIZE*2, :])
lstm.lstm.bias_ih_l0.data[NET_SIZE:NET_SIZE*2] = torch.ones_like(lstm.lstm.bias_ih_l0.data[NET_SIZE:NET_SIZE*2])
lstm.lstm.bias_hh_l0.data[NET_SIZE:NET_SIZE*2] = torch.ones_like(lstm.lstm.bias_hh_l0.data[NET_SIZE:NET_SIZE*2])

lstm.lstm.flatten_parameters()


def train(num_epochs, model, Ns, network_number):
    M_MIN = Ns[-1] + 2
    M_MAX = M_MIN + 3 * Ns[-1]

    # stats
    losses = []
    accuracies = []

    # save init
    lstm_subdir = _save_lstm(model, BASE_PATH, None, network_number, init=True)

    # Train the model
    total_step = TRAINING_STEPS
    for epoch in range(num_epochs):
        losses_step = []
        for i in range(TRAINING_STEPS):
            optimizer.zero_grad()

            M = np.random.randint(M_MIN, M_MAX)

            sequences, labels = make_batch_Nbit_pair_parity(Ns, M, BATCH_SIZE)
            sequences = sequences.to(device)
            labels = [l.to(device) for l in labels]

            # Forward pass
            out, out_class = model(sequences)

            # Backward and optimize
            loss = 0.
            for N_i in range(len(Ns)):
                loss += criterion(out_class[N_i], labels[N_i])

            losses_step.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)  # gradient clipping
            optimizer.step()
        losses.append(np.mean(losses_step))

        # Test and measure accuracy
        correct_N = np.zeros_like(Ns)
        total = 0
        for j in range(TEST_STEPS):
            with torch.no_grad():
                M = np.random.randint(M_MIN, M_MAX)

                sequences, labels = make_batch_Nbit_pair_parity(Ns, M, BATCH_SIZE)
                sequences = sequences.to(device)
                labels = [l.to(device) for l in labels]

                out, out_class = model(sequences)

                for N_i in range(len(Ns)):
                    predicted = torch.max(out_class[N_i], 1)[1]

                    correct_N[N_i] += (predicted == labels[N_i]).sum()
                    total += labels[N_i].size(0)

        accuracy = 100 * correct_N / float(total) * len(Ns)
        accuracies.append(accuracy)

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy {:.4f}  %'
              .format(epoch + 1, num_epochs, i + 1, total_step, losses[-1], np.mean(accuracy)), flush=True)
        print('({N}, accuracy):\n' + ''.join([f'({Ns[i]}, {accuracy[i]:.4f})\n' for i in range(len(Ns))]), flush=True)

        stats = {'loss': losses,
                 'accuracy': accuracies}
        np.save(f'{lstm_subdir}/stats.npy', stats)
        # np.save(f'{MAIN_DIR}/{EXPERIMENT_NAME}/stats.npy', stats)

        # curriculum stuff + save
        if np.mean(accuracy) > 98.:
            if accuracy[-1] > 98.:
                print(f'Finishing training for N = ' + str(Ns) + '...', flush=True)

                _save_lstm(model, BASE_PATH, Ns[-1], network_number, init=False)

                # append new curriculum
                # multi-head
                Ns = Ns + [Ns[-1] + 1 + i for i in range(NHEADS)]

                # single-head
                # Ns = [Ns[-1] + 1]

                M_MIN = Ns[-1] + 2
                M_MAX = M_MIN + 3 * Ns[-1]
                print(f'N = {Ns[0]}, {Ns[-1]}', flush=True)

    return stats


NHEADS = 1
# BASE_PATH = '../../trained_models'

if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('-b', '--base_path', type=str, dest='base_path',
                        help='The base path to save results.')
    parser.add_argument('-n', '--network_number', type=int, dest='network_number',
                        help='The run number of the network, to be used as a naming suffix for savefiles.')
    args = parser.parse_args()

    BASE_PATH = args.base_path
    train(num_epochs=100, model=lstm, Ns=[2], network_number=args.network_number)
