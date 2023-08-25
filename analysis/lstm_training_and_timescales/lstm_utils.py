import os

import numpy as np
import torch
from torch import nn, einsum

from analysis.timescales.timescales_utils import make_batch_Nbit_parity

EPOCHS = 1000
INPUT_SIZE = 1

NET_SIZE = 500
NUM_LAYERS = 1
NUM_CLASSES = 2
BIAS = True

NUM_READOUT_HEADS = 100


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

    def forward(self, data, h0=None, c0=None, device='cpu', save_hidden_states=False):
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
        if save_hidden_states is True:
            cell_states = []
            forget_gates = []
            for t in range(data.size(1)):
                forget_gates.append(
                    torch.sigmoid(
                        einsum(
                            "hi,ni->nh",
                            self.lstm.weight_ih_l0[self.hidden_size:2 * self.hidden_size, :],
                            data[:, t, :],
                        ) +
                        einsum(
                            "hg,ng->nh",
                            self.lstm.weight_hh_l0[self.hidden_size:2 * self.hidden_size, :],
                            h0[0, :, :],
                        ) +
                        self.lstm.bias_ih_l0[self.hidden_size:2 * self.hidden_size] +
                        self.lstm.bias_hh_l0[self.hidden_size:2 * self.hidden_size]
                    )
                )
                output, (h_0, c_0) = self.lstm(data[:, t, :].unsqueeze(1), (h0, c0))
                cell_states.append(c_0[0])

        else:
            output, (h_n, c_n) = self.lstm(data, (h0, c0))

        # readout = [self.fc[i](h_n[-1]) for i in range(self.num_readout_heads)]
        readout = [self.fc[i](output[:, -1, :]) for i in range(self.num_readout_heads)]

        if save_hidden_states is True:
            return output, readout, cell_states, forget_gates
        else:
            return output, readout


def load_lstm(base_path, n_max, network_number, n_min=2, init=False, device='cpu'):
    lstm_subdir = os.path.join(base_path, f'lstm_network_{network_number}')
    if init:
        rnn_name = f'lstm_init'
    else:
        rnn_name = f'lstm_N{n_min:d}_N{n_max:d}'

    lstm = LSTM_custom(
        input_size=INPUT_SIZE,
        hidden_size=NET_SIZE,
        num_layers=NUM_LAYERS,
        bias=BIAS,
        num_readout_heads=NUM_READOUT_HEADS
    ).to(device)

    checkpoint = torch.load(os.path.join(lstm_subdir, rnn_name), map_location=device)
    lstm.load_state_dict(checkpoint['state_dict'], strict=False)

    return lstm


def simulate_lstm_binary(
    lstm,
    M,
    BATCH_SIZE,
    device='cpu',
):
    # preparing training data and labels
    sequences = make_batch_Nbit_parity(M, BATCH_SIZE)
    # sequences = sequences.permute(1, 0, 2).to(device)

    with torch.no_grad():
        output, readout, cell_states, forget_gates = lstm.forward(sequences, save_hidden_states=True)

    # dict of {layer_i: array(timerseries)} where timeseries is shape [timesteps, batch_size, num_neurons]
    # save_dict = {f'l{str(i_l).zfill(2)}': np.array([h_n[i_l].cpu().numpy() for h_n in h_ns]) for i_l in
    #             range(len(NET_SIZE))}
    save_dict = {
        # [batch_size, timesteps, num_neurons]
        'cell_states': np.stack([c_n.cpu().numpy() for c_n in cell_states], axis=1),
        # [batch_size, timesteps, num_neurons]
        'forget_gates': np.stack([f_n.cpu().numpy() for f_n in forget_gates], axis=1),
    }
    return save_dict
