import os

import numpy as np
import torch
from torch import nn, einsum


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

    def forward(
        self,
        data,
        h0=None,
        c0=None,
        device='cpu',
        save_readouts=False,
        save_hidden_states=False,
        save_cell_states=False,
        save_forget_gates=False,
        save_output_gates=False,
    ):
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
        if (save_hidden_states or save_forget_gates or save_cell_states or save_output_gates) is True:
            B = data.size(0)  # batch size
            T = data.size(1)  # sequence length
            hidden_states = np.zeros((B, T, self.hidden_size)) if save_hidden_states is True else None
            cell_states = np.zeros((B, T, self.hidden_size)) if save_cell_states is True else None
            forget_gates = np.zeros((B, T, self.hidden_size)) if save_forget_gates is True else None
            output_gates = np.zeros((B, T, self.hidden_size)) if save_output_gates is True else None
            output = torch.zeros((B, T, self.hidden_size))
            for t in range(data.size(1)):
                if t % 10000 == 0:
                    print(f't = {t}')
                if save_forget_gates is True:
                    forget_gates[:, t, :] = torch.sigmoid(
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
                if save_output_gates is True:
                    output_gates[:, t, :] = torch.sigmoid(
                        einsum(
                            "hi,ni->nh",
                            self.lstm.weight_ih_l0[3 * self.hidden_size:4 * self.hidden_size, :],
                            data[:, t, :],
                        ) +
                        einsum(
                            "hg,ng->nh",
                            self.lstm.weight_hh_l0[3 * self.hidden_size:4 * self.hidden_size, :],
                            h0[0, :, :],
                        ) +
                        self.lstm.bias_ih_l0[3 * self.hidden_size:4 * self.hidden_size] +
                        self.lstm.bias_hh_l0[3 * self.hidden_size:4 * self.hidden_size]
                    )
                output_single, (h_0, c_0) = self.lstm(data[:, t, :].unsqueeze(1), (h0, c0))
                output[:, t, :] = output_single[:, 0, :]
                if save_hidden_states is True:
                    hidden_states[:, t, :] = h_0[0]
                if save_cell_states is True:
                    cell_states[:, t, :] = c_0[0]

        else:
            output, (h_n, c_n) = self.lstm(data, (h0, c0))

        # readout = [self.fc[i](h_n[-1]) for i in range(self.num_readout_heads)]
        if save_readouts:
            # num_readout_heads x time x batch_size x 2
            readout = [np.stack([self.fc[i](output[:, t, :]) for t in range(data.size(1))], axis=0) for i in range(self.num_readout_heads)]
        else:
            readout = [self.fc[i](output[:, -1, :]) for i in range(self.num_readout_heads)]

        return_list = [output, readout]
        if save_hidden_states:
            # hidden_states = np.stack([h_n.cpu().numpy() for h_n in hidden_states], axis=1)
            return_list.append(hidden_states)
        if save_cell_states:
            # cell_states = np.stack([c_n.cpu().numpy() for c_n in cell_states], axis=1)
            return_list.append(cell_states)
        if save_forget_gates:
            # forget_gates = np.stack([f_n.cpu().numpy() for f_n in forget_gates], axis=1)
            return_list.append(forget_gates)
        if save_output_gates:
            # output_gates = np.stack([o_n.cpu().numpy() for o_n in output_gates], axis=1)
            return_list.append(output_gates)

        return tuple(return_list)


def load_lstm(base_path, n_max, network_number, curriculum_type, n_min=2, init=False, device='cpu'):
    lstm_subdir = os.path.join(base_path, f'lstm_{curriculum_type}_network_{network_number}')
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
    sequences = generate_binary_sequence(M, BATCH_SIZE)
    # sequences = sequences.permute(1, 0, 2).to(device)

    with torch.no_grad():
        _, _, cell_states, forget_gates = lstm.forward(
            sequences,
            save_cell_states=True,
            save_forget_gates=True,
        )

    # dict of {layer_i: array(timerseries)} where timeseries is shape [timesteps, batch_size, num_neurons]
    # save_dict = {f'l{str(i_l).zfill(2)}': np.array([h_n[i_l].cpu().numpy() for h_n in h_ns]) for i_l in
    #             range(len(NET_SIZE))}
    save_dict = {
        # [batch_size, timesteps, num_neurons]
        'cell_states': cell_states,
        # [batch_size, timesteps, num_neurons]
        'forget_gates': forget_gates,
    }
    return save_dict


def generate_binary_sequence(M, bs):
    # return (torch.rand(M) < torch.rand(1)) * 1.
    return torch.stack([((torch.rand(M) < 0.5) * 1.)*2 - 1 for _ in range(bs)], dim=0).unsqueeze(-1)


def make_batch_Nbit_pair_parity(Ns, M, bs):
    with torch.no_grad():
        # sequences = [generate_binary_sequence(M).unsqueeze(-1) for i in range(bs)]
        sequences = generate_binary_sequence(M, bs)

        labels = [torch.stack([get_parity(sequences[i, :, 0], N) for i in range(sequences.size(0))]) for N in Ns]

    return sequences, labels


def get_parity(vec, N):
    return (((vec + 1)/2)[-N:].sum() % 2).long()
    # return (vec[-N:].sum() % 2).long()
