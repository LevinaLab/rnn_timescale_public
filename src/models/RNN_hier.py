import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN_Stack(nn.Module):
    '''
    Custom RNN class so that we can make RNNs with layers with different sizes,
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
    tau: float, if train_tau = False
        decay time constant of a single neuron outside the network
    train_tau: boolean, default False
        if we train the taus
    '''

    def __init__(self, input_size=28,
                 net_size=[28, 28, 28],
                 num_classes=2,
                 bias=True,
                 num_readout_heads=1,
                 tau=1.,
                 train_tau=False
                 ):
        super(RNN_Stack, self).__init__()
        # todo: create the full network with moduless right off the bat.
        self.input_size = input_size
        self.net_size = net_size
        self.num_classes = num_classes
        self.bias = bias
        self.num_readout_heads = num_readout_heads
        self.tau = tau
        self.train_tau = train_tau

        self.input_layers = [nn.Linear(input_size, net_size[0], bias=bias)]

        # recurrent connections
        self.w_hh = [nn.Linear(net_size[i], net_size[i], bias=bias)
                     for i in range(len(net_size))]
        # forward connections
        self.w_hh_i = [nn.Linear(net_size[i], net_size[i + 1], bias=bias)
                       for i in range(len(net_size) - 1)]

        # setting the single neuron tau
        if self.train_tau:
            # fixed trainble tau
            self.taus = [nn.Parameter(1 + 1. * torch.rand(net_size[i]), requires_grad=True) for i in range(len(net_size))]
        else:
            # fixed tau
            self.taus = [nn.Parameter(torch.Tensor([self.tau]), requires_grad=False) for i in range(len(net_size))]

        self.fc = [nn.Linear(net_size[-1], num_classes, bias=bias) for i in range(num_readout_heads)]

        self.parameterlist = nn.ParameterList(self.taus)
        self.modulelist = nn.ModuleList(self.input_layers + self.w_hh + self.w_hh_i + self.fc)

        self.afunc = nn.LeakyReLU()


    def forward(self, data, hs=None, classify_in_time=False, savetime=False, index_in_head=None):
        '''
        input: data [batch_size, sequence length, input_features]
        output:
                hs: list of hidden layer states at final time
                hs_t (if savetime): list of hidden layer states for all time points,
                    shape=[time, num_layers]
                out: readout from the fc layers at the end,
                    shape=[batch_size, self.num_classes],
                or if classify_in_time == True:
                    shape=[time, batch_size, self.num_classes]
        index_in_head: if we want to have faster evaluation,
            we can pass the index of the head that we want to return
        '''
        if hs is None:
            # hs = [torch.zeros(data.size(1), self.net_size[i]).to(device) for i in range(len(self.net_size))]
            hs = [0.1 * torch.rand(data.size(1), self.net_size[i]).to(device) for i in range(len(self.net_size))]

        hs_t = [[h.clone() for h in hs]]
        x = torch.stack([input_layer(data) for input_layer in self.input_layers]).mean(dim=0) # [ *, H_in] -> [*, H_out]

        if classify_in_time:
            out = []

        # putting self connenctions to zero
        for i in range(len(self.net_size)):
            self.w_hh[i].weight.data.fill_diagonal_(0.)

        for t in range(data.size(0)):
            inp = x[t, ...]
            for i in range(len(self.net_size)):  # net_size is normally just 1.

                if self.train_tau:
                     ## with training taus
                    if i == 0:
                        hs[i] = (1 - 1/torch.clamp(self.taus[i], min=1.)) * hs[i] + \
                                (self.w_hh[i](hs[i]) + \
                                inp)/torch.clamp(self.taus[i], min=1.)
                    else:
                        hs[i] = (1 - 1/torch.clamp(self.taus[i], min=1.)) * hs[i] + \
                                (self.w_hh[i](hs[i]) + self.w_hh_i[i-1](hs[i-1]) + \
                                inp)/torch.clamp(self.taus[i], min=1.)

                else:
                    ## with fixed tau
                    if i == 0:
                        hs[i] = (1 - 1/(self.taus[i])) * hs[i] + \
                                (self.w_hh[i](hs[i]) + inp)/(self.taus[i])
                    else:
                        hs[i] = (1 - 1/(self.taus[i])) * hs[i] + \
                                (self.w_hh[i](hs[i]) + self.w_hh_i[i-1](hs[i-1]) + inp)/(self.taus[i])


                hs[i] = self.afunc(hs[i])
            if savetime:
                # hs_t.append([h.detach().to('cpu') for h in hs])
                hs_t.append([h.clone() for h in hs])
            if classify_in_time:
                if index_in_head is None:
                    out.append([self.fc[i](hs[-1]) for i in range(self.num_readout_heads)])
                else:
                    out.append([self.fc[index_in_head](hs[-1])])

        if not classify_in_time:
            if index_in_head is None:
                out = [self.fc[i](hs[-1]) for i in range(self.num_readout_heads)]
            else:
                out = [self.fc[index_in_head](hs[-1])]

        if savetime:
            return hs_t, out

        return hs, out


def init_model(
    INPUT_SIZE=1,
    NET_SIZE=[500],
    NUM_CLASSES=2,
    BIAS=True,
    NUM_READOUT_HEADS=100,
    TRAIN_TAU=True,
    DEVICE='cpu',
):
    # init new model
    rnn = RNN_Stack(input_size=INPUT_SIZE,
                    net_size=NET_SIZE,
                    num_classes=NUM_CLASSES,
                    bias=BIAS,
                    num_readout_heads=NUM_READOUT_HEADS,
                    tau=1.,
                    train_tau=TRAIN_TAU
                    ).to(DEVICE)

    return rnn
