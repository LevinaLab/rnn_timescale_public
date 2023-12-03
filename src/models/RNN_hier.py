from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN_Hierarchical(nn.Module):
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

    def __init__(self, depth=1,  # how many "modules" there are in the hierarchy.
                 input_size=28,
                 net_size=[100],
                 num_classes=2,
                 bias=True,
                 num_readout_heads=1,
                 tau=1.,
                 train_tau=False,
                 ):
        super(RNN_Hierarchical, self).__init__()

        self.input_size = input_size
        self.net_size = net_size
        self.num_classes = num_classes
        self.bias = bias
        self.num_readout_heads = num_readout_heads
        self.tau = tau
        self.train_tau = train_tau
        self.depth = depth  # todo: since there is 1 read-out head per module, depth = num_readout_heads so one is redundant
        self.current_depth = 0

        self.afunc = nn.LeakyReLU()

        self.module_dict = defaultdict(dict)  # module_dict[network][layer] = layer_object
        self.modules = nn.ModuleList()

        for d in range(depth):

            self.module_dict[d]['input_layers'] = [nn.Linear(input_size, net_size[0], bias=bias)]
            # recurrent connections
            self.module_dict[d]['w_hh'] = [nn.Linear(net_size[i], net_size[i], bias=bias)
                                           for i in range(len(net_size))]


            # forward connections from another module in the hierarchy
            # todo: assumes that the superior module is single layer and of the same size as this current module
            self.module_dict[d]['w_ff_in'] = nn.Linear(net_size[0], net_size[0], bias=bias)

            # setting the single neuron tau
            if self.train_tau:
                # fixed trainble tau
                self.module_dict[d]['taus'] = [nn.Parameter(1 + 1. * torch.rand(net_size[i]), requires_grad=True) for i in range(len(net_size))]
            else:
                # fixed tau
                self.module_dict[d]['taus'] = [nn.Parameter(torch.Tensor([self.tau]), requires_grad=False) for i in range(len(net_size))]

            self.module_dict[d]['fc'] = [nn.Linear(net_size[-1], num_classes, bias=bias) for i in range(num_readout_heads)]

            self.parameterlist = nn.ParameterList(self.module_dict[j]['taus'])  # todo: what is this good for? it's not used anywhere.
            for k, v in self.module_dict[d].items():
                self.modules.extend(v)  # todo: not sure if/why its necessary to have them all declared in a nn.ModuleList()


    def forward(self, data, hier_signal, net_hs=None, hs=None, classify_in_time=False, savetime=False, index_in_head=None):
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
        if net_hs is None:  # todo: when is the function ever called with net_hs =! None?
            # hs = [torch.zeros(data.size(1), self.net_size[i]).to(device) for i in range(len(self.net_size))]
            net_hs = []
            for d in range(self.current_depth):
                hs = [0.1 * torch.rand(data.size(1), self.net_size[i]).to(device) for i in range(len(self.net_size))]
                net_hs.append(hs)

        net_hs_t = [[h.clone() for h in hs] for hs in net_hs]
        net_x = []
        for d in range(self.current_depth):
            # todo: what does this stack().mean() do? Why would ther be any averaging anyways?
            x = torch.stack([input_layer(data) for input_layer in self.module_dict[d]['input_layers']]).mean(dim=0) # [ *, H_in] -> [*, H_out]
            net_x.append(x)
        # if classify_in_time: # todo: this seems to be always False in the existing training code so i'll remove.
        out = []

        # putting self connenctions to zero
        for d in range(self.current_depth):
            for i in range(len(self.net_size)):
                self.module_dict[d]['w_hh'][i].weight.data.fill_diagonal_(0.)

        for t in range(data.size(0)):
            inp = x[t, ...]

            for j in range(self.current_depth):
                hs = net_hs[j]
                taus = self.module_dict[j]['taus']
                w_hh = self.module_dict[j]['w_hh']
                w_ff_in = self.module_dict[j]['w_ff_in']
                for i in range(len(self.net_size)):  # net_size is normally just 1.
                    if self.train_tau:
                        raise NotImplementedError
                    else:
                        ## with fixed tau
                        if i == 0:
                            hs[i] = (1 - 1/taus[i]) * hs[i] + \
                                    (w_hh[i](hs[i]) + w_ff_in(hier_signal) + inp)/(taus[i])
                        else:
                            hs[i] = (1 - 1/(taus[i])) * hs[i] + \
                                    (w_hh[i](hs[i]) + w_ff_in[i-1](hs[i-1]) +
                                     w_ff_in(hier_signal) + inp)/(taus[i])

                    hs[i] = self.afunc(hs[i])
            if savetime: # todo: why append? Do we want to want to save the hidden layers' states before and after the update?
                # hs_t.append([h.detach().to('cpu') for h in hs])
                net_hs_t.append([[h.clone() for h in hs] for hs in net_hs])
            # if classify_in_time:  # todo: let's just assume classify_in_time == False for now.
            #     if index_in_head is None:
            #         out.append([self.module_dict['fc'][i](net_hs[i][-1]) for i in range(self.current_depth)])
            #     else:
            #         out.append([self.fc[index_in_head](hs[-1])])

        if not classify_in_time:
            if index_in_head is None:
                out = [self.module_dict['fc'][i](hs[-1]) for i in range(self.current_depth)]
            else:
                out = [self.module_dict['fc'][index_in_head](hs[-1])]

        if savetime:
            return net_hs_t, out

        return net_hs, out


def init_model(
    INPUT_SIZE=1,
    NET_SIZE=[50],
    DEPTH = 10,
    NUM_CLASSES=2,
    BIAS=True,
    NUM_READOUT_HEADS=100,
    TRAIN_TAU=True,
    DEVICE='cpu',
):
    # init new model
    rnn = RNN_Hierarchical(depth=1,
                    input_size=INPUT_SIZE,
                    net_size=NET_SIZE,
                    num_classes=NUM_CLASSES,
                    bias=BIAS,
                    num_readout_heads=NUM_READOUT_HEADS,
                    tau=1.,
                    train_tau=TRAIN_TAU
                    ).to(DEVICE)

    return rnn
