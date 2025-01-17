"""RNN model with continuous time dynamics."""

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN_Continuous(nn.Module):
    """
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
    afunc: activation function, default nn.LeakyReLU
    train_tau: boolean, default False
        if we train the taus
    init_tau: float, if provided, it will initialize the taus to this mean value, default None
    """
    def __init__(
        self, input_size=28,
        net_size=[28, 28, 28],
        num_classes=2,
        bias=True,
        num_readout_heads=1,
        tau=1.,
        afunc=nn.LeakyReLU,
        train_tau=True,
        init_tau=None,
    ):
        super(RNN_Continuous, self).__init__()

        self.input_size = input_size
        self.net_size = net_size
        self.num_classes = num_classes
        self.bias = bias
        self.num_readout_heads = num_readout_heads
        self.tau = tau
        self.afunc = afunc()
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
            # fixed trainable tau
            self.taus = [nn.Parameter(1 + 1. * torch.rand(net_size[i]), requires_grad=True) for i in range(len(net_size))]
        else:
            # fixed tau
            self.taus = [nn.Parameter(torch.Tensor([self.tau]), requires_grad=False) for i in range(len(net_size))]
        if init_tau is not None:
            for i in range(len(net_size)):
                self.taus[i].data = self.taus[i] * (init_tau / self.taus[i].data.mean())

        self.fc = [nn.Linear(net_size[-1], num_classes, bias=bias) for i in range(num_readout_heads)]

        self.parameterlist = nn.ParameterList(self.taus)
        self.modulelist = nn.ModuleList(self.input_layers + self.w_hh + self.w_hh_i + self.fc)

    def forward(
        self,
        data,
        k_data,
        hs=None,
        classify_in_time=False,
        savetime=False,
        index_in_head=None,
    ):
        """
        input:
            - data [batch_size, sequence length, input_features]
            - k_data: integer >= 1
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
        """
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
            for i in range(len(self.net_size)):

                if self.train_tau:
                    # with training taus rescaled by k_data
                    taus = torch.clamp(self.taus[i] * k_data, min=1.)
                    if i == 0:
                        hs[i] = ((1 - 1/taus) * hs[i]
                                 + self.afunc(self.w_hh[i](hs[i]) + inp) / taus)
                    else:
                        hs[i] = ((1 - 1/taus) * hs[i]
                                 + self.afunc(self.w_hh[i](hs[i]) + self.w_hh_i[i-1](hs[i-1]) + inp) / taus)

                else:
                    # with fixed tau rescaled by k_data
                    if i == 0:
                        hs[i] = ((1 - 1/(self.taus[i] * k_data)) * hs[i]
                                 + self.afunc(self.w_hh[i](hs[i]) + inp) / taus[i])
                    else:
                        hs[i] = ((1 - 1/taus[i]) * hs[i]
                                 + self.afunc(self.w_hh[i](hs[i]) + self.w_hh_i[i-1](hs[i-1]) + inp) / taus[i])

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


def init_model_continuous(
    INPUT_SIZE=1,
    NET_SIZE=[500],
    NUM_CLASSES=2,
    BIAS=True,
    NUM_READOUT_HEADS=100,
    A_FUNC=nn.LeakyReLU,
    TRAIN_TAU=True,
    DEVICE='cpu',
):
    # init new model
    rnn = RNN_Continuous(
        input_size=INPUT_SIZE,
        net_size=NET_SIZE,
        num_classes=NUM_CLASSES,
        bias=BIAS,
        num_readout_heads=NUM_READOUT_HEADS,
        tau=1.,
        afunc=A_FUNC,
        train_tau=TRAIN_TAU
    ).to(DEVICE)

    return rnn
