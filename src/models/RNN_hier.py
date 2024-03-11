from collections import defaultdict

import torch
import torch.nn as nn


class RNN_Hierarchical(nn.Module):
    '''
    Custom RNN class so that we can make RNNs with layers with different sizes,
    and also to save hidden_layer states through time.

     Parameters
    -----------
    input_size: int
        size of input (it has been always 1)
    net_size: list of lists of ints
        list of number of neurons per layer of each module: net_size[module][layer]
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

    def __init__(self, max_depth, net_size, device,
                 input_size=1,
                 num_classes=2,
                 bias=True,
                 num_readout_heads_per_mod=1,
                 fixed_tau_val=1.,
                 train_tau=True,
                 ):
        """

        Args:
            max_depth: [int] how many modules there are in the hierarchy
            net_size:  [list of lists of ints], number of neurons per module, per layer. i.e. Each list is a module.
            input_size: [int] size of input (it has been always 1)
            num_classes: [int] number of classes for classification at the end of each "head"
            bias: [bool] if we include bias terms
            num_readout_heads_per_mod: [int] number of readout heads per module
            fixed_tau_val: [float] for when train_tau = False
            train_tau: [bool] if we let backprop train the taus or not
        """
        super(RNN_Hierarchical, self).__init__()

        self.input_size = input_size
        self.net_size = net_size
        self.num_classes = num_classes
        self.bias = bias
        self.num_readout_heads_per_mod = num_readout_heads_per_mod
        self.fixed_tau = fixed_tau_val
        self.train_tau = train_tau
        self.max_depth = max_depth  # todo: since there is 1 read-out head per module, depth = num_readout_heads so one is redundant
        self.current_depth = nn.Parameter(torch.tensor([1]), requires_grad=False)  #  The network starts with a single module and therefore depth=1. We register it as a parameter so that it is saved in the model.  #
        self.device = device
        self.afunc = nn.LeakyReLU()
        self.taus = defaultdict()
        self.modules = defaultdict()  # module_dict[network][layer] = layer_object  # todo: should probably use nn.ModuleDict() instead. BUt likely doesn't support nested dicts. Can get around it using tuple keys: (module number[int], parameter name[str])

        for d in range(self.max_depth):
            # todo: for now just do single layer modules. Later we can add multi-layer modules.
            self.modules[f'{d}:input_layers'] = nn.Linear(input_size, net_size[d][0], bias=bias)
            # recurrent connections
            self.modules[f'{d}:w_hh'] = nn.Linear(net_size[d][0], net_size[d][0], bias=bias)
                                           #for i in range(len(net_size))]

            # forward connections from another module in the hierarchy
            if d > 0:
                # only defined for d > 0 since the very first module doesn't receive any inputs other than input data.
                self.modules[f'{d}:w_ff_in'] = nn.Linear(net_size[d - 1][-1], net_size[d][0], bias=bias)  # make this a list for consistency, even though only one element.

            # setting the single neuron tau
            if self.train_tau: # todo: specify the depth
                # fixed trainble tau  # todo: is this fixed or trainable? It can't be both.
                self.taus[f'{d}'] = nn.Parameter(1 + 1. * torch.rand(net_size[d][0]), requires_grad=True)
            else:
                # fixed tau
                self.taus[f'{d}'] = nn.Parameter(torch.Tensor([self.fixed_tau]), requires_grad=False)

            # todo: for 'grow' curriculum it only makes sense to have 1 read out head per module, each with their own fc parameter sets.
            self.modules[f'{d}:fc'] = nn.Linear(net_size[d][-1], num_classes, bias=bias)


        self.parameter_dict = nn.ParameterDict(self.taus)  # so that parameters are registered by torch.
        self.non_trained_params = nn.ParameterList([self.current_depth])
        self.module_dict = nn.ModuleDict(self.modules)


    def forward(self, data, net_hs=None, hs=None, classify_in_time=False, savetime=False, index_in_head=None):
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
                hs = [0.1 * torch.rand(data.size(1), net_size).to(self.device) for net_size in self.net_size[d]]
                net_hs.append(hs)

        # net_hs_t = [[h.clone() for h in hs] for hs in net_hs]  # todo: this isn't used anywhere, except save_time!!
        net_x = []
        for d in range(self.current_depth):
            # todo: what does this stack().mean() do? Why would ther be any averaging anyways?
            # todo: NEED TO GET RID OF THIS. This was a hack to network work with signal that is more than 1D.
            x = torch.stack([self.modules[f'{d}:input_layers'](data)]).mean(dim=0) # [ *, H_in] -> [*, H_out]
            net_x.append(x)
        # if classify_in_time: # todo: this seems to be always False in the existing training code so i'll remove.
        out = []

        # putting self connenctions to zero
        for d in range(self.current_depth):
            for i in range(len(self.net_size[d])):
                self.modules[f'{d}:w_hh'].weight.data.fill_diagonal_(0.)

        net_hs_t = []
        for t in range(data.size(0)):
            for d in range(self.current_depth):
                  # todo: for now just do 1-layer modules and get the values from this single layer immediately
                if d > 0:
                    # if we are in the second module or higher, we have input from the previous (j-1) module
                    hier_signal = self.modules[f'{d}:w_ff_in'](net_hs[d - 1][-1])
                else:
                    hier_signal = 0

                for i in range(len(self.net_size[d])):  # net_size[d] is normally just 1.
                    if self.train_tau:
                        if d == 0:  # todo: this is redundant if i'm already setting hier_signal to 0 for d > 0.
                            hs = (1 - 1 / torch.clamp(self.parameter_dict[f'{d}'], min=1.)) * net_hs[d][0] + \
                                 self.afunc(self.modules[f'{d}:w_hh'](net_hs[d][0]) + net_x[d][t, ...]) / (torch.clamp(self.parameter_dict[f'{d}'], min=1.))
                        else:
                            hs = (1 - 1 / (torch.clamp(self.parameter_dict[f'{d}'], min=1.))) * net_hs[d][0] + \
                                 self.afunc(self.modules[f'{d}:w_hh'](net_hs[d][0]) + hier_signal + net_x[d][t, ...]) / (torch.clamp(self.parameter_dict[f'{d}'], min=1.))
                    else:
                        ## with fixed tau
                        # Note: Every layer in the module gets the hier_signal from previous module. #todo: correct?
                        if d == 0: # todo: this is redundant if i'm already setting hier_signal to 0 for d > 0.
                            hs = (1 - 1/self.parameter_dict[f'{d}']) * net_hs[d][0] + \
                                    self.afunc(self.modules[f'{d}:w_hh'](net_hs[d][0]) + net_x[d][t, ...])/(self.parameter_dict[f'{d}'])
                        else:
                            hs = (1 - 1/(self.parameter_dict[f'{d}'])) * net_hs[d][0] + \
                                    self.afunc(self.modules[f'{d}:w_hh'](net_hs[d][0]) + hier_signal + net_x[d][t, ...])/(self.parameter_dict[f'{d}'])
                net_hs[d][0] = hs
            if t == data.size(0) - 1:  # todo: do we need this if statement? Just put it outside the loop.
                out = [self.modules[f'{d_i}:fc'](net_hs[d_i][0]) for d_i in range(self.current_depth)]

            if savetime:  # todo: why append? Do we want to save the hidden layers' states before and after the update?
                # hs_t.append([h.detach().to('cpu') for h in hs])
                net_hs_t.append([hs_[0].clone() for hs_ in net_hs])
            # if classify_in_time:  # todo: let's just assume classify_in_time == False for now.
            #     if index_in_head is None:
            #         out.append([self.module_dict['fc'][i](net_hs[i][-1]) for i in range(self.current_depth)])
            #     else:
            #         out.append([self.fc[index_in_head](hs[-1])])


        # if not classify_in_time:
        #     if index_in_head is None:
        #         out = [self.modules[f'{d}:fc'](hs) for d in range(self.current_depth)]
        #     else:
        #         out = [self.modules[f'{d}:fc'][index_in_head](hs)]

        if savetime:
            return net_hs_t, out

        return net_hs, out


def init_model(
    INPUT_SIZE=1,
    NET_SIZE=[50],
    DEPTH = 10,
    NUM_CLASSES=2,
    BIAS=True,
    NUM_READOUT_HEADS_PER_MOD=1,
    TRAIN_TAU=True,
    DEVICE='cpu',
):
    # init new model
    rnn = RNN_Hierarchical(max_depth=5,
                           input_size=INPUT_SIZE,
                           net_size=NET_SIZE,
                           num_classes=NUM_CLASSES,
                           bias=BIAS,
                           num_readout_heads_per_mod=NUM_READOUT_HEADS_PER_MOD,
                           fixed_tau_val=1.,
                           train_tau=TRAIN_TAU
                           ).to(DEVICE)

    return rnn
