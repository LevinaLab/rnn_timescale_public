import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RNN_stack(nn.Module):
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

    def __init__(self, input_size=1, net_size=[28, 28, 28], num_classes=2, bias=True, num_readout_heads=1, tau = 1., train_tau = False):
        super(RNN_stack, self).__init__()
        
        # defining parameters for the class
        self.input_size = input_size
        self.net_size = net_size
        self.num_classes = num_classes
        self.bias = bias
        self.num_readout_heads = num_readout_heads
        self.tau = tau 
        self.train_tau = train_tau

        #------------- initializing the layers (torch has its automatic way, it's a uniform distribution)
        # input layer
        self.input_layers = [nn.Linear(input_size, net_size[0], bias=bias)]
        
        # connectivity within a recurrent to itself (e.g., 500*500)
        self.w_hh = [nn.Linear(net_size[i], net_size[i], bias=bias) for i in range(len(net_size))]
        
        # connectivity from one recurrent layer to another recurrent layer (multi-layer network)
        self.w_hh_i = [nn.Linear(net_size[i], net_size[i + 1], bias=bias) for i in range(len(net_size) - 1)]
        
        # setting the single neuron tau
        if self.train_tau:
            # trainable tau 
            self.taus = [nn.Parameter(1 + 1. * torch.rand(net_size[i]), requires_grad=True) for i in range(len(net_size))]  
        else:
            # fixed tau 
            self.taus = [nn.Parameter(torch.Tensor([self.tau]), requires_grad=False) for i in range(len(net_size))]
         
        
        # readout layer
        self.fc = [nn.Linear(net_size[-1], num_classes, bias=bias) for i in range(num_readout_heads)]

      

        self.parameterlist = nn.ParameterList(self.taus)
        self.modulelist = nn.ModuleList(self.input_layers + self.w_hh + self.w_hh_i + self.fc)
        # self.parameterlist = nn.ParameterList(self.input_layers + self.w_hh + self.w_hh_i + self.fc + self.taus)
        # self.modulelist = nn.ModuleList(self.input_layers + self.w_hh + self.w_hh_i + self.fc + self.taus)
        
        # defining the activation function
        self.afunc = nn.LeakyReLU()


    def forward(self, data, hs=None, classify_in_time=False, savetime=False):
        '''
        input: data [batch_size, sequence length, input_features]
        output:
                hs: list of hidden layer states for all layers at the final time
                    [l for l in layers]
                hs_t (if savetime): list in time of list of hidden layer states for all layers in sequence length:
                                    [[l for l in layers] for layers in time]
                out: readout from the fc layers at the end, of size [batch_size, self.num_classes],
                or [time, batch_size, self.num_classes] if classify_in_time == True
        '''
        if hs is None:
            hs = [torch.zeros(data.size(1), self.net_size[i]).to(device) for i in range(len(self.net_size))]
         
        hs_t = [[h.clone() for h in hs]]
        x = torch.stack([input_layer(data) for input_layer in self.input_layers]).mean(dim=0) # [ *, H_in] -> [*, H_out]

        if classify_in_time:
            out = []


        for t in range(data.size(0)):
            inp = x[t, ...]
            
            # loop iterating over layers (if we only have one layer then there is no iteration)
            for i in range(len(self.net_size)):
                
                # putting self connenctions to zero
                self.w_hh[i].weight.data.fill_diagonal_(0.)
                
                              
               
                if self.train_tau:
                     ## with training taus
                    if i == 0:
                        hs[i] = (1 - 1/torch.clip(self.taus[i], min=1.)) * hs[i] + \
                                (self.w_hh[i](hs[i]) + \
                                inp) /torch.clip(self.taus[i], min=1.)
                    else:
                        hs[i] = (1 - 1/torch.clip(self.taus[i], min=1.)) * hs[i] + \
                                (self.w_hh[i](hs[i]) + self.w_hh_i[i-1](hs[i-1]) +\
                                inp) /torch.clip(self.taus[i], min=1.)

                
                else:
                    ## with fixed tau
                    if i == 0:
                        hs[i] = (1- 1/(self.taus[i])) * hs[i] + (self.w_hh[i](hs[i]) + inp)/(self.taus[i])
                    else:
                        hs[i] = (1- 1/(self.taus[i])) * hs[i] +\
                        (self.w_hh[i](hs[i]) + self.w_hh_i[i-1](hs[i-1]) + inp)/(self.taus[i])


                # applying non-linearity
                hs[i] = self.afunc(hs[i])
            if savetime:            
                hs_t.append([h.clone() for h in hs])
            
             # this part was for reading out at every time step
            if classify_in_time:
                out.append( [self.fc[i](hs[-1]) for i in range(self.num_readout_heads)] )
       
        # this part was for reading out at the end 
        if not classify_in_time:
            out = [self.fc[i](hs[-1]) for i in range(self.num_readout_heads)]       

        if savetime:
            return hs_t, out

        return hs, out
    
