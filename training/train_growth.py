import sys
import os
import time

# Get the absolute path of the parent directory of 'src'
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add both the parent directory and the 'src' directory to the module search path
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

import torch
import torch.nn as nn

import numpy as np
import argparse
from tqdm import tqdm

from src.models.RNN_hier import RNN_Hierarchical
from src.models.RNN_Stack import RNN_Stack
import src.tasks as tasks
from src.utils import save_model, generate_subdir, save_configs
import config_parser


def train():
    """Only to be run from __main__ here to guarantee that the global variables are defined.

    Global variables:
    MODEL: RNN_Hierarchical
    CONFIGS: dict of hyperparameters
    CRITERION: nn.CrossEntropyLoss
    OPTIMIZERS: dict of torch.optim.SGD
    SCHEDULERS: dict of torch.optim.lr_scheduler.ExponentialLR
    TASK_FUNCTION: function that returns input data and their labels.
    NS: list of N's that the model should solve.

    todo: these should all be passed as arguments to the function, but I also don't want there to be a danger of
          mismatch between attributes of the model that is passed and what the config says.
    """
    losses = []
    accuracies = []


    # Train the model
    for epoch in tqdm(range(num_epochs)):
        losses_step = []
        for i in range(TRAINING_STEPS):
            zero_grad_helper(OPTIMIZERS)
            sequences, labels = TASK_FUNCTION(Ns, BATCH_SIZE)
            sequences = sequences.permute(1, 0, 2).to(device)
            labels = [l.to(device) for l in labels]

            # Forward pass
            out, out_class = MODEL(sequences)

            # Backward and optimize
            loss = 0.
            for N_i in range(len(Ns)):
                # todo: forward pass for all N_i, requires custom forward pass, that takes as input number of N's you actually have this far into the encephalization.
                loss += CRITERION(out_class[N_i], labels[N_i])
            loss.backward()
            losses_step.append(loss.item())
            # gradient clipping
            nn.utils.clip_grad_norm_(MODEL.parameters(), max_norm=2.0, norm_type=2)
            # OPTIMIZER.step()
            # Step the optimizers in every training step:
            stepper(stepper_object=OPTIMIZERS, max_depth=MODEL.current_depth.item(), num_steps=1)

        # Step the schedulers in every epoch:
        # stepper(stepper_object=SCHEDULERS, max_depth=model.current_depth.item(), num_steps=1)

        losses.append(np.mean(losses_step))

        correct_N = torch.zeros_like(torch.tensor(Ns)).to(CONFIGS['DEVICE'])
        total = 0
        for j in range(CONFIGS['TEST_STEPS']):
            with torch.no_grad():
                sequences, labels = TASK_FUNCTION(Ns, CONFIGS['BATCH_SIZE'])
                sequences = sequences.permute(1, 0, 2).to(CONFIGS['DEVICE'])
                labels = [l.to(CONFIGS['DEVICE']) for l in labels]

                out, out_class = MODEL(sequences)

                for N_i in range(len(Ns)):
                    predicted = torch.max(out_class[N_i], 1)[1]
                    correct_N[N_i] += (predicted == labels[N_i]).sum()
                    total += labels[N_i].size(0)

        accuracy = 100 * correct_N / float(total) * len(Ns)
        accuracies.append(accuracy)

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy {:.4f}  %'
              .format(epoch + 1, CONFIGS["NUM_EPOCHS"], i + 1, CONFIGS['TRAINING_STEPS'], losses[-1], accuracy.mean()), flush=True)
        print('({N}, accuracy):\n' + ''.join([f'({Ns[i]}, {accuracy[i]:.4f})\n' for i in range(len(Ns))]), flush=True)

        stats = {'loss': losses,
                 'accuracy': accuracies,
                 'time': time.time(),
                 'epoch': epoch,
                 'N': Ns,}

        np.save(f'{subdir}/stats.npy', stats)

        # curriculum stuff + save
        if accuracy.mean() > 98.:  # so it doesn't forget the older tasks
            if accuracy[-1] > 98.:
                print(f'Saving model for N = ' + str(Ns) + '...', flush=True)
                save_model(MODEL, rnn_subdir=subdir, N_max=Ns[-1], N_min=Ns[0])

                if CONFIGS['CURRICULUM'] == 'grow':
                    Ns = Ns + [Ns[-1] + 1]  # grow by 1 module/head each time.
                    MODEL.current_depth.data += 1  # change to MODEL.current_depth.data += 1. Register as parameter so torch dumps it.

                    d_int = MODEL.current_depth.item()
                    for layer_name in ['input_layers', 'w_hh', 'w_ff_in', 'fc']:
                        new_layer = MODEL.modules[f'{d_int}:{layer_name}']
                        last_layer = MODEL.modules[f'{d_int - 1}:{layer_name}']
                        new_layer.weight.data = last_layer.weight.data * (1 + CONFIGS['WEIGHT_NOISE'] * torch.randn_like(last_layer.weight.data))
                        new_layer.bias.data = last_layer.bias.data * (1 + CONFIGS['BIAS_NOISE'] * torch.randn_like(last_layer.bias.data))
                    new_taus = MODEL.taus[f'{d_int}']
                    last_taus = MODEL.taus[f'{d_int - 1}']
                    new_taus.data = last_taus.data * (1 + CONFIGS['TAUS_NOISE'] * torch.randn_like(last_taus.data))

                    stepper(stepper_object=SCHEDULERS, max_depth=d_int - 1, num_steps=CONFIGS['FREEZING_STEPS'])
                else:
                    raise Exception(f"train_growth.py only supports the 'grow' curriculum, you provided: {CONFIGS['CURRICULUM']}")
                print(f'N = {Ns[0]}, {Ns[-1]}', flush=True)

    return stats


def zero_grad_helper(optimizers):
    for name, opt in optimizers.items():
        opt.zero_grad()


def stepper(stepper_object, max_depth, num_steps):
    for d in range(max_depth):
        for m in ['input_layers', 'w_hh', 'w_ff_in', 'fc']:
            try:
                for _ in range(num_steps):
                    stepper_object[f'{d}:{m}'].step()
            # so I allow for w_ff_in to not exist for d == 0 but still raise if layer is not found for d > 0.
            except KeyError:
                if 'w_ff_in' in m and d == 0:
                    continue
                else:
                    print(f'Error: {d}:{m} not found in stepper_object.', flush=True)
                    raise

###############################################################

if __name__ == '__main__':

    # Create an ArgumentParser object
    CONFIGS = config_parser.parse_args()

    # Access the values of the arguments
    print('CURRICULUM:', CONFIGS["CURRICULUM"])
    print('TASK:', CONFIGS["TASK"])
    print('SEED:', CONFIGS["SEED"])

    SEED = CONFIGS["SEED"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    # Figure out which task
    if CONFIGS["TASK"] == 'parity':
        TASK_FUNCTION = tasks.make_batch_Nbit_pair_parity
    elif CONFIGS["TASK"] == 'dms':
        TASK_FUNCTION = tasks.make_batch_multihead_dms
    else:
        print('Unrecognized task:', CONFIGS["TASK"])


    if CONFIGS['CURRICULUM'] == 'grow':
        Ns_init = [2]
        INIT_HEADS = 1
    else:
        raise Exception(f"Unrecognized curriculum type: {CONFIGS['CURRICULUM']}")
    ###############################################################


    CRITERION = nn.CrossEntropyLoss()

    tau_affix = f"train_tau={CONFIGS['TRAIN_TAU']}" if not CONFIGS['TRAIN_TAU'] else ""
    AFFIXES = [tau_affix]

    MODEL = RNN_Hierarchical(max_depth=CONFIGS['MAX_DEPTH'],
                           input_size=CONFIGS['INPUT_SIZE'],
                           net_size=CONFIGS['NET_SIZE'],
                           num_classes=CONFIGS['NUM_CLASSES'],
                           bias=CONFIGS['BIAS'],
                           num_readout_heads_per_mod=CONFIGS['NUM_READOUT_HEADS_PER_MOD'],
                           tau=1.,
                           train_tau=CONFIGS['TRAIN_TAU']
                           )

    MODEL.to(CONFIGS['DEVICE'])

    OPTIMIZERS = {}
    SCHEDULERS = {}
    for d in range(CONFIGS['MAX_DEPTH']):
        opt_params = {}
        for m in ['input_layers', 'w_hh', 'w_ff_in', 'fc']:
            if f'{d}:{m}' in MODEL.modules.keys():  # to control for the fact that w_ff_in only exists for d > 0
                OPTIMIZERS[f'{d}:{m}'] = torch.optim.SGD(MODEL.modules[f'{d}:{m}'].parameters(), lr=CONFIGS['LEARNING_RATE'],
                                                         momentum=CONFIGS['MOMENTUM'], nesterov=True)
                SCHEDULERS[f'{d}:{m}'] = torch.optim.lr_scheduler.ExponentialLR(OPTIMIZERS[f'{d}:{m}'], gamma=CONFIGS['GAMMA'])

    NS = list(range(2, CONFIGS['MAX_DEPTH']))

    BASE_PATH = os.path.join(parent_dir, 'trained_models')
    subdir = generate_subdir(curriculum_type=CONFIGS['CURRICULUM'],
                             n_heads=len(NS),
                             n_forget=CONFIGS['NUM_FORGET'],
                             task=CONFIGS['task'],
                             network_number=0,
                             base_path=BASE_PATH,
                             affixes=AFFIXES,
                             timestamp_subdir_fmt="%Y-%b-%d-%H_%M_%S")
    save_configs(subdir, CONFIGS)
    # save init
    save_model(MODEL, rnn_subdir=subdir, N_max=NS[-1], N_min=NS[0], init=True)

    stats = train()