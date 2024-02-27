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
from src.utils import save_model, generate_subdir


def train(model,
          curriculum_type,
          task,
          num_epochs,
          Ns,  # List of parities that are being trained/tested. Grows with curriculum.
          run_number):
    # stats
    losses = []
    accuracies = []

    subdir = generate_subdir(curriculum_type=curriculum_type,
                             n_heads=len(Ns),
                             n_forget=NUM_FORGET,
                             task=task,
                             network_number=run_number,
                             base_path=BASE_PATH,
                             affixes=AFFIXES,
                             timestamp_subdir_fmt="%Y-%b-%d-%H_%M_%S")
    # save init
    save_model(model, rnn_subdir=subdir, N_max=Ns[-1], N_min=Ns[0], init=True)

    # Train the model
    for epoch in tqdm(range(num_epochs)):
        losses_step = []
        for i in range(TRAINING_STEPS):
            zero_grad_helper(OPTIMIZERS)
            sequences, labels = task_function(Ns, BATCH_SIZE)
            sequences = sequences.permute(1, 0, 2).to(device)
            labels = [l.to(device) for l in labels]

            # Forward pass
            out, out_class = model(sequences)

            # Backward and optimize
            loss = 0.
            for N_i in range(len(Ns)):
                # todo: forward pass for all N_i, requires custom forward pass, that takes as input number of N's you actually have this far into the encephalization.
                loss += CRITERION(out_class[N_i], labels[N_i])
            loss.backward()
            losses_step.append(loss.item())
            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            # OPTIMIZER.step()
            # Step the optimizers in every training step:
            stepper(stepper_object=OPTIMIZERS, max_depth=model.current_depth.item(), num_steps=1)

        # Step the schedulers in every epoch:
        # stepper(stepper_object=SCHEDULERS, max_depth=model.current_depth.item(), num_steps=1)

        losses.append(np.mean(losses_step))

        correct_N = torch.zeros_like(torch.tensor(Ns)).to(device)
        total = 0
        for j in range(TEST_STEPS):
            with torch.no_grad():
                sequences, labels = task_function(Ns, BATCH_SIZE)
                sequences = sequences.permute(1, 0, 2).to(device)
                labels = [l.to(device) for l in labels]

                out, out_class = model(sequences)

                for N_i in range(len(Ns)):
                    predicted = torch.max(out_class[N_i], 1)[1]
                    correct_N[N_i] += (predicted == labels[N_i]).sum()
                    total += labels[N_i].size(0)

        accuracy = 100 * correct_N / float(total) * len(Ns)
        accuracies.append(accuracy)

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy {:.4f}  %'
              .format(epoch + 1, num_epochs, i + 1, TRAINING_STEPS, losses[-1], accuracy.mean()), flush=True)
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
                save_model(model, rnn_subdir=subdir, N_max=Ns[-1], N_min=Ns[0])

                if curriculum_type == 'cumulative':
                    Ns = Ns + [Ns[-1] + 1 + i for i in range(NUM_ADD)]

                if curriculum_type == 'sliding':
                    Ns = Ns[NUM_FORGET:] + [Ns[-1] + 1 + i for i in range(NUM_FORGET)]

                if curriculum_type == 'single':
                    Ns = [Ns[0] + 1]

                if curriculum_type == 'grow':
                    Ns = Ns + [Ns[-1] + 1]  # grow by 1 module/head each time.
                    model.current_depth.data += 1  # change to model.current_depth.data += 1. Register as parameter so torch dumps it.

                    d_int = model.current_depth.item()
                    for layer_name in ['input_layers', 'w_hh', 'w_ff_in', 'fc']:
                        new_layer = model.modules[f'{d_int}:{layer_name}']
                        last_layer = model.modules[f'{d_int - 1}:{layer_name}']
                        new_layer.weight.data = last_layer.weight.data * (1 + WEIGHT_NOISE * torch.randn_like(last_layer.weight.data))
                        new_layer.bias.data = last_layer.bias.data * (1 + BIAS_NOISE * torch.randn_like(last_layer.bias.data))
                    new_taus = model.taus[f'{d_int}']
                    last_taus = model.taus[f'{d_int - 1}']
                    new_taus.data = last_taus.data * (1 + TAUS_NOISE * torch.randn_like(last_taus.data))

                    stepper(stepper_object=SCHEDULERS, max_depth=d_int - 1, num_steps=FREEZING_STEPS)

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
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    # Add arguments to the parser
    parser.add_argument('-b', '--base_path', type=str, dest='base_path',
                        help='The base path to save results.')
    parser.add_argument('-a', '--agent_type', type=str, dest='agent_type',
                        help='agent type: (hierarchical, stack)')
    parser.add_argument('-c', '--curriculum_type', type=str, dest='curriculum_type',
                        help='Curriculum type: (cumulative, sliding, single)')
    parser.add_argument('-t', '--task', type=str, dest='task',
                        help='Task: (parity, dms)')
    parser.add_argument('-r', '--runs', type=int, dest='runs',
                        help='Number of independent runs.')
    parser.add_argument('-ih', '--init_heads', type=int, dest='init_heads',
                        help='Number of heads to start with.')
    parser.add_argument('-dh', '--add_heads', type=int, dest='add_heads',
                        help='Number of heads to add per new curricula.')
    parser.add_argument('-fh', '--forget_heads', type=int, dest='forget_heads',
                        help='Number of heads to forget for the sliding window curriculum type.')
    parser.add_argument('-s', '--seed', type=int, dest='seed',
                        help='Random seed.')

    parser.set_defaults(agent_type='hierarchical',
                        curriculum_type='grow',
                        task='parity',
                        runs=1,
                        init_heads=1,
                        add_heads=1,
                        forget_heads=1,
                        seed=np.random.choice(2 ** 16 - 1)
                        )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    print('agent_type:', args.agent_type)
    print('curriculum_type:', args.curriculum_type)
    print('task:', args.task)
    print('runs:', args.runs)

    BASE_PATH = args.base_path

    # USER ARGUMENTS (curriculum type/task and related params)
    CURRICULUM = args.curriculum_type
    TASK = args.task
    RUNS = args.runs

    INIT_HEADS = args.init_heads  # how many heads/tasks to start with
    NUM_ADD = args.add_heads  # how many heads/tasks to add per new curricula (only relevant for cumulative curriculum)
    NUM_FORGET = args.forget_heads  # how many heads to forget per new curricula (only relevant for sliding curriculum)

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    # Figure out which task
    if TASK == 'parity':
        task_function = tasks.make_batch_Nbit_pair_parity
    elif TASK == 'dms':
        task_function = tasks.make_batch_multihead_dms
    else:
        print('Unrecognized task:', TASK)

    # Set up the correct curriculum
    if CURRICULUM == 'cumulative':
        Ns_init = list(np.arange(2, 2 + INIT_HEADS))
    elif CURRICULUM == 'sliding':
        if INIT_HEADS < NUM_FORGET:
            INIT_HEADS = NUM_FORGET
        Ns_init = list(np.arange(2, 2 + INIT_HEADS))
    elif CURRICULUM == 'single':
        Ns_init = [2]
        INIT_HEADS = 1
        NUM_FORGET = 1  # todo: This doesn't appear to be used when curriculum_type == 'single', not sure why it's declared here.
    elif CURRICULUM == 'grow':
        Ns_init = [2]
        INIT_HEADS = 1
    else:
        raise Exception(f'Unrecognized curriculum type: {CURRICULUM}')
    ###############################################################

    # MODEL PARAMS
    MAX_DEPTH = 55
    INPUT_SIZE = 1
    NET_SIZE = [20]
    NUM_CLASSES = 2
    BIAS = True
    NUM_READOUT_HEADS_PER_MOD = 1  # for hierarchical/growing model
    TRAIN_TAU = True

    WEIGHT_NOISE = 0.03
    BIAS_NOISE = 0.03
    TAUS_NOISE = 0.02
    NUM_READOUT_HEADS = 100  # for basic stack model

    # TRAINING PARAMS
    NUM_EPOCHS = 1000
    BATCH_SIZE = 256
    TRAINING_STEPS = 600
    TEST_STEPS = 50
    CRITERION = nn.CrossEntropyLoss()
    device = 'cuda'

    tau_affix = f'train_tau={TRAIN_TAU}' if not TRAIN_TAU else ''
    AFFIXES = [tau_affix]
    for r_idx in range(1, RUNS + 1):
        # init new model
        if args.agent_type == 'hierarchical':
            rnn = RNN_Hierarchical(max_depth=MAX_DEPTH,
                                   input_size=INPUT_SIZE,
                                   net_size=NET_SIZE,
                                   num_classes=NUM_CLASSES,
                                   bias=BIAS,
                                   num_readout_heads_per_mod=NUM_READOUT_HEADS_PER_MOD,
                                   tau=1.,
                                   train_tau=TRAIN_TAU
                                   )
        elif args.agent_type == 'stack':
            rnn = RNN_Stack(input_size=INPUT_SIZE,
                            net_size=NET_SIZE,
                            num_classes=NUM_CLASSES,
                            bias=BIAS,
                            num_readout_heads=NUM_READOUT_HEADS,
                            tau=1.,
                            train_tau=TRAIN_TAU
                            )
        rnn.to(device)

        # SGD Optimizer
        LEARNING_RATE = 0.05
        MOMENTUM = 0.1
        FREEZING_STEPS = 25  # how many scheduling steps are taken upon successful completion of curriculum step.
        GAMMA = 0.95  # learning rate decay factor upon every scheduling step.

        if args.agent_type == 'stack':
            OPTIMIZER = torch.optim.SGD(rnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
        elif args.agent_type == 'hierarchical':
            OPTIMIZERS = {}
            SCHEDULERS = {}
            for d in range(MAX_DEPTH):
                opt_params = {}
                for m in ['input_layers', 'w_hh', 'w_ff_in', 'fc']:
                    if f'{d}:{m}' in rnn.modules.keys():  # to control for the fact that w_ff_in only exists for d > 0
                        OPTIMIZERS[f'{d}:{m}'] = torch.optim.SGD(rnn.modules[f'{d}:{m}'].parameters(), lr=LEARNING_RATE,
                                                                 momentum=MOMENTUM, nesterov=True)
                        SCHEDULERS[f'{d}:{m}'] = torch.optim.lr_scheduler.ExponentialLR(OPTIMIZERS[f'{d}:{m}'], gamma=GAMMA)

        stats = train(rnn,
                      curriculum_type=CURRICULUM,
                      task=TASK,
                      num_epochs=NUM_EPOCHS,
                      Ns=Ns_init,
                      run_number=r_idx
                      )
