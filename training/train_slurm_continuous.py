import sys
import os

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

from src.models import RNN_Continuous
import src.tasks as tasks
from src.utils import save_model


def _train(
    model,
    curriculum_type,
    task,
    num_epochs,
    Ns,
    run_number,
):
    # stats
    losses = []
    accuracies = []

    # save init
    subdir = save_model(
        model,
        curriculum_type=curriculum_type,
        n_heads=len(Ns),
        n_forget=NUM_FORGET,
        task=task,
        network_number=run_number,
        N_max=Ns[-1],
        N_min=Ns[0],
        init=True,
        base_path=BASE_PATH,
        affixes=AFFIXES
    )

    # Train the model
    for epoch in tqdm(range(num_epochs)):
        losses_step = []
        for i in range(TRAINING_STEPS):
            for duplicate in DUPLICATES:
                OPTIMIZER.zero_grad()
                sequences, labels = TASK_FUNCTION(Ns, BATCH_SIZE, duplicate=duplicate)
                sequences = sequences.to(device)
                labels = [l.to(device) for l in labels]

                # Forward pass
                out, out_class = model(sequences, k_data=duplicate)

                # Backward and optimize
                loss = 0.
                for N_i in range(len(Ns)):
                    loss += CRITERION(out_class[N_i], labels[N_i])
                loss.backward()
                losses_step.append(loss.item())
                # gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                OPTIMIZER.step()

        losses.append(np.mean(losses_step))

        correct_N = np.zeros_like(Ns)
        total = 0
        for j in range(TEST_STEPS):
            with torch.no_grad():
                for duplicate in DUPLICATES:
                    sequences, labels = TASK_FUNCTION(Ns, BATCH_SIZE, duplicate=duplicate)
                    sequences = sequences.to(device)
                    labels = [l.to(device) for l in labels]

                    out, out_class = model(sequences, k_data=duplicate)

                    for N_i in range(len(Ns)):
                        predicted = torch.max(out_class[N_i], 1)[1]

                        correct_N[N_i] += (predicted == labels[N_i]).sum()
                        total += labels[N_i].size(0)

        accuracy = 100 * correct_N / float(total) * len(Ns)
        accuracies.append(accuracy)

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy {:.4f}  %'
              .format(epoch + 1, num_epochs, i + 1, TRAINING_STEPS, losses[-1], np.mean(accuracy)), flush=True)
        print('({N}, accuracy):\n' + ''.join([f'({Ns[i]}, {accuracy[i]:.4f})\n' for i in range(len(Ns))]), flush=True)

        stats = {'loss': losses,
                 'accuracy': accuracies}
        np.save(f'{subdir}/stats.npy', stats)

        # curriculum stuff + save
        if np.mean(accuracy) > 98.:
            if accuracy[-1] > 98.:
                print(f'Saving model for N = ' + str(Ns) + '...', flush=True)
                save_model(
                    model,
                    curriculum_type=curriculum_type,
                    n_heads=len(Ns),
                    n_forget=NUM_FORGET,
                    task=task,
                    network_number=run_number,
                    N_max=Ns[-1],
                    N_min=Ns[0],
                    base_path=BASE_PATH,
                    affixes=AFFIXES,
                    )

                if curriculum_type == 'cumulative':
                    Ns = Ns + [Ns[-1] + 1 + i for i in range(NUM_ADD)]

                if curriculum_type == 'sliding':
                    Ns = Ns[NUM_FORGET:] + [Ns[-1] + 1 + i for i in range(NUM_FORGET)]

                if curriculum_type == 'single':
                    Ns = [Ns[0] + 1]

                print(f'N = {Ns[0]}, {Ns[-1]}', flush=True)

    return stats


###############################################################

if __name__ == '__main__':

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    def _list_of_ints(arg):
        return list(map(int, arg.split(',')))

    # Add arguments to the parser
    parser.add_argument('-b', '--base_path', type=str, dest='base_path',
                        help='The base path to save results.')
    parser.add_argument('-a', '--afunc', type=str, dest='afunc',
                        help='Acitvation functions: (leakyrelu, relu, tanh, sigmoid)')
    parser.add_argument('-c', '--curriculum_type', type=str, dest='curriculum_type',
                        help='Curriculum type: (cumulative, sliding, single)')
    parser.add_argument('-t', '--task', type=str, dest='task',
                        help='Task: (parity, dms)')
    parser.add_argument('-n', '--network_number', type=int, dest='network_number',
                        help='The run number of the network, to be used as a naming suffix for savefiles.')
    parser.add_argument('-ih', '--init_heads', type=int, dest='init_heads',
                        help='Number of heads to start with.')
    parser.add_argument('-dh', '--add_heads', type=int, dest='add_heads',
                        help='Number of heads to add per new curricula.')
    parser.add_argument('-fh', '--forget_heads', type=int, dest='forget_heads',
                        help='Number of heads to forget for the sliding window curriculum type.')
    parser.add_argument('-s', '--seed', type=int, dest='seed',
                        help='Random seed.')
    parser.add_argument('-dup', '--duplicates', type=_list_of_ints, dest='duplicates',
                        help='Time discretization: duplicate samples this many times.'
                             'Must be list of ints, e.g. -dup 2,3,4')
    parser.add_argument('-it', '--init_tau', type=float, dest='init_tau',
                        help='Initial mean value of tau.')

    parser.set_defaults(
        afunc='leakyrelu',
        curriculum_type='cumulative',
        task='parity',
        network_number=1,
        init_heads=1,
        add_heads=1,
        forget_heads=1,
        seed=np.random.choice(2 ** 32),
        duplicates=None,
        init_tau=None,
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    BASE_PATH = args.base_path
    AFFIXES = []

    # USER ARGUMENTS (curriculum type/task and related params)
    AFUNC = args.afunc
    CURRICULUM = args.curriculum_type
    TASK = args.task
    DUPLICATES = args.duplicates
    if DUPLICATES is None:
        raise ValueError('Must specify duplicates as command-line argument.')
    else:
        AFFIXES.append(f'duplicates{DUPLICATES}')
    INIT_TAU = args.init_tau
    if INIT_TAU is not None:
        AFFIXES.append(f'tau{INIT_TAU}')
    NETWORK_NUMBER = args.network_number

    INIT_HEADS = args.init_heads  # how many heads/tasks to start with
    NUM_ADD = args.add_heads  # how many heads/tasks to add per new curricula (only relevant for cumulative curriculum)
    NUM_FORGET = args.forget_heads  # how many heads to forget per new curricula (only relevant for sliding curriculum)

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    # Figure out which task
    if TASK == 'parity':
        TASK_FUNCTION = tasks.make_batch_Nbit_pair_parity
    elif TASK == 'dms':
        raise NotImplementedError('DMS not implemented for continuous time model.')
        TASK_FUNCTION = tasks.make_batch_multihead_dms
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
        NUM_FORGET = 1
    else:
        print('Unrecognized curriculum type: ', CURRICULUM)
    ###############################################################

    # MODEL PARAMS
    INPUT_SIZE = 1
    NET_SIZE = [500]
    NUM_CLASSES = 2
    BIAS = True
    NUM_READOUT_HEADS = 100
    TRAIN_TAU = True

    # TRAINING PARAMS
    NUM_EPOCHS = 1000
    BATCH_SIZE = 256
    TRAINING_STEPS = 600
    TEST_STEPS = 50
    CRITERION = nn.CrossEntropyLoss()
    device = 'cuda'

    # init new model
    AFFIXES += ['mod', AFUNC]
    if AFUNC == 'leakyrelu':
        AFUNC = nn.LeakyReLU
    elif AFUNC == 'relu':
        AFUNC = nn.ReLU
    elif AFUNC == 'sigmoid':
        AFUNC = nn.Sigmoid
    elif AFUNC == 'tanh':
        AFUNC = nn.Tanh
    else:
        print('Unrecognized activation function: ', AFUNC)

    rnn = RNN_Continuous(
        input_size=INPUT_SIZE,
        net_size=NET_SIZE,
        num_classes=NUM_CLASSES,
        bias=BIAS,
        num_readout_heads=NUM_READOUT_HEADS,
        tau=1.,
        afunc=AFUNC,
        train_tau=TRAIN_TAU,
        init_tau=INIT_TAU,
        ).to(device)
    rnn.to(device)

    # SGD Optimizer
    learning_rate = 0.05
    momentum = 0.1
    OPTIMIZER = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)

    # print values of global variables
    print('BASE_PATH: ', BASE_PATH)
    print('AFFIXES: ', AFFIXES)
    print('AFUNC: ', AFUNC)
    print('CURRICULUM: ', CURRICULUM)
    print('TASK: ', TASK)
    print('DUPLICATES: ', DUPLICATES)
    print('INIT_TAU: ', INIT_TAU)
    print('NETWORK_NUMBER: ', NETWORK_NUMBER)
    print('INIT_HEADS: ', INIT_HEADS)
    print('NUM_ADD: ', NUM_ADD)
    print('NUM_FORGET: ', NUM_FORGET)
    print('SEED: ', SEED)
    print('INPUT_SIZE: ', INPUT_SIZE)
    print('NET_SIZE: ', NET_SIZE)
    print('NUM_CLASSES: ', NUM_CLASSES)
    print('BIAS: ', BIAS)
    print('NUM_READOUT_HEADS: ', NUM_READOUT_HEADS)
    print('TRAIN_TAU: ', TRAIN_TAU)
    print('NUM_EPOCHS: ', NUM_EPOCHS)
    print('BATCH_SIZE: ', BATCH_SIZE)
    print('TRAINING_STEPS: ', TRAINING_STEPS)
    print('TEST_STEPS: ', TEST_STEPS)
    print('CRITERION: ', CRITERION)
    print('device: ', device)

    # Train the model
    stats = _train(
        rnn,
        curriculum_type=CURRICULUM,
        task=TASK,
        num_epochs=NUM_EPOCHS,
        Ns=Ns_init,
        run_number=NETWORK_NUMBER
    )

