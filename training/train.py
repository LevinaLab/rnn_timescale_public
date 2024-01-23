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

from src.models import RNN_Stack, RNN_Mod
import src.tasks as tasks
from src.utils import save_model

def train(model,
          curriculum_type,
          task,
          num_epochs,
          Ns,
          run_number):

    # stats
    losses = []
    accuracies = []

    # save init
    subdir = save_model(model,
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
            OPTIMIZER.zero_grad()
            args = (Ns, BATCH_SIZE)
            sequences, labels = task_function(*args)
            sequences = sequences.to(device)
            labels = [l.to(device) for l in labels]

            # Forward pass
            out, out_class = model(sequences)

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
                args = (Ns, BATCH_SIZE)
                sequences, labels = task_function(*args)
                sequences = sequences.to(device)
                labels = [l.to(device) for l in labels]

                out, out_class = model(sequences)

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
                save_model(model,
                           curriculum_type=curriculum_type,
                           n_heads=len(Ns),
                           n_forget=NUM_FORGET,
                           task=task,
                           network_number=run_number,
                           N_max=Ns[-1],
                           N_min=Ns[0],
                           base_path=BASE_PATH,
                           affixes=AFFIXES
                           )

                if curriculum_type == 'cumulative':
                    Ns = Ns + [Ns[-1] + 1 + i for i in range(NUM_ADD)]

                if curriculum_type == 'sliding':
                    Ns = Ns[NUM_FORGET:] + [Ns[-1] + 1 + i for i in range(NUM_FORGET)]

                if curriculum_type == 'single':
                    Ns = [Ns[0] + 1]

                if curriculum_type == 'single_nocurr':
                    break

                print(f'N = {Ns[0]}, {Ns[-1]}', flush=True)

    return stats


###############################################################

if __name__ == '__main__':

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('-b', '--base_path', type=str, dest='base_path',
                        help='The base path to save results. (str)')
    parser.add_argument('-nn', '--num_neurons', type=int, dest='num_neurons',
                        help='The number of hidden neurons in the RNN. (int)')
    parser.add_argument('-ni', '--ns_init', type=int, dest='ns_init',
                        help='The starting value of N for the task. (int)')
    parser.add_argument('-m', '--model_type', type=str, dest='model_type',
                        help='Model types: (default, mod). (str)')
    parser.add_argument('-a', '--afunc', type=str, dest='afunc',
                        help='Acitvation functions: (leakyrelu, relu, tanh, sigmoid). (str)')
    parser.add_argument('-c', '--curriculum_type', type=str, dest='curriculum_type',
                        help='Curriculum type: (cumulative, sliding, single, single_nocurr). (str)')
    parser.add_argument('-t', '--task', type=str, dest='task',
                        help='Task: (parity, dms). (str)')
    parser.add_argument('-T', '--tau', type=float, dest='tau',
                        help='The value of tau each neuron starts with. If set, taus will not be trainable. '
                             'Default = None. (float > 1)')
    parser.add_argument('-n', '--network_number', type=int, dest='network_number',
                        help='The run number of the network, to be used as a naming suffix for savefiles. (int)')
    parser.add_argument('-ih', '--init_heads', type=int, dest='init_heads',
                        help='Number of heads to start with. (int)')
    parser.add_argument('-dh', '--add_heads', type=int, dest='add_heads',
                        help='Number of heads to add per new curricula. (int)')
    parser.add_argument('-fh', '--forget_heads', type=int, dest='forget_heads',
                        help='Number of heads to forget for the sliding window curriculum type. (int)')
    parser.add_argument('-s', '--seed', type=int, dest='seed',
                        help='Random seed. (int)')

    parser.set_defaults(
        model_type='default',
        num_neurons=500,
        afunc='leakyrelu',
        curriculum_type='cumulative',
        task='parity',
        tau=None,
        network_number=1,
        init_heads=1,
        add_heads=1,
        forget_heads=1,
        seed=np.random.choice(2 ** 32),
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    print('num_neurons:', args.num_neurons)
    print('curriculum_type:', args.curriculum_type)
    print('task:', args.task)
    print('network number:', args.network_number)

    BASE_PATH = args.base_path
    NS_INIT = args.ns_init
    NUM_NEURONS = args.num_neurons
    AFFIXES = []

    # USER ARGUMENTS (curriculum type/task and related params)
    MODEL = args.model_type
    AFUNC = args.afunc
    CURRICULUM = args.curriculum_type
    TASK = args.task
    NETWORK_NUMBER = args.network_number
    TAU = args.tau
    if TAU is not None:
        TRAIN_TAU = False
    else:
        TAU = 1.
        TRAIN_TAU = True

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
        NUM_CLASSES = 2
    elif TASK == 'dms':
        task_function = tasks.make_batch_multihead_dms
        NUM_CLASSES = 2
    else:
        print('Unrecognized task:', TASK)

    # Set up the correct curriculum
    if CURRICULUM == 'cumulative':
        Ns_init = list(np.arange(2, 2 + INIT_HEADS))
    elif CURRICULUM == 'sliding':
        if INIT_HEADS < NUM_FORGET:
            INIT_HEADS = NUM_FORGET
        Ns_init = list(np.arange(2, 2 + INIT_HEADS))
    elif CURRICULUM == 'single' or CURRICULUM == 'single_nocurr':
        Ns_init = [2]
        INIT_HEADS = 1
        NUM_FORGET = 1
    else:
        print('Unrecognized curriculum type: ', CURRICULUM)
    ###############################################################

    if NS_INIT is not None:
        Ns_init = [N - 2 + NS_INIT for N in Ns_init]

    # MODEL PARAMS
    INPUT_SIZE = 1
    NET_SIZE = [NUM_NEURONS]
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
    if MODEL == 'mod':
        AFFIXES += ['mod', AFUNC]
        if args.tau is not None:
            AFFIXES += ['T', str(TAU)]

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

        rnn = RNN_Mod(
            input_size=INPUT_SIZE,
            net_size=NET_SIZE,
            num_classes=NUM_CLASSES,
            bias=BIAS,
            num_readout_heads=NUM_READOUT_HEADS,
            tau=TAU,
            afunc=AFUNC,
            train_tau=TRAIN_TAU,
        ).to(device)

    elif MODEL == 'default':

        if NUM_NEURONS != 500:
            AFFIXES += ['size', str(NUM_NEURONS)]
        if args.tau is not None:
            AFFIXES += ['T', str(TAU)]

        rnn = RNN_Stack(
            input_size=INPUT_SIZE,
            net_size=NET_SIZE,
            num_classes=NUM_CLASSES,
            bias=BIAS,
            num_readout_heads=NUM_READOUT_HEADS,
            tau=TAU,
            train_tau=TRAIN_TAU,
        ).to(device)

        rnn.to(device)
    else:
        print('Unrecognized model type: ', MODEL)

    # SGD Optimizer
    learning_rate = 0.05
    momentum = 0.1
    OPTIMIZER = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)


    stats = train(rnn,
                  curriculum_type=CURRICULUM,
                  task=TASK,
                  num_epochs=NUM_EPOCHS,
                  Ns=Ns_init,
                  run_number=NETWORK_NUMBER
                  )