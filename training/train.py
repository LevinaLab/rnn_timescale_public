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

from src.models.RNN_hier import RNN_Hierarchical
import src.tasks as tasks
from src.utils import save_model


def train(model,
          curriculum_type,
          task,
          num_epochs,
          Ns,  # List of parities that are being trained/tested. Grows with curriculum.
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
                        init=True
                      )

    # Train the model
    for epoch in tqdm(range(num_epochs)):
        losses_step = []
        for i in range(TRAINING_STEPS):
            OPTIMIZER.zero_grad()
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
            OPTIMIZER.step()

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
                 'accuracy': accuracies}
        np.save(f'{subdir}/stats.npy', stats)

        # curriculum stuff + save
        if accuracy.mean() > 98.:  # so it doesn't forget the older tasks
            if accuracy[-1] > 98.:
                print(f'Saving model for N = ' + str(Ns) + '...', flush=True)
                save_model(model,
                           curriculum_type=curriculum_type,
                           n_heads=len(Ns),
                           n_forget=NUM_FORGET,
                           task=task,
                           network_number=run_number,
                           N_max=Ns[-1],
                           N_min=Ns[0]
                           )

                if curriculum_type == 'cumulative':
                    Ns = Ns + [Ns[-1] + 1 + i for i in range(NUM_ADD)]

                if curriculum_type == 'sliding':
                    Ns = Ns[NUM_FORGET:] + [Ns[-1] + 1 + i for i in range(NUM_FORGET)]

                if curriculum_type == 'single':
                    Ns = [Ns[0] + 1]

                if curriculum_type == 'grow':
                    Ns = Ns + [Ns[-1] + 1]  # grow by 1 module/head each time.
                    model.current_depth += 1

                print(f'N = {Ns[0]}, {Ns[-1]}', flush=True)

    return stats


###############################################################

if __name__ == '__main__':

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
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

    parser.set_defaults(curriculum_type='grow',
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
    print('curriculum_type:', args.curriculum_type)
    print('task:', args.task)
    print('runs:', args.runs)

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
    INPUT_SIZE = 1
    NET_SIZE = [100]
    NUM_CLASSES = 2
    BIAS = True
    NUM_READOUT_HEADS_PER_MOD = 1
    # NUM_READOUT_HEADS = 100
    TRAIN_TAU = False

    # TRAINING PARAMS
    NUM_EPOCHS = 1000
    BATCH_SIZE = 256
    TRAINING_STEPS = 600
    TEST_STEPS = 50
    CRITERION = nn.CrossEntropyLoss()
    device = 'cuda'

    for r_idx in range(1, RUNS+1):
        # init new model
        rnn = RNN_Hierarchical(max_depth=5,
                               input_size=INPUT_SIZE,
                               net_size=NET_SIZE,
                               num_classes=NUM_CLASSES,
                               bias=BIAS,
                               num_readout_heads_per_mod=NUM_READOUT_HEADS_PER_MOD,
                               tau=1.,
                               train_tau=TRAIN_TAU
                               )
        rnn.to(device)


        # SGD Optimizer
        learning_rate = 0.05
        momentum = 0.1
        OPTIMIZER = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)

        stats = train(rnn,
                      curriculum_type=CURRICULUM,
                      task=TASK,
                      num_epochs=NUM_EPOCHS,
                      Ns=Ns_init,
                      run_number=r_idx
                      )
