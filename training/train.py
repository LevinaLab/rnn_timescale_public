import torch
import torch.nn as nn

import numpy as np
import sys
import argparse
from tqdm import tqdm

from src.models import RNN_Stack
import src.tasks as tasks
import src.utils as utils


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
    subdir = utils.save_model(model,
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
            sequences, labels = tasks.make_batch_multihead_dms(Ns, BATCH_SIZE)
            sequences = sequences.permute(1, 0, 2).to(device)
            labels = [l.to(device) for l in labels]

            # Forward pass
            out, out_class = model(sequences)

            # Backward and optimize
            loss = 0.
            for N_i in range(len(Ns)):
                head_idx = N_i
                loss += CRITERION(out_class[head_idx], labels[head_idx])
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
                sequences, labels = tasks.make_batch_multihead_dms(Ns, BATCH_SIZE)
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
              .format(epoch + 1, num_epochs, i + 1, TRAINING_STEPS, losses[-1], np.mean(accuracy)), flush=True)
        print('({N}, accuracy):\n' + ''.join([f'({Ns[i]}, {accuracy[i]:.4f})\n' for i in range(len(Ns))]), flush=True)

        stats = {'loss': losses,
                 'accuracy': accuracies}
        np.save(f'{subdir}/stats.npy', stats)

        # curriculum stuff + save
        if np.mean(accuracy) > 98.:
            if accuracy[-1] > 98.:
                print(f'Saving model for N = ' + str(Ns) + '...', flush=True)
                utils.save_model(model,
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

    parser.set_defaults(curriculum_type='cumulative',
                        task='parity',
                        runs=1,
                        init_heads=1,
                        add_heads=1,
                        forget_heads=1,
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
        NUM_FORGET = 1
    else:
        print('Unrecognized curriculum type: ', CURRICULUM)
    ###############################################################

    # MODEL PARAMS
    INPUT_SIZE = 1
    NET_SIZE = [500]
    NUM_CLASSES = 2
    BIAS = True
    NUM_READOUT_HEADS = 10
    TRAIN_TAU = True

    # TRAINING PARAMS
    NUM_EPOCHS = 1000
    BATCH_SIZE = 256
    TRAINING_STEPS = 600
    TEST_STEPS = 50
    CRITERION = nn.CrossEntropyLoss()
    device = 'cuda'

    for r_idx in range(1, RUNS+1):
        # init new model
        rnn = RNN_Stack(input_size=INPUT_SIZE,
                        net_size=NET_SIZE,
                        num_classes=NUM_CLASSES,
                        bias=BIAS,
                        num_readout_heads=NUM_READOUT_HEADS,
                        tau=1.,
                        train_tau=TRAIN_TAU
                        ).to(device)

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
