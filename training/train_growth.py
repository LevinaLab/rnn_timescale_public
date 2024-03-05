import sys
import os
import time

from src.utils.git_logger import report_git_status

# Get the absolute path of the parent directory of 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add both the parent directory and the 'src' directory to the module search path
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from src.models.RNN_hier import RNN_Hierarchical
import src.tasks as tasks
from src.utils import save_model, generate_subdir, save_configs
from src.utils import slurm
import config_parser


def train(network_number, output_path):
    """Only to be run from __main__ here to guarantee that the global variables are defined.

    Global variables:
    MODEL: RNN_Hierarchical
    CONFIGS: dict of hyperparameters
    CRITERION: nn.CrossEntropyLoss
    OPTIMIZERS: dict of torch.optim.SGD
    SCHEDULERS: dict of torch.optim.lr_scheduler.ExponentialLR
    TASK_FUNCTION: function that returns input data and their labels.
    todo: these should all be passed as arguments to the function, but I also don't want there to be a danger of
          mismatch between attributes of the model that is passed and what the config says.
    """
    if CONFIGS['CURRICULUM'] != 'grow':
        raise Exception(f"train_growth.py only supports the 'grow' curriculum, you provided: {CONFIGS['CURRICULUM']}")

    losses = []
    accuracies = []
    times = []
    Ns = [2]

    start_time = time.time()
    # Train the model
    for epoch in tqdm(range(CONFIGS['NUM_EPOCHS']), desc='Epochs', position=0):
        losses_step = []
        for i in tqdm(range(CONFIGS['TRAINING_STEPS']), desc='Steps', position=1):
            zero_grad_helper(OPTIMIZERS)
            sequences, labels = TASK_FUNCTION(Ns, CONFIGS['BATCH_SIZE'])
            sequences = sequences.permute(1, 0, 2).to(CONFIGS['DEVICE'])
            labels = [l.to(CONFIGS['DEVICE']) for l in labels]

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
            sys.stdout.flush()

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
        times.append(time.time() - start_time)
        print("Size of modules trained:", NET_SIZE[0:MODEL.current_depth])
        print(f"Epoch [{epoch + 1}/{CONFIGS['NUM_EPOCHS']}], Step [{i + 1}/{CONFIGS['TRAINING_STEPS']}], "
              f"Loss: {losses[-1]:.4f}, Accuracy {accuracy.mean():.4f}%", flush=True)
        print('({N}, accuracy):\n' + ''.join([f'({Ns[i]}, {accuracy[i]:.4f})\n' for i in range(len(Ns))]), flush=True)

        stats = {'loss': losses,
                 'accuracy': accuracies,
                 'time': times,
                 'epochs': epoch,
                 'max_N': Ns,}

        np.save(f'{output_path}/stats.npy', stats)

        # curriculum stuff + save
        if accuracy.mean() > 98.:  # so it doesn't forget the older tasks
            if accuracy[-1] > 98.:
                print(f'Saving model for N = ' + str(Ns) + '...', flush=True)
                save_model(MODEL, rnn_subdir=output_path, network_number=network_number, stage=Ns[-1], init=False)

                Ns = Ns + [Ns[-1] + 1]  # grow by 1 module/head each time.
                MODEL.current_depth.data += 1  # change to MODEL.current_depth.data += 1. Register as parameter so torch dumps it.
                duplication_layers = [k.replace('DUPLICATE_', '').lower()
                                      for k, v in CONFIGS.items() if k.startswith('DUPLICATE_') and v is True]
                d_int = MODEL.current_depth.item()

                for layer_name in duplication_layers:
                    if MODEL.taus[f'{d_int}'].data.shape != MODEL.taus[f'{d_int - 1}'].data.shape:
                        # All DUPLICATION_ configs have to be False when a module size scheduling is taking place.
                        raise Exception(f'Error: Modules {d_int} and {d_int - 1} have different shapes.'
                                        f'Either turn of size scheduling, or disable all duplications.')

                    if layer_name == 'taus':  # todo: this is ugly, but I couldn't find a way to register taus under model.modules
                        MODEL.taus[f'{d_int}'].data = MODEL.taus[f'{d_int - 1}'].data * (
                                    1 + CONFIGS['TAUS_NOISE'] * torch.randn_like(MODEL.taus[f'{d_int - 1}'].data))
                    MODEL.modules[f'{d_int}:{layer_name}'].weight.data = MODEL.modules[f'{d_int - 1}:{layer_name}'].weight.data * (1 + CONFIGS['WEIGHT_NOISE'] * torch.randn_like(MODEL.modules[f'{d_int - 1}:{layer_name}'].weight.data))
                    MODEL.modules[f'{d_int}:{layer_name}'].bias.data = MODEL.modules[f'{d_int - 1}:{layer_name}'].bias.data * (1 + CONFIGS['BIAS_NOISE'] * torch.randn_like(MODEL.modules[f'{d_int - 1}:{layer_name}'].bias.data))

                # stepper(stepper_object=SCHEDULERS, max_depth=d_int - 1, num_steps=CONFIGS['FREEZING_STEPS'])
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
    for k in ['CURRICULUM', 'NET_SIZE', 'TASK', 'SEED', 'DEVICE']:
        print(f"{k}: {CONFIGS[k]}")

    print("Duplicating: ", [k.replace('DUPLICATE_', '').lower()
     for k, v in CONFIGS.items() if k.startswith('DUPLICATE_') and v is True])

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


    if CONFIGS['CURRICULUM'] != 'grow':
        raise Exception(f"Unrecognized curriculum type: {CONFIGS['CURRICULUM']}")

    ###############################################################


    CRITERION = nn.CrossEntropyLoss()

    # NET_SIZE = list(map(int, [CONFIGS['NET_SIZE']]))  # todo: fix
    NET_SIZE_MIN = 2
    NET_SIZE_MAX = 50
    # NET_SIZES = [6, 7, 8, 8, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # NET_SIZES = NET_SIZES + [50] * (CONFIGS['MAX_DEPTH'] - len(NET_SIZES))
    # NET_SIZES = [20] * CONFIGS['MAX_DEPTH']
    NET_SIZES = [5] * CONFIGS['MAX_DEPTH']
    NET_SIZE = [[s] for s in NET_SIZES]

    # todo: must be replaced with a path provided via an environment variable.
    BASE_PATH = slurm.get_log_dir(results_dir=os.path.join(project_root, 'trained_models'))
    print(f"Logging results to: {BASE_PATH}", flush=True)
    env_vars = slurm.get_env_vars()
    # BASE_PATH = os.path.join(project_root, 'trained_models')
    subdir = generate_subdir(configs=CONFIGS,
                             base_path=BASE_PATH,
                             env_vars=env_vars,
                             affixes=[],
                             timestamp_subdir_fmt="%Y-%b-%d-%H_%M_%S")
    commit_hash, files_modified = report_git_status(os.path.join(subdir, "git_report_log.txt"))
    CONFIGS['COMMIT_HASH'] = commit_hash
    CONFIGS['FILES_MODIFIED'] = files_modified
    save_configs(subdir, CONFIGS)

    for network_number in range(CONFIGS['REPLICAS']):
        print("Replica #", network_number + 1, flush=True)
        MODEL = RNN_Hierarchical(max_depth=CONFIGS['MAX_DEPTH'],
                                 input_size=CONFIGS['INPUT_SIZE'],
                                 net_size=NET_SIZE,  # todo: fix
                                 device=CONFIGS['DEVICE'],
                                 num_classes=CONFIGS['NUM_CLASSES'],
                                 bias=CONFIGS['BIAS'],
                                 num_readout_heads_per_mod=CONFIGS['NUM_READOUT_HEADS_PER_MOD'],
                                 fixed_tau_val=1.,
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


        # save init
        save_model(MODEL, rnn_subdir=subdir, network_number=network_number, stage=None, init=True)

        stats = train(network_number=network_number + 1, output_path=subdir)
        print("Finished training replica #", network_number, flush=True)
        print("Results saved in", subdir, flush=True)