import json
from datetime import datetime

import torch
import os


def save_model(
        model,
        rnn_subdir,
        network_number,
        stage,
        init=False,
):

    """Save the RNNs for the given type and network_name.

    Saves into {base_path}/{curriculum_type}_{task}_network_{network_number}/rnn_N{N_min:d}_N{N_max:d}

    Args:
        model: torch RNN_Stack model
        rnn_subdir: the directory to save the model to.
        network_number: the number of the network
        stage: the stage of the curriculum, corresponding to the depth of the network.
        init: If True, saves with a different file name to show this is before training.
    """

    if init:
        rnn_name = f'rnn_{network_number:d}_init'
    else:
        rnn_name = f'rnn_{network_number:d}_N{stage:d}'

    filename = os.path.join(
        rnn_subdir,
        rnn_name
    )
    torch.save({'state_dict': model.state_dict()}, filename)

    return filename


def save_configs(subdir, configs):
    """Saves the configurations used to a a json file in the rnn_subdir.

    """
    filename = os.path.join(subdir, 'configs.json')
    with open(filename, 'w') as f:
        json.dump(configs, f, indent=4)


def generate_subdir(configs,
                    base_path, affixes,
                    timestamp_subdir_fmt="%Y-%b-%d-%H_%M_%S"):
    """Creates a directory for saving results to.

    If the directory already exists, it will print a warning and may overwrite files.

    Parameters
    ----------
    timestamp_subdir_fmt: "%Y-%b-%d-%H_%M_%S" by default, can be None to omit in directory name.
    """
    curriculum_type = configs['CURRICULUM']
    task = configs['TASK']
    tau_affix = f"train_tau={configs['TRAIN_TAU']}" if not configs['TRAIN_TAU'] else ""
    pieces = [curriculum_type, task, tau_affix] + affixes

    if timestamp_subdir_fmt:
        pieces.append(datetime.now().strftime(timestamp_subdir_fmt) + '_')

    dir_name = '_'.join(pieces)

    path = os.path.abspath(os.path.join(base_path, dir_name))
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"Warning: {path} already exists. Files may be overwritten.")

    return path

