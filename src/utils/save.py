import json
from datetime import datetime

import torch
import os

def save_model(
        model,
        rnn_subdir,
        N_max: int,
        N_min: int = 2,
        init=False,
):

    """Save the RNNs for the given type and network_name.

    Saves into {base_path}/{curriculum_type}_{task}_network_{network_number}/rnn_N{N_min:d}_N{N_max:d}

    Args:
        model: torch RNN_Stack model
        N_max: N that the network should be able to solve
        N_min: minimum N, potentially depending on curriculum_type
        init: If True, saves with a different file name to show this is before training.
    """

    if init:
        rnn_name = f'rnn_init'
    else:
        rnn_name = f'rnn_N{N_min:d}_N{N_max:d}'

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


def generate_subdir(curriculum_type, n_heads, task,
                    network_number, base_path, affixes,
                    timestamp_subdir_fmt="%Y-%b-%d-%H_%M_%S"):
    """Creates a directory for saving results to.

    If the directory already exists, it will print a warning and may overwrite files.

    Parameters
    ----------
    timestamp_subdir_fmt: "%Y-%b-%d-%H_%M_%S" by default, can be None to omit in directory name.
    """

    affix_str = '_'
    if len(affixes) > 0:
        affix_str += '_'.join(affixes) + '_'
    if timestamp_subdir_fmt:
        affix_str += datetime.now().strftime(timestamp_subdir_fmt) + '_'

    if curriculum_type == 'sliding':
        rnn_subdir = os.path.join(
            base_path,
            f'{curriculum_type}_{n_heads}_{task}{affix_str}network_{network_number}'
        )
    else:
        rnn_subdir = os.path.join(
            base_path,
            f'{curriculum_type}_{task}{affix_str}network_{network_number}'
        )

    if not os.path.exists(rnn_subdir):
        os.makedirs(rnn_subdir)
    else:
        print(f"Warning: {rnn_subdir} already exists. Files may be overwritten.")

    return os.path.abspath(rnn_subdir)
