import torch
import os

def save_model(
        model,
        curriculum_type: str,
        n_heads: int,
        n_forget: int,
        task: str,
        network_number: int,
        N_max: int,
        N_min: int = 2,
        base_path="../trained_models",
        init=False
):

    """Save the RNNs for the given type and network_name.

    Saves into {base_path}/{curriculum_type}_{task}_network_{network_number}/rnn_N{N_min:d}_N{N_max:d}

    Args:
        model: torch RNN_Stack model
        curriculum_type: 'cumulative', f'sliding_{n_heads}_{n_forget}', 'single'
        n_heads: (for sliding) number of training heads
        n_forget: (for sliding) number of heads forgotten per curricula
        task: 'parity' or 'dms'
        network_number: 1, 2, 3, ...
        N_max: N that the network should be able to solve
        N_min: minimum N, potentially depending on curriculum_type
        init: If True, saves with a different file name to show this is before training.
    """
    if curriculum_type == 'sliding':
        rnn_subdir = os.path.join(
            base_path,
            f'{curriculum_type}_{n_heads}_{n_forget}_{task}_network_{network_number}'
        )
    else:
        rnn_subdir = os.path.join(
            base_path,
            f'{curriculum_type}_{task}_network_{network_number}'
        )
    if init:
        rnn_name = f'rnn_init'
    else:
        rnn_name = f'rnn_N{N_min:d}_N{N_max:d}'


    if not os.path.exists(rnn_subdir):
        os.makedirs(rnn_subdir)

    filename = os.path.join(
        rnn_subdir,
        rnn_name
    )
    torch.save({'state_dict': model.state_dict()}, filename)

    return rnn_subdir