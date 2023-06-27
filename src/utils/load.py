import torch
import torch.nn as nn
import os
from src.models import init_model, init_model_mod


def load_model(
        curriculum_type: str, task: str, network_number: int, N_max: int, N_min: int = 2, device="cpu", base_path="./trained_models", strict=False,
        mod_model=False, mod_afunc=nn.LeakyReLU
):
    """Load the RNNs for the given type and network_name.

    Loads from {base_path}/{curriculum_type}_{task}_network_{network_number}/rnn_N{N_min:d}_N{N_max:d}

    Args:
        curriculum_type: 'cumulative', f'sliding_{n_heads}_{n_forget}', 'single'
        task: 'parity' or 'dms'
        network_number: 1, 2, 3, ...
        N_max: N that the network should be able to solve
        N_min: minimum N, potentially depending on curriculum_type
        device: 'cpu' or 'cuda'
        mod_model: 'modified model or default model'
    """
    rnn_path = os.path.join(
        base_path,
        f'{curriculum_type}_{task}_network_{network_number}',
        f'rnn_N{N_min:d}_N{N_max:d}',
    )
    if mod_model:
        rnn = init_model_mod(A_FUNC=mod_afunc, DEVICE=device)
    else:
        rnn = init_model(DEVICE=device)
    rnn.load_state_dict(torch.load(rnn_path, map_location=device)['state_dict'], strict = strict)
    return rnn

