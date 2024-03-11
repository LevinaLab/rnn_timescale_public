import torch
import numpy as np

def generate_binary_sequence(M, balanced=False):
    if balanced:
        # for dms if the input sequence is correlated it'll make one class very likely
        return (torch.rand(M) < 0.5) * 1.
    else:
        # doesn't seem to have the same effect for nbit-parity
        return (torch.rand(M) < torch.rand(1)) * 1.

def batched_generate_binary_sequence(M, BS, balanced=False):
    if balanced:
        # for dms if the input sequence is correlated it'll make one class very likely
        return (torch.rand(BS, M) < 0.5) * 1.
    else:
        # doesn't seem to have the same effect for nbit-parity
        return (torch.rand(BS, M) < torch.rand(1)) * 1.


############ N_PARITY TASKS ##############

def get_parity(vec, N):
    """

    Parameters
    ----------
    vec: [torch.Tensor]
    N: how many steps back it needs to remember in order to check for parity

    Returns
    -------

    """
    return (vec[-N:].sum() % 2).long()

def get_batched_parity(mat, N):
    """

    Parameters
    ----------
    mat: [torch.Tensor: BATCH_SIZE x TIME]
    N: how many steps back it needs to remember in order to check for parity

    Returns
    -------

    """
    return (mat[:, -N:].squeeze().sum(dim=1) % 2).long()

def make_batch_Nbit_pair_parity(Ns, bs):
    """ Generates input sequences and labels for the N-bit parity task.
    Parameters
    ----------
    Ns: list of how many different N you want the labels for.
    bs: how many batches.

    Returns
    -------
    tuple (sequence, labels):
        sequence: torch.Tensor of shape [batch_size, sequence length, input_features]
        labels: list of torch.Tensor of shape (bs, 1)
    """
    M_min = Ns[-1] + 2
    M_max = M_min + 1.5 * Ns[-1]  # todo: consider reducing 3 to 1.5 just to save time.
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        # sequences = [generate_binary_sequence(M).unsqueeze(-1) for i in range(bs)]
        # labels = [torch.stack([get_parity(s, N) for s in sequences]) for N in Ns]
        sequences = batched_generate_binary_sequence(M, bs).unsqueeze(-1)
        labels = [get_batched_parity(sequences, N) for N in Ns]
    return sequences, labels


############ DMS TASKS ##################

def get_match(vec, N):
    return (vec[-N] == vec[-1]).long()


def make_batch_multihead_dms(Ns, bs):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = [generate_binary_sequence(M, balanced=True).unsqueeze(-1) for i in range(bs)]
        labels = [torch.stack([get_match(s, N) for s in sequences]).squeeze() for N in Ns]

    return torch.stack(sequences), labels