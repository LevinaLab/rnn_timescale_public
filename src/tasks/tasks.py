import torch
import numpy as np

def generate_binary_sequence(M, balanced=False):
    if balanced:
        # for dms if the input sequence is correlated it'll make one class very likely
        return (torch.rand(M) < 0.5) * 1.
    else:
        # doesn't seem to have the same effect for nbit-parity
        return (torch.rand(M) < torch.rand(1)) * 1.


############ N_PARITY TASKS ##############

def get_parity(vec, N):
    return (vec[-N:].sum() % 2).long()

def make_batch_Nbit_pair_parity(Ns, bs):
    M_min = Ns[-1] + 2
    M_max = M_min + 3 * Ns[-1]
    M = np.random.randint(M_min, M_max)
    with torch.no_grad():
        sequences = [generate_binary_sequence(M).unsqueeze(-1) for i in range(bs)]
        labels = [torch.stack([get_parity(s, N) for s in sequences]) for N in Ns]
    return torch.stack(sequences), labels


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