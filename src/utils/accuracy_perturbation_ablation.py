"""Utilities for calculating accuracy in analysis of perturbation and ablation."""
import copy

import numpy as np
import torch
import sys
sys.path.append('../../')
from src.utils import load_model


def calculate_accuracy(
        network_type,
        network_name,
        input_length,
        warmup_length,
        N,
        perturbation_type,
        perturbation,
        n_trials,
        base_path,
        p_input=0.5,
        trained_task="parity",
):
    """Calculate accuracy of a perturbed or ablated network.

    Args:
        network_type (str): 'single' or 'cumulative'
        network_name (int): 1, 2, 3, ...
        input_length (int): length of the input sequence for measuring accuracy
        warmup_length (int): length of the warmup sequence for measuring accuracy
        N (int): N that the network should be able to solve
        perturbation_type (str): Ablation or perturbation type
            - 'multiplicative rnn'
            - 'normalized rnn'
            - 'additive tau'
            - 'normalized tau'
            - 'normalized tau abs'
            - 'ablation'
            - 'ablation readout'
        perturbation (float): ablation index or perturbation strength (0 for no perturbation, -1 for no ablation)
        n_trials (int): number of trials for measuring accuracy
        base_path (str): path to the folder where the networks are stored, e.g. '../../trained_models/'
        p_input (float, optional): probability for Bernoulli input
        trained_task (str, optional): 'parity' or 'dms'

    Returns:
        list[float]: accuracies (length n_trials)
    """
    try:
        rnn_load = load_model(
            curriculum_type=network_type,
            task=trained_task,
            network_number=network_name,
            N_max=N,
            base_path=base_path,
        )
        match network_type:
            case 'single':
                index_in_head = 0
            case 'cumulative':
                index_in_head = N - 2
            case _:
                raise NotImplementedError
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {network_type}, network: {network_name}, {N}, {str(e)}")
        return None
    accuracies = []
    for i_trial in range(n_trials):
        rnn = perturb_network_(rnn_load, perturbation_type, perturbation)
        accuracy = calculate_accuracy_(rnn, network_type, input_length, warmup_length, N, trained_task=trained_task, p_input=p_input, index_in_head=index_in_head)
        accuracies.append(accuracy)
    return accuracies


def perturb_network_(rnn, perturbation_type, perturbation):
    rnn = copy.deepcopy(rnn)
    with torch.no_grad():
        match perturbation_type:
            case 'multiplicative rnn':
                if perturbation > 0:
                    rnn.w_hh[0].weight *= 1 + perturbation * torch.rand_like(rnn.w_hh[0].weight)
            case 'normalized rnn':
                # https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
                # equation (5)
                if perturbation > 0:
                    direction = torch.randn_like(rnn.w_hh[0].weight)
                    direction_norm = torch.linalg.norm(direction, ord='fro')
                    weight_norm = torch.linalg.norm(rnn.w_hh[0].weight, ord='fro')
                    rnn.w_hh[0].weight += perturbation / direction_norm * weight_norm * direction
            case 'additive tau':
                if perturbation > 0:
                    rnn.taus[0] += perturbation * torch.randn_like(rnn.taus[0])
            case 'normalized tau':
                # https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
                # equation (5)
                if perturbation > 0:
                    direction = torch.randn_like(rnn.taus[0])
                    direction_norm = torch.linalg.norm(direction)
                    tau_norm = torch.linalg.norm(rnn.taus[0])
                    rnn.taus[0] += perturbation / direction_norm * tau_norm * direction
            case 'normalized tau abs':
                # https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
                # equation (5)
                if perturbation > 0:
                    direction = torch.abs(torch.randn_like(rnn.taus[0]))
                    direction_norm = torch.linalg.norm(direction)
                    tau_norm = torch.linalg.norm(rnn.taus[0])
                    rnn.taus[0] += perturbation / direction_norm * tau_norm * direction

            case 'ablation':
                if perturbation > -1:
                    for idx in [perturbation] if isinstance(perturbation, (int, np.integer)) else perturbation:
                        rnn.w_hh[0].weight.data[idx, :] = 0
                        rnn.w_hh[0].weight.data[:, idx] = 0
                        rnn.fc[0].weight.data[:, idx] = 0
                        rnn.input_layers[0].weight.data[idx, :] = 0
            case 'ablation readout':
                if perturbation > -1:
                    for idx in [perturbation] if isinstance(perturbation, (int, np.integer)) else perturbation:
                        rnn.fc[0].weight.data[:, idx] = 0
    return rnn


def calculate_accuracy_(
        rnn, network_type, input_length, warmup_length, N, trained_task='parity', p_input=0.5, index_in_head=None
):
    inputs = torch.bernoulli(p_input * torch.ones((input_length + warmup_length, 1)))
    with torch.no_grad():
        if network_type == 'single-head':
            _, outputs = rnn(inputs, classify_in_time=True)
        else:
            _, outputs = rnn(inputs, classify_in_time=True, index_in_head=index_in_head)
        outputs = torch.vstack([o[0] for o in outputs])[warmup_length:].detach().numpy()
        outputs_predict = (outputs[:, 0] < outputs[:, 1]) * 1
    match trained_task:
        case 'parity':
            inputs_cumsum = inputs.detach().numpy().flatten().cumsum(-1)
            partial_sums = inputs_cumsum[warmup_length:] - inputs_cumsum[warmup_length - N:-N]
            outputs_correct = (partial_sums % 2)
            accuracy = (outputs_predict == outputs_correct).mean()
            return accuracy
        case 'dms':
            inputs_detach = inputs.detach().numpy().flatten()
            inputs_match = inputs_detach[warmup_length:] == inputs_detach[warmup_length - N + 1:-N +1]
            accuracy = (outputs_predict == inputs_match).mean()
            return accuracy
        case _:
            raise ValueError(f'Unknown task: {trained_task}')
