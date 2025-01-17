"""Perturb tau and measure accuracy."""
import numpy as np
import pandas as pd
import torch

import sys
sys.path.append('../../')
from src.utils import calculate_accuracy

torch.manual_seed(42000)

# input
input_length = 1000
warmup_length = 100

Ns = [30]

network_names = list(range(1, 5))  # [f'network_{i}' for i in range(1, 6)]
network_types = ['single', 'cumulative']  # ['single-head', 'cumulative']

n_batch_perturbation = 10
perturbations = [0] + list(np.geomspace(1e-3, 1e1, 13))

model_path = '../../trained_models'
df_filename = '../../results/df_perturb_tau_normalized_abs.pkl'

# %% accuracy with perturbation
df = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [network_types, network_names, Ns, perturbations],
        names=['network_type', 'network', 'N', 'perturbation'],
    ),
    columns=['accuracy'],
)
for index in df.index:
    network_type, network, N, perturbation = index
    print(f"network_type = {network_type}, network = {network}, N = {N}, perturbation = {perturbation}")
    accuracies = df.at[index, 'accuracy'] = calculate_accuracy(
        network_type,
        network,
        input_length,
        warmup_length,
        N,
        'normalized tau abs',
        perturbation,
        n_batch_perturbation,
        base_path=model_path,
    )
    if accuracies is None:
        df.drop(index, inplace=True)
    else:
        df.at[index, 'accuracy'] = accuracies
if len(df) == 0:
    raise ValueError("Dataframe is empty, because no data was found.")
df.to_pickle(df_filename)
