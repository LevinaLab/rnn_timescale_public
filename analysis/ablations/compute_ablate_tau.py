"""Ablate based on tau.

20 fastest and 20 slowest
"""
import numpy as np
import pandas as pd

import sys
sys.path.append('../../')
from src.utils import load_model
from src.utils import calculate_accuracy

path_ids_ablation = 'ids_ablation_tau_effective/'

network_names = list(range(1, 5))  # [f'network_{i}' for i in range(1, 9)]
network_types = ['single', 'cumulative']
Ns = [5, 30, 35]

# parameters for accuracy measurement
input_length = 1000
warmup_length = 100
perturbation_type = 'ablation'
n_batch_perturbation = 10

model_path = '../../trained_models'
df_filename = '../../results/df_ablate_tau.pkl'

# %% accuracy from ablation
df = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [network_types, network_names, Ns],
        names=['network_type', 'network', 'N'],
    ),
    columns=['baseline_accuracy', 'ids_ablated', 'tau_ablated', 'accuracies'],
)
for index in df.index:
    network_type, network_name, N = index
    print(f"network_type = {network_type}, network = {network_name}, N = {N}")
    try:
        rnn = load_model(
            curriculum_type=network_type,
            task='parity',
            network_number=network_name,
            N_max=N,
            base_path=model_path,
        )
    except (FileNotFoundError, NotImplementedError) as e:
        print("File not found")
        print(e)
        df.drop(index, inplace=True)
        continue
    taus = rnn.taus[0].detach().numpy()
    ids_ablated = np.argsort(taus)[:20].tolist() + np.argsort(taus)[-20:].tolist()
    tau_ablated = taus[ids_ablated]

    df.at[index, 'ids_ablated'] = ids_ablated
    df.at[index, 'tau_ablated'] = tau_ablated
    df.at[index, 'accuracies'] = np.zeros((len(ids_ablated), n_batch_perturbation))
    df.at[index, 'baseline_accuracy'] = calculate_accuracy(
        network_type,
        network_name,
        input_length,
        warmup_length,
        N,
        perturbation_type,
        -1,
        n_batch_perturbation,
        base_path=model_path,
    )
    for i_idx, id_ablate in enumerate(ids_ablated):
        print(f"i_idx = {i_idx}, id_ablate = {id_ablate}, tau_ablate = {tau_ablated[i_idx]}")
        perturbation = id_ablate
        accuracies = calculate_accuracy(
            network_type,
            network_name,
            input_length,
            warmup_length,
            N,
            perturbation_type,
            perturbation,
            n_batch_perturbation,
            base_path=model_path,
        )
        df.at[index, 'accuracies'][i_idx] = accuracies
if len(df) == 0:
    raise ValueError("Dataframe is empty, because no data was found.")
df.to_pickle(df_filename)
