import os
import pickle

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def _load_saved_ac(base_path, network_number, n_max, curriculum_type):
    """
    Load the saved taus for the given network number and N_max.
    """
    # Load the saved taus
    with open(
        os.path.join(
            base_path,
            f'lstm_{curriculum_type}_network_{network_number}_N{n_max}_acs_taus.pkl',
        ),
        'rb',
    ) as f:
        data = pickle.load(f)
    return data['ac_all']


fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
ax.set_yscale('log')
ac_all = _load_saved_ac(
    base_path='../../results',
    network_number=1,
    n_max=8,
    curriculum_type='single',
)
ax.plot(ac_all[:, :].T)
fig.show()
