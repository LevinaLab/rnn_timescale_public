# Summary of analysis codes

Codes for running different analysis on trained networks.


### Timescale analysis
Codes are located in the "timescales" folder. "compute_taus.py" script contains the code for loading different networks, simulating them forward and saving $\tau$ and $\tau_{net}$ as described in the manuscript.

### Network dynamics analysis
Codes are located in the "network_dynamics" folder. "compute_pca.py" script contains the code for loading different networks, simulating them forward and computing the dimensionality of population activity using PCA.

### Ablation analysis
Codes are located in the "ablations" folder.
Plot function works out of the box, based on the precomputed data.
To recompute the data, train your own networks, and run "compute_ablate_tau.py" script.

### Perturbation and retraining analysis
Codes are located in the "perturbation_retraining" folder.
Plot function works out of the box, based on the precomputed data.
To recompute the data, train your own networks, and run "compute_perturb_rnn_normalized.py" and 
"compute_perturb_tau_normalized_abs.py" scripts.
