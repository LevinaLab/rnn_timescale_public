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

### Continous RNN analysis
To train a continuous RNN run `train_slurm_continuous.py` script or use `train_slurm_continuous.sh` script to submit a job to the cluster.
Parameters to train the three networks in the paper are:
```
train_slurm_continuous.sh -c cumulative -t parity -dup [7,8,9,10] -n 99
train_slurm_continuous.sh -c single -t parity -dup [7,8,9,10] -n 99
train_slurm_continuous.sh -c cumulative -t parity -dup [10] -n 1
```
More arguments through `python train_slurm_continous.py --help`.

For analysis run
- `exp003_test_correct.py`: computes and saves the accuracies
- `exp006_plot_ICLR_suppl.py`: recreates the paper figure
