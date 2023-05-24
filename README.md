# Check Your Timescales: Single-Neuron and Network-Mediated Mechanisms in Recurrent Networks for Long Memory Tasks

Codes for the implementation of the NeurIPS 2023 submission titled "Check Your Timescales: Single-Neuron and Network-Mediated Mechanisms in Recurrent Networks for Long Memory Tasks".



## Requirements
- Python: 
- PyTorch:
- Numpy: 1.21.5 
- Scipy: 1.7.3
- Scikit-Learn: 1.1.3
- Pandas: 1.4.3

### Visualization
- Matplotlib: 3.5.1
- Seaborn: 0.11.2


## Training
To train a model, run ```./training/train.py```.
The script accepts the following arguments:

    -h, --help: Show the help message and exit.
    -c, --curriculum_type, str: Specify the curriculum type: (cumulative, sliding, single).
    -t, --task, str: Specify the task: (parity, dms).
    -r, --runs, int: Number of independent runs.
    -ih, --init_heads int: Number of heads to start with.
    -dh, --add_heads int: Number of heads to add per new curricula.
    -fh, --forget_heads int: Number of heads to forget for the sliding window curriculum type.

If an argument is not provided, the script uses the following default values:

    curriculum_type: cumulative
    task: parity
    runs: 1
    init_heads: 1
    add_heads: 1
    forget_heads: 1

Models are saved in the ```./trained_folders``` directory where models trained above 98% accuracy are saved as the curriculum grows.
## Evaluation


## Analysis
Codes for running different analysis on trained networks. It includes:
- Measuring $\tau$ and $\tau_{net}$
- Measuring dimensionality of population activity using PCA
- Running ablation and perturbation analysis

More details are provided in the README inside the "analysis" folder.


## Example trained models
