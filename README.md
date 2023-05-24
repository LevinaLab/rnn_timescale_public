# Check Your Timescales: Single-Neuron and Network-Mediated Mechanisms in Recurrent Networks for Long Memory Tasks

Codes for the implementation of the NeurIPS 2023 submission titled "Check Your Timescales: Single-Neuron and Network-Mediated Mechanisms in Recurrent Networks for Long Memory Tasks".



## Requirements
- Python: 3.7.13
- PyTorch: 1.3.1
- Numpy: 1.21.5 
- Scipy: 1.7.3
- Scikit-Learn: 1.1.3
- Pandas: 1.4.3

### Visualization
- Matplotlib: 3.5.1
- Seaborn: 0.11.2


## Training
To train a model, run, `cd ./training` `python train.py --{args}`.
The script accepts the following arguments:

    -h, --help: Show the help message and exit.
    -c, --curriculum_type, str: Specify the curriculum type: (cumulative, sliding, single).
    -t, --task, str: Specify the task: (parity, dms).
    -r, --runs, int: Number of independent runs.
    -ih, --init_heads int: Number of heads to start with.
    -dh, --add_heads int: Number of heads to add per new curricula.
    -fh, --forget_heads int: Number of heads to forget for the sliding window curriculum type.
    -s, --seed int: Set the random seed.

For example

```python train.py -c cumulative -t parity -s 0```

If an argument is not provided, the script uses the following default values:

    curriculum_type: cumulative
    task: parity
    runs: 1
    init_heads: 1
    add_heads: 1
    forget_heads: 1
    seed: np.random.choice(2 ** 32)

Models are saved in the `./trained_models` directory where models trained above 98% accuracy are saved as the curriculum grows.
## Evaluation
To load and test a trained model, you can run the Jupyter Notebook `./training/eval.ipynb` to load
an example model and evaluate it on the task it was trained on. 

## Analysis
Codes for running different analysis on trained networks. It includes:
- Measuring $\tau$ and $\tau_{net}$
- Measuring dimensionality of population activity using PCA
- Running ablation and perturbation analysis

More details are provided in the README inside `./analysis`.

The results of the analysis are saved in `./results`. 


## Example trained models
Example trained models are saved in the `./trained_models` directory.
All example models are trained with ```-seed 0```.

The naming convention for the saved models is:
`{curriculum_type}_{task}_network_{network_number}/rnn_N{N_min}_N{N_max}`. 


To load a model, import the ```load_model``` function ```from src.utils import load_model```
which takes the following arguments

    curriculum_type (str): 'cumulative' (a.k.a. multi-head)
                            f'sliding_{n_heads}_{n_forget}' (a.k.a. multi-head sliding)
                            'single' (a.k.a single-head)
    task (str): 'parity' or 'dms'
    network_number (int): 1, 2, 3, ...
    N_max (int): N that the network should be able to solve (N_max = N_min for curriculum_type='single')
    N_min (int): minimum N, potentially depending on curriculum_type

For example: ```load_model('cumulative', 'parity', 1, 2, 50)```
