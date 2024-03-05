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

# Running growing model on slurm: 

## Locally

### Option 1: Run a single model

`python train_growth.py`  # This will run a single model with default parameters under `config_parser.py`

### Option 2: Run multiple in parallel

1. Make sure there is a `param_file` in `./training/param_files/prod_sweep` with the desired hyperparameters.

You can generate this by modifying `sweep_generator.py` as desired and running:

`python sweep_generator.py --group="prod"`

2. Spawn a subprocess for each param combo in parallel using:

`python parallel_dispatcher.py --group="prod" --path_to_script="train_growth.py"`

Be careful not to have too many parameters in the file, as this will spawn a subprocess for each.


## On Cluster

### Option 1: GPU

0. Make sure the default parameters in `config_parser.py` are desired. Push changes and pull on the cluster.
1. Make sure the slurm file `train.sh` is set up to request the desired time limit etc.
2. Run: `sbatch train.sh`

### Option 2: Hyperparameter search with CPU

0. Modify `sweep_generator.py` to include the desired hyperparameters.
1. Create hyperparameter grid using 

`python sweep_generator.py --group="prod"`

2. Modify slurm file `train_parallel.sh` to pick number of nodes, cpus, and memory requested. 
3. Run: `sbatch train_parallel.sh`.
4. Monitor job using `squeue` and by inspecting the log files.  

    * Useful command to watch status of slurm run:


```bash
tmux new-session \; split-window -h \; send-keys 'watch -n 1 squeue --user=<username>' C-m \; select-pane -t 0 \; send-keys "jobid=$(sacct -n -o JobID,Submit --format=JobID,Submit | awk -F'[_. ]' '{print $1, $2}' | sort -k2,2r | uniq | tail -n 1 | awk '{print $1}')" C-m
```

* __NOTE: Replace `<username>` with your username.__


To find the parameter combo with the largest N so far, run:

```bash
echo $(find . -type f -name "*_N*" | awk -F '_N' '{print $0 " " $(NF)}' | sort -t ' ' -k2 -n | tail -1 | cut -d ' ' -f1)
```


Transferring files to Raven for analysis: 

```bash
src="./SLURM_ARRAY_JOB_ID=7669330_Mar-05-2024-13_14_48"
dest="/raven/u/mhami/rnn_timescale_public/trained_models/"
rsync --info=progress2 -ah --partial "$src" "$dest"
```

For example: ```load_model('cumulative', 'parity', 1, 2, 50)```

Update rule: 

$r_i(t) = (1 -\frac{1}{\tau})r_i(t-1) + \frac{1}{2}r_j(t-1)$  

r_i(t)  =  w_{ij}r_j(t)   + 

input is 1, 500, 


