# Emergent mechanisms for long timescales depend on training curriculum and affect performance in memory tasks (2024)

Codes for the implementation of the ICLR 2024 paper titled ["Emergent mechanisms for long timescales depend on training curriculum and affect performance in memory tasks"](https://openreview.net/forum?id=xwKt6bUkXj).


If you use this code for a scietific publication, please cite the paper:

Khajehabdollahi*, S., Zeraati*, R., Giannakakis, E., Schäfer, T. J., Martius, G., & Levina, A. (2024). Emergent mechanisms for long timescales depend on training curriculum and affect performance in memory tasks. International Conference on Learning Representations (ICLR).

```
@inproceedings{
khajehabdollahi2024emergent,
title={Emergent mechanisms for long timescales depend on training curriculum and affect performance in memory tasks},
author={Sina Khajehabdollahi and Roxana Zeraati and Emmanouil Giannakakis and Tim Jakob Sch{\"a}fer and Georg Martius and Anna Levina},
booktitle={International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=xwKt6bUkXj}
}
```


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
To train a model, run `python training/train.py --{args}`.
The script accepts the following arguments:

    -h, --help: Show the help message and exit.
    -b, --base_path, str: The base path to save results.
    -m, --model_type, str: Standard RNN model or modified location of non-linearity ([default], mod)
    -a, --afunc, str: Activation functions: ([leakyrelu], relu, tanh, sigmoid)
    -nn, --num_neurons, int: The number of hidden neurons in the RNN. 
    -ni, --ns_init, int: The starting value of N for the task [default=2].
    -c, --curriculum_type, str: Specify the curriculum type: ([cumulative], sliding, single).
    -n, --network_number, int: 'The run number of the network, to be used as a naming suffix for savefiles.
    -t, --task, str: Specify the task: ([parity], dms).
    -T, --tau, float: The value of tau each neuron starts with. If set, taus will not be trainable. Default = None. (float > 1)
    -ih, --init_heads int: Number of heads to start with.
    -dh, --add_heads int: Number of heads to add per new curricula.
    -fh, --forget_heads int: Number of heads to forget for the sliding window curriculum type.
    -s, --seed int: Set the random seed.

For example

```python training/train.py -c cumulative -t parity -s 0 -b "../trained_models"```

If an argument is not provided, the script uses the following default values:
    
    model_type: default
    activation_function: 'leakyrelu'
    num_neurons: 500
    curriculum_type: cumulative
    task: parity
    tau: 1.
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
- Continuous RNN training and analysis
- LSTM training and analysis

More details are provided in the README inside `./analysis`.

The results of the analysis are saved in `./results`. 


## Example trained models
Example trained models are saved in the `./trained_models` directory.
All example models are trained with ```-seed 0```.

The naming convention for the saved models is:
`{curriculum_type}_{task}_network_{network_number}/rnn_N{N_min}_N{N_max}`. 


To load a model, import the ```load_model``` function ```from src.utils import load_model```
which takes the following arguments:

    curriculum_type (str): 'cumulative' (a.k.a. multi-head)
                            f'sliding_{n_heads}_{n_forget}' (a.k.a. multi-head sliding)
                            'single' (a.k.a single-head)
    task (str): 'parity' or 'dms'
    network_number (int): 1, 2, 3, ...
    N_max (int): N that the network should be able to solve (N_max = N_min for curriculum_type='single')
    N_min (int): minimum N, potentially depending on curriculum_type

For example: ```load_model(curriculum_type='cumulative', task='parity', network_number=1, N_min=2, N_max=50)```

## Contact
If you have any questions, please contact us through GitHub. 


