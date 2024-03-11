"""
This file is modified from: https://github.com/CSCfi/slurm-hyper-search/blob/master/generate_params.py

Usage:

python sweep_generator.py --group=prod
"""
import argparse
import os
from itertools import product

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

space = {
    'prod': {
        'SEED': [0],
        'NUM_EPOCHS': [200],
        'BATCH_SIZE': [256],
        'TRAINING_STEPS': [500],
        'TEST_STEPS': [50],
        'DEVICE': ['cpu'],
        'CURRICULUM': ['grow'],
        # Optimizer & Scheduling
        'LEARNING_RATE': [0.2, 0.4],
        'MOMENTUM': [0.1],
        'FREEZING_STEPS': [1],
        "GAMMA": [0.95, 0.10],
        "SCHEDULE_INPUT_LAYERS": (False, bool),
        "SCHEDULE_W_HH": [True, False],
        "SCHEDULE_W_FF_IN": [True, False],
        "SCHEDULE_FC": [True, False],
        "SCHEDULE_TAUS": [True, False],
        # Duplication Scheme
        'WEIGHT_NOISE': [0.05, 0.3],
        # 'BIAS_NOISE': [0],  # todo: setting these to the same as weight noise for now
        # 'TAUS_NOISE': [0],  # todo: setting these to the same as weight noise for now
        "DUPLICATE_INPUT_LAYERS": [True, False],
        # "DUPLICATE_W_HH": [True, False],    # todo: setting these to the same as DUPLICATE_INPUT_LAYERS for now
        # "DUPLICATE_W_FF_IN": [True, False], # todo: setting these to the same as DUPLICATE_INPUT_LAYERS for now
        "DUPLICATE_FC": [False],
        "DUPLICATE_TAUS": [True, False],
        # "DUPLICATE_TAUS": [True, False], # Setting these to same as DUPLICATE_W_HH for now
        # Agent
        "MAX_DEPTH": [50],
        "INPUT_SIZE": [1],
        "NET_SIZE": [5, 10],
        "NUM_CLASSES": [2],
        "BIAS": [True],
        "NUM_READOUT_HEADS_PER_MOD": [1],
        "TRAIN_TAU": [True],
        "TASK": ['parity'],
    },
}


def parameter_grid(param_dict):
    """Generate all combinations of parameters from a dictionary or a list of dictionaries."""
    # If param_dict is a dictionary, wrap it in a list
    if isinstance(param_dict, dict):
        param_dict = [param_dict]

    # Iterate over dictionaries in the list
    for params in param_dict:
        # Extract keys and values; sort keys to ensure consistent ordering
        keys = sorted(params)
        values = [params[key] for key in keys]

        # Generate all combinations of parameter values
        for combination in product(*values):
            yield dict(zip(keys, combination))


def main(args, output_dir):

    group = args.group

    os.makedirs(output_dir, exist_ok=True)

    file_name = f'{group}_sweep'
    fn = os.path.join(output_dir, file_name)

    ps = list(parameter_grid(space[group]))

    with open(fn, 'w') as fp:
        for p in ps:
            p_str = ' '.join([args.format.format(name=k, value=v) for k, v in p.items()])
            if args.extra:
                p_str = args.extra + ' ' + p_str
            fp.write(p_str + '\n')

    print(f"{len(ps)} parameter combinations written to {os.path.abspath(fn)}")


if __name__ == '__main__':
    print("Project Root:", root_dir)
    parser = argparse.ArgumentParser(
        description='Generate a set of hyper parameters')
    parser.add_argument('--group', type=str,
                        help='choose among "test", "prod", "sac"', required=True)
    parser.add_argument('--format', type=str, default='--{name}={value}',
                        help='format for parameter arguments, default is '
                        '--{name}={value}')
    parser.add_argument('--extra', type=str, help='Extra arguments to add')

    args = parser.parse_args()
    params_dir = os.path.join(root_dir, 'training', 'param_files')
    print("File will be saved to:", params_dir)
    main(args, params_dir)