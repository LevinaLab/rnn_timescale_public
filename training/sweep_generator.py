"""
This file is modified from: https://github.com/CSCfi/slurm-hyper-search/blob/master/generate_params.py

Usage:

python sweep_generator.py --group=prod
"""
import argparse
import itertools
import os

import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class EqualTo:
    def __init__(self, field):
        self.field = field


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
        "GAMMA": [0.0, 0.9],
        "SCHEDULE_INPUT_LAYERS": [False],
        "SCHEDULE_W_HH": [True, False],
        "SCHEDULE_W_FF_IN": EqualTo(field='SCHEDULE_W_HH'),  # [True, False],
        "SCHEDULE_FC": [True, False],
        "SCHEDULE_TAUS": [True, False],
        # Duplication Scheme
        'WEIGHT_NOISE': [0.05, 0.3],
        'BIAS_NOISE': EqualTo(field='WEIGHT_NOISE'),
        'TAUS_NOISE': EqualTo(field='WEIGHT_NOISE'),
        "DUPLICATE_W_HH": [True, False],
        "DUPLICATE_INPUT_LAYERS": EqualTo(field='DUPLICATE_W_HH'),
        "DUPLICATE_W_FF_IN": EqualTo(field='DUPLICATE_W_HH'),
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


def parameter_grid_with_conditionals(param_dict):

    param_dict_no_cond = {k: v for k, v in param_dict.items() if isinstance(v, list)}
    combos = itertools.product(*param_dict_no_cond.values())
    df = pd.DataFrame(combos, columns=param_dict_no_cond.keys())

    param_dict_cond = {k: v for k, v in param_dict.items() if isinstance(v, EqualTo)}
    for k, v in param_dict_cond.items():
        df[k] = df[v.field]

    return df


def write_to_disk(ps, output_dir, file_name):
    fn = os.path.join(output_dir, file_name)

    with open(fn, 'w') as fp:
        for p in ps:
            p_str = ' '.join([args.format.format(name=k, value=v) for k, v in p.items()])
            if args.extra:
                p_str = args.extra + ' ' + p_str
            fp.write(p_str + '\n')

    print(f"{len(ps)} parameter combinations written to {os.path.abspath(fn)}")


def main(args, output_dir):
    group = args.group
    os.makedirs(output_dir, exist_ok=True)
    df = parameter_grid_with_conditionals(space[group])
    ps = df.to_dict(orient='records')
    file_name = f'{group}_sweep'
    write_to_disk(ps, output_dir, file_name)


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