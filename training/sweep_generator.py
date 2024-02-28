"""
This file is from: https://github.com/CSCfi/slurm-hyper-search/blob/master/generate_params.py

Usage:

python generate_params.py --production all  # generates full grid of prod hyperparams under params/prod_sweep
python generate_params.py all               # generates full grid of test hyperparams under params/test_sweep
"""
import argparse
import os
import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid


OUT_DIR = 'params'
space = {
    'prod': {
        'env_N': [6],
        'env_K': [0],
        'learning_rate': [1e-3],
        'lr_rho': [1e-3],
        'beta_1': [np.inf],
        'beta_3': [2, 4, 8, 16, 32, 64, 1028],
        'beta_beh': [0.5, 1,  2, 4, 8, 16, 32, 64, 1028],
        'n_envs_per_param': [1000],
        'n_seeds_per_param': [1],
        'seed_per_env': [True],
        "reward_noise_std": [0.05],
        'pre_built_env_folder': [''],
        'pre_built_env': [False],
        'fitness_type': ['k0_exponential'],
        "mask": [None],
        "n_actions_masked": [0],
        "mask_length": [0],
        "replay_memory": [1],
        "replay_batch_size": [1],
    },
}


def main(args):
    group = args.group
    os.makedirs(OUT_DIR, exist_ok=True)

    file_name = f'{group}_sweep'

    fn = os.path.join(OUT_DIR, file_name)

    if args.n.lower() == 'all':
        ps = ParameterGrid(space[group])
    else:
        n = int(args.n)
        rng = np.random.RandomState(args.seed)
        ps = ParameterSampler(space[group], n_iter=n, random_state=rng)

    print(f"{len(ps)} parameters.")
    print("Divisors:")
    for i in range(2, len(ps)-1):
        if len(ps) % i == 0:
            print(f"    {i} * {len(ps) / i}")

    with open(fn, 'w') as fp:
        for p in ps:
            p_str = ' '.join([args.format.format(name=k, value=v) for k, v in p.items()])
            if args.extra:
                p_str = args.extra + ' ' + p_str
            fp.write(p_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a set of hyper parameters')
    parser.add_argument('--group', type=str,
                        help='choose among "test", "prod", "sac"')
    parser.add_argument('--n', type=str, default='all',
                        help='random search: number of hyper parameter sets '
                        'to sample, for grid search: set to "all"')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for deterministic runs')
    parser.add_argument('--format', type=str, default='--{name}={value}',
                        help='format for parameter arguments, default is '
                        '--{name}={value}')
    parser.add_argument('--extra', type=str, help='Extra arguments to add')

    args = parser.parse_args()
    main(args)