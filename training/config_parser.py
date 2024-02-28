import argparse


hyperparameter_defaults = {
    # Training
    "SEED": (0, int),
    "NUM_EPOCHS": (1000, int),
    "BATCH_SIZE": (256, int),
    "TRAINING_STEPS": (600, int),
    "TEST_STEPS": (50, int),
    "DEVICE": ('cuda', str),
    "CURRICULUM": ('grow', bool),
    # Optimizer & Scheduling
    "LEARNING_RATE": (0.05, float),
    "MOMENTUM": (0.1, float),
    "FREEZING_STEPS": (25, int),  # how many scheduling steps are taken upon successful completion of curriculum step.
    "GAMMA": (0.95, float),  # learning rate decay factor upon every scheduling step.
    # Duplication Scheme
    "WEIGHT_NOISE": (0.03, float),
    "BIAS_NOISE": (0.03, float),
    "TAUS_NOISE": (0.02, float),
    # Agent
    "MAX_DEPTH": (50, int),
    "INPUT_SIZE": (1, int),   # learning rate for updates to p(a).
    "NET_SIZE": (10, float),
    "NUM_CLASSES": (2, int),
    "BIAS": (True, bool),
    "NUM_READOUT_HEADS_PER_MOD": (1, int),
    "TRAIN_TAU": (True, bool),
    # TASK
    "TASK": ('parity', str),
    # "INIT_HEADS": (1, int),
    # "NUM_ADD": (0, int),
    # "NUM_FORGET": (0, int),

    }


def _over_write_defaults(args):
    user_input = vars(args)
    user_input = {k: v for k, v in user_input.items() if v is not None}
    config = {k: v[0] for k, v in hyperparameter_defaults.items()}
    for k, v in user_input.items():
        if k in config.keys():
            config[k] = v
        else:
            raise Exception(f"parameter {k} is not valid, see config_default.hyperparameter_defaults")
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Set hyperparameters for RL env and agent.')
    for k, v in hyperparameter_defaults.items():
        parser.add_argument('--' + k, type=v[1], required=False)

    args = parser.parse_args()
    config = _over_write_defaults(args)
    return config