import os
from datetime import datetime
import glob


def get_log_dir(results_dir):
    """ safe log directory creation

    given multiple nodes and multiple subprocesses per node attempting to make this directory and write to it, first
    I check whether there is already a directory with the current SLURM ID that is being run (across all nodes, and
    all subprocesses per node), and if so I just return it. But I also do another try/except block in case two
    subprocesses (perhaps from different nodes) attempt to make this directory for the first time with exactly the
    same time-stamp. In this case, whoever is second will just get the same path as the existing folder.

    log directory structure:
    logs/slurm-[slurm_array_job_id]-[datetime]/N=[N]_K=[K]...=[slurm_task_id]_[log-type.suffix]

    where log-type.suffix = [summary.logs|rewards.npy|losses.npy|mi.npy] i.e. 4 files per hyperparameter combo.
    :param config:
    :return:
    """
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    slurm_array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID', None)
    dir_name = f"SLURM_ARRAY_JOB_ID={slurm_array_job_id}"
    dir_regex = os.path.join(results_dir, f"SLURM_ARRAY_JOB_ID={slurm_array_job_id}_*")
    hits = glob.glob(dir_regex)
    if len(hits) == 1:  # assumes every run has a different SLURM_ARRAY_JOB_ID
        return hits[0]
    elif len(hits) == 0:
        print(f"regex: {dir_regex} returned 0 hits, creating directory now")
        timestamp = datetime.now().strftime("%b-%d-%Y-%H_%M_%S")
        dir_path = os.path.join(results_dir, f"{dir_name}_{timestamp}")
        try:   # in the rare event that two subprocesses from two different nodes try mkdir with exact same timestamp
            os.mkdir(dir_path)
            return os.path.abspath(dir_path)
        except FileExistsError as e:
            return os.path.abspath(dir_path)
    else:
        raise Exception(f"Found multiple directories matching regex {dir_regex}: {hits}")


def get_env_vars():
    slurm_array_job_id = os.environ.get('SLURM_ARRAY_JOB_ID', None)
    slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', None)
    subproc_id = os.environ.get('SUBPROC_ID', None)
    params_id = os.environ.get('PARAMS_ID', None)
    logs_path = os.environ.get('LOGS_PATH', None)
    env_vars = {'SLURM_ARRAY_JOB_ID': slurm_array_job_id, 'SLURM_ARRAY_TASK_ID': slurm_task_id,
                   'SUBPROC_ID': subproc_id, 'PARAMS_ID': params_id, 'LOGS_PATH': logs_path}
    print("Running train.py with:", env_vars)
    return env_vars

