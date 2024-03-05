"""
Usage:

python slurm_hyper_search_dispatcher.py ./params/test_sweep 1
"""
import sys
from datetime import datetime
from subprocess import Popen
import multiprocessing
import os, time
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add both the parent directory and the 'src' directory to the module search path
sys.path.insert(0, project_root)

from src.utils import slurm

parser = argparse.ArgumentParser(description='Run multiple processes to sweep over parameters')

parser.add_argument('--params_file', type=str,
                    help='File containing parameters to sweep over, each line is one set of'
                         ' parameters executed by a separate subprocess')

parser.add_argument('--path_to_script', type=str, default='train.py',
                    help='Path to script to run')
parser.add_argument('--test', action='store_true',
                    help='Test it on the first 10 hyperparam combos only')

args = parser.parse_args()


def all_done(procs):
    done_status = [p.poll() is not None for p in
                   procs]  # When p.poll is NOT None == process is finished. When ALL are NOT None then all have finished.
    return done_status


if __name__ == '__main__':
    N_PARAMS_TEST_MAX = 10
    print("Parameter File: ", args.params_file)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    current_directory = os.path.abspath(os.path.curdir)
    os.environ['PYTHONPATH'] = current_directory
    print("Current directory: ", current_directory, flush=True)
    print("CWD: ", os.getcwd())
    params = open(f"./param_files/{args.params_file}_sweep", 'r').read().splitlines()
    n_params_for_test = min([len(params), N_PARAMS_TEST_MAX])
    n_params_used = len(params) if not args.test else n_params_for_test
    print(f"Running in TEST MODE (only 10 out of {len(params)}) hyperparams"
          if args.test else f"Running in PRODUCTION MODE with all {len(params)} hyperparams.", flush=True)
    num_cores = multiprocessing.cpu_count()
    print("Number of cores available: ", num_cores, flush=True)
    slurm_job_id = os.environ.get('SLURM_ARRAY_JOB_ID', None)
    if slurm_job_id:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        n_slurm_nodes = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
        max_jobs_per_node = n_params_used // n_slurm_nodes
        print(f"# Current array task id: {task_id} / {n_slurm_nodes}", flush=True)
        print(
            f"There are {n_slurm_nodes} slurm nodes and {n_params_used} total jobs to dispatch from ./params/{args.params_file}. \n"
            f"There will have to be {max_jobs_per_node} jobs per node.", flush=True)
        if max_jobs_per_node > num_cores:
            print(f"WARNING: # of subprocesses per node ({max_jobs_per_node}) exceeds # of cores ({num_cores})",
                  flush=True)
    else:
        # todo: overcome this limitation e.g. by substituting slurm's parallel nodes with serial executions.
        slurm_job_id = 'local'
        task_id = 0
        n_slurm_nodes = 0
        max_jobs_per_node = num_cores - 1  # leave one core for the OS
        print(f"# No slurm job detected. Running locally on {num_cores} cores.", flush=True)

    logs_path = slurm.get_log_dir(results_dir=os.path.join(project_root, 'trained_models'))  # the path for logs from the entire array
    print(f"Logging results to: {logs_path}", flush=True)
    os.environ['LOGS_PATH'] = logs_path

    # Determine which lines in the params file this node should run
    start_index = task_id * max_jobs_per_node  # todo: relies on 0-indexed task_id's.
    num_jobs_this_node = max_jobs_per_node if task_id < n_slurm_nodes - 1 else n_params_used - start_index
    end_index = start_index + num_jobs_this_node

    python_executable = sys.executable

    print(f"# Current slurm job id: {slurm_job_id}", flush=True)
    print(f"Running lines: {start_index} to {end_index} from {args.params_file} "
          f"on node {task_id} of {n_slurm_nodes} nodes.", flush=True)
    print("Python executable: ", python_executable, flush=True)

    # todo: If n_total > n_slurm_nodes * max_jobs_per_node:
    #       allow for more cycles of subprocess launches, serially, after first batch is done.
    processes = []
    for param_id in range(start_index, end_index):
        os.environ['SUBPROC_ID'] = str(param_id)
        os.environ['PARAMS_ID'] = str(param_id)
        current_env = os.environ.copy()
        params_list = params[param_id].split(' ')
        print(
            f"Node {task_id}: Running {args.path_to_script} with arguments from line # {param_id} from {args.params_file}: {params_list}",
            flush=True)
        proc = Popen([python_executable, args.path_to_script] + params_list, env=current_env)
        processes.append(proc)
        print(f"Node {task_id}: Dispatched subprocess-{param_id}", flush=True)

    print("Done with dispatch", flush=True)
    DONE_TASKS = 0
    FIRST = True
    while not all(all_done(processes)):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if sum(all_done(processes)) > DONE_TASKS or FIRST:
            message = f"Node {task_id}: {sum(all_done(processes))} / {len(all_done(processes))} subprocesses are done. Waiting for all to finish..."
            print(f"{timestamp} -- {message}", flush=True)
        else:
            print(".", end="", flush=True)
        time.sleep(30)
        FIRST = False
