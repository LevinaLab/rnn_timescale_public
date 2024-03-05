#!/bin/bash
#SBATCH -J rnn_growth        # A single job name for the array
#SBATCH --partition=general                     # stands for --partition
#SBATCH --mem=60G
#SBATCH --cpus-per-task=40
#SBATCH -t 0-48:00                      # Maximum execution time (D-HH:MM)
#SBATCH --array=0-100                     # maps 1 to N to SLURM_ARRAY_TASK_ID below
#SBATCH --output=/u/mhami/rnn_timescale_public/logs/%A_%a.out # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/u/mhami/rnn_timescale_public/logs/%A_%a.err   # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=manihamidi@gmail.com   # Email to which notifications will be sent

# some bug
source $HOME/.bashrc

conda activate rnn_timescale
# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
echo "---------------------------------"

save_path="/u/mhami/rnn_timescale_public/trained_models"

# insert your commands here
# takes the same arguments as train.py but instead of number of runs you need to label the run number manually with -n
# e.g. python train.py -c cumulative -t parity -s 0 -n 0
#python "./training/train_growth.py"
python "./training/parallel_dispatcher.py" --path_to_script="training/train_growth.py" --params_file="prod"

conda deactivate
