#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=00-24:00            # Runtime in D-HH:MM
# Node feature:
#SBATCH --constraint="gpu"
# Specify type and number of GPUs to use:
#  GPU type can be v100 or rtx5000
#SBATCH --gres=gpu:rtx5000:1       # If using only 1 GPU of a shared node
#SBATCH --ntasks-per-node=20    # If using only 1 GPU of a shared node
#SBATCH --mem=16G             # Memory is necessary if using only 1 GPU
#SBATCH --open-mode=append        # update the output file periodically (?)
#SBATCH --output=/u/mhami/rnn_timescale_public/logs/%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/u/mhami/rnn_timescale_public/logs/%j.err   # File to which STDERR will be written - make sure this is not on $HOME
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
python "./training/train.py" -b "$save_path" "$@"
conda deactivate
