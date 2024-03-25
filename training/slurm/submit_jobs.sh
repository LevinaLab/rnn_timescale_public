#!/bin/bash

# Submit jobs to train the model with different values of n (network number)
save_path="/mnt/qb/levina/rnn_timescale_public/trained_models/ALIFE2024"

# sparsity experiment
for n in {1..5}
do
   sbatch rnn_timescale_public/training/slurm/train.sh -c cumulative -n $n -nn 500 -m sparse -st 0.028 -b "$save_path"
done

for n in {1..5}
do
   sbatch rnn_timescale_public/training/slurm/train.sh -c cumulative -n $n -nn 330 -m sparse -st 0.028 -b "$save_path"
done

#for n in {1..5}
#do
#   sbatch rnn_timescale_public/training/slurm/train.sh -c cumulative -n $n -nn 175 -m sparse -st 0.028 -b "$save_path"
#done
#
#for n in {1..5}
#do
#   sbatch rnn_timescale_public/training/slurm/train.sh -c cumulative -n $n -nn 55 -m sparse -st 0.028 -b "$save_path"
#done

# 20 neuron experiments
#for n in {1..5}
#do
#   sbatch rnn_timescale_public/training/slurm/train.sh -c cumulative -n $n -nn 20 -b "$save_path"
#done
#
## 54 neuron experiments
#for n in {1..5}
#do
#   sbatch rnn_timescale_public/training/slurm/train.sh -c cumulative -n $n -nn 54 -b "$save_path"
#done
#
## 91 neuron experiments
#for n in {1..5}
#do
#   sbatch rnn_timescale_public/training/slurm/train.sh -c cumulative -n $n -nn 91 -b "$save_path"
#done

## 128 neuron experiments
#for n in {1..5}
#do
#   sbatch rnn_timescale_public/training/slurm/train.sh -c cumulative -n $n -nn 128 -b "$save_path"
#done
