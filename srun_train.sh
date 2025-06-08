#!/bin/bash

NUM_GPU="2"
JOB_NAME="173"
EXP_NAME="options/train/AlignFormer/AlignFormer_train.yml"

srun -p priority \
    --mpi=pmi2 \
    --job-name=$JOB_NAME \
    --gres=gpu:$NUM_GPU \
    --ntasks=$NUM_GPU \
    --ntasks-per-node=$NUM_GPU \
    --cpus-per-task=2 \
    --kill-on-bad-exit=1 \
python -u basicsr/train.py -opt $EXP_NAME --launcher="slurm"