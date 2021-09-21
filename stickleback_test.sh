#!/bin/bash
#
#SBATCH --job-name=batch_cv_all
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G

SIF=$GROUP_HOME/$USER/sing/sb-test_latest.sif
SCRIPT=$HOME/stickleback_test.py
DATA=$GROUP_SCRATCH/lunges.pkl 
WINDOW=64
FOLDS=4
TREES=8

srun singularity exec $SIF python3 $SCRIPT $DATA $WINDOW $FOLDS $TREES
