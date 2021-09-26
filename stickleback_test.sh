#!/bin/bash
#
#SBATCH --job-name=stickleback_test
#
#SBATCH --time=0:08:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=ALL

SIF=$GROUP_HOME/$USER/sing/sb-test_latest.sif
SCRIPT=/stickleback/stickleback_test.py
DATA=$SCRATCH/stickleback/bw_breaths_sensors_events.pkl
WINDOW=50
FOLDS=2
TREES=16
NUMBA_CACHE=$SCRATCH/.numba_cache

singularity exec --env NUMBA_CACHE_DIR=$NUMBA_CACHE $SIF python3 $SCRIPT $DATA $WINDOW $FOLDS $TREES $SLURM_NTASKS_PER_NODE