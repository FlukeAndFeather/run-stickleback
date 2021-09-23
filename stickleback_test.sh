#!/bin/bash
#
#SBATCH --job-name=stickleback_test
#
#SBATCH --time=0:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --mail-type=ALL

SIF=$GROUP_HOME/$USER/sing/sb-test_latest.sif
SCRIPT=$SCRATCH/stickleback/stickleback_test.py
DATA=$SCRATCH/stickleback/bw_breaths_sensors_events.pkl
WINDOW=100
FOLDS=5
TREES=120

ml python/3.9
pip install --user --upgrade --force-reinstall git+git://github.com/FlukeAndFeather/stickleback.git
python3 $SCRIPT $DATA $WINDOW $FOLDS $TREES $SLURM_NTASKS_PER_NODE
