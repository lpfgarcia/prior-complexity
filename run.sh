#!/bin/bash

#SBATCH -J prior        # Job name
#SBATCH -o job.%j.out   # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1            # Total number of nodes requested
#SBATCH -n 1            # Total number of mpi tasks requested
#SBATCH -t 12:00:00     # Run time (hh:mm:ss)
#SBATCH -p mi2508x      # Desired partition

export STAGE=$1
export DATASETS=$2
export JOBS=$3
export PRIOR=$4

uv run --no-sync jupyter nbconvert --to notebook --execute \
    --inplace --ExecutePreprocessor.timeout=-1 \
    --output notebooks/analysis_stage${STAGE}.ipynb analysis.ipynb
