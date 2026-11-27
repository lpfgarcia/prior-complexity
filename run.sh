#!/bin/bash

#SBATCH -J prior        # Job name
#SBATCH -o job.%j.out   # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1            # Total number of nodes requested
#SBATCH -n 1            # Total number of mpi tasks requested
#SBATCH -t 24:00:00     # Run time (hh:mm:ss)
#SBATCH -p mi2104x      # Desired partition

export MODE=$1

if [ "$MODE" = "real" ]; then
    export JOBS=$2
    NAME=real
elif [ "$MODE" = "synthetic" ]; then
    export STAGE=$2
    export DATASETS=$3
    export JOBS=$4
    NAME=synthetic_stage$STAGE
else
    echo "uso: run.sh real JOBS | run.sh synthetic STAGE DATASETS JOBS"
    exit 1
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

uv run --no-sync jupyter nbconvert --to notebook --execute --output 02_extract_$NAME.ipynb \
    --ExecutePreprocessor.timeout=-1 notebooks/02_extract.ipynb