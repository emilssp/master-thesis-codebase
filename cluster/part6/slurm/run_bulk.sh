#!/bin/bash
#SBATCH --job-name=bulk
#SBATCH --output=logs/bulk.out
#SBATCH --error=logs/bulk.err
#SBATCH --cpus-per-task=12
#SBATCH --time=32:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data

which python
python --version

python ./scripts/run_bulk.py