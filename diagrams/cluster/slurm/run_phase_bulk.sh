#!/bin/bash
#SBATCH --job-name=bulk
#SBATCH --output=logs/bulk-%A_%a.out
#SBATCH --error=logs/bulk-%A_%a.err
#SBATCH --array=0-44
#SBATCH --cpus-per-task=6
#SBATCH --time=62:00:00
#SBATCH --partition=habanaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/muT_V15_bulk

which python
python --version

python ./scripts/run_phase_bulk.py ${SLURM_ARRAY_TASK_ID}