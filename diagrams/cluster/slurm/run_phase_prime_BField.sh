#!/bin/bash
#SBATCH --job-name=BField
#SBATCH --output=logs/BField-%A_%a.out
#SBATCH --error=logs/BField-%A_%a.err
#SBATCH --array=0-24
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --partition=habanaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/muT_V15_prime_BField

which python
python --version

python ./scripts/run_phase_prime_BField.py ${SLURM_ARRAY_TASK_ID}