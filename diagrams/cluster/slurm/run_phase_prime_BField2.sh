#!/bin/bash
#SBATCH --job-name=BField2
#SBATCH --output=logs/BField2-%A_%a.out
#SBATCH --error=logs/BField2-%A_%a.err
#SBATCH --array=0-47
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/muT_V2_prime_BField data/muT_V2_prime_BField/raw

which python
python --version

python ./scripts/run_phase_prime_BField2.py ${SLURM_ARRAY_TASK_ID}