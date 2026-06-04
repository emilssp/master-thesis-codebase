#!/bin/bash
#SBATCH --job-name=batch3
#SBATCH --output=logs/batch3-%A_%a.out
#SBATCH --error=logs/batch3-%A_%a.err
#SBATCH --array=0-9
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/muT_V2

which python
python --version

python ./scripts/run_phase_batch3.py ${SLURM_ARRAY_TASK_ID}