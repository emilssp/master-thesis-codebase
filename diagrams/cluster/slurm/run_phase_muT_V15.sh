#!/bin/bash
#SBATCH --job-name=muT_V1.5
#SBATCH --output=logs/muT_V1.5-%A_%a.out
#SBATCH --error=logs/muT_V1.5-%A_%a.err
#SBATCH --array=0-49
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --partition=habanaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/muT_V15

which python
python --version

python ./scripts/run_phase_muT_V15.py ${SLURM_ARRAY_TASK_ID}