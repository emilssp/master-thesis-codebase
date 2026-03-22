#!/bin/bash
#SBATCH --job-name=SP
#SBATCH --output=logs/SP-%A_%a.out
#SBATCH --error=logs/SP-%A_%a.err
#SBATCH --array=0-210
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=habanaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/SP

which python
python --version

python ./scripts/run_sp.py ${SLURM_ARRAY_TASK_ID}