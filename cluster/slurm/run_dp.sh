#!/bin/bash
#SBATCH --job-name=DP
#SBATCH --output=logs/DP-%A_%a.out
#SBATCH --error=logs/DP-%A_%a.err
#SBATCH --array=0-230
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/DP

which python
python --version

python ./scripts/run_dp.py ${SLURM_ARRAY_TASK_ID}