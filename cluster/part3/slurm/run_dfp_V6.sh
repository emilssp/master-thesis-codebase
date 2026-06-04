#!/bin/bash
#SBATCH --job-name=DFP_V6
#SBATCH --output=logs/DFP_V6-%A_%a.out
#SBATCH --error=logs/DFP_V6-%A_%a.err
#SBATCH --array=0-262
#SBATCH --cpus-per-task=4
#SBATCH --time=32:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/DFP_V6

which python
python --version

python ./scripts/run_dfp_V6.py ${SLURM_ARRAY_TASK_ID}