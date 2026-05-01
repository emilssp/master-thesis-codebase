#!/bin/bash
#SBATCH --job-name=DFP_hx1
#SBATCH --output=logs/DFP_hx1-%A_%a.out
#SBATCH --error=logs/DFP_hx1-%A_%a.err
#SBATCH --array=0-204
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/DFP_hx1

which python
python --version

python ./scripts/run_dfp_hx1.py ${SLURM_ARRAY_TASK_ID}