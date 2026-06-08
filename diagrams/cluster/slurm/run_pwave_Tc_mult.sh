#!/bin/bash
#SBATCH --job-name=pwave
#SBATCH --output=logs/pwave-%A_%a.out
#SBATCH --error=logs/pwave-%A_%a.err
#SBATCH --array=0-863
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/pwave_Tc_mult

which python
python --version

python ./scripts/run_pwave_Tc_mult.py ${SLURM_ARRAY_TASK_ID}