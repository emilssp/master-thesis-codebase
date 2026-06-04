#!/bin/bash
#SBATCH --job-name=mult_V40
#SBATCH --output=logs/mult_V40-%A_%a.out
#SBATCH --error=logs/mult_V40-%A_%a.err
#SBATCH --array=0-2576
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/DFP_hx_mult_V40

which python
python --version

python ./scripts/run_dfp_hx_mult_V40.py ${SLURM_ARRAY_TASK_ID}