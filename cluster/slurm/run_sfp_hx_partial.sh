#!/bin/bash
#SBATCH --job-name=SFP_hx_partial
#SBATCH --output=logs/SFP_hx_partial-%A_%a.out
#SBATCH --error=logs/SFP_hx_partial-%A_%a.err
#SBATCH --array=0-216
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=habanaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/SFP_hx_partial

which python
python --version

python ./scripts/run_sfp_hx_partial.py ${SLURM_ARRAY_TASK_ID}