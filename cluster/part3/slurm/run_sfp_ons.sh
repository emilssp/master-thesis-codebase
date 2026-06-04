#!/bin/bash
#SBATCH --job-name=SFP_ons
#SBATCH --output=logs/SFP_ons-%A_%a.out
#SBATCH --error=logs/SFP_ons-%A_%a.err
#SBATCH --array=0-239
#SBATCH --cpus-per-task=4
#SBATCH --time=32:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/SFP_ons

which python
python --version

python ./scripts/run_sfp_ons.py ${SLURM_ARRAY_TASK_ID}