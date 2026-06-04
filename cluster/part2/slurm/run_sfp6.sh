#!/bin/bash
#SBATCH --job-name=SFP_U6
#SBATCH --output=logs/SFP_U6-%A_%a.out
#SBATCH --error=logs/SFP_U6-%A_%a.err
#SBATCH --array=0-239
#SBATCH --cpus-per-task=4
#SBATCH --time=32:00:00
#SBATCH --partition=habanaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/SFP_U6

which python
python --version

python ./scripts/run_sfp6.py ${SLURM_ARRAY_TASK_ID}