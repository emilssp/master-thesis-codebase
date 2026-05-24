#!/bin/bash
#SBATCH --job-name=SFP_hx
#SBATCH --output=logs/SFP_hx-%A_%a.out
#SBATCH --error=logs/SFP_hx-%A_%a.err
#SBATCH --array=0-216
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/SFP_hx

which python
python --version

python ./scripts/run_sfp_hx.py ${SLURM_ARRAY_TASK_ID}