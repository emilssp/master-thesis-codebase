#!/bin/bash
#SBATCH --job-name=SFP
#SBATCH --output=logs/SFP-%A_%a.out
#SBATCH --error=logs/SFP-%A_%a.err
#SBATCH --array=0-231
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=milanq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/SFP

which python
python --version

python ./scripts/run_sfp.py ${SLURM_ARRAY_TASK_ID}