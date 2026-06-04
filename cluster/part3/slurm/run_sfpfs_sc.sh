#!/bin/bash
#SBATCH --job-name=SFPFS_sc
#SBATCH --output=logs/SFPFS_sc-%A_%a.out
#SBATCH --error=logs/SFPFS_sc-%A_%a.err
#SBATCH --array=0-249
#SBATCH --cpus-per-task=4
#SBATCH --time=32:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/SFPFS_sc

which python
python --version

python ./scripts/run_sfpfs_sc.py ${SLURM_ARRAY_TASK_ID}