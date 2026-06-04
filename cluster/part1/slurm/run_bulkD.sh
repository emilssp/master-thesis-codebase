#!/bin/bash
#SBATCH --job-name=bulkD
#SBATCH --output=logs/bulkD-%A_%a.out
#SBATCH --error=logs/bulkD-%A_%a.err
#SBATCH --array=0-238
#SBATCH --cpus-per-task=4
#SBATCH --time=32:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/bulkD

which python
python --version

python ./scripts/run_bulkD.py ${SLURM_ARRAY_TASK_ID}