#!/bin/bash
#SBATCH --job-name=NP
#SBATCH --output=logs/NP-%A_%a.out
#SBATCH --error=logs/NP-%A_%a.err
#SBATCH --array=0-262
#SBATCH --cpus-per-task=4
#SBATCH --time=32:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/NP_part
which python
python --version

python ./scripts/run_np.py ${SLURM_ARRAY_TASK_ID}