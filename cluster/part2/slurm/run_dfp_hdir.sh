#!/bin/bash
#SBATCH --job-name=hdir
#SBATCH --output=logs/DFP_hdir-%A_%a.out
#SBATCH --error=logs/DFP_hdir-%A_%a.err
#SBATCH --array=0-2629
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=fpgaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/DFP_hdir

which python
python --version

python ./scripts/run_dfp_hdir.py ${SLURM_ARRAY_TASK_ID}