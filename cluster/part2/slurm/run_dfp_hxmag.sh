#!/bin/bash
#SBATCH --job-name=hxmag
#SBATCH --output=logs/DFP_hxmag-%A_%a.out
#SBATCH --error=logs/DFP_hxmag-%A_%a.err
#SBATCH --array=0-2629
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=habanaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/DFP_hxmag

which python
python --version

python ./scripts/run_dfp_hxmag.py ${SLURM_ARRAY_TASK_ID}