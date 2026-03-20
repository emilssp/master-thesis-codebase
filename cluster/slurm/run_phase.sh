#!/bin/bash
#SBATCH --job-name=bdg_phase
#SBATCH --output=logs/phase-%A_%a.out
#SBATCH --error=logs/phase-%A_%a.err
#SBATCH --array=0-51%8
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --partition=habanaq

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source ~/D1/venv/bin/activate

mkdir -p logs data data/temps

which python
python --version

python ./scripts/run_phase.py ${SLURM_ARRAY_TASK_ID}