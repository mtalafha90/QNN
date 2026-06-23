#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=qnn_sft
#SBATCH --output=qnn_sft_%j.out
#SBATCH --error=qnn_sft_%j.err
#SBATCH --mail-type=ALL

# Use all the CPU cores we asked for (the old value of 1 throttled the run to a
# single thread). Fall back to 8 if SLURM_CPUS_PER_TASK is unset.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# These module names are not present on every cluster; don't let a missing
# module abort the job (conda is activated directly from $HOME below).
module load gnu8 2>/dev/null || true
module unload openmpi3/3.1.4 2>/dev/null || true
module load anaconda 2>/dev/null || true

# Activate conda correctly
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda activate qnn_sft

# Force cluster to use the active conda env's C++ runtime first.
# (CONDA_PREFIX points at the env we just activated; the old hard-coded
# "pinns-sft" path referred to a different environment.)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python main.py
