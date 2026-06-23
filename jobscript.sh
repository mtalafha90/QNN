#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=qnn_sft
#SBATCH --output=qnn_sft_%j.out
#SBATCH --error=qnn_sft_%j.err
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=1

module load gnu8
module unload openmpi3/3.1.4
module load anaconda
# Activate conda correctly
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda activate qnn_sft

# Force cluster to use conda C++ runtime first
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/pinns-sft/lib:$LD_LIBRARY_PATH

python main.py
