#!/bin/bash
#SBATCH -J retrain_unpruned_physics_features
#SBATCH -o logs/retrain_unpruned_physics_features_%j.out
#SBATCH -e logs/retrain_unpruned_physics_features_%j.err
#SBATCH -p compute
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate committor_env

set -euo pipefail
mkdir -p logs

python /scratch/mma9420/q_pipeline/scripts/retrain_unpruned_physics_features.py
