#!/bin/bash
#SBATCH -J sweep_balance_ratio_weighted
#SBATCH -o logs/sweep_balance_ratio_weighted_%j.out
#SBATCH -e logs/sweep_balance_ratio_weighted_%j.err
#SBATCH -p compute
#SBATCH -n 2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate committor_env

set -euo pipefail
mkdir -p logs

python /scratch/mma9420/q_pipeline/scripts/sweep_balance_ratio_weighted.py
