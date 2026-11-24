#!/bin/bash
#SBATCH -J build_and_retrain_lit_4_5_7
#SBATCH -o logs/build_and_retrain_lit_4_5_7_%j.out
#SBATCH -e logs/build_and_retrain_lit_4_5_7_%j.err
#SBATCH -p compute
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate committor_env

set -euo pipefail
mkdir -p logs

python /scratch/mma9420/q_pipeline/scripts/build_and_retrain_lit_priorities_4_5_7.py
