#!/bin/bash
#SBATCH -J build_and_retrain_lit_asp2n_gly6o
#SBATCH -o logs/build_and_retrain_lit_asp2n_gly6o_%j.out
#SBATCH -e logs/build_and_retrain_lit_asp2n_gly6o_%j.err
#SBATCH -p compute
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate committor_env

set -euo pipefail
mkdir -p logs

python /scratch/mma9420/q_pipeline/scripts/build_and_retrain_lit_asp2n_gly6o.py
