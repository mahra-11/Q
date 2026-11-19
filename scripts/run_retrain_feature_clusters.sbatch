#!/bin/bash
#SBATCH -J retrain_feature_clusters
#SBATCH -o logs/retrain_feature_clusters_%j.out
#SBATCH -e logs/retrain_feature_clusters_%j.err
#SBATCH -p compute
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate committor_env

set -euo pipefail
mkdir -p logs

python /scratch/mma9420/q_pipeline/scripts/retrain_feature_clusters.py
