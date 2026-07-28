#!/bin/bash
#SBATCH -J pinn_committor
#SBATCH -o logs/pinn_committor_%j.out
#SBATCH -e logs/pinn_committor_%j.err
#SBATCH -p compute
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# Chignolin is small (64 features, MLP) -- CPU is expected to be fine. If
# training is too slow in practice, switch to a GPU node instead:
#   #SBATCH -p gpu
#   #SBATCH --gres=gpu:1

CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate committor_env

set -euo pipefail
mkdir -p logs

# PyTorch may not already be in committor_env -- everything else in this
# pipeline so far is sklearn/LightGBM/MDTraj, not deep learning. Install a
# CPU-only build on first run if it's missing.
python -c "import torch" 2>/dev/null || \
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

python scripts/build_pinn_dataset.py
python scripts/train_pinn_committor.py
