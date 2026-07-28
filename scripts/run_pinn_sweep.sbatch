#!/bin/bash
#SBATCH -J pinn_sweep
#SBATCH -o logs/pinn_sweep_%j.out
#SBATCH -e logs/pinn_sweep_%j.err
#SBATCH -p compute
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate committor_env

set -euo pipefail
mkdir -p logs

python -c "import torch" 2>/dev/null || \
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

# Assumes pinn_dataset.csv already exists (build_pinn_dataset.py already ran).
python scripts/sweep_pinn_hparams.py
