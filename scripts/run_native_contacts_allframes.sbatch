#!/bin/bash
#SBATCH -J native_contacts_q_allframes
#SBATCH -o logs/native_contacts_allframes_%j.out
#SBATCH -e logs/native_contacts_allframes_%j.err
#SBATCH -p compute
#SBATCH -n 2
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Activate conda environment
CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate committor_env

set -euo pipefail

# ── configurable ──────────────────────────────────────────────────────────
SYSTEM="chignolin"       # or "ww_domain"
STRIDE_OVERRIDE=1        # 1 = every single frame, overrides config's stride
# ─────────────────────────────────────────────────────────────────────────

mkdir -p logs

python - <<PYEOF
import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.native_contacts import compute_q

with open(f"config/${SYSTEM}.yaml") as f:
    cfg = yaml.safe_load(f)

SYSTEM_NAME = cfg["system"]
PDB    = cfg["topology_pdb"]
FILES  = cfg["trajectory_xtc"]
STRIDE = ${STRIDE_OVERRIDE}
NATIVE = cfg.get("native_pdb") or PDB

print(f"System: {SYSTEM_NAME}")
print(f"PDB:    {PDB}")
print(f"Native: {NATIVE}")
print(f"Stride: {STRIDE} (override -- config stride is {cfg['stride']})")

q, n_contacts = compute_q(FILES, PDB, native_pdb=NATIVE, stride=STRIDE)
print(f"Native contacts: {n_contacts}")
print(f"Frames:          {len(q)}")
print(f"Q range:         [{q.min():.3f}, {q.max():.3f}]")

Q_FOLDED   = cfg.get("q_folded_threshold", 0.8)
Q_UNFOLDED = cfg.get("q_unfolded_threshold", 0.2)
folded    = (q >= Q_FOLDED).sum()
unfolded  = (q <= Q_UNFOLDED).sum()
print(f"Folded   (Q >= {Q_FOLDED}): {folded:,}  ({folded/len(q):.1%})")
print(f"Unfolded (Q <= {Q_UNFOLDED}): {unfolded:,}  ({unfolded/len(q):.1%})")

out_dir = cfg.get("q_output_dir") or f"data/{SYSTEM_NAME}"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/Q_values_allframes.csv"

# Frame = raw trajectory frame index (== sample index here since stride=1),
# matching the "frame_index" convention used in the MDTraj feature table.
pd.DataFrame({
    "Frame": np.arange(len(q)) * STRIDE,
    "Time_ps": np.arange(len(q)) * 200 * STRIDE,
    "Q": q,
}).to_csv(out_path, index=False)
print(f"Saved: {out_path}")
PYEOF
