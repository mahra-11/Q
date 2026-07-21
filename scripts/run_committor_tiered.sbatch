#!/bin/bash
#SBATCH -J committor_tiered
#SBATCH -o logs/committor_tiered_%j.out
#SBATCH -e logs/committor_tiered_%j.err
#SBATCH -p compute
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

CONDA_BASE="/share/apps/NYUAD5/miniconda/3-4.11.0"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate committor_env

set -euo pipefail
mkdir -p logs

python - <<'PYEOF'
"""
Per-cell (Rg x Q grid) committor estimate via tiered step-sampling:
frames within each cell are ordered by frame index, and every Nth frame
is kept, where N depends on how many frames are in the cell:
  <=100      -> step 1  (use all)
  101-500    -> step 2  (every other)
  501-1000   -> step 5
  1001-4000  -> step 10
  4001-10000 -> step 20
  >10000     -> step 100
Separate output from committor_grid_visits.csv, for comparison.
"""
import sys
import numpy as np
import pandas as pd

MDTRAJ_CSV = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
FATE_CSV   = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate.csv"
OUT_CSV    = "/scratch/mma9420/committor_check/Q/committor_grid_tiered.csv"
BINS       = 60


def tiered_step(n):
    if n <= 100:
        return 1
    elif n <= 500:
        return 2
    elif n <= 1000:
        return 5
    elif n <= 4000:
        return 10
    elif n <= 10000:
        return 20
    else:
        return 100


print(f"Loading MDTraj features: {MDTRAJ_CSV}")
df_feat = pd.read_csv(MDTRAJ_CSV)
print(f"Loading fate labels:     {FATE_CSV}")
df_fate = pd.read_csv(FATE_CSV)

for col in ["frame_index", "radius_gyration"]:
    if col not in df_feat.columns:
        sys.exit(f"ERROR: '{col}' not found in {MDTRAJ_CSV}. Columns: {list(df_feat.columns)}")
for col in ["Time_ps", "Q", "Fate"]:
    if col not in df_fate.columns:
        sys.exit(f"ERROR: '{col}' not found in {FATE_CSV}. Columns: {list(df_fate.columns)}")

df_fate["frame_index"] = (df_fate["Time_ps"] / 200).round().astype(int)

merged = pd.merge(
    df_feat[["frame_index", "radius_gyration"]],
    df_fate[["frame_index", "Q", "Fate"]],
    on="frame_index", how="inner",
).sort_values("frame_index").reset_index(drop=True)

print(f"Merged: {len(merged):,} frames")
merged = merged.dropna(subset=["Fate"]).reset_index(drop=True)
print(f"Frames with a determined Fate (0/1): {len(merged):,}")

# --- Assign each frame to an (Rg, Q) grid cell (same BINS as the heatmap) ---
merged["rg_bin"] = pd.cut(merged["radius_gyration"], bins=BINS, labels=False)
merged["q_bin"]  = pd.cut(merged["Q"], bins=BINS, labels=False)
merged["cell_id"] = merged["rg_bin"].astype(str) + "_" + merged["q_bin"].astype(str)

n_frames_by_cell = merged.groupby("cell_id").size().rename("n_frames")

def sample_cell(group):
    n = len(group)
    step = tiered_step(n)
    sampled = group.iloc[::step]
    return pd.Series({
        "step_used": step,
        "n_sampled_tiered": len(sampled),
        "p_folded_tiered": sampled["Fate"].mean(),
        "std_tiered": sampled["Fate"].std(ddof=1) if len(sampled) > 1 else np.nan,
    })

tiered_by_cell = (
    merged.sort_values("frame_index")
    .groupby("cell_id", group_keys=False)
    .apply(sample_cell)
)

result = (
    pd.concat([n_frames_by_cell, tiered_by_cell], axis=1)
    .reset_index()
    .rename(columns={"index": "cell_id"})
    .sort_values("n_frames", ascending=False)
)
result.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV} ({len(result):,} cells)")

print(f"\nTotal sampled frames across all cells: {result['n_sampled_tiered'].sum():,} "
      f"(vs. {len(merged):,} total frames)")
print(f"Mean sampled frames per cell: {result['n_sampled_tiered'].mean():.1f}")
print(f"Median sampled frames per cell: {result['n_sampled_tiered'].median():.0f}")

print("\nTop 15 largest cells (by raw frame count):")
cols = ["cell_id", "n_frames", "step_used", "n_sampled_tiered", "p_folded_tiered", "std_tiered"]
print(result[cols].head(15).to_string(index=False))
PYEOF
