#!/bin/bash
#SBATCH -J committor_visits
#SBATCH -o logs/committor_visits_%j.out
#SBATCH -e logs/committor_visits_%j.err
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
Per-cell (Rg x Q grid) committor estimate via visit-based sampling:
group consecutive frames that stay in the same cell into one "visit"
(same logic as src/committor.py's existing count-once approach, applied
to the 2D grid instead of clusters), take the middle frame of each visit
as its representative, average Fate (0/1) across visits per cell.

Sample size adapts automatically -- a cell visited once for 40,000 frames
= 1 sample; a cell visited 400 separate times = 400 samples.
"""
import sys
import numpy as np
import pandas as pd

MDTRAJ_CSV = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
FATE_CSV   = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate.csv"
OUT_CSV    = "/scratch/mma9420/committor_check/Q/committor_grid_visits.csv"
BINS       = 60

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

# --- Segment into visits: consecutive frames sharing the same cell ---
same_as_prev = merged["cell_id"] == merged["cell_id"].shift()
merged["visit_id"] = (~same_as_prev).cumsum()

visit_reps = merged.groupby("visit_id").agg(
    cell_id=("cell_id", "first"),
    Fate=("Fate", lambda s: s.iloc[len(s) // 2]),  # middle frame of the visit
)

n_frames_by_cell = merged.groupby("cell_id").size().rename("n_frames")

visits_by_cell = visit_reps.groupby("cell_id").agg(
    n_visits=("Fate", "size"),
    p_folded_visits=("Fate", "mean"),
    std_visits=("Fate", lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
)

result = (
    pd.concat([n_frames_by_cell, visits_by_cell], axis=1)
    .reset_index()
    .rename(columns={"index": "cell_id"})
    .sort_values("n_frames", ascending=False)
)
result.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV} ({len(result):,} cells)")

print(f"\nTotal visits across all cells: {result['n_visits'].sum():,} "
      f"(vs. {len(merged):,} total frames)")
print(f"Mean visits per cell: {result['n_visits'].mean():.1f}")
print(f"Median visits per cell: {result['n_visits'].median():.0f}")

print("\nTop 15 largest cells (by raw frame count):")
cols = ["cell_id", "n_frames", "n_visits", "p_folded_visits", "std_visits"]
print(result[cols].head(15).to_string(index=False))
PYEOF
