"""
Log-color-scale heatmap of Q vs. radius of gyration (all-frames data).
The frame counts span ~4 orders of magnitude, so a linear color scale
washes out everything except the single most-populated cell.
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

MDTRAJ_CSV = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
Q_CSV      = "/scratch/mma9420/committor_check/Q/Q_values_allframes.csv"
OUT_PNG    = "/scratch/mma9420/committor_check/Q/rg_vs_Q_heatmap_allframes_logscale.png"

print(f"Loading MDTraj features: {MDTRAJ_CSV}")
df_feat = pd.read_csv(MDTRAJ_CSV)

print(f"Loading Q values:        {Q_CSV}")
df_q = pd.read_csv(Q_CSV)

if "frame_index" not in df_feat.columns:
    sys.exit(f"ERROR: 'frame_index' not found in {MDTRAJ_CSV}. Columns: {list(df_feat.columns)}")
if "radius_gyration" not in df_feat.columns:
    sys.exit(f"ERROR: 'radius_gyration' not found in {MDTRAJ_CSV}. Columns: {list(df_feat.columns)}")
if "Time_ps" not in df_q.columns or "Q" not in df_q.columns:
    sys.exit(f"ERROR: expected 'Time_ps' and 'Q' columns in {Q_CSV}. Columns: {list(df_q.columns)}")

df_q["frame_index"] = (df_q["Time_ps"] / 200).round().astype(int)

merged = pd.merge(
    df_feat[["frame_index", "radius_gyration"]],
    df_q[["frame_index", "Q"]],
    on="frame_index", how="inner",
)

print(f"Merged on frame index: {len(merged):,} matching frames "
      f"(features={len(df_feat):,}, Q={len(df_q):,})")

if len(merged) == 0:
    sys.exit("ERROR: no overlapping frame indices -- check that both files come from the same trajectory.")

rg = merged["radius_gyration"].values
q  = merged["Q"].values

plt.figure(figsize=(7, 5))
# LogNorm: bins with 0 frames are automatically masked -- only cells with
# >=1 frame get colored.
h = plt.hist2d(rg, q, bins=60, cmap="viridis", norm=LogNorm())
cbar = plt.colorbar(h[3])
cbar.ax.set_ylabel("Frame count (log scale)")
plt.xlabel("Radius of gyration (nm)")
plt.ylabel("Q (fraction of native contacts)")
plt.title(f"Chignolin — Q vs. Rg, all frames, log color scale ({len(merged):,} frames)")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Saved: {OUT_PNG}")
