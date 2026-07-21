"""
Heatmap of Q (fraction of native contacts) vs. radius of gyration,
using the existing MDTraj feature table and the computed Q values
(all-frames / stride=1 version).
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MDTRAJ_CSV = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
Q_CSV      = "/scratch/mma9420/committor_check/Q/Q_values_allframes.csv"
OUT_PNG    = "/scratch/mma9420/committor_check/Q/rg_vs_Q_heatmap_allframes.png"

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

print(f"MDTraj feature table: {df_feat.shape[0]:,} rows, {df_feat.shape[1]} columns")
print(f"Q value table:        {df_q.shape[0]:,} rows")

# IMPORTANT: Q_values*.csv's own "Frame" column can be a 0..N-1 sample
# counter rather than the raw trajectory frame index when stride > 1 --
# it does not necessarily line up with MDTraj's frame_index. Time_ps
# *does* already account for stride (Time_ps = raw_frame_index * 200), so
# derive the true raw frame index from that instead of trusting "Frame".
df_q["frame_index"] = (df_q["Time_ps"] / 200).round().astype(int)

merged = pd.merge(
    df_feat[["frame_index", "radius_gyration"]],
    df_q[["frame_index", "Q"]],
    on="frame_index", how="inner",
)

print(f"Merged on frame index: {len(merged):,} matching frames "
      f"(features={len(df_feat):,}, Q={len(df_q):,})")

if len(merged) == 0:
    sys.exit("ERROR: no overlapping frame indices between the two files — "
              "check that both were generated from the same trajectory.")
if len(merged) != len(df_q):
    print(f"WARNING: only {len(merged):,} of {len(df_q):,} Q rows found a matching "
          "frame_index in the feature table. Plotting the overlapping frames only.")
else:
    print("All Q rows matched a frame_index in the feature table. Good.")

rg = merged["radius_gyration"].values
q  = merged["Q"].values

plt.figure(figsize=(7, 5))
h = plt.hist2d(rg, q, bins=60, cmap="viridis")
cbar = plt.colorbar(h[3])
cbar.ax.set_ylabel("Frame count")
plt.xlabel("Radius of gyration (nm)")
plt.ylabel("Q (fraction of native contacts)")
plt.title(f"Chignolin — Q vs. Rg, all frames ({len(merged):,} frames)")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Saved: {OUT_PNG}")
