"""
Full frame-count grid (matching the Q vs. radius-of-gyration heatmap bins)
plus a summary report.
"""
import sys
import numpy as np
import pandas as pd

MDTRAJ_CSV = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
Q_CSV      = "/scratch/mma9420/committor_check/Q/Q_values_allframes.csv"
GRID_CSV   = "/scratch/mma9420/committor_check/Q/bin_counts_grid_allframes.csv"
SUMMARY_TXT = "/scratch/mma9420/committor_check/Q/bin_counts_summary_allframes.txt"
BINS       = 60   # same bin count as the heatmap

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

print(f"Merged: {len(merged):,} frames (features={len(df_feat):,}, Q={len(df_q):,})")
if len(merged) == 0:
    sys.exit("ERROR: no overlapping frame indices -- check both files come from the same trajectory.")

rg = merged["radius_gyration"].values
q  = merged["Q"].values

# counts[i, j] = frame count with radius_gyration in rg bin i and Q in bin j
counts, rg_edges, q_edges = np.histogram2d(rg, q, bins=BINS)

# --- Full grid, including empty (zero) bins ---
rg_labels = [f"rg_{rg_edges[i]:.4f}_{rg_edges[i+1]:.4f}" for i in range(BINS)]
q_labels  = [f"q_{q_edges[j]:.4f}_{q_edges[j+1]:.4f}"   for j in range(BINS)]

df_grid = pd.DataFrame(counts.astype(int), index=rg_labels, columns=q_labels)
df_grid.index.name = "radius_gyration_bin"
df_grid.to_csv(GRID_CSV)
print(f"\nSaved full grid ({BINS} x {BINS}, including empty bins): {GRID_CSV}")

# --- Summary stats ---
total_bins = counts.size
occupied = counts[counts > 0]

top10 = (
    pd.DataFrame({
        "rg_bin_low": np.repeat(rg_edges[:-1], BINS),
        "rg_bin_high": np.repeat(rg_edges[1:], BINS),
        "q_bin_low": np.tile(q_edges[:-1], BINS),
        "q_bin_high": np.tile(q_edges[1:], BINS),
        "frame_count": counts.flatten().astype(int),
    })
    .sort_values("frame_count", ascending=False)
    .head(10)
)

lines = []
lines.append(f"Grid: {BINS} x {BINS} = {total_bins} bins")
lines.append(f"Total frames counted: {counts.sum():,.0f} (should equal {len(merged):,})")
lines.append(f"Occupied bins: {len(occupied):,} / {total_bins:,} ({len(occupied)/total_bins:.1%})")
lines.append(f"Empty bins:    {total_bins - len(occupied):,}")
lines.append("")
lines.append("All bins (including empty):")
lines.append(f"  min={counts.min():.0f}  max={counts.max():.0f}  mean={counts.mean():.2f}")
lines.append("")
lines.append("Occupied bins only:")
lines.append(f"  min={occupied.min():.0f}  max={occupied.max():.0f}  mean={occupied.mean():.2f}  median={np.median(occupied):.0f}")
lines.append("")
lines.append(f"radius_gyration range: [{rg.min():.4f}, {rg.max():.4f}] nm")
lines.append(f"Q range:               [{q.min():.4f}, {q.max():.4f}]")
lines.append("")
lines.append("Top 10 most populated bins:")
lines.append(top10.to_string(index=False))

summary_text = "\n".join(lines)
print("\n" + summary_text)

with open(SUMMARY_TXT, "w") as f:
    f.write(summary_text + "\n")
print(f"\nSaved summary report: {SUMMARY_TXT}")
