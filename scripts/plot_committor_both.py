"""
Side-by-side heatmaps of the visit-based and tiered committor estimates
over the Rg x Q grid, plus their difference. Same bwr / 0-1 color scale
used for both, so they're directly comparable at a glance.

Note: for a more reliable, readable 2D view, prefer plot_committor_scatter.py
(bubble plot) -- the filled-grid heatmap here can look mostly empty/uninformative
since only ~1/3 of the grid cells are typically occupied.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MDTRAJ_CSV = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
FATE_CSV   = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate.csv"
VISITS_CSV = "/scratch/mma9420/committor_check/Q/committor_grid_visits.csv"
TIERED_CSV = "/scratch/mma9420/committor_check/Q/committor_grid_tiered.csv"
OUT_PNG    = "/scratch/mma9420/committor_check/Q/committor_both_heatmaps.png"
BINS       = 60

print(f"Loading MDTraj features: {MDTRAJ_CSV}")
df_feat = pd.read_csv(MDTRAJ_CSV)
print(f"Loading fate labels:     {FATE_CSV}")
df_fate = pd.read_csv(FATE_CSV)
print(f"Loading visits results:  {VISITS_CSV}")
df_visits = pd.read_csv(VISITS_CSV)
print(f"Loading tiered results:  {TIERED_CSV}")
df_tiered = pd.read_csv(TIERED_CSV)

df_fate["frame_index"] = (df_fate["Time_ps"] / 200).round().astype(int)
merged = pd.merge(
    df_feat[["frame_index", "radius_gyration"]],
    df_fate[["frame_index", "Q", "Fate"]],
    on="frame_index", how="inner",
).dropna(subset=["Fate"])

_, rg_edges = pd.cut(merged["radius_gyration"], bins=BINS, retbins=True)
_, q_edges  = pd.cut(merged["Q"], bins=BINS, retbins=True)

def build_grid(df, value_col):
    grid = np.full((BINS, BINS), np.nan)
    rg_idx = df["cell_id"].str.split("_").str[0].astype(int)
    q_idx  = df["cell_id"].str.split("_").str[1].astype(int)
    for i, j, v in zip(rg_idx, q_idx, df[value_col]):
        grid[i, j] = v
    return grid

p_grid_visits = build_grid(df_visits, "p_folded_visits")
p_grid_tiered = build_grid(df_tiered, "p_folded_tiered")
diff_grid = p_grid_visits - p_grid_tiered

fig, axes = plt.subplots(1, 3, figsize=(19, 5))

im0 = axes[0].pcolormesh(rg_edges, q_edges, p_grid_visits.T, cmap="bwr", vmin=0, vmax=1, shading="flat")
fig.colorbar(im0, ax=axes[0]).ax.set_ylabel("P(folded first)")
axes[0].set_xlabel("Radius of gyration (nm)")
axes[0].set_ylabel("Q")
axes[0].set_title("Visit-based")

im1 = axes[1].pcolormesh(rg_edges, q_edges, p_grid_tiered.T, cmap="bwr", vmin=0, vmax=1, shading="flat")
fig.colorbar(im1, ax=axes[1]).ax.set_ylabel("P(folded first)")
axes[1].set_xlabel("Radius of gyration (nm)")
axes[1].set_ylabel("Q")
axes[1].set_title("Tiered step-sampling")

im2 = axes[2].pcolormesh(rg_edges, q_edges, diff_grid.T, cmap="PuOr", vmin=-0.2, vmax=0.2, shading="flat")
fig.colorbar(im2, ax=axes[2]).ax.set_ylabel("visits - tiered")
axes[2].set_xlabel("Radius of gyration (nm)")
axes[2].set_ylabel("Q")
axes[2].set_title("Difference")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"\nSaved: {OUT_PNG}")

valid_diff = diff_grid[~np.isnan(diff_grid)]
print(f"Mean abs difference across all cells: {np.abs(valid_diff).mean():.4f}")
print(f"Max abs difference: {np.abs(valid_diff).max():.4f}")
