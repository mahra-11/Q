"""
Heatmap of the visit-based committor estimate over the Rg x Q grid.
Colors: blue ~ 0 (unfolded), red ~ 1 (folded), white ~ 0.5 (transition
state) -- matches src/contours.py's existing committor color convention.
A second panel shows n_visits per cell, as a rough confidence indicator --
cells with very few visits should be trusted less.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

MDTRAJ_CSV   = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
FATE_CSV     = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate.csv"
VISITS_CSV   = "/scratch/mma9420/committor_check/Q/committor_grid_visits.csv"
OUT_PNG      = "/scratch/mma9420/committor_check/Q/committor_visits_heatmap.png"
BINS         = 60

print(f"Loading MDTraj features: {MDTRAJ_CSV}")
df_feat = pd.read_csv(MDTRAJ_CSV)
print(f"Loading fate labels:     {FATE_CSV}")
df_fate = pd.read_csv(FATE_CSV)
print(f"Loading visit results:   {VISITS_CSV}")
df_visits = pd.read_csv(VISITS_CSV)

df_fate["frame_index"] = (df_fate["Time_ps"] / 200).round().astype(int)
merged = pd.merge(
    df_feat[["frame_index", "radius_gyration"]],
    df_fate[["frame_index", "Q", "Fate"]],
    on="frame_index", how="inner",
).dropna(subset=["Fate"])

_, rg_edges = pd.cut(merged["radius_gyration"], bins=BINS, retbins=True)
_, q_edges  = pd.cut(merged["Q"], bins=BINS, retbins=True)

rg_idx = df_visits["cell_id"].str.split("_").str[0].astype(int)
q_idx  = df_visits["cell_id"].str.split("_").str[1].astype(int)

p_grid   = np.full((BINS, BINS), np.nan)
n_grid   = np.full((BINS, BINS), np.nan)
for i, j, p, n in zip(rg_idx, q_idx, df_visits["p_folded_visits"], df_visits["n_visits"]):
    p_grid[i, j] = p
    n_grid[i, j] = n

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

im0 = axes[0].pcolormesh(rg_edges, q_edges, p_grid.T, cmap="bwr", vmin=0, vmax=1, shading="flat")
cbar0 = fig.colorbar(im0, ax=axes[0])
cbar0.ax.set_ylabel("P(folded first) -- visit-based")
axes[0].set_xlabel("Radius of gyration (nm)")
axes[0].set_ylabel("Q (fraction of native contacts)")
axes[0].set_title("Committor estimate (visit-based)")

im1 = axes[1].pcolormesh(rg_edges, q_edges, n_grid.T, cmap="viridis", norm=LogNorm(), shading="flat")
cbar1 = fig.colorbar(im1, ax=axes[1])
cbar1.ax.set_ylabel("n_visits (log scale)")
axes[1].set_xlabel("Radius of gyration (nm)")
axes[1].set_ylabel("Q (fraction of native contacts)")
axes[1].set_title("Sample size per cell (n_visits)")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"\nSaved: {OUT_PNG}")

n_low_confidence = (df_visits["n_visits"] <= 2).sum()
print(f"Cells with n_visits <= 2 (low confidence): {n_low_confidence:,} / {len(df_visits):,}")
