"""
Scatter/bubble plot of the visit-based and tiered committor estimates over
Rg x Q: one dot per occupied cell (not a filled grid), colored by committor
value (blue=unfolded, red=folded), sized by sample count. Avoids drawing
the mostly-empty grid a heatmap forces.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MDTRAJ_CSV = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
FATE_CSV   = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate.csv"
VISITS_CSV = "/scratch/mma9420/committor_check/Q/committor_grid_visits.csv"
TIERED_CSV = "/scratch/mma9420/committor_check/Q/committor_grid_tiered.csv"
OUT_PNG    = "/scratch/mma9420/committor_check/Q/committor_scatter.png"
BINS       = 60

df_feat = pd.read_csv(MDTRAJ_CSV)
df_fate = pd.read_csv(FATE_CSV)
df_visits = pd.read_csv(VISITS_CSV)
df_tiered = pd.read_csv(TIERED_CSV)

df_fate["frame_index"] = (df_fate["Time_ps"] / 200).round().astype(int)
merged = pd.merge(
    df_feat[["frame_index", "radius_gyration"]],
    df_fate[["frame_index", "Q", "Fate"]],
    on="frame_index", how="inner",
).dropna(subset=["Fate"])

# Same bin edges used to build the grid CSVs -- recomputed deterministically
_, rg_edges = pd.cut(merged["radius_gyration"], bins=BINS, retbins=True)
_, q_edges  = pd.cut(merged["Q"], bins=BINS, retbins=True)
rg_centers = (rg_edges[:-1] + rg_edges[1:]) / 2
q_centers  = (q_edges[:-1] + q_edges[1:]) / 2

def cell_coords(df):
    rg_idx = df["cell_id"].str.split("_").str[0].astype(int).values
    q_idx  = df["cell_id"].str.split("_").str[1].astype(int).values
    return rg_centers[rg_idx], q_centers[q_idx]

def bubble_sizes(n):
    # log-scaled so a 40,000-sample cell doesn't blot out everything else
    n = n.astype(float)
    return 15 + 200 * (np.log10(n + 1) / np.log10(n.max() + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rg_v, q_v = cell_coords(df_visits)
sc0 = axes[0].scatter(rg_v, q_v, c=df_visits["p_folded_visits"], cmap="bwr", vmin=0, vmax=1,
                       s=bubble_sizes(df_visits["n_visits"].values),
                       edgecolors="black", linewidths=0.3, alpha=0.85)
fig.colorbar(sc0, ax=axes[0]).ax.set_ylabel("P(folded first)")
axes[0].set_xlabel("Radius of gyration (nm)")
axes[0].set_ylabel("Q")
axes[0].set_title("Visit-based (dot size = n_visits)")

rg_t, q_t = cell_coords(df_tiered)
sc1 = axes[1].scatter(rg_t, q_t, c=df_tiered["p_folded_tiered"], cmap="bwr", vmin=0, vmax=1,
                       s=bubble_sizes(df_tiered["n_sampled_tiered"].values),
                       edgecolors="black", linewidths=0.3, alpha=0.85)
fig.colorbar(sc1, ax=axes[1]).ax.set_ylabel("P(folded first)")
axes[1].set_xlabel("Radius of gyration (nm)")
axes[1].set_ylabel("Q")
axes[1].set_title("Tiered (dot size = n_sampled_tiered)")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Saved: {OUT_PNG}")
