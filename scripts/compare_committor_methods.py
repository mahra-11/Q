"""
Merge the visit-based and tiered committor results (2D Rg x Q grid) and
compare them.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VISITS_CSV = "/scratch/mma9420/committor_check/Q/committor_grid_visits.csv"
TIERED_CSV = "/scratch/mma9420/committor_check/Q/committor_grid_tiered.csv"
OUT_CSV    = "/scratch/mma9420/committor_check/Q/committor_methods_comparison.csv"
OUT_PNG    = "/scratch/mma9420/committor_check/Q/committor_methods_comparison.png"

df_v = pd.read_csv(VISITS_CSV)
df_t = pd.read_csv(TIERED_CSV)

merged = pd.merge(df_v, df_t, on="cell_id", suffixes=("", "_t"))
assert (merged["n_frames"] == merged["n_frames_t"]).all(), "n_frames mismatch between the two files!"
merged = merged.drop(columns=["n_frames_t"])

merged["diff"] = merged["p_folded_visits"] - merged["p_folded_tiered"]
merged = merged.sort_values("n_frames", ascending=False)
merged.to_csv(OUT_CSV, index=False)

print(f"Saved: {OUT_CSV} ({len(merged):,} cells)")
print(f"\nOverall agreement:")
print(f"  Mean abs difference: {merged['diff'].abs().mean():.4f}")
print(f"  Median abs difference: {merged['diff'].abs().median():.4f}")
print(f"  Max abs difference:  {merged['diff'].abs().max():.4f}")
print(f"  Correlation (Pearson r): {merged['p_folded_visits'].corr(merged['p_folded_tiered']):.4f}")

print("\nBiggest disagreements (top 10 by |diff|):")
cols = ["cell_id", "n_frames", "n_visits", "p_folded_visits",
        "n_sampled_tiered", "p_folded_tiered", "diff"]
print(merged.reindex(merged["diff"].abs().sort_values(ascending=False).index)[cols].head(10).to_string(index=False))

print("\nLargest cells (top 10 by n_frames) -- where the methods differ most philosophically:")
print(merged[cols].head(10).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sc = axes[0].scatter(merged["p_folded_visits"], merged["p_folded_tiered"],
                      c=np.log10(merged["n_frames"]), cmap="viridis", s=15, alpha=0.7)
axes[0].plot([0, 1], [0, 1], "r--", linewidth=1, label="perfect agreement")
axes[0].set_xlabel("P(folded) -- visit-based")
axes[0].set_ylabel("P(folded) -- tiered")
axes[0].set_title("Per-cell agreement between methods")
axes[0].legend()
cbar = fig.colorbar(sc, ax=axes[0])
cbar.ax.set_ylabel("log10(n_frames)")

axes[1].hist(merged["diff"], bins=50, color="steelblue")
axes[1].set_xlabel("p_folded_visits - p_folded_tiered")
axes[1].set_ylabel("Number of cells")
axes[1].set_title("Distribution of per-cell differences")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"\nSaved: {OUT_PNG}")
