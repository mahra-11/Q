"""
Plot the empirical committor-vs-Q curve (from committor_vs_Q_binned.csv,
produced by compute_frame_fate.py). A good reaction coordinate should show
a sigmoid rising from ~0 to ~1, crossing 0.5 near the true transition state.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVE_CSV = "/scratch/mma9420/committor_check/Q/committor_vs_Q_binned.csv"
OUT_PNG   = "/scratch/mma9420/committor_check/Q/committor_vs_Q_curve.png"

print(f"Loading: {CURVE_CSV}")
df = pd.read_csv(CURVE_CSV)

for col in ["q_bin_center", "n_frames", "p_folded"]:
    if col not in df.columns:
        sys.exit(f"ERROR: '{col}' not found. Columns: {list(df.columns)}")

df = df.dropna(subset=["p_folded"]).sort_values("q_bin_center")

fig, ax1 = plt.subplots(figsize=(8, 5))

sizes = 20 + 200 * (df["n_frames"] / df["n_frames"].max())
ax1.scatter(df["q_bin_center"], df["p_folded"], s=sizes, c="crimson",
            edgecolors="black", linewidths=0.5, zorder=3, label="Binned committor estimate")
ax1.plot(df["q_bin_center"], df["p_folded"], color="crimson", alpha=0.4, zorder=2)

ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, zorder=1)
ax1.set_xlabel("Q (fraction of native contacts)")
ax1.set_ylabel("P(folded first)  --  empirical committor")
ax1.set_ylim(-0.05, 1.05)
ax1.set_title("Empirical committor vs. Q (Chignolin)")

crossing = None
p = df["p_folded"].values
q = df["q_bin_center"].values
for i in range(len(p) - 1):
    if (p[i] - 0.5) * (p[i + 1] - 0.5) < 0:
        frac = (0.5 - p[i]) / (p[i + 1] - p[i])
        crossing = q[i] + frac * (q[i + 1] - q[i])
        break

if crossing is not None:
    ax1.axvline(crossing, color="black", linestyle=":", linewidth=1)
    ax1.annotate(f"Q* ~ {crossing:.3f}", xy=(crossing, 0.5),
                 xytext=(crossing + 0.03, 0.6),
                 arrowprops=dict(arrowstyle="->"))
    print(f"\nEstimated transition-state Q (where curve crosses 0.5): {crossing:.4f}")
else:
    print("\nNo 0.5 crossing found in this curve (it may not span both folded and unfolded).")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Saved: {OUT_PNG}")
