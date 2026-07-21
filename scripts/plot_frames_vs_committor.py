"""
Histogram: number of frames vs. their assigned committor value (each frame
gets the p_folded_visits estimate of the Q-bin it falls into, from the
visit-based 1D committor curve).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FATE_CSV   = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate.csv"
CURVE_CSV  = "/scratch/mma9420/committor_check/Q/committor_1d_visits_sem.csv"
OUT_PNG    = "/scratch/mma9420/committor_check/Q/frames_vs_committor.png"
N_BINS     = 40   # must match the N_BINS used to build committor_1d_visits_sem.csv

df_fate = pd.read_csv(FATE_CSV)
df_curve = pd.read_csv(CURVE_CSV)

# Recompute the exact same Q bins used to build the committor curve
df_fate["q_bin"] = pd.cut(df_fate["Q"], bins=N_BINS, labels=False)

# Map each frame's q_bin to that bin's committor estimate
bin_to_committor = df_curve.set_index("q_bin")["p_folded_visits"]
df_fate["committor_value"] = df_fate["q_bin"].map(bin_to_committor)

valid = df_fate.dropna(subset=["committor_value"])
print(f"Frames with an assigned committor value: {len(valid):,} / {len(df_fate):,}")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(valid["committor_value"], bins=40, range=(0, 1), color="steelblue", edgecolor="black", linewidth=0.3)
ax.set_yscale("log")
ax.set_xlabel("Committor value (P(folded first))")
ax.set_ylabel("Number of frames (log scale)")
ax.set_title("Frame count by committor value (visit-based, folded=0.8/unfolded=0.2)")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Saved: {OUT_PNG}")

print("\nFrame counts by committor range:")
for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)]:
    n = ((valid["committor_value"] >= lo) & (valid["committor_value"] < hi)).sum()
    print(f"  [{lo:.1f}, {hi:.1f}): {n:,} frames ({n/len(valid):.1%})")
