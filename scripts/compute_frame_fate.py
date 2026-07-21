"""
Per-frame "fate" label (0=unfolded, 1=folded) via single-trajectory forward
scan, plus a binned empirical committor-vs-Q curve.

IMPORTANT: the per-frame Fate column is a single observed outcome from your
one recorded trajectory -- NOT a true committor probability. A frame's
committor is only meaningfully estimated by averaging Fate over many frames
with similar Q (see the binned curve below).
"""
import sys
import yaml
import numpy as np
import pandas as pd

Q_CSV        = "/scratch/mma9420/committor_check/Q/Q_values_allframes.csv"
CONFIG_YAML  = "/scratch/mma9420/q_pipeline/config/chignolin.yaml"
OUT_FATE_CSV = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate.csv"
OUT_CURVE_CSV = "/scratch/mma9420/committor_check/Q/committor_vs_Q_binned.csv"
N_BINS       = 40   # number of Q bins for the empirical committor curve

print(f"Loading Q values: {Q_CSV}")
df = pd.read_csv(Q_CSV)
if "Frame" not in df.columns or "Q" not in df.columns:
    sys.exit(f"ERROR: expected 'Frame' and 'Q' columns. Columns: {list(df.columns)}")

df = df.sort_values("Frame").reset_index(drop=True)
if not (df["Frame"].values == np.arange(len(df))).all():
    print("WARNING: Frame column is not a simple 0..N-1 sequence -- "
          "make sure this is the all-frames (stride=1) Q file, not the stride-10 one.")

with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)
Q_FOLDED   = cfg.get("q_folded_threshold", 0.8)
Q_UNFOLDED = cfg.get("q_unfolded_threshold", 0.2)
print(f"Folded threshold:   Q >= {Q_FOLDED}")
print(f"Unfolded threshold: Q <= {Q_UNFOLDED}")

q = df["Q"].values
n = len(q)

# --- Backward scan: for each frame, find the *next* basin (folded/unfolded)
#     it reaches, moving forward in time. ---
fate = np.full(n, np.nan)
last_known = np.nan
for i in range(n - 1, -1, -1):
    if q[i] >= Q_FOLDED:
        last_known = 1.0
    elif q[i] <= Q_UNFOLDED:
        last_known = 0.0
    fate[i] = last_known

df["Fate"] = fate

n_folded_first   = (fate == 1.0).sum()
n_unfolded_first = (fate == 0.0).sum()
n_undetermined   = np.isnan(fate).sum()
print(f"\nFrames whose next basin is FOLDED (Fate=1):   {n_folded_first:,} ({n_folded_first/n:.1%})")
print(f"Frames whose next basin is UNFOLDED (Fate=0): {n_unfolded_first:,} ({n_unfolded_first/n:.1%})")
print(f"Undetermined (trajectory ends before either basin is reached again): "
      f"{n_undetermined:,} ({n_undetermined/n:.1%})")

df.to_csv(OUT_FATE_CSV, index=False)
print(f"\nSaved per-frame fate labels: {OUT_FATE_CSV}")

# --- Bonus: bin by Q, average Fate within each bin -> empirical committor(Q) ---
valid = df.dropna(subset=["Fate"]).copy()
valid["Q_bin"] = pd.cut(valid["Q"], bins=N_BINS)
curve = (
    valid.groupby("Q_bin", observed=True)
    .agg(q_bin_center=("Q", "mean"), n_frames=("Fate", "size"), p_folded=("Fate", "mean"))
    .reset_index(drop=True)
)
curve.to_csv(OUT_CURVE_CSV, index=False)
print(f"Saved binned committor-vs-Q curve: {OUT_CURVE_CSV}")
print("\nBinned committor curve (Q bin center, n frames, P(folded first)):")
print(curve.to_string(index=False))
