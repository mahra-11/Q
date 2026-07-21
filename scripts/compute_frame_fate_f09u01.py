"""
Same as compute_frame_fate.py, but with folded=0.9 / unfolded=0.1 instead
of the config defaults (0.8 / 0.2). Separate output files -- doesn't touch
the original 0.8/0.2 results.
"""
import numpy as np
import pandas as pd

Q_CSV        = "/scratch/mma9420/committor_check/Q/Q_values_allframes.csv"
OUT_FATE_CSV = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate_f09u01.csv"

Q_FOLDED   = 0.9
Q_UNFOLDED = 0.1

print(f"Loading Q values: {Q_CSV}")
df = pd.read_csv(Q_CSV).sort_values("Frame").reset_index(drop=True)

print(f"Folded threshold:   Q >= {Q_FOLDED}")
print(f"Unfolded threshold: Q <= {Q_UNFOLDED}")

q = df["Q"].values
n = len(q)

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
print(f"Undetermined: {n_undetermined:,} ({n_undetermined/n:.1%})")

df.to_csv(OUT_FATE_CSV, index=False)
print(f"\nSaved: {OUT_FATE_CSV}")
