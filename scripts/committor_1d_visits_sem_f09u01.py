"""
Visit-based 1D committor-vs-Q curve with SEM, using the folded=0.9/unfolded=0.1
Fate labels. Separate output from the 0.8/0.2 version.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FATE_CSV = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate_f09u01.csv"
OUT_CSV  = "/scratch/mma9420/committor_check/Q/committor_1d_visits_sem_f09u01.csv"
OUT_PNG  = "/scratch/mma9420/committor_check/Q/committor_1d_visits_sem_f09u01.png"
N_BINS   = 40

df = pd.read_csv(FATE_CSV).sort_values("Frame").reset_index(drop=True)
df = df.dropna(subset=["Fate"]).reset_index(drop=True)
print(f"Frames with a determined Fate: {len(df):,}")

df["q_bin"] = pd.cut(df["Q"], bins=N_BINS, labels=False)
_, q_edges = pd.cut(df["Q"], bins=N_BINS, retbins=True)
q_centers = (q_edges[:-1] + q_edges[1:]) / 2

same_as_prev = df["q_bin"] == df["q_bin"].shift()
df["visit_id"] = (~same_as_prev).cumsum()

visit_reps = df.groupby("visit_id").agg(
    q_bin=("q_bin", "first"),
    Fate=("Fate", lambda s: s.iloc[len(s) // 2]),
)

visits_by_bin = visit_reps.groupby("q_bin").agg(
    n_visits=("Fate", "size"),
    p_folded_visits=("Fate", "mean"),
    std_visits=("Fate", lambda s: s.std(ddof=1) if len(s) > 1 else np.nan),
)
visits_by_bin["sem_visits"] = visits_by_bin["std_visits"] / np.sqrt(visits_by_bin["n_visits"])

n_frames_by_bin = df.groupby("q_bin").size().rename("n_frames")

result = pd.concat([n_frames_by_bin, visits_by_bin], axis=1).reset_index()
result["q_bin_center"] = q_centers[result["q_bin"].astype(int)]
result = result.sort_values("q_bin_center")
result.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
print(result[["q_bin_center", "n_frames", "n_visits", "p_folded_visits",
              "std_visits", "sem_visits"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.errorbar(result["q_bin_center"], result["p_folded_visits"], yerr=result["sem_visits"],
            fmt="o-", color="crimson", ecolor="crimson", elinewidth=1.2, capsize=3,
            markersize=5, alpha=0.9, label="Visit-based (mean +/- SEM), folded=0.9/unfolded=0.1")

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)

p = result["p_folded_visits"].values
q = result["q_bin_center"].values
crossing = None
for i in range(len(p) - 1):
    if not np.isnan(p[i]) and not np.isnan(p[i + 1]) and (p[i] - 0.5) * (p[i + 1] - 0.5) < 0:
        frac = (0.5 - p[i]) / (p[i + 1] - p[i])
        crossing = q[i] + frac * (q[i + 1] - q[i])
        break
if crossing is not None:
    ax.axvline(crossing, color="black", linestyle=":", linewidth=1)
    ax.annotate(f"Q* ~ {crossing:.3f}", xy=(crossing, 0.5),
                xytext=(crossing + 0.05, 0.65), arrowprops=dict(arrowstyle="->"))
    print(f"\nEstimated transition-state Q: {crossing:.4f}")

ax.set_xlabel("Q (fraction of native contacts)")
ax.set_ylabel("P(folded first)")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Empirical committor vs. Q -- folded=0.9 / unfolded=0.1")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"\nSaved: {OUT_PNG}")
