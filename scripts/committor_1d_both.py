"""
1D committor-vs-Q curves for both the visit-based and tiered methods,
computed directly in Q-only space (not derived from the 2D Rg x Q grid,
since visit/tier segmentation differs between 1D and 2D).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FATE_CSV = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate.csv"
OUT_CSV  = "/scratch/mma9420/committor_check/Q/committor_1d_both.csv"
OUT_PNG  = "/scratch/mma9420/committor_check/Q/committor_1d_both.png"
N_BINS   = 40


def tiered_step(n):
    if n <= 100:
        return 1
    elif n <= 500:
        return 2
    elif n <= 1000:
        return 5
    elif n <= 4000:
        return 10
    elif n <= 10000:
        return 20
    else:
        return 100


df = pd.read_csv(FATE_CSV).sort_values("Frame").reset_index(drop=True)
df = df.dropna(subset=["Fate"]).reset_index(drop=True)
print(f"Frames with a determined Fate: {len(df):,}")

df["q_bin"] = pd.cut(df["Q"], bins=N_BINS, labels=False)
_, q_edges = pd.cut(df["Q"], bins=N_BINS, retbins=True)
q_centers = (q_edges[:-1] + q_edges[1:]) / 2

# --- Visit-based: segment by consecutive same-Q-bin frames ---
same_as_prev = df["q_bin"] == df["q_bin"].shift()
df["visit_id"] = (~same_as_prev).cumsum()

visit_reps = df.groupby("visit_id").agg(
    q_bin=("q_bin", "first"),
    Fate=("Fate", lambda s: s.iloc[len(s) // 2]),
)
visits_by_bin = visit_reps.groupby("q_bin").agg(
    n_visits=("Fate", "size"),
    p_folded_visits=("Fate", "mean"),
)

# --- Tiered: step-sample within each Q-bin, ordered by frame ---
def sample_bin(group):
    n = len(group)
    step = tiered_step(n)
    sampled = group.iloc[::step]
    return pd.Series({
        "n_sampled_tiered": len(sampled),
        "p_folded_tiered": sampled["Fate"].mean(),
    })

tiered_by_bin = (
    df.sort_values("Frame")
    .groupby("q_bin", group_keys=False)
    .apply(sample_bin)
)

n_frames_by_bin = df.groupby("q_bin").size().rename("n_frames")

result = pd.concat([n_frames_by_bin, visits_by_bin, tiered_by_bin], axis=1).reset_index()
result["q_bin_center"] = q_centers[result["q_bin"].astype(int)]
result = result.sort_values("q_bin_center")
result.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
print(result[["q_bin_center", "n_frames", "n_visits", "p_folded_visits",
              "n_sampled_tiered", "p_folded_tiered"]].to_string(index=False))

# --- Plot both curves overlaid ---
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(result["q_bin_center"], result["p_folded_visits"], "o-",
        color="crimson", label="Visit-based", markersize=5)
ax.plot(result["q_bin_center"], result["p_folded_tiered"], "s--",
        color="steelblue", label="Tiered step-sampling", markersize=5, alpha=0.85)

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Q (fraction of native contacts)")
ax.set_ylabel("P(folded first)")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Empirical committor vs. Q -- visit-based vs. tiered")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"\nSaved: {OUT_PNG}")
