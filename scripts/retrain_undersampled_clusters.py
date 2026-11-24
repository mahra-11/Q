"""
Feature-based clustering, retried with two changes from the failed attempt
in retrain_feature_clusters.py (R^2=-3.14 on the 101-feature set):

1. Clusters on the literature-curated 21-feature reduced set
   (retrain_literature_reduced.py) instead of the full 101/109-feature set.
   Rejected for DIRECT regression (lower R^2 than the full set there), but
   that's a different question from whether it gives K-means a cleaner,
   more physically meaningful space to find structure in -- untested until
   now.

2. The K-means FIT is done on an undersampled (balance-ratio-controlled)
   subset of training frames, not the full, heavily end-state-dominated
   population. With 99%+ of frames sitting in the two end-state basins,
   unweighted K-means mostly spends its cluster budget carving those two
   blobs up; the thin transition-region sliver gets absorbed into the edges
   of nearby end-state clusters rather than forming its own cluster. Fitting
   on a ratio-controlled sample gives transition frames real influence over
   where cluster boundaries land. After fitting, EVERY frame (the full,
   un-undersampled population) is assigned to its nearest cluster via
   .predict(), and the visit-corrected real-Fate committor average per
   cluster (same logic as retrain_feature_clusters.py) is computed over all
   of them -- undersampling only changes where the cluster boundaries are
   drawn, not what data the final labels/training use.

Both changes are bundled in one run; if this helps, a follow-up could
isolate which one mattered. Sweeps the undersampling ratio
(end-state:transition, same convention as sweep_balance_ratio_weighted.py)
so we see whether any ratio helps versus the extremes.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score
import lightgbm as lgb

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
FATE_CSV = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate.csv"
ASP2N_GLY6O_CSV = "/scratch/mma9420/committor_check/Q/physics_features/retrain_lit_asp2n_gly6o/lit_d_Asp2N_Gly6O.csv"
PRIORITIES_4_5_7_CSV = "/scratch/mma9420/committor_check/Q/physics_features/retrain_lit_priorities_4_5_7/new_features.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_undersampled_clusters/"
os.makedirs(OUT_DIR, exist_ok=True)

LITERATURE_REDUCED_FEATURES = [
    "hb_6GLYN_2ASPO", "hb_9TYRN_0TYRO", "lit_d_Asp2N_Gly6O", "lit_d_Tyr1N_Trp8O",
    "lit_d_Tyr0_Asp2_closest_heavy", "hb_2ASPN_7THRO", "hb_5THROG1_2ASPOD1",
    "hb_5THROG1_2ASPOD2", "hb_7THRN_2ASPO", "saltbridge_Nterm_Asp2",
    "cos_phi_res6", "sin_phi_res6", "cos_psi_res6", "sin_psi_res6",
    "cos_phi_res7", "sin_phi_res7", "sc_dist_res1_res8", "sin_chi1_res1",
    "sc_dist_res1_res7", "sc_dist_res1_res6", "cos_chi1_res1",
]

UNDERSAMPLE_RATIOS = [0.5, 1, 2, 5, 10, "all"]
CANDIDATE_K = [10, 20, 30, 50, 75]
RANDOM_STATE = 42
EPS = 1e-4
N_WEIGHT_BINS = 20
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8
SILHOUETTE_SAMPLE_CAP = 20000

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

REF_101FEAT_NO_UNDERSAMPLE = -3.1437
REF_109FEAT_FULL_TRAINING = 0.1909
REF_109FEAT_TRANSONLY = 0.4040

# ------------------------------------------------------------------
# Step 1: load features + raw per-frame Fate.
# ------------------------------------------------------------------
needed_from_merged = [c for c in LITERATURE_REDUCED_FEATURES
                       if c not in ("lit_d_Asp2N_Gly6O", "lit_d_Tyr1N_Trp8O", "lit_d_Tyr0_Asp2_closest_heavy")]
df = pd.read_csv(MERGED_CSV, usecols=["frame_index"] + needed_from_merged + ["Committor_prob"])
n_before = len(df)
df = df.merge(pd.read_csv(ASP2N_GLY6O_CSV), on="frame_index", how="left")
df = df.merge(pd.read_csv(PRIORITIES_4_5_7_CSV)[["frame_index", "lit_d_Tyr1N_Trp8O", "lit_d_Tyr0_Asp2_closest_heavy"]],
              on="frame_index", how="left")
assert len(df) == n_before
df = df.sort_values("frame_index").reset_index(drop=True)
n = len(df)
print(f"Feature set: {len(LITERATURE_REDUCED_FEATURES)} features, {n:,} rows")

print(f"\nLoading raw per-frame Fate from {FATE_CSV}")
df_fate = pd.read_csv(FATE_CSV, usecols=["Time_ps", "Fate"])
df_fate["frame_index"] = (df_fate["Time_ps"] / 200).round().astype(int)
df_fate = df_fate[["frame_index", "Fate"]].drop_duplicates(subset="frame_index")
n_before = len(df)
df = df.merge(df_fate, on="frame_index", how="left")
assert len(df) == n_before
n_matched = df["Fate"].notna().sum()
print(f"Fate join: {n_matched:,} / {len(df):,} frames matched ({100 * n_matched / len(df):.2f}%)")

# ------------------------------------------------------------------
# Step 2: standard train/test split (identical to every other script).
# ------------------------------------------------------------------
y_old = df["Committor_prob"].values
block_size = n // N_BLOCKS
block_id = np.minimum(np.arange(n) // block_size, N_BLOCKS - 1)
rng_split = np.random.RandomState(RANDOM_STATE)
test_blocks = sorted(rng_split.choice(N_BLOCKS, size=N_TEST_BLOCKS, replace=False))
print(f"Test blocks: {test_blocks}")

is_test = np.isin(block_id, test_blocks)
buffer_idx = set()
for b in test_blocks:
    idx_in_block = np.where(block_id == b)[0]
    lo, hi = idx_in_block.min(), idx_in_block.max()
    buffer_idx.update(range(max(0, lo - BUFFER_FRAMES), lo))
    buffer_idx.update(range(hi + 1, min(n, hi + 1 + BUFFER_FRAMES)))
is_buffer = np.isin(np.arange(n), list(buffer_idx))
is_train = (~is_test) & (~is_buffer)
idx_train = np.where(is_train)[0]
idx_test = np.where(is_test)[0]
print(f"Train: {is_train.sum():,}  Test: {is_test.sum():,}  Buffer: {is_buffer.sum():,}")

y_old_test = y_old[idx_test]
mask_test_tr = (y_old_test > TRANSITION_LOW) & (y_old_test < TRANSITION_HIGH)
print(f"Transition-region test rows: {mask_test_tr.sum():,}")

y_old_train = y_old[idx_train]
is_trans_train = (y_old_train > TRANSITION_LOW) & (y_old_train < TRANSITION_HIGH)
trans_train_idx = idx_train[is_trans_train]
end_state_train_idx = idx_train[~is_trans_train]
n_trans = len(trans_train_idx)
n_end_state_available = len(end_state_train_idx)
print(f"Transition-region training rows: {n_trans:,}  End-state training rows available: {n_end_state_available:,}")

X_all = df[LITERATURE_REDUCED_FEATURES].values


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


results = []
for ratio in UNDERSAMPLE_RATIOS:
    label = str(ratio)
    print(f"\n{'=' * 70}\nUndersample ratio: {label}\n{'=' * 70}")

    if ratio == "all":
        n_end_state_keep = n_end_state_available
    else:
        n_end_state_keep = min(int(round(ratio * n_trans)), n_end_state_available)

    rng_ratio = np.random.RandomState(RANDOM_STATE)
    if n_end_state_keep > 0:
        end_state_fit_idx = rng_ratio.choice(end_state_train_idx, size=n_end_state_keep, replace=False)
        fit_idx = np.concatenate([trans_train_idx, end_state_fit_idx])
    else:
        fit_idx = trans_train_idx
    print(f"K-means fit set: {len(fit_idx):,} ({n_trans:,} transition + {n_end_state_keep:,} end-state)")

    # scaler fit on the fit set only, then applied to every frame
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_all[fit_idx])
    X_all_scaled = scaler.transform(X_all)

    # silhouette scan on the fit set (subsampled further only if it's still too big)
    if len(fit_idx) > SILHOUETTE_SAMPLE_CAP:
        sil_sample_idx = rng_ratio.choice(len(fit_idx), size=SILHOUETTE_SAMPLE_CAP, replace=False)
        X_sil = X_fit_scaled[sil_sample_idx]
    else:
        X_sil = X_fit_scaled

    silhouette_scores = {}
    for k in CANDIDATE_K:
        if k >= len(X_sil):
            continue
        km_sample = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels_sample = km_sample.fit_predict(X_sil)
        silhouette_scores[k] = silhouette_score(X_sil, labels_sample)
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Silhouette scores: {silhouette_scores}")
    print(f"Best k: {best_k}")

    # final K-means fit on the (undersampled) fit set
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    km_final.fit(X_fit_scaled)

    # assign EVERY frame (full, un-undersampled population) to its nearest cluster
    df["cluster_id"] = km_final.predict(X_all_scaled)
    cluster_sizes = df["cluster_id"].value_counts().sort_index()
    print(f"Cluster sizes on full data: min={cluster_sizes.min():,}, median={cluster_sizes.median():.0f}, "
          f"max={cluster_sizes.max():,}")

    # visit-corrected per-cluster committor average, using ALL frames' real Fate
    same_as_prev = df["cluster_id"] == df["cluster_id"].shift()
    df["visit_id"] = (~same_as_prev).cumsum()
    visit_reps = df.dropna(subset=["Fate"]).groupby("visit_id").agg(
        cluster_id=("cluster_id", "first"),
        Fate=("Fate", lambda s: s.iloc[len(s) // 2]),
    )
    cluster_committor = visit_reps.groupby("cluster_id")["Fate"].agg(["mean", "count"])
    cluster_committor.columns = ["cluster_committor", "n_visits"]
    print(f"Clusters with at least one visit: {len(cluster_committor):,} / {best_k}")

    df_labeled = df.drop(columns=[c for c in ["cluster_committor"] if c in df.columns])
    df_labeled = df_labeled.merge(cluster_committor[["cluster_committor"]], on="cluster_id", how="left")
    y_new = df_labeled["cluster_committor"].values

    train_mask = is_train & ~np.isnan(y_new)
    train_idx_labeled = np.where(train_mask)[0]
    print(f"Train rows with a valid cluster label: {len(train_idx_labeled):,} / {is_train.sum():,}")

    y_new_clipped = np.clip(y_new, EPS, 1 - EPS)
    y_new_logit = np.log(y_new_clipped / (1 - y_new_clipped))

    bin_edges = np.linspace(0, 1, N_WEIGHT_BINS + 1)
    train_y_for_weights = y_new[train_idx_labeled]
    bin_idx_w = np.clip(np.digitize(train_y_for_weights, bin_edges) - 1, 0, N_WEIGHT_BINS - 1)
    bin_counts = np.bincount(bin_idx_w, minlength=N_WEIGHT_BINS)
    train_weights = np.array([1.0 / bin_counts[b] if bin_counts[b] > 0 else 0.0 for b in bin_idx_w])
    train_weights = train_weights / train_weights.mean()

    X_lgb = df_labeled[LITERATURE_REDUCED_FEATURES]
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=2
    )
    model.fit(X_lgb.iloc[train_idx_labeled], y_new_logit[train_idx_labeled], sample_weight=train_weights)

    pred_test = inv_logit(model.predict(X_lgb.iloc[idx_test]))
    r2_test_overall = r2_score(y_old_test, pred_test)
    r2_test_trans = r2_score(y_old_test[mask_test_tr], pred_test[mask_test_tr])

    print(f"Test overall R^2: {r2_test_overall:.4f}  Test transition-region R^2: {r2_test_trans:.4f}")

    results.append({
        "undersample_ratio": label,
        "best_k": best_k,
        "n_fit_rows": len(fit_idx),
        "n_clusters_with_visits": len(cluster_committor),
        "r2_overall": r2_test_overall,
        "r2_transition": r2_test_trans,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR + "undersample_ratio_sweep.csv", index=False)

print("\n=== Full sweep ===")
print(results_df.to_string(index=False))

best_row = results_df.loc[results_df["r2_transition"].idxmax()]
print(f"\nBest undersample ratio: {best_row['undersample_ratio']} "
      f"(transition R^2 = {best_row['r2_transition']:.4f})")

print("\n=== Comparison ===")
print(f"{'run':65s} {'transition-region test R^2':>28s}")
print(f"{'Previous clustering attempt (101 feat, no undersampling)':65s} {REF_101FEAT_NO_UNDERSAMPLE:>28.4f}")
print(f"{'109-feature direct regression, full training':65s} {REF_109FEAT_FULL_TRAINING:>28.4f}")
print(f"{'109-feature direct regression, transition-only':65s} {REF_109FEAT_TRANSONLY:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
for _, row in results_df.iterrows():
    print(f"{'THIS: literature-feat clusters, undersample ratio=%s' % row['undersample_ratio']:65s} "
          f"{row['r2_transition']:>28.4f}")

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Feature set: {LITERATURE_REDUCED_FEATURES}\n\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nBest undersample ratio: {best_row['undersample_ratio']} "
            f"(transition R^2 = {best_row['r2_transition']:.4f})\n")

print(f"\nSaved outputs to {OUT_DIR}")
