"""
Implements the PI's feature-based re-clustering + cluster-averaged
committor relabeling, replacing the project's original Rg/Q-grid binning
(committor_grid_visits.csv) with clustering on the reduced local-feature
space itself.

Why this is different from just averaging the existing Committor_prob:
Committor_prob is already a Rg/Q-bin average (see build_physics_features.py
docstring / project history) -- re-clustering by features and averaging
THAT would be an average of an average, muddying exactly the thing the PI
flagged: two frames can share an Rg/Q bin while having different local
features, or share similar local features while landing in different Rg/Q
bins. So this script goes back to each frame's own raw, per-frame
fold/unfold outcome (Fate, 0 or 1, from Q_values_allframes_with_fate.csv --
the forward-scanning result used to build the ORIGINAL labels, before any
grid-averaging) and re-averages that directly within feature-similarity
clusters.

Also mirrors the project's established "visit" correction (same logic as
run_committor_visits.sbatch): naively averaging every frame's Fate within
a cluster would massively overweight long dwell periods in one cluster
(temporal autocorrelation). Consecutive same-cluster frames are collapsed
into one "visit" (representative = the visit's middle frame) before
averaging, exactly as the original Rg/Q-grid committor was built.

Clusters on the FULL 101-feature pruned set, not a PCA-reduced subset --
an earlier version depended on retrain_balanced_pca_features.py's
PCA-selected top-10 features, but that PCA selection was later shown to
be unsupervised and uninformative (it picked whatever varied most across
the whole trajectory, not what predicts the committor, and scored far
worse than using all 101 features). Nothing tested since has shown that
cutting to 10 features beats using all 101, so there's no basis for
forcing that cut before clustering either.

Applies the project's standard inverse-frequency sample weighting when
training on the cluster-averaged label -- an earlier round of balance/
PCA experiments omitted this by oversight and produced a spurious,
catastrophic-looking collapse that weighting alone fixed; this script
carries that lesson forward rather than repeating the omission.

Evaluation stays against the EXISTING, established Committor_prob on the
project's standard held-out test set -- not the new cluster-based label --
so this result is directly comparable to every other number in this
project's history. The new cluster-averaged label is used only as the
TRAINING target, to test whether a cleaner, feature-similarity-grouped
label makes a better training signal than the noisier, dilutable Rg/Q-grid
one.
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
PRUNED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_feature_clusters/"
os.makedirs(OUT_DIR, exist_ok=True)

SILHOUETTE_SAMPLE_SIZE = 20000
CANDIDATE_K = [10, 20, 30, 50, 75, 100, 150]
RANDOM_STATE = 42
EPS = 1e-4
N_WEIGHT_BINS = 20
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

REF_BASELINE_TRANSITION = 0.0305
REF_CHECK_C_TRANSITION = 0.3224
REF_CONFIDENCE_FILTERED = 0.3440
REF_PRUNED_FULL_TRANSITION = 0.1583
REF_PRUNED_TRANSONLY_TRANSITION = 0.3914
REF_CONFIDENCE_FILTERED_NEW_FULLTEST = 0.3333
REF_CONFIDENCE_FILTERED_NEW_HICONF = 0.4609

# ------------------------------------------------------------------
# Step 1: the full 101-feature pruned set (not a reduced subset -- see
# docstring), plus raw per-frame Fate (NOT the existing grid-averaged
# Committor_prob).
# ------------------------------------------------------------------
pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
feature_cols = [c for c in pruned_cols if c != "Committor_prob"]
print(f"Clustering on the full pruned feature set ({len(feature_cols)}): {feature_cols}")

print(f"\nLoading {MERGED_CSV}")
df = pd.read_csv(MERGED_CSV, usecols=["frame_index"] + feature_cols + ["Committor_prob"])
df = df.sort_values("frame_index").reset_index(drop=True)
n = len(df)
print(f"{n:,} rows")

print(f"\nLoading raw per-frame Fate from {FATE_CSV}")
df_fate = pd.read_csv(FATE_CSV, usecols=["Time_ps", "Fate"])
df_fate["frame_index"] = (df_fate["Time_ps"] / 200).round().astype(int)
df_fate = df_fate[["frame_index", "Fate"]].drop_duplicates(subset="frame_index")

n_before = len(df)
df = df.merge(df_fate, on="frame_index", how="left")
assert len(df) == n_before, "join changed row count -- frame_index is not unique somewhere"
n_matched = df["Fate"].notna().sum()
print(f"Fate join: {n_matched:,} / {len(df):,} frames matched ({100 * n_matched / len(df):.2f}%)")
if n_matched / len(df) < 0.99:
    print("WARNING: >1% of frames have no Fate value -- check FATE_CSV coverage before trusting results below.")

# ------------------------------------------------------------------
# Step 2: standardize features (clustering uses Euclidean distance --
# this is the step scaling actually matters for, unlike the LightGBM
# retrains elsewhere in this project).
# ------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols].values)

# ------------------------------------------------------------------
# Step 3: pick cluster count via silhouette score, scanning on a sample
# for tractability (534K rows is too large for an O(n^2) silhouette
# computation at full size).
# ------------------------------------------------------------------
print(f"\nScanning cluster counts {CANDIDATE_K} via silhouette score on a "
      f"{SILHOUETTE_SAMPLE_SIZE:,}-row sample...")
rng = np.random.RandomState(RANDOM_STATE)
sample_idx = rng.choice(n, size=min(SILHOUETTE_SAMPLE_SIZE, n), replace=False)
X_sample = X_scaled[sample_idx]

silhouette_scores = {}
for k in CANDIDATE_K:
    km_sample = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels_sample = km_sample.fit_predict(X_sample)
    score = silhouette_score(X_sample, labels_sample)
    silhouette_scores[k] = score
    print(f"  k={k:4d}  silhouette={score:.4f}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nBest k by silhouette score: {best_k}")

pd.Series(silhouette_scores, name="silhouette_score").to_csv(OUT_DIR + "silhouette_scan.csv")

# ------------------------------------------------------------------
# Step 4: final clustering on the FULL dataset with the chosen k.
# ------------------------------------------------------------------
print(f"\nFitting final KMeans(n_clusters={best_k}) on all {n:,} rows...")
km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df["cluster_id"] = km_final.fit_predict(X_scaled)

cluster_sizes = df["cluster_id"].value_counts().sort_index()
print(f"Cluster sizes: min={cluster_sizes.min():,}, median={cluster_sizes.median():.0f}, "
      f"max={cluster_sizes.max():,}")

# ------------------------------------------------------------------
# Step 5: visit-corrected per-cluster committor average (same logic as
# run_committor_visits.sbatch, applied to cluster_id instead of cell_id).
# ------------------------------------------------------------------
print("\nComputing visit-corrected per-cluster committor averages...")
same_as_prev = df["cluster_id"] == df["cluster_id"].shift()
df["visit_id"] = (~same_as_prev).cumsum()

visit_reps = df.dropna(subset=["Fate"]).groupby("visit_id").agg(
    cluster_id=("cluster_id", "first"),
    Fate=("Fate", lambda s: s.iloc[len(s) // 2]),
)
cluster_committor = visit_reps.groupby("cluster_id")["Fate"].agg(["mean", "count"])
cluster_committor.columns = ["cluster_committor", "n_visits"]
cluster_committor.to_csv(OUT_DIR + "cluster_committor_averages.csv")
print(f"Clusters with at least one visit: {len(cluster_committor):,} / {best_k}")
print(f"n_visits per cluster: min={cluster_committor['n_visits'].min()}, "
      f"median={cluster_committor['n_visits'].median():.0f}, max={cluster_committor['n_visits'].max()}")

df = df.merge(cluster_committor[["cluster_committor"]], on="cluster_id", how="left")
n_label_missing = df["cluster_committor"].isna().sum()
print(f"Frames with no cluster-committor label (cluster had zero visits with a Fate value): "
      f"{n_label_missing:,} ({100 * n_label_missing / len(df):.2f}%)")

# ------------------------------------------------------------------
# Step 6: identical split to the rest of the project.
# ------------------------------------------------------------------
y_old = df["Committor_prob"].values  # established label, used for TEST evaluation only
y_new = df["cluster_committor"].values  # new label, used for TRAINING only

block_size = n // N_BLOCKS
block_id = np.minimum(np.arange(n) // block_size, N_BLOCKS - 1)
rng_split = np.random.RandomState(RANDOM_STATE)
test_blocks = sorted(rng_split.choice(N_BLOCKS, size=N_TEST_BLOCKS, replace=False))
print(f"\nTest blocks: {test_blocks}")

is_test = np.isin(block_id, test_blocks)
buffer_idx = set()
for b in test_blocks:
    idx_in_block = np.where(block_id == b)[0]
    lo, hi = idx_in_block.min(), idx_in_block.max()
    buffer_idx.update(range(max(0, lo - BUFFER_FRAMES), lo))
    buffer_idx.update(range(hi + 1, min(n, hi + 1 + BUFFER_FRAMES)))
is_buffer = np.isin(np.arange(n), list(buffer_idx))
is_train = (~is_test) & (~is_buffer) & (~np.isnan(y_new))
idx_train = np.where(is_train)[0]
idx_test = np.where(is_test)[0]
print(f"Train (with a valid cluster label): {is_train.sum():,}  Test: {is_test.sum():,}  "
      f"Buffer: {is_buffer.sum():,}")

X = df[feature_cols]
y_new_clipped = np.clip(y_new, EPS, 1 - EPS)
y_new_logit = np.log(y_new_clipped / (1 - y_new_clipped))

# Inverse-frequency sample weighting on the TRAINING target's own
# distribution (cluster_committor among training rows) -- the project's
# standard scheme, carried forward after an earlier round of experiments
# omitted it by oversight and produced a spurious collapse that weighting
# alone fixed. Computed only over valid (non-NaN) training labels.
train_y_for_weights = y_new[idx_train]
bin_edges = np.linspace(0, 1, N_WEIGHT_BINS + 1)
bin_idx = np.clip(np.digitize(train_y_for_weights, bin_edges) - 1, 0, N_WEIGHT_BINS - 1)
bin_counts = np.bincount(bin_idx, minlength=N_WEIGHT_BINS)
train_weights = np.array([1.0 / bin_counts[b] if bin_counts[b] > 0 else 0.0 for b in bin_idx])
train_weights = train_weights / train_weights.mean()
print(f"\nComputed inverse-frequency weights on the training label distribution ({N_WEIGHT_BINS} bins)")

y_old_test = y_old[idx_test]
mask_test_tr = (y_old_test > TRANSITION_LOW) & (y_old_test < TRANSITION_HIGH)
print(f"Transition-region test rows (by the established Committor_prob): {mask_test_tr.sum():,}")


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


# ------------------------------------------------------------------
# Step 7: train on the new cluster-averaged label, evaluate against the
# EXISTING established Committor_prob on the standard test set.
# ------------------------------------------------------------------
print("\nTraining on cluster-averaged committor labels (weighted)...")
model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=2
)
model.fit(X.iloc[idx_train], y_new_logit[idx_train], sample_weight=train_weights)

pred_test = inv_logit(model.predict(X.iloc[idx_test]))
r2_test_overall = r2_score(y_old_test, pred_test)
r2_test_trans = r2_score(y_old_test[mask_test_tr], pred_test[mask_test_tr])

print(f"\nTest overall R^2 (vs. established Committor_prob): {r2_test_overall:.4f}")
print(f"Test transition-region R^2 (vs. established Committor_prob): {r2_test_trans:.4f}")

print("\n=== Comparison ===")
print(f"{'run':65s} {'transition-region test R^2':>28s}")
print(f"{'Old baseline (64 features, full training)':65s} {REF_BASELINE_TRANSITION:>28.4f}")
print(f"{'Old Check C (64 features, transition-only train)':65s} {REF_CHECK_C_TRANSITION:>28.4f}")
print(f"{'Old confidence-filtered (n_visits>=84)':65s} {REF_CONFIDENCE_FILTERED:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'New physics feat (101), full training':65s} {REF_PRUNED_FULL_TRANSITION:>28.4f}")
print(f"{'New physics feat (101), transition-only training':65s} {REF_PRUNED_TRANSONLY_TRANSITION:>28.4f}")
print(f"{'New confidence-filtered, full test':65s} {REF_CONFIDENCE_FILTERED_NEW_FULLTEST:>28.4f}")
print(f"{'New confidence-filtered, hi-conf test':65s} {REF_CONFIDENCE_FILTERED_NEW_HICONF:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'THIS: feature-based clusters (k=%d), cluster-avg label train' % best_k:65s} {r2_test_trans:>28.4f}")

feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance.csv", header=["importance"])
print(f"\nFeature importance:")
print(feature_importance.to_string())

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Feature set ({len(feature_cols)}): {feature_cols}\n")
    f.write(f"Best k (silhouette): {best_k}\n")
    f.write(f"Silhouette scan: {silhouette_scores}\n\n")
    f.write(f"Train rows: {is_train.sum():,}  Test rows: {is_test.sum():,}\n")
    f.write(f"Transition-region test rows: {mask_test_tr.sum():,}\n\n")
    f.write(f"Test overall R^2 (vs. established Committor_prob): {r2_test_overall:.4f}\n")
    f.write(f"Test transition-region R^2 (vs. established Committor_prob): {r2_test_trans:.4f}\n\n")
    f.write("References:\n")
    f.write(f"  Old baseline:                   {REF_BASELINE_TRANSITION:.4f}\n")
    f.write(f"  Old Check C:                    {REF_CHECK_C_TRANSITION:.4f}\n")
    f.write(f"  Old confidence-filtered:        {REF_CONFIDENCE_FILTERED:.4f}\n")
    f.write(f"  New physics feat, full:         {REF_PRUNED_FULL_TRANSITION:.4f}\n")
    f.write(f"  New physics feat, trans-only:   {REF_PRUNED_TRANSONLY_TRANSITION:.4f}\n")
    f.write(f"  New confidence-filtered, full:  {REF_CONFIDENCE_FILTERED_NEW_FULLTEST:.4f}\n")
    f.write(f"  New confidence-filtered, hiconf:{REF_CONFIDENCE_FILTERED_NEW_HICONF:.4f}\n")

df[["frame_index", "cluster_id", "cluster_committor"]].to_csv(OUT_DIR + "frame_cluster_assignments.csv", index=False)
print(f"\nSaved outputs to {OUT_DIR}")
