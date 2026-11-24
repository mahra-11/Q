"""
Implements the PI's balance + feature-reduction recipe from the meeting
notes, combined rather than tested in isolation -- balancing alone was
already tried on the old 64-feature set and made things worse (transition-
region R^2 dropped to -0.05, from a project working document), which the
PI's own diagnosis explains: with too many features relative to the
amount of real transition-region data, removing the easy-state "anchor"
frames doesn't help if the model still has to fit a high-dimensional
feature space on a tiny, noisy subset. His recipe pairs the two fixes:

  1. Reduce the feature set to a small, information-dense set (~10)
     via PCA: standardize features, fit PCA, and rank each original
     feature by its PCA-weighted contribution (sum of explained-variance-
     ratio-weighted squared loadings across all components), not by using
     PCA-transformed components as inputs -- the PI's description keeps
     training on the original (interpretable) features, just fewer of
     them.
  2. Balance the TRAINING set so transition-region frames aren't
     swamped by the (much more numerous) folded/unfolded frames --
     downsample the end-state training rows so their total count is
     comparable to the transition-region training row count, rather
     than capping each class separately at a fixed number.

Evaluated on the project's standard, UNBALANCED, contiguous time-block
test set (same split as every other retrain in this project), so the
R^2 numbers stay comparable to every reference below.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

PRUNED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"
MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_balanced_pca/"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8
N_PCA_FEATURES = 10

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
REF_OLD_DOWNSAMPLE_ALONE = -0.05  # old 64-feature set, no Q, downsample-only -- the result being retested here

# ------------------------------------------------------------------
# Step 1: load the pruned (101-feature) set and rank features by PCA
# contribution, keep the top N_PCA_FEATURES.
# ------------------------------------------------------------------
pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
feature_cols_full = [c for c in pruned_cols if c != "Committor_prob"]
print(f"Starting from the pruned feature set: {len(feature_cols_full)} features")

df = pd.read_csv(MERGED_CSV, usecols=["frame_index"] + feature_cols_full + ["Committor_prob"])
df = df.sort_values("frame_index").reset_index(drop=True)
n = len(df)
print(f"{n:,} rows")

print("\nRunning PCA on standardized features to rank feature importance...")
scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(df[feature_cols_full].values)

# IMPORTANT: must restrict to a SUBSET of components, not the full rank.
# For standardized features, summing (explained_variance_ratio * loading^2)
# across ALL components reconstructs each feature's own variance, which is
# exactly 1.0 for every standardized feature -- so a full-rank sum gives
# every feature the identical score (1 / n_features) and the "ranking" is
# meaningless. Restricting to the top components that explain most of the
# variance is what actually differentiates features.
pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
pca.fit(X_scaled_full)
print(f"Kept {pca.n_components_} components to explain "
      f"{pca.explained_variance_ratio_.sum():.1%} of variance")

# Score each feature by its PCA-weighted contribution across only the
# retained top components: sum of (explained_variance_ratio * squared
# loading). A feature that loads heavily onto high-variance components
# scores higher.
loadings = pca.components_  # shape (n_components_kept, n_features)
evr = pca.explained_variance_ratio_
feature_scores = pd.Series(
    np.sum((loadings ** 2) * evr[:, None], axis=0),
    index=feature_cols_full,
).sort_values(ascending=False)

feature_scores.to_csv(OUT_DIR + "pca_feature_scores.csv", header=["pca_weighted_score"])
print(f"\nTop {N_PCA_FEATURES} features by PCA-weighted contribution:")
print(feature_scores.head(N_PCA_FEATURES).to_string())

feature_cols = feature_scores.head(N_PCA_FEATURES).index.tolist()
print(f"\nReduced feature set ({len(feature_cols)} features): {feature_cols}")

# ------------------------------------------------------------------
# Step 2: identical split to the rest of the project.
# ------------------------------------------------------------------
y_raw = df["Committor_prob"].values
y_clipped = np.clip(y_raw, EPS, 1 - EPS)
y_logit = np.log(y_clipped / (1 - y_clipped))

block_size = n // N_BLOCKS
block_id = np.minimum(np.arange(n) // block_size, N_BLOCKS - 1)
rng = np.random.RandomState(RANDOM_STATE)
test_blocks = sorted(rng.choice(N_BLOCKS, size=N_TEST_BLOCKS, replace=False))
print(f"\nTest blocks: {test_blocks}")

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

X = df[feature_cols]
y_raw_test = y_raw[idx_test]
mask_test_tr = (y_raw_test > TRANSITION_LOW) & (y_raw_test < TRANSITION_HIGH)
print(f"Transition-region test rows (unbalanced, as in every other retrain): {mask_test_tr.sum():,}")

# ------------------------------------------------------------------
# Step 3: balance the TRAINING set only -- downsample end-state
# (non-transition-region) training rows so their count is comparable to
# the transition-region training row count. Test set stays untouched.
# ------------------------------------------------------------------
y_raw_train = y_raw[idx_train]
is_trans_train = (y_raw_train > TRANSITION_LOW) & (y_raw_train < TRANSITION_HIGH)
trans_train_idx = idx_train[is_trans_train]
end_state_train_idx = idx_train[~is_trans_train]

n_trans = len(trans_train_idx)
print(f"\nTransition-region training rows: {n_trans:,}")
print(f"End-state (folded/unfolded) training rows before balancing: {len(end_state_train_idx):,}")

rng_balance = np.random.RandomState(RANDOM_STATE)
n_end_state_keep = min(n_trans, len(end_state_train_idx))
end_state_keep_idx = rng_balance.choice(end_state_train_idx, size=n_end_state_keep, replace=False)

balanced_train_idx = np.concatenate([trans_train_idx, end_state_keep_idx])
rng_balance.shuffle(balanced_train_idx)
print(f"Balanced training set: {len(balanced_train_idx):,} rows "
      f"({n_trans:,} transition-region + {n_end_state_keep:,} end-state, "
      f"target ratio ~50/50)")


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


# ------------------------------------------------------------------
# Step 4: train on the balanced, PCA-reduced-feature set; evaluate on
# the standard unbalanced test set.
# ------------------------------------------------------------------
print("\nTraining on balanced + PCA-reduced feature set...")
model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=2
)
model.fit(X.iloc[balanced_train_idx], y_logit[balanced_train_idx])

pred_train = inv_logit(model.predict(X.iloc[balanced_train_idx]))
pred_test = inv_logit(model.predict(X.iloc[idx_test]))

r2_train = r2_score(y_raw[balanced_train_idx], pred_train)
r2_test_overall = r2_score(y_raw_test, pred_test)
r2_test_trans = r2_score(y_raw_test[mask_test_tr], pred_test[mask_test_tr])

print(f"\nOwn (balanced) training R^2: {r2_train:.4f}")
print(f"Test overall R^2: {r2_test_overall:.4f}")
print(f"Test transition-region R^2: {r2_test_trans:.4f}")

print("\n=== Comparison ===")
print(f"{'run':65s} {'transition-region test R^2':>28s}")
print(f"{'Old baseline (64 features, full training)':65s} {REF_BASELINE_TRANSITION:>28.4f}")
print(f"{'Old Check C (64 features, transition-only train)':65s} {REF_CHECK_C_TRANSITION:>28.4f}")
print(f"{'Old confidence-filtered (n_visits>=84)':65s} {REF_CONFIDENCE_FILTERED:>28.4f}")
print(f"{'Old downsample-alone (64 feat, no Q) -- being retested here':65s} {REF_OLD_DOWNSAMPLE_ALONE:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'New physics feat (101), full training':65s} {REF_PRUNED_FULL_TRANSITION:>28.4f}")
print(f"{'New physics feat (101), transition-only training':65s} {REF_PRUNED_TRANSONLY_TRANSITION:>28.4f}")
print(f"{'New physics feat, confidence-filtered train, full test':65s} {REF_CONFIDENCE_FILTERED_NEW_FULLTEST:>28.4f}")
print(f"{'New physics feat, confidence-filtered train, hi-conf test':65s} {REF_CONFIDENCE_FILTERED_NEW_HICONF:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'THIS: balanced training + PCA-reduced (%d feat)' % len(feature_cols):65s} {r2_test_trans:>28.4f}")

feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance.csv", header=["importance"])
print(f"\nFeature importance (reduced set, by LightGBM split count):")
print(feature_importance.to_string())

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"PCA-reduced feature set ({len(feature_cols)} features): {feature_cols}\n\n")
    f.write(f"Transition-region training rows: {n_trans:,}\n")
    f.write(f"End-state training rows kept (balanced): {n_end_state_keep:,}\n")
    f.write(f"Balanced training set: {len(balanced_train_idx):,} rows\n")
    f.write(f"Test: {is_test.sum():,} rows ({mask_test_tr.sum():,} transition-region, unbalanced)\n\n")
    f.write(f"Own (balanced) training R^2: {r2_train:.4f}\n")
    f.write(f"Test overall R^2: {r2_test_overall:.4f}\n")
    f.write(f"Test transition-region R^2: {r2_test_trans:.4f}\n\n")
    f.write("References:\n")
    f.write(f"  Old baseline:                          {REF_BASELINE_TRANSITION:.4f}\n")
    f.write(f"  Old Check C:                           {REF_CHECK_C_TRANSITION:.4f}\n")
    f.write(f"  Old confidence-filtered:                {REF_CONFIDENCE_FILTERED:.4f}\n")
    f.write(f"  Old downsample-alone (being retested):  {REF_OLD_DOWNSAMPLE_ALONE:.4f}\n")
    f.write(f"  New physics feat, full training:        {REF_PRUNED_FULL_TRANSITION:.4f}\n")
    f.write(f"  New physics feat, transition-only:      {REF_PRUNED_TRANSONLY_TRANSITION:.4f}\n")
    f.write(f"  New confidence-filtered, full test:     {REF_CONFIDENCE_FILTERED_NEW_FULLTEST:.4f}\n")
    f.write(f"  New confidence-filtered, hi-conf test:  {REF_CONFIDENCE_FILTERED_NEW_HICONF:.4f}\n")

print(f"\nSaved outputs to {OUT_DIR}")
