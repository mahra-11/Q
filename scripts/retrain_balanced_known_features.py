"""
Isolation test: same balanced-training recipe as retrain_balanced_pca_features.py,
but swapping PCA-selected features for the 10 features ALREADY KNOWN to be
predictive from Step 8's own supervised feature_importances_ ranking
(sc_dist_res1_res8, sc_dist_res1_res7, sc_dist_res1_res6, sin_chi1_res1,
cos_chi1_res1, contact_res1_res7, contact_res1_res8, hb_2ASPN_7THRO,
sc_dist_res0_res8, saltbridge_Nterm_Cterm9 -- see retrain_physics_features.py
output and the session report's Step 8 section).

Purpose: the PCA-reduced balanced retrain scored transition-region R^2 =
-1.04, far worse than every other approach, and PCA is unsupervised -- it
picked features that vary the most across the whole trajectory, not
features that predict the committor (none of PCA's top 10 overlapped with
Step 8's actual top 10 by importance). This script holds the balancing
methodology IDENTICAL and only changes the feature set, to isolate whether
balancing itself is the problem or whether it was specifically PCA's
choice of (likely uninformative) features.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_balanced_known_features/"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

# Step 8's own top-10 features by feature_importances_ (known predictive,
# not PCA-selected) -- held fixed so only the balancing variable changes
# relative to retrain_physics_features.py's full-101-feature, unbalanced
# full-training run.
FEATURE_COLS = [
    "sc_dist_res1_res8", "sc_dist_res1_res7", "sc_dist_res1_res6",
    "sin_chi1_res1", "cos_chi1_res1", "contact_res1_res7",
    "contact_res1_res8", "hb_2ASPN_7THRO", "sc_dist_res0_res8",
    "saltbridge_Nterm_Cterm9",
]

REF_BASELINE_TRANSITION = 0.0305
REF_CHECK_C_TRANSITION = 0.3224
REF_CONFIDENCE_FILTERED = 0.3440
REF_PRUNED_FULL_TRANSITION = 0.1583
REF_PRUNED_TRANSONLY_TRANSITION = 0.3914
REF_CONFIDENCE_FILTERED_NEW_FULLTEST = 0.3333
REF_CONFIDENCE_FILTERED_NEW_HICONF = 0.4609
REF_PCA_BALANCED = -1.0377  # the result this script is isolating against

print(f"Known-predictive feature set ({len(FEATURE_COLS)}): {FEATURE_COLS}")

df = pd.read_csv(MERGED_CSV, usecols=["frame_index"] + FEATURE_COLS + ["Committor_prob"])
df = df.sort_values("frame_index").reset_index(drop=True)
n = len(df)
print(f"{n:,} rows")

# ------------------------------------------------------------------
# Identical split to the rest of the project.
# ------------------------------------------------------------------
y_raw = df["Committor_prob"].values
y_clipped = np.clip(y_raw, EPS, 1 - EPS)
y_logit = np.log(y_clipped / (1 - y_clipped))

block_size = n // N_BLOCKS
block_id = np.minimum(np.arange(n) // block_size, N_BLOCKS - 1)
rng = np.random.RandomState(RANDOM_STATE)
test_blocks = sorted(rng.choice(N_BLOCKS, size=N_TEST_BLOCKS, replace=False))
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

X = df[FEATURE_COLS]
y_raw_test = y_raw[idx_test]
mask_test_tr = (y_raw_test > TRANSITION_LOW) & (y_raw_test < TRANSITION_HIGH)
print(f"Transition-region test rows (unbalanced, as in every other retrain): {mask_test_tr.sum():,}")

# ------------------------------------------------------------------
# Balance the TRAINING set only -- IDENTICAL logic to
# retrain_balanced_pca_features.py (downsample end-state training rows
# to match the transition-region training row count). Test set untouched.
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
      f"({n_trans:,} transition-region + {n_end_state_keep:,} end-state, target ratio ~50/50)")


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


# ------------------------------------------------------------------
# Train on balanced + known-predictive-feature set; evaluate on the
# standard unbalanced test set.
# ------------------------------------------------------------------
print("\nTraining on balanced + known-predictive feature set...")
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
print(f"{'-' * 65} {'-' * 28}")
print(f"{'New physics feat (101), full training':65s} {REF_PRUNED_FULL_TRANSITION:>28.4f}")
print(f"{'New physics feat (101), transition-only training':65s} {REF_PRUNED_TRANSONLY_TRANSITION:>28.4f}")
print(f"{'New confidence-filtered, full test':65s} {REF_CONFIDENCE_FILTERED_NEW_FULLTEST:>28.4f}")
print(f"{'New confidence-filtered, hi-conf test':65s} {REF_CONFIDENCE_FILTERED_NEW_HICONF:>28.4f}")
print(f"{'Balanced + PCA-reduced features (10 feat) -- being isolated against':65s} {REF_PCA_BALANCED:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'THIS: balanced training + known-predictive features (10 feat)':65s} {r2_test_trans:>28.4f}")

feature_importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance.csv", header=["importance"])
print(f"\nFeature importance (known-predictive set, balanced training):")
print(feature_importance.to_string())

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Known-predictive feature set ({len(FEATURE_COLS)}): {FEATURE_COLS}\n\n")
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
    f.write(f"  New physics feat, full training:        {REF_PRUNED_FULL_TRANSITION:.4f}\n")
    f.write(f"  New physics feat, transition-only:      {REF_PRUNED_TRANSONLY_TRANSITION:.4f}\n")
    f.write(f"  New confidence-filtered, full test:     {REF_CONFIDENCE_FILTERED_NEW_FULLTEST:.4f}\n")
    f.write(f"  New confidence-filtered, hi-conf test:  {REF_CONFIDENCE_FILTERED_NEW_HICONF:.4f}\n")
    f.write(f"  Balanced + PCA-reduced (isolated against): {REF_PCA_BALANCED:.4f}\n")

print(f"\nSaved outputs to {OUT_DIR}")
