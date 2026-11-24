"""
Second isolation test, holding feature count fixed and changing only the
training population (the opposite of the previous script, which held
population fixed and changed features).

Trains on transition-region rows ONLY (no end-state frames at all -- not
a 50/50 balance, a pure exclusion, mirroring the project's established
"Check C" / "transition-only training" methodology from
retrain_physics_features.py), but using only the 10 known-predictive
features instead of all 101.

Purpose: retrain_balanced_known_features.py (known-predictive features +
50/50 balanced training, i.e. transition rows PLUS an equal number of
end-state rows) scored transition-region R^2 = 0.0979 -- far below the
already-established 0.3914 from training on transition-only rows with
the FULL 101-feature set. Two things changed between those two results
(101->10 features, and adding end-state frames back in), so it's not
clear which one cost the performance. This script changes only the
feature count (101->10) while keeping the training population identical
to the 0.3914 reference (transition-only, no end-state frames) -- if
this lands near 0.39, the feature cut was free and the 50/50 balance
step is what hurt; if it lands well below 0.39, cutting to 10 features
costs real signal on its own.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_known_features_transonly/"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

# Identical 10-feature set to retrain_balanced_known_features.py -- Step 8's
# own top-10 by feature_importances_.
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
REF_PRUNED_TRANSONLY_TRANSITION = 0.3914  # the exact reference being isolated against: 101 feat, transition-only train
REF_CONFIDENCE_FILTERED_NEW_FULLTEST = 0.3333
REF_CONFIDENCE_FILTERED_NEW_HICONF = 0.4609
REF_PCA_BALANCED = -1.0377
REF_KNOWN_FEATURES_BALANCED = 0.0979  # the other half of this isolation pair

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
print(f"Transition-region test rows: {mask_test_tr.sum():,}")

# ------------------------------------------------------------------
# Transition-region-ONLY training rows -- no end-state frames at all,
# identical population logic to retrain_physics_features.py's
# "transition-region-only training" variant (the 0.3914 reference).
# ------------------------------------------------------------------
y_raw_train = y_raw[idx_train]
train_tr_idx = idx_train[(y_raw_train > TRANSITION_LOW) & (y_raw_train < TRANSITION_HIGH)]
print(f"\nTransition-region-only training rows: {len(train_tr_idx):,} "
      f"(should match the 4,406 used in retrain_balanced_known_features.py's balanced set)")


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


# ------------------------------------------------------------------
# Train on transition-only rows, known-predictive 10-feature set;
# evaluate on the standard test set.
# ------------------------------------------------------------------
print("\nTraining on transition-only rows, known-predictive feature set...")
model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=2
)
model.fit(X.iloc[train_tr_idx], y_logit[train_tr_idx])

pred_train = inv_logit(model.predict(X.iloc[train_tr_idx]))
pred_test = inv_logit(model.predict(X.iloc[idx_test]))

r2_train = r2_score(y_raw[train_tr_idx], pred_train)
r2_test_overall = r2_score(y_raw_test, pred_test)
r2_test_trans = r2_score(y_raw_test[mask_test_tr], pred_test[mask_test_tr])

print(f"\nOwn (transition-only) training R^2: {r2_train:.4f}")
print(f"Test overall R^2: {r2_test_overall:.4f}")
print(f"Test transition-region R^2: {r2_test_trans:.4f}")

print("\n=== Comparison ===")
print(f"{'run':70s} {'transition-region test R^2':>28s}")
print(f"{'Old baseline (64 features, full training)':70s} {REF_BASELINE_TRANSITION:>28.4f}")
print(f"{'Old Check C (64 features, transition-only train)':70s} {REF_CHECK_C_TRANSITION:>28.4f}")
print(f"{'Old confidence-filtered (n_visits>=84)':70s} {REF_CONFIDENCE_FILTERED:>28.4f}")
print(f"{'-' * 70} {'-' * 28}")
print(f"{'New physics feat (101), full training':70s} {REF_PRUNED_FULL_TRANSITION:>28.4f}")
print(f"{'New physics feat (101), transition-only training -- isolating against':70s} {REF_PRUNED_TRANSONLY_TRANSITION:>28.4f}")
print(f"{'New confidence-filtered, full test':70s} {REF_CONFIDENCE_FILTERED_NEW_FULLTEST:>28.4f}")
print(f"{'New confidence-filtered, hi-conf test':70s} {REF_CONFIDENCE_FILTERED_NEW_HICONF:>28.4f}")
print(f"{'Balanced + PCA-reduced features (10 feat)':70s} {REF_PCA_BALANCED:>28.4f}")
print(f"{'Balanced + known-predictive features (10 feat) -- other isolation half':70s} {REF_KNOWN_FEATURES_BALANCED:>28.4f}")
print(f"{'-' * 70} {'-' * 28}")
print(f"{'THIS: transition-only training, known-predictive features (10 feat)':70s} {r2_test_trans:>28.4f}")

feature_importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance.csv", header=["importance"])
print(f"\nFeature importance:")
print(feature_importance.to_string())

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Known-predictive feature set ({len(FEATURE_COLS)}): {FEATURE_COLS}\n\n")
    f.write(f"Transition-region-only training rows: {len(train_tr_idx):,}\n")
    f.write(f"Test: {is_test.sum():,} rows ({mask_test_tr.sum():,} transition-region)\n\n")
    f.write(f"Own (transition-only) training R^2: {r2_train:.4f}\n")
    f.write(f"Test overall R^2: {r2_test_overall:.4f}\n")
    f.write(f"Test transition-region R^2: {r2_test_trans:.4f}\n\n")
    f.write("References:\n")
    f.write(f"  Old baseline:                                {REF_BASELINE_TRANSITION:.4f}\n")
    f.write(f"  Old Check C:                                 {REF_CHECK_C_TRANSITION:.4f}\n")
    f.write(f"  Old confidence-filtered:                     {REF_CONFIDENCE_FILTERED:.4f}\n")
    f.write(f"  New physics feat, full training:             {REF_PRUNED_FULL_TRANSITION:.4f}\n")
    f.write(f"  New physics feat, transition-only (101 feat): {REF_PRUNED_TRANSONLY_TRANSITION:.4f}\n")
    f.write(f"  New confidence-filtered, full test:          {REF_CONFIDENCE_FILTERED_NEW_FULLTEST:.4f}\n")
    f.write(f"  New confidence-filtered, hi-conf test:       {REF_CONFIDENCE_FILTERED_NEW_HICONF:.4f}\n")
    f.write(f"  Balanced + PCA-reduced (10 feat):            {REF_PCA_BALANCED:.4f}\n")
    f.write(f"  Balanced + known-predictive (10 feat):       {REF_KNOWN_FEATURES_BALANCED:.4f}\n")

print(f"\nSaved outputs to {OUT_DIR}")
