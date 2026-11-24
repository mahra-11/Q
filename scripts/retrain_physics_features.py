"""
Step 8: retrain on the pruned physics feature set (101 features, after the
both-passes-must-agree Pearson pruning fix), using the same setup as every
other retrain in this project -- logit-transformed target, inverse-frequency
sample weighting, and the contiguous time-block train/test split that became
the standard after the original random-split baseline was found to risk
leakage via temporal autocorrelation between adjacent frames.

Reports two numbers to compare against the project's references:
  - full pruned feature set (this is the one to beat: baseline 0.0305,
    Check C 0.3224/0.306, confidence-filtered 0.344 -- all on the OLD
    64-feature set, all without Q as an input)
  - trained on transition-region-only rows (mirrors "Check C"), evaluated
    on the same held-out test set as the full model, restricted to its
    transition-region rows

Does NOT include the third historical variant (transition-region +
n_visits >= 84, the high-confidence subset) -- that needs an extra join
back to committor_grid_visits.csv via reconstructed Rg/Q bins that isn't
part of this pruned dataset. Left as a follow-up rather than risking
another subtle alignment bug in this script.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
PRUNED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain/"

os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
N_WEIGHT_BINS = 20
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

# References from the old 64-feature set (no Q as input), for comparison:
REF_BASELINE_TRANSITION = 0.0305      # individual-frame regression, block split
REF_CHECK_C_TRANSITION = 0.3224       # train on transition-region-only, block split
REF_CONFIDENCE_FILTERED = 0.344       # transition-region + n_visits >= 84, high-confidence half

# --- Load the pruned feature set, but keep frame_index (dropped from the pruned
#     file itself) by pulling the same final column set out of the unpruned
#     merged file, which still has it. ---
pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
feature_cols = [c for c in pruned_cols if c != "Committor_prob"]
print(f"Pruned feature set: {len(feature_cols)} features")

print(f"Loading {MERGED_CSV}")
df = pd.read_csv(MERGED_CSV, usecols=["frame_index"] + feature_cols + ["Committor_prob"])
df = df.sort_values("frame_index").reset_index(drop=True)
n = len(df)
print(f"{n:,} rows")

y_raw = df["Committor_prob"].values
y_clipped = np.clip(y_raw, EPS, 1 - EPS)
y_logit = np.log(y_clipped / (1 - y_clipped))

bin_edges = np.linspace(0, 1, N_WEIGHT_BINS + 1)
bin_idx = np.clip(np.digitize(y_raw, bin_edges) - 1, 0, N_WEIGHT_BINS - 1)
bin_counts = np.bincount(bin_idx, minlength=N_WEIGHT_BINS)
weights = np.array([1.0 / bin_counts[b] for b in bin_idx])
weights = weights / weights.mean()

# --- Contiguous time-block split (10 blocks, 2 held out as test, 2000-frame
#     buffer around test-block boundaries) -- the project's standard scheme
#     since the original random split was found to risk leakage via
#     temporal autocorrelation between adjacent frames. ---
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

print(f"Train: {is_train.sum():,}  Test: {is_test.sum():,}  Buffer (excluded): {is_buffer.sum():,}")

idx_train = np.where(is_train)[0]
idx_test = np.where(is_test)[0]

X = df[feature_cols]
y_raw_test = y_raw[idx_test]
mask_test_tr = (y_raw_test > TRANSITION_LOW) & (y_raw_test < TRANSITION_HIGH)
print(f"Transition-region test rows: {mask_test_tr.sum():,}")


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


def fit_and_score(train_idx, label):
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=2
    )
    model.fit(X.iloc[train_idx], y_logit[train_idx], sample_weight=weights[train_idx])

    pred_train = inv_logit(model.predict(X.iloc[train_idx]))
    pred_test = inv_logit(model.predict(X.iloc[idx_test]))

    r2_train = r2_score(y_raw[train_idx], pred_train)
    r2_test_overall = r2_score(y_raw_test, pred_test)
    r2_test_tr = r2_score(y_raw_test[mask_test_tr], pred_test[mask_test_tr])

    print(f"\n[{label}] train rows: {len(train_idx):,}")
    print(f"  Train R^2:                  {r2_train:.4f}")
    print(f"  Test overall R^2:           {r2_test_overall:.4f}")
    print(f"  Test transition-region R^2: {r2_test_tr:.4f}")
    return model, pred_test, r2_train, r2_test_overall, r2_test_tr


print("\n=== Full pruned feature set, all training rows ===")
model_full, pred_full, r2tr_full, r2o_full, r2tr_test_full = fit_and_score(idx_train, "full")

print("\n=== Trained on transition-region-only rows (mirrors Check C) ===")
train_tr_idx = idx_train[(y_raw[idx_train] > TRANSITION_LOW) & (y_raw[idx_train] < TRANSITION_HIGH)]
model_trans, pred_trans, r2tr_trans, r2o_trans, r2tr_test_trans = fit_and_score(
    train_tr_idx, "transition-region-only training"
)

print("\n=== Comparison against references (old 64-feature set, no Q as input) ===")
print(f"{'run':45s} {'transition-region test R^2':>28s}")
print(f"{'baseline (individual-frame, block split)':45s} {REF_BASELINE_TRANSITION:>28.4f}")
print(f"{'Check C (transition-only train, block split)':45s} {REF_CHECK_C_TRANSITION:>28.4f}")
print(f"{'confidence-filtered (n_visits>=84 subset)':45s} {REF_CONFIDENCE_FILTERED:>28.4f}")
print(f"{'-'*45} {'-'*28}")
print(f"{'THIS: full pruned physics feature set':45s} {r2tr_test_full:>28.4f}")
print(f"{'THIS: transition-region-only training':45s} {r2tr_test_trans:>28.4f}")

feature_importance = pd.Series(model_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance_full_model.csv", header=["importance"])
print(f"\nTop 15 features by importance (full model):")
print(feature_importance.head(15).to_string())

test_df = df.iloc[idx_test].copy()
test_df["pred_full"] = pred_full
# predictions from the transition-only-trained model, for every test row (not just transition ones)
test_df["pred_transition_only_train"] = inv_logit(model_trans.predict(X.iloc[idx_test]))
test_df[["frame_index", "Committor_prob", "pred_full", "pred_transition_only_train"]].to_csv(
    OUT_DIR + "test_predictions.csv", index=False
)

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Pruned feature set: {len(feature_cols)} features\n")
    f.write(f"Train: {is_train.sum():,}  Test: {is_test.sum():,}  Buffer: {is_buffer.sum():,}\n")
    f.write(f"Test blocks: {test_blocks}\n\n")
    f.write("Full pruned feature set:\n")
    f.write(f"  Train R^2:                  {r2tr_full:.4f}\n")
    f.write(f"  Test overall R^2:           {r2o_full:.4f}\n")
    f.write(f"  Test transition-region R^2: {r2tr_test_full:.4f}\n\n")
    f.write("Transition-region-only training:\n")
    f.write(f"  Train R^2:                  {r2tr_trans:.4f}\n")
    f.write(f"  Test overall R^2:           {r2o_trans:.4f}\n")
    f.write(f"  Test transition-region R^2: {r2tr_test_trans:.4f}\n\n")
    f.write("References (old 64-feature set, no Q as input):\n")
    f.write(f"  Baseline (block split):                    {REF_BASELINE_TRANSITION:.4f}\n")
    f.write(f"  Check C (transition-only train, block split): {REF_CHECK_C_TRANSITION:.4f}\n")
    f.write(f"  Confidence-filtered (n_visits>=84):        {REF_CONFIDENCE_FILTERED:.4f}\n")

print(f"\nSaved outputs to {OUT_DIR}")
