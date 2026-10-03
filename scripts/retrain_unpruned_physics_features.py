"""
Step 8 variant: retrain on ALL candidate physics features (no Pearson
pruning at all), to directly test whether the pruning step is costing
performance rather than helping. Same methodology as
retrain_physics_features.py (logit target, inverse-frequency weighting,
contiguous time-block split, same block-selection random seed) so the two
runs are a clean apples-to-apples comparison -- only the feature set
differs.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_unpruned/"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
N_WEIGHT_BINS = 20
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

LEAKY_AND_LABEL_COLS = ["frame_index", "radius_gyration", "rmsd_to_folded", "Q",
                         "rg_bin", "q_bin", "cell_id", "Committor_prob"]

# References for comparison -- old 64-feature set (no Q as input), and the
# already-completed run on the 101-feature PRUNED physics set:
REF_BASELINE_TRANSITION = 0.0305
REF_CHECK_C_TRANSITION = 0.3224
REF_CONFIDENCE_FILTERED = 0.344
REF_PRUNED_FULL_TRANSITION = 0.1583       # 101 pruned features, full training
REF_PRUNED_TRANSONLY_TRANSITION = 0.3914  # 101 pruned features, transition-only training

print(f"Loading {MERGED_CSV}")
df = pd.read_csv(MERGED_CSV)
df = df.sort_values("frame_index").reset_index(drop=True)
n = len(df)
print(f"{n:,} rows, {df.shape[1]} total columns")

candidate_cols = [c for c in df.columns if c not in LEAKY_AND_LABEL_COLS]
nunique = df[candidate_cols].nunique()
zero_var = nunique[nunique <= 1].index.tolist()
if zero_var:
    print(f"Dropping {len(zero_var)} zero-variance columns: {zero_var}")
feature_cols = [c for c in candidate_cols if c not in zero_var]
print(f"Using ALL {len(feature_cols)} candidate features (no Pearson pruning)")

y_raw = df["Committor_prob"].values
y_clipped = np.clip(y_raw, EPS, 1 - EPS)
y_logit = np.log(y_clipped / (1 - y_clipped))

bin_edges = np.linspace(0, 1, N_WEIGHT_BINS + 1)
bin_idx = np.clip(np.digitize(y_raw, bin_edges) - 1, 0, N_WEIGHT_BINS - 1)
bin_counts = np.bincount(bin_idx, minlength=N_WEIGHT_BINS)
weights = np.array([1.0 / bin_counts[b] for b in bin_idx])
weights = weights / weights.mean()

# --- Same contiguous time-block split as retrain_physics_features.py,
#     same random seed, so the test set is identical between the two runs. ---
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


print("\n=== ALL candidate features (no pruning), all training rows ===")
model_full, pred_full, r2tr_full, r2o_full, r2tr_test_full = fit_and_score(idx_train, "full, unpruned")

print("\n=== ALL candidate features (no pruning), trained on transition-region-only rows ===")
train_tr_idx = idx_train[(y_raw[idx_train] > TRANSITION_LOW) & (y_raw[idx_train] < TRANSITION_HIGH)]
model_trans, pred_trans, r2tr_trans, r2o_trans, r2tr_test_trans = fit_and_score(
    train_tr_idx, "transition-region-only training, unpruned"
)

print("\n=== Comparison: does pruning help or hurt? ===")
print(f"{'run':50s} {'transition-region test R^2':>28s}")
print(f"{'old baseline (64 features, block split)':50s} {REF_BASELINE_TRANSITION:>28.4f}")
print(f"{'old Check C (64 features, transition-only train)':50s} {REF_CHECK_C_TRANSITION:>28.4f}")
print(f"{'old confidence-filtered (n_visits>=84)':50s} {REF_CONFIDENCE_FILTERED:>28.4f}")
print(f"{'-'*50} {'-'*28}")
print(f"{'PRUNED (101 feat), full training':50s} {REF_PRUNED_FULL_TRANSITION:>28.4f}")
print(f"{'PRUNED (101 feat), transition-only training':50s} {REF_PRUNED_TRANSONLY_TRANSITION:>28.4f}")
print(f"{'-'*50} {'-'*28}")
print(f"{'THIS: UNPRUNED (%d feat), full training' % len(feature_cols):50s} {r2tr_test_full:>28.4f}")
print(f"{'THIS: UNPRUNED (%d feat), transition-only training' % len(feature_cols):50s} {r2tr_test_trans:>28.4f}")

feature_importance = pd.Series(model_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance_full_model.csv", header=["importance"])
print(f"\nTop 15 features by importance (full model, unpruned):")
print(feature_importance.head(15).to_string())

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"ALL candidate features (no pruning): {len(feature_cols)} features\n")
    f.write(f"Train: {is_train.sum():,}  Test: {is_test.sum():,}  Buffer: {is_buffer.sum():,}\n")
    f.write(f"Test blocks: {test_blocks}\n\n")
    f.write("Full unpruned feature set:\n")
    f.write(f"  Train R^2:                  {r2tr_full:.4f}\n")
    f.write(f"  Test overall R^2:           {r2o_full:.4f}\n")
    f.write(f"  Test transition-region R^2: {r2tr_test_full:.4f}\n\n")
    f.write("Transition-region-only training, unpruned:\n")
    f.write(f"  Train R^2:                  {r2tr_trans:.4f}\n")
    f.write(f"  Test overall R^2:           {r2o_trans:.4f}\n")
    f.write(f"  Test transition-region R^2: {r2tr_test_trans:.4f}\n\n")
    f.write("Comparison references:\n")
    f.write(f"  Old baseline (64 features):              {REF_BASELINE_TRANSITION:.4f}\n")
    f.write(f"  Old Check C (64 features):                {REF_CHECK_C_TRANSITION:.4f}\n")
    f.write(f"  Old confidence-filtered:                  {REF_CONFIDENCE_FILTERED:.4f}\n")
    f.write(f"  Pruned (101 features), full training:     {REF_PRUNED_FULL_TRANSITION:.4f}\n")
    f.write(f"  Pruned (101 features), transition-only:   {REF_PRUNED_TRANSONLY_TRANSITION:.4f}\n")

print(f"\nSaved outputs to {OUT_DIR}")
