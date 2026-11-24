"""
Rescues 3 features that were legitimately pruned by the both-passes-must-
agree rule (correlated with something in both the all-frames and
transition-region views, just different partners in each) but which
independent literature (3 papers studying CLN025 committors/folding
mechanism specifically) identifies as primary transition-state-defining
coordinates:

  hb_5THRN_2ASPOD1, hb_5THRN_2ASPOD2  -- Asp2 side-chain O to Thr5
    backbone N: the "TS-up" transition-state hydrogen bond (Kang, Trizio
    & Parrinello 2024; pairwise-interaction committor model); dropped for
    correlating with saltbridge_Nterm_Cterm9/cos_psi_res3 (all-frames) and
    hb_4GLUN_2ASPOD1/2 (transition-region) -- plausible but not literature-
    informed redundancy calls.
  hb_6GLYN_2ASPO  -- Asp2 backbone O to Gly6 backbone N: a turn-forming
    hydrogen bond reported to rise from ~6% (unfolded) to ~48% (transition
    state) to ~72% (folded) occupancy (Sobieraj & Setny 2022, Granger
    causality); dropped for the same kind of reasons.

These are re-added as EXTRA columns on top of the existing 101-feature
pruned set (104 total), not a replacement -- see the chat discussion:
nothing supports abandoning the already-validated 101-feature set, only
supplementing it with specific, literature-flagged coordinates that
correlation pruning discarded for reasons unrelated to physical
importance. No new MDTraj computation needed -- these columns already
exist in the unpruned merged CSV, just excluded from the pruned file.

Reports both full-training and transition-region-only-training variants,
directly comparable to retrain_physics_features.py's 0.1583 / 0.3914
(same methodology, same split, same weighting), to isolate whether these
3 rescued features add anything on top of the existing 101.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
PRUNED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_rescued/"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
N_WEIGHT_BINS = 20
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

RESCUED_FEATURES = ["hb_5THRN_2ASPOD1", "hb_5THRN_2ASPOD2", "hb_6GLYN_2ASPO"]

REF_BASELINE_TRANSITION = 0.0305
REF_CHECK_C_TRANSITION = 0.3224
REF_CONFIDENCE_FILTERED = 0.3440
REF_PRUNED_FULL_TRANSITION = 0.1583       # 101 features, full training -- isolating against
REF_PRUNED_TRANSONLY_TRANSITION = 0.3914  # 101 features, transition-only training -- isolating against
REF_CONFIDENCE_FILTERED_NEW_FULLTEST = 0.3333
REF_CONFIDENCE_FILTERED_NEW_HICONF = 0.4609

pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
base_feature_cols = [c for c in pruned_cols if c != "Committor_prob"]
print(f"Base pruned feature set: {len(base_feature_cols)} features")

feature_cols = base_feature_cols + RESCUED_FEATURES
print(f"Rescued features being added: {RESCUED_FEATURES}")
print(f"Total feature set: {len(feature_cols)} features")

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


print("\n=== Full feature set (101 + 3 rescued), all training rows ===")
model_full, pred_full, r2tr_full, r2o_full, r2tr_test_full = fit_and_score(idx_train, "full")

print("\n=== Full feature set (101 + 3 rescued), transition-region-only training ===")
train_tr_idx = idx_train[(y_raw[idx_train] > TRANSITION_LOW) & (y_raw[idx_train] < TRANSITION_HIGH)]
model_trans, pred_trans, r2tr_trans, r2o_trans, r2tr_test_trans = fit_and_score(
    train_tr_idx, "transition-region-only training"
)

print("\n=== Comparison ===")
print(f"{'run':65s} {'transition-region test R^2':>28s}")
print(f"{'Old baseline (64 features, full training)':65s} {REF_BASELINE_TRANSITION:>28.4f}")
print(f"{'Old Check C (64 features, transition-only train)':65s} {REF_CHECK_C_TRANSITION:>28.4f}")
print(f"{'Old confidence-filtered (n_visits>=84)':65s} {REF_CONFIDENCE_FILTERED:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'New physics feat (101), full training -- isolating against':65s} {REF_PRUNED_FULL_TRANSITION:>28.4f}")
print(f"{'New physics feat (101), transition-only -- isolating against':65s} {REF_PRUNED_TRANSONLY_TRANSITION:>28.4f}")
print(f"{'New confidence-filtered, full test':65s} {REF_CONFIDENCE_FILTERED_NEW_FULLTEST:>28.4f}")
print(f"{'New confidence-filtered, hi-conf test':65s} {REF_CONFIDENCE_FILTERED_NEW_HICONF:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'THIS: 101+3 rescued, full training':65s} {r2tr_test_full:>28.4f}")
print(f"{'THIS: 101+3 rescued, transition-only training':65s} {r2tr_test_trans:>28.4f}")

feature_importance = pd.Series(model_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance_full_model.csv", header=["importance"])
print(f"\nTop 15 features by importance (full model, 101+3 rescued):")
print(feature_importance.head(15).to_string())
print(f"\nRescued feature ranks (out of {len(feature_cols)}):")
rank_series = feature_importance.reset_index()
rank_series.columns = ["feature", "importance"]
for feat in RESCUED_FEATURES:
    rank = rank_series.index[rank_series["feature"] == feat].tolist()
    rank_str = f"#{rank[0] + 1}" if rank else "not used in any split"
    print(f"  {feat}: {rank_str}")

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Rescued features added: {RESCUED_FEATURES}\n")
    f.write(f"Total feature set: {len(feature_cols)} features\n\n")
    f.write("Full feature set (101+3), full training:\n")
    f.write(f"  Train R^2:                  {r2tr_full:.4f}\n")
    f.write(f"  Test overall R^2:           {r2o_full:.4f}\n")
    f.write(f"  Test transition-region R^2: {r2tr_test_full:.4f}\n\n")
    f.write("Full feature set (101+3), transition-region-only training:\n")
    f.write(f"  Train R^2:                  {r2tr_trans:.4f}\n")
    f.write(f"  Test overall R^2:           {r2o_trans:.4f}\n")
    f.write(f"  Test transition-region R^2: {r2tr_test_trans:.4f}\n\n")
    f.write("References:\n")
    f.write(f"  New physics feat (101), full training:        {REF_PRUNED_FULL_TRANSITION:.4f}\n")
    f.write(f"  New physics feat (101), transition-only:      {REF_PRUNED_TRANSONLY_TRANSITION:.4f}\n")
    f.write(f"  New confidence-filtered, full test:           {REF_CONFIDENCE_FILTERED_NEW_FULLTEST:.4f}\n")
    f.write(f"  New confidence-filtered, hi-conf test:        {REF_CONFIDENCE_FILTERED_NEW_HICONF:.4f}\n")

print(f"\nSaved outputs to {OUT_DIR}")
