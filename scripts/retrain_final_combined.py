"""
Final consolidation: combines every literature-informed addition validated
so far into one feature set and retrains once, instead of juggling partial
combinations across separate jobs.

Stack (109 features total):
  101 base (pruned set)
  + 3 previously rescued (hb_5THRN_2ASPOD1, hb_5THRN_2ASPOD2, hb_6GLYN_2ASPO)
  + 1 newly rescued (hb_9TYRN_0TYRO, priority #6)
  + 1 computed in retrain_lit_asp2n_gly6o.py (lit_d_Asp2N_Gly6O, priority #3)
  + 3 computed in build_and_retrain_lit_priorities_4_5_7.py
    (lit_d_Tyr0O_Trp8O #4, lit_d_Tyr1N_Trp8O #5, lit_d_Tyr0_Asp2_closest_heavy #7)

Each of these was independently verified first (see retrain_rescued_features.py,
retrain_lit_asp2n_gly6o.py, build_and_retrain_lit_priorities_4_5_7.py) before
being stacked here. This script does no new MDTraj computation -- it only
merges the already-saved per-feature CSVs from those two jobs' output dirs.

Reports both full and transition-region-only training, directly comparable to
every prior step in the stack.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
PRUNED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"
ASP2N_GLY6O_CSV = "/scratch/mma9420/committor_check/Q/physics_features/retrain_lit_asp2n_gly6o/lit_d_Asp2N_Gly6O.csv"
PRIORITIES_4_5_7_CSV = "/scratch/mma9420/committor_check/Q/physics_features/retrain_lit_priorities_4_5_7/new_features.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_final_combined/"
import os
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
N_WEIGHT_BINS = 20
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

PREVIOUSLY_RESCUED = ["hb_5THRN_2ASPOD1", "hb_5THRN_2ASPOD2", "hb_6GLYN_2ASPO"]
NEWLY_RESCUED = ["hb_9TYRN_0TYRO"]
LIT_ASP2N_GLY6O = ["lit_d_Asp2N_Gly6O"]
LIT_PRIORITIES_4_5_7 = ["lit_d_Tyr0O_Trp8O", "lit_d_Tyr1N_Trp8O", "lit_d_Tyr0_Asp2_closest_heavy"]
ALL_NEW_FEATURES = PREVIOUSLY_RESCUED + NEWLY_RESCUED + LIT_ASP2N_GLY6O + LIT_PRIORITIES_4_5_7

REF_PRUNED_FULL = 0.1583
REF_PRUNED_TRANSONLY = 0.3914
REF_RESCUED_104_FULL = 0.1726
REF_RESCUED_104_TRANSONLY = 0.3969
REF_105_FULL = 0.1829
REF_105_TRANSONLY = 0.3956
REF_108_FULL = 0.1891
REF_108_TRANSONLY = 0.3979

pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
base_feature_cols = [c for c in pruned_cols if c != "Committor_prob"]

df = pd.read_csv(
    MERGED_CSV,
    usecols=["frame_index"] + base_feature_cols + PREVIOUSLY_RESCUED + NEWLY_RESCUED + ["Committor_prob"],
)
n_before = len(df)

asp2n_gly6o = pd.read_csv(ASP2N_GLY6O_CSV)
df = df.merge(asp2n_gly6o, on="frame_index", how="left")
assert len(df) == n_before

priorities_df = pd.read_csv(PRIORITIES_4_5_7_CSV)
df = df.merge(priorities_df, on="frame_index", how="left")
assert len(df) == n_before

feature_cols = base_feature_cols + ALL_NEW_FEATURES
print(f"Total feature set: {len(feature_cols)} features "
      f"({len(base_feature_cols)} base + {len(ALL_NEW_FEATURES)} literature-informed additions)")

for feat in LIT_ASP2N_GLY6O + LIT_PRIORITIES_4_5_7:
    n_matched = df[feat].notna().sum()
    print(f"{feat} join: {n_matched:,} / {len(df):,} matched ({100 * n_matched / len(df):.2f}%)")

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
    return model, r2_train, r2_test_overall, r2_test_tr


print(f"\n=== Final combined feature set ({len(feature_cols)}), all training rows ===")
model_full, r2tr_full, r2o_full, r2tr_test_full = fit_and_score(idx_train, "full")

print(f"\n=== Final combined feature set ({len(feature_cols)}), transition-region-only training ===")
train_tr_idx = idx_train[(y_raw[idx_train] > TRANSITION_LOW) & (y_raw[idx_train] < TRANSITION_HIGH)]
model_trans, r2tr_trans, r2o_trans, r2tr_test_trans = fit_and_score(train_tr_idx, "transition-region-only training")

print("\n=== Comparison (full stacking history) ===")
print(f"{'run':65s} {'transition-region test R^2':>28s}")
print(f"{'101 base, full training':65s} {REF_PRUNED_FULL:>28.4f}")
print(f"{'101 base, transition-only':65s} {REF_PRUNED_TRANSONLY:>28.4f}")
print(f"{'101+3 rescued (104), full training':65s} {REF_RESCUED_104_FULL:>28.4f}")
print(f"{'101+3 rescued (104), transition-only':65s} {REF_RESCUED_104_TRANSONLY:>28.4f}")
print(f"{'+lit_d_Asp2N_Gly6O (105), full training':65s} {REF_105_FULL:>28.4f}")
print(f"{'+lit_d_Asp2N_Gly6O (105), transition-only':65s} {REF_105_TRANSONLY:>28.4f}")
print(f"{'+#6 rescued +#4/#5/#7 computed (108), full training':65s} {REF_108_FULL:>28.4f}")
print(f"{'+#6 rescued +#4/#5/#7 computed (108), transition-only':65s} {REF_108_TRANSONLY:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'FINAL: %d features, full training' % len(feature_cols):65s} {r2tr_test_full:>28.4f}")
print(f"{'FINAL: %d features, transition-only training' % len(feature_cols):65s} {r2tr_test_trans:>28.4f}")

feature_importance = pd.Series(model_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance_full_model.csv", header=["importance"])
rank = feature_importance.reset_index()
rank.columns = ["feature", "importance"]
print(f"\nAll literature-informed feature ranks (out of {len(feature_cols)}):")
for feat in ALL_NEW_FEATURES:
    r = rank.index[rank["feature"] == feat].tolist()
    print(f"  {feat}: #{r[0] + 1}" if r else f"  {feat}: not used in any split")

print("\nTop 15 features overall (full model):")
print(feature_importance.head(15).to_string())

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Final combined feature set: {len(feature_cols)} features\n")
    f.write(f"Literature-informed additions: {ALL_NEW_FEATURES}\n\n")
    f.write(f"Full training: transition-region R^2 = {r2tr_test_full:.4f}\n")
    f.write(f"Transition-only training: transition-region R^2 = {r2tr_test_trans:.4f}\n")

print(f"\nSaved outputs to {OUT_DIR}")
