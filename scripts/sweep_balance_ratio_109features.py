"""
Re-runs the balance-ratio sweep (end-state:transition resampling ratio,
with the project's standard inverse-frequency sample weighting) on the
current best 109-feature set, instead of the old 10-feature set it was
originally tested on.

Context: sweep_balance_ratio_weighted.py found that, on the old 10-feature
set, weighted training was roughly flat across ratios 0-10 (0.30-0.35),
peaking slightly at ratio=1 (0.346), and only degraded at the extreme
ratio="all" (0.047). But "full training" in every later script (104, 105,
108, 109 features) is effectively ratio="all" (every available row, no
resampling) -- and with the richer 109-feature set, that "full training"
result (0.1909) is actually BETTER than "transition-only" (ratio=0) used
to be relative to the 10-feature set's own numbers, suggesting the
ratio="all" degradation seen on the 10-feature set may have been a
weak-feature-set artifact, not a general problem with using all the data.
This sweep checks directly whether that holds on the 109-feature set: does
some intermediate ratio actually beat both ratio=0 (transition-only,
0.4040) and ratio="all" (full training, 0.1909)?

Same resampling logic as sweep_balance_ratio_weighted.py: weights are
computed once from the full dataset, then subset to whichever training
rows a given ratio uses, so results stay directly comparable.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
PRUNED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"
ASP2N_GLY6O_CSV = "/scratch/mma9420/committor_check/Q/physics_features/retrain_lit_asp2n_gly6o/lit_d_Asp2N_Gly6O.csv"
PRIORITIES_4_5_7_CSV = "/scratch/mma9420/committor_check/Q/physics_features/retrain_lit_priorities_4_5_7/new_features.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/sweep_balance_ratio_109features/"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
N_WEIGHT_BINS = 20
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

RATIOS = [0, 0.5, 1, 2, 5, 10, 20, 50, "all"]

PREVIOUSLY_RESCUED = ["hb_5THRN_2ASPOD1", "hb_5THRN_2ASPOD2", "hb_6GLYN_2ASPO"]
NEWLY_RESCUED = ["hb_9TYRN_0TYRO"]
LIT_ASP2N_GLY6O = ["lit_d_Asp2N_Gly6O"]
LIT_PRIORITIES_4_5_7 = ["lit_d_Tyr0O_Trp8O", "lit_d_Tyr1N_Trp8O", "lit_d_Tyr0_Asp2_closest_heavy"]
ALL_NEW_FEATURES = PREVIOUSLY_RESCUED + NEWLY_RESCUED + LIT_ASP2N_GLY6O + LIT_PRIORITIES_4_5_7

REF_109_FULL_RATIO_ALL_EQUIV = 0.1909   # from retrain_final_combined.py, "full training"
REF_109_RATIO0_EQUIV = 0.4040           # from retrain_final_combined.py, "transition-only training"
REF_10FEAT_RATIO1_WEIGHTED = 0.346      # from sweep_balance_ratio_weighted.py, old 10-feature set

pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
base_feature_cols = [c for c in pruned_cols if c != "Committor_prob"]
FEATURE_COLS = base_feature_cols + ALL_NEW_FEATURES
print(f"Feature set: {len(FEATURE_COLS)} features")

df = pd.read_csv(
    MERGED_CSV,
    usecols=["frame_index"] + base_feature_cols + PREVIOUSLY_RESCUED + NEWLY_RESCUED + ["Committor_prob"],
)
n_before = len(df)
df = df.merge(pd.read_csv(ASP2N_GLY6O_CSV), on="frame_index", how="left")
df = df.merge(pd.read_csv(PRIORITIES_4_5_7_CSV), on="frame_index", how="left")
assert len(df) == n_before
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

X = df[FEATURE_COLS]
y_raw_test = y_raw[idx_test]
mask_test_tr = (y_raw_test > TRANSITION_LOW) & (y_raw_test < TRANSITION_HIGH)
print(f"Transition-region test rows: {mask_test_tr.sum():,}")

y_raw_train = y_raw[idx_train]
is_trans_train = (y_raw_train > TRANSITION_LOW) & (y_raw_train < TRANSITION_HIGH)
trans_train_idx = idx_train[is_trans_train]
end_state_train_idx = idx_train[~is_trans_train]
n_trans = len(trans_train_idx)
n_end_state_available = len(end_state_train_idx)
print(f"Transition-region training rows: {n_trans:,}")
print(f"End-state training rows available: {n_end_state_available:,}")


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


results = []
for ratio in RATIOS:
    if ratio == "all":
        n_end_state_keep = n_end_state_available
        label = "all"
    else:
        n_end_state_keep = min(int(round(ratio * n_trans)), n_end_state_available)
        label = str(ratio)

    rng_ratio = np.random.RandomState(RANDOM_STATE)
    if n_end_state_keep > 0:
        end_state_keep_idx = rng_ratio.choice(end_state_train_idx, size=n_end_state_keep, replace=False)
        train_idx = np.concatenate([trans_train_idx, end_state_keep_idx])
    else:
        train_idx = trans_train_idx
    rng_ratio.shuffle(train_idx)

    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=2
    )
    model.fit(X.iloc[train_idx], y_logit[train_idx], sample_weight=weights[train_idx])

    pred_test = inv_logit(model.predict(X.iloc[idx_test]))
    r2_overall = r2_score(y_raw_test, pred_test)
    r2_trans = r2_score(y_raw_test[mask_test_tr], pred_test[mask_test_tr])

    print(f"\nratio={label}: train={len(train_idx):,} "
          f"({n_trans:,} transition + {n_end_state_keep:,} end-state) "
          f"-> overall R^2={r2_overall:.4f}, transition R^2={r2_trans:.4f}")

    results.append({
        "ratio": label,
        "n_train": len(train_idx),
        "n_transition": n_trans,
        "n_end_state": n_end_state_keep,
        "end_state_fraction": n_end_state_keep / len(train_idx),
        "r2_overall": r2_overall,
        "r2_transition": r2_trans,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR + "balance_ratio_sweep_109features.csv", index=False)

print("\n=== Full sweep on 109-feature set ===")
print(results_df.to_string(index=False))

best_row = results_df.loc[results_df["r2_transition"].idxmax()]
print(f"\nBest ratio: {best_row['ratio']} (transition R^2 = {best_row['r2_transition']:.4f})")

print("\n=== Comparison ===")
print(f"ratio=0 (this sweep, 109 feat)    = {results_df.loc[results_df['ratio'] == '0', 'r2_transition'].values[0]:.4f} "
      f"  vs. transition-only training (retrain_final_combined.py) = {REF_109_RATIO0_EQUIV:.4f}")
print(f"ratio=all (this sweep, 109 feat)  = {results_df.loc[results_df['ratio'] == 'all', 'r2_transition'].values[0]:.4f} "
      f"  vs. full training (retrain_final_combined.py)            = {REF_109_FULL_RATIO_ALL_EQUIV:.4f}")
print(f"ratio=1 (this sweep, 109 feat)    = {results_df.loc[results_df['ratio'] == '1', 'r2_transition'].values[0]:.4f} "
      f"  vs. ratio=1 on the old 10-feature set                    = {REF_10FEAT_RATIO1_WEIGHTED:.4f}")

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Feature set: {len(FEATURE_COLS)} features (same as retrain_final_combined.py)\n\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nBest ratio: {best_row['ratio']} (transition R^2 = {best_row['r2_transition']:.4f})\n")

print(f"\nSaved outputs to {OUT_DIR}")
