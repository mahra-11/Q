"""
Diagnoses WHY mixing end-state (folded/unfolded) frames back into
transition-region training hurts performance, rather than just observing
that it does. Scans the ratio of end-state-to-transition-region training
rows from 0 (pure transition-only, the best result so far: R^2=0.3266)
up through the full natural imbalance (~94x, equivalent to ordinary full
unbalanced training but with only 10 features), to see whether
performance degrades gradually or falls off a cliff at some specific
ratio.

Uses the same 10 features as retrain_known_features_transonly.py and
retrain_balanced_known_features.py (held fixed -- this sweep is only
about the training population, not feature selection, which was already
shown to matter much less).
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/sweep_balance_ratio/"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

# end-state : transition-region row ratio in the training set. 0 = pure
# transition-only (the 0.3266 reference). The largest value uses ALL
# available end-state rows (the full natural imbalance, ~94x).
RATIOS = [0, 0.5, 1, 2, 5, 10, 20, 50, "all"]

FEATURE_COLS = [
    "sc_dist_res1_res8", "sc_dist_res1_res7", "sc_dist_res1_res6",
    "sin_chi1_res1", "cos_chi1_res1", "contact_res1_res7",
    "contact_res1_res8", "hb_2ASPN_7THRO", "sc_dist_res0_res8",
    "saltbridge_Nterm_Cterm9",
]

REF_TRANSONLY_10FEAT = 0.3266  # ratio=0 should reproduce this exactly
REF_BALANCED_10FEAT = 0.0979   # ratio=1 should reproduce this exactly
REF_TRANSONLY_101FEAT = 0.3914
REF_PRUNED_FULL_101FEAT = 0.1583  # full natural imbalance, 101 features -- compare to ratio="all" with 10 features

print(f"Feature set ({len(FEATURE_COLS)}): {FEATURE_COLS}")

df = pd.read_csv(MERGED_CSV, usecols=["frame_index"] + FEATURE_COLS + ["Committor_prob"])
df = df.sort_values("frame_index").reset_index(drop=True)
n = len(df)
print(f"{n:,} rows")

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
    model.fit(X.iloc[train_idx], y_logit[train_idx])

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
results_df.to_csv(OUT_DIR + "balance_ratio_sweep.csv", index=False)

print("\n=== Full sweep ===")
print(results_df.to_string(index=False))

print("\n=== Sanity checks against known references ===")
print(f"ratio=0 should match REF_TRANSONLY_10FEAT={REF_TRANSONLY_10FEAT}: "
      f"got {results_df.loc[results_df['ratio'] == '0', 'r2_transition'].values[0]:.4f}")
print(f"ratio=1 should match REF_BALANCED_10FEAT={REF_BALANCED_10FEAT}: "
      f"got {results_df.loc[results_df['ratio'] == '1', 'r2_transition'].values[0]:.4f}")
print(f"(for reference: 101-feature transition-only={REF_TRANSONLY_101FEAT}, "
      f"101-feature full training={REF_PRUNED_FULL_101FEAT})")

print(f"\nSaved outputs to {OUT_DIR}")
