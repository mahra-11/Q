"""
Literature-informed feature reduction.

Two prior attempts at cutting the feature set to ~10 both performed worse than
the full set: PCA-selected 10 (R^2=-1.0377) and raw-importance-selected top-10
(R^2=0.0979 full / 0.3266 transition-only), vs. today's full 109-feature stack
(R^2=0.1909 / 0.4040). Automated selection (PCA loadings, or raw importance
alone) has not worked for this problem. This script instead anchors the
reduced set to literature-confirmed physical relevance first, fills remaining
slots with the highest-ranked remaining features second, then runs a
correlation-redundancy pass (the project's standard both-passes-must-agree
rule: a pair must be correlated > 0.8 in BOTH the all-frames view AND the
transition-region-only view before either is dropped) -- the same reduction
instruction the PI gave (correlation + "PCA"), but anchored to domain
knowledge instead of blind automated loadings.

LITERATURE_CORE: every feature flagged as mechanistically important across
the three CLN025 committor papers reviewed (McKiernan/Husic/Pande 2017,
Sobieraj & Setny 2022, Kang/Trizio/Parrinello 2024, pairwise-interaction
committor paper) -- today's 8 newly added/rescued features, plus the features
the literature review separately flagged as "already present and confirmed
important". Any name in this list that doesn't actually exist in the data is
reported and skipped, not silently assumed.

TOP_FILL: after the literature core, remaining slots up to TARGET_SIZE are
filled with the highest-importance-ranked features (from today's 109-feature
final model) not already in LITERATURE_CORE.
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
IMPORTANCE_CSV = "/scratch/mma9420/committor_check/Q/physics_features/retrain_final_combined/feature_importance_full_model.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_literature_reduced/"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
N_WEIGHT_BINS = 20
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8
CORR_THRESHOLD = 0.8
TARGET_SIZE = 25

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

PREVIOUSLY_RESCUED = ["hb_5THRN_2ASPOD1", "hb_5THRN_2ASPOD2", "hb_6GLYN_2ASPO"]
NEWLY_RESCUED = ["hb_9TYRN_0TYRO"]
LIT_ASP2N_GLY6O = ["lit_d_Asp2N_Gly6O"]
LIT_PRIORITIES_4_5_7 = ["lit_d_Tyr0O_Trp8O", "lit_d_Tyr1N_Trp8O", "lit_d_Tyr0_Asp2_closest_heavy"]
TODAYS_NEW_FEATURES = PREVIOUSLY_RESCUED + NEWLY_RESCUED + LIT_ASP2N_GLY6O + LIT_PRIORITIES_4_5_7

ALREADY_PRESENT_KEEP_LIST = [
    "hb_2ASPN_7THRO", "hb_5THROG1_2ASPOD1", "hb_5THROG1_2ASPOD2", "hb_7THRN_2ASPO",
    "saltbridge_Nterm_Asp2", "saltbridge_Nterm_Cterm9",
    "cos_phi_res6", "sin_phi_res6", "cos_psi_res6", "sin_psi_res6",
    "cos_phi_res7", "sin_phi_res7",
]

LITERATURE_CORE_CANDIDATES = TODAYS_NEW_FEATURES + ALREADY_PRESENT_KEEP_LIST

REF_101_FULL = 0.1583
REF_101_TRANSONLY = 0.3914
REF_109_FULL = 0.1909
REF_109_TRANSONLY = 0.4040
REF_PCA10_FULL = -1.0377
REF_KNOWN10_FULL = 0.0979
REF_KNOWN10_TRANSONLY = 0.3266

# ------------------------------------------------------------------
# Step 1: load everything and assemble the full 109-feature table.
# ------------------------------------------------------------------
pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
base_feature_cols = [c for c in pruned_cols if c != "Committor_prob"]

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

full_feature_cols = base_feature_cols + TODAYS_NEW_FEATURES
print(f"Full stack available: {len(full_feature_cols)} features, {n:,} rows")

# ------------------------------------------------------------------
# Step 2: check which literature-core candidates actually exist.
# ------------------------------------------------------------------
present_cols = set(df.columns)
literature_core = [f for f in LITERATURE_CORE_CANDIDATES if f in present_cols]
missing = [f for f in LITERATURE_CORE_CANDIDATES if f not in present_cols]
print(f"\nLiterature-core candidates present: {len(literature_core)} of {len(LITERATURE_CORE_CANDIDATES)}")
if missing:
    print(f"MISSING (skipped, not silently assumed): {missing}")
print(f"Literature core: {literature_core}")

# ------------------------------------------------------------------
# Step 3: fill remaining slots with top-ranked remaining features.
# ------------------------------------------------------------------
importance_rank = pd.read_csv(IMPORTANCE_CSV, index_col=0)
ranked_features = importance_rank.sort_values("importance", ascending=False).index.tolist()

candidate_set = list(literature_core)
for feat in ranked_features:
    if len(candidate_set) >= TARGET_SIZE:
        break
    if feat not in candidate_set and feat in present_cols:
        candidate_set.append(feat)

print(f"\nCandidate set before correlation pruning: {len(candidate_set)} features")
print(candidate_set)

# ------------------------------------------------------------------
# Step 4: correlation-redundancy pass (both-passes-must-agree rule),
# preferring to keep literature-core features over top-fill features.
# ------------------------------------------------------------------
y_raw_all = df["Committor_prob"].values
is_transition_all = (y_raw_all > TRANSITION_LOW) & (y_raw_all < TRANSITION_HIGH)

corr_all = df[candidate_set].corr().abs()
corr_trans = df.loc[is_transition_all, candidate_set].corr().abs()

rank_pos = {f: i for i, f in enumerate(ranked_features)}


def priority(feat):
    # lower = keep preferentially: literature-core first, then by importance rank
    return (0 if feat in literature_core else 1, rank_pos.get(feat, 10**6))


dropped = set()
cols = candidate_set
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        f1, f2 = cols[i], cols[j]
        if f1 in dropped or f2 in dropped:
            continue
        if corr_all.loc[f1, f2] > CORR_THRESHOLD and corr_trans.loc[f1, f2] > CORR_THRESHOLD:
            loser = f1 if priority(f1) > priority(f2) else f2
            dropped.add(loser)
            print(f"  dropping {loser} (correlated {corr_all.loc[f1, f2]:.3f} all-frames / "
                  f"{corr_trans.loc[f1, f2]:.3f} transition-region with "
                  f"{f2 if loser == f1 else f1})")

feature_cols = [f for f in candidate_set if f not in dropped]
print(f"\nFinal reduced feature set: {len(feature_cols)} features (dropped {len(dropped)} redundant)")
print(feature_cols)

# ------------------------------------------------------------------
# Step 5: train/test split (identical methodology to every other script).
# ------------------------------------------------------------------
y_raw = y_raw_all
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


print(f"\n=== Literature-reduced feature set ({len(feature_cols)}), all training rows ===")
model_full, r2tr_full, r2o_full, r2tr_test_full = fit_and_score(idx_train, "full")

print(f"\n=== Literature-reduced feature set ({len(feature_cols)}), transition-region-only training ===")
train_tr_idx = idx_train[(y_raw[idx_train] > TRANSITION_LOW) & (y_raw[idx_train] < TRANSITION_HIGH)]
model_trans, r2tr_trans, r2o_trans, r2tr_test_trans = fit_and_score(train_tr_idx, "transition-region-only training")

print("\n=== Comparison ===")
print(f"{'run':65s} {'transition-region test R^2':>28s}")
print(f"{'101 base, full training':65s} {REF_101_FULL:>28.4f}")
print(f"{'101 base, transition-only':65s} {REF_101_TRANSONLY:>28.4f}")
print(f"{'PCA-selected 10, balanced, full training':65s} {REF_PCA10_FULL:>28.4f}")
print(f"{'Raw-importance top-10, balanced, full training':65s} {REF_KNOWN10_FULL:>28.4f}")
print(f"{'Raw-importance top-10, transition-only training':65s} {REF_KNOWN10_TRANSONLY:>28.4f}")
print(f"{'Final combined stack (109), full training':65s} {REF_109_FULL:>28.4f}")
print(f"{'Final combined stack (109), transition-only':65s} {REF_109_TRANSONLY:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'THIS: literature-reduced (%d), full training' % len(feature_cols):65s} {r2tr_test_full:>28.4f}")
print(f"{'THIS: literature-reduced (%d), transition-only' % len(feature_cols):65s} {r2tr_test_trans:>28.4f}")

feature_importance = pd.Series(model_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance_full_model.csv", header=["importance"])
print(f"\nFinal reduced set, ranked by importance:")
print(feature_importance.to_string())

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Literature-reduced feature set: {len(feature_cols)} features\n")
    f.write(f"Literature core used: {literature_core}\n")
    f.write(f"Missing from literature core (skipped): {missing}\n")
    f.write(f"Dropped as correlation-redundant: {sorted(dropped)}\n")
    f.write(f"Final features: {feature_cols}\n\n")
    f.write(f"Full training: transition-region R^2 = {r2tr_test_full:.4f} (vs. {REF_109_FULL:.4f} with 109 features)\n")
    f.write(f"Transition-only training: transition-region R^2 = {r2tr_test_trans:.4f} (vs. {REF_109_TRANSONLY:.4f} with 109 features)\n")

print(f"\nSaved outputs to {OUT_DIR}")
