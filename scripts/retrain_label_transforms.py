"""
Label-transformation comparison, isolating exactly one variable: how the
committor probability is transformed before training.

Hypothesis: training on logit(y) with squared-error loss may systematically
under-penalize errors in the transition region. d(raw probability)/d(logit)
= raw*(1-raw), which is MAXIMIZED at y=0.5 and drops toward 0 near y=0/1 --
so a fixed logit-space error translates to a LARGER raw-probability error
near the transition region than near the end-states. If true, logit-space
MSE loss is misaligned with what we actually care about (accuracy inside
the transition region specifically).

Three variants compared, same 109-feature set, same split, same
inverse-frequency sample weighting (computed from raw-probability bins,
held fixed across variants so only the training target transform changes):

  1. logit:  log(y / (1-y))            -- current project standard
  2. raw:    y itself, no transform    -- removes the squashing entirely
  3. asin_sqrt: arcsin(sqrt(y))        -- standard variance-stabilizing
     transform for proportions/probabilities, a middle ground between the
     other two

All three are evaluated in raw-probability space (after inverting back),
so R^2 values are directly comparable across variants and to every other
result in this project.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
PRUNED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"
ASP2N_GLY6O_CSV = "/scratch/mma9420/committor_check/Q/physics_features/retrain_lit_asp2n_gly6o/lit_d_Asp2N_Gly6O.csv"
PRIORITIES_4_5_7_CSV = "/scratch/mma9420/committor_check/Q/physics_features/retrain_lit_priorities_4_5_7/new_features.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_label_transforms/"
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

REF_109_FULL = 0.1909
REF_109_TRANSONLY = 0.4040

pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
base_feature_cols = [c for c in pruned_cols if c != "Committor_prob"]
feature_cols = base_feature_cols + ALL_NEW_FEATURES

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
print(f"Feature set: {len(feature_cols)} features, {n:,} rows")

y_raw = df["Committor_prob"].values
y_clipped = np.clip(y_raw, EPS, 1 - EPS)

TRANSFORMS = {
    "logit": {
        "forward": lambda y: np.log(y / (1 - y)),
        "inverse": lambda z: 1.0 / (1.0 + np.exp(-z)),
    },
    "raw": {
        "forward": lambda y: y,
        "inverse": lambda z: z,
    },
    "asin_sqrt": {
        "forward": lambda y: np.arcsin(np.sqrt(y)),
        "inverse": lambda z: np.sin(z) ** 2,
    },
}

# inverse-frequency sample weighting computed once from raw-probability bins,
# held fixed across all three transform variants
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

results = {}

for name, t in TRANSFORMS.items():
    y_t = t["forward"](y_clipped)

    def fit_and_score(train_idx, label):
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=2
        )
        model.fit(X.iloc[train_idx], y_t[train_idx], sample_weight=weights[train_idx])

        pred_train = t["inverse"](model.predict(X.iloc[train_idx]))
        pred_test = t["inverse"](model.predict(X.iloc[idx_test]))

        r2_train = r2_score(y_raw[train_idx], pred_train)
        r2_test_overall = r2_score(y_raw_test, pred_test)
        r2_test_tr = r2_score(y_raw_test[mask_test_tr], pred_test[mask_test_tr])

        print(f"\n[{name} / {label}] train rows: {len(train_idx):,}")
        print(f"  Train R^2:                  {r2_train:.4f}")
        print(f"  Test overall R^2:           {r2_test_overall:.4f}")
        print(f"  Test transition-region R^2: {r2_test_tr:.4f}")
        return r2_train, r2_test_overall, r2_test_tr

    print(f"\n=== Transform: {name}, full training ===")
    r2tr_full, r2o_full, r2tr_test_full = fit_and_score(idx_train, "full")

    print(f"\n=== Transform: {name}, transition-region-only training ===")
    train_tr_idx = idx_train[(y_raw[idx_train] > TRANSITION_LOW) & (y_raw[idx_train] < TRANSITION_HIGH)]
    r2tr_trans, r2o_trans, r2tr_test_trans = fit_and_score(train_tr_idx, "transition-only")

    results[name] = {
        "full_transition_r2": r2tr_test_full,
        "transonly_transition_r2": r2tr_test_trans,
    }

print("\n=== Comparison ===")
print(f"{'transform':15s} {'full training TR R^2':>22s} {'transition-only training TR R^2':>34s}")
for name, r in results.items():
    print(f"{name:15s} {r['full_transition_r2']:>22.4f} {r['transonly_transition_r2']:>34.4f}")
print(f"{'-' * 15} {'-' * 22} {'-' * 34}")
print(f"{'(reference: current logit baseline)':15s} {REF_109_FULL:>22.4f} {REF_109_TRANSONLY:>34.4f}")

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Feature set: {len(feature_cols)} features (same as retrain_final_combined.py)\n\n")
    for name, r in results.items():
        f.write(f"{name}: full training TR R^2 = {r['full_transition_r2']:.4f}, "
                f"transition-only training TR R^2 = {r['transonly_transition_r2']:.4f}\n")
    f.write(f"\nReference (logit, from retrain_final_combined.py): "
            f"full={REF_109_FULL:.4f}, transonly={REF_109_TRANSONLY:.4f}\n")

print(f"\nSaved outputs to {OUT_DIR}")
