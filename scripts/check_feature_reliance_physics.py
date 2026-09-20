"""
Adapted from check_feature_reliance.py for the pruned physics-feature set
(regression_dataset_pruned.csv, 72 features after Pearson pruning) instead
of the original 64-feature regression_dataset.csv. Two checks on whether
the model is effectively just leaning on one or two features:

1. Train a model on the FULL pruned feature set to get a real reference R^2
   for this dataset (the old script hardcoded the 64-feature model's known
   numbers -- that reference doesn't apply here, so it's computed fresh),
   then train on just the top-1/2/3 most important features (by LightGBM
   importance from that same full-feature model) and compare. If a
   single feature gets nearly the same score as the full 72-feature model,
   the rest of the features aren't adding much.
2. Check whether those top features actually vary within the transition
   region (committor 0.2-0.8), or whether they're already saturated
   (mostly one value) there.

Same train/test split methodology as the original script (random 80/20)
for a like-for-like rerun; the project later adopted a contiguous
time-block split as more rigorous for leakage -- swap TEST_FRAC's
train_test_split for that scheme if this needs to match those numbers
directly.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATA_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"

EPS = 1e-4
N_WEIGHT_BINS = 20
TEST_FRAC = 0.2
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8
N_TOP_FEATURES = 3

df = pd.read_csv(DATA_CSV)
y_raw = df["Committor_prob"].values
X = df.drop(columns=["Committor_prob"])
print(f"Loaded {DATA_CSV}: {len(df):,} rows, {X.shape[1]} candidate features")

y_clipped = np.clip(y_raw, EPS, 1 - EPS)
y_logit = np.log(y_clipped / (1 - y_clipped))

bin_edges = np.linspace(0, 1, N_WEIGHT_BINS + 1)
bin_idx = np.clip(np.digitize(y_raw, bin_edges) - 1, 0, N_WEIGHT_BINS - 1)
bin_counts = np.bincount(bin_idx, minlength=N_WEIGHT_BINS)
weights = np.array([1.0 / bin_counts[b] for b in bin_idx])
weights = weights / weights.mean()

idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=TEST_FRAC, random_state=RANDOM_STATE
)
y_logit_train = y_logit[idx_train]
y_raw_train, y_raw_test = y_raw[idx_train], y_raw[idx_test]
w_train = weights[idx_train]

mask_test_tr = (y_raw_test > TRANSITION_LOW) & (y_raw_test < TRANSITION_HIGH)


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


def fit_and_score(feats):
    X_train_sub = X.iloc[idx_train][feats]
    X_test_sub = X.iloc[idx_test][feats]
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=1
    )
    model.fit(X_train_sub, y_logit_train, sample_weight=w_train)
    pred_test = inv_logit(model.predict(X_test_sub))
    r2_overall = r2_score(y_raw_test, pred_test)
    r2_tr = (
        r2_score(y_raw_test[mask_test_tr], pred_test[mask_test_tr])
        if mask_test_tr.sum() > 1 else float("nan")
    )
    return model, r2_overall, r2_tr


print("\n=== Full pruned feature set (reference for this dataset) ===")
full_model, full_r2_overall, full_r2_tr = fit_and_score(list(X.columns))
print(f"Overall test R^2:           {full_r2_overall:.4f}")
print(f"Transition-region test R^2: {full_r2_tr:.4f}")

importances = pd.Series(full_model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = importances.head(N_TOP_FEATURES).index.tolist()
print(f"\nTop {N_TOP_FEATURES} features by importance in the full model:")
print(importances.head(N_TOP_FEATURES).to_string())

print("\n=== Reduced models (top-K most important features) ===")
for n_feat in range(1, N_TOP_FEATURES + 1):
    feats = top_features[:n_feat]
    _, r2_overall, r2_tr = fit_and_score(feats)
    gap_overall = full_r2_overall - r2_overall
    gap_tr = full_r2_tr - r2_tr
    print(f"Using top {n_feat} feature(s) {feats}:")
    print(f"  Overall test R^2:           {r2_overall:.4f}  (full model - this: {gap_overall:+.4f})")
    print(f"  Transition-region test R^2: {r2_tr:.4f}  (full model - this: {gap_tr:+.4f})")
    if n_feat == 1 and gap_tr < 0.05:
        print("  WARNING: a single feature nearly matches the full 72-feature transition-region "
              "R^2 -- treat this as a leakage/triviality flag and check this feature's relationship "
              "to how the committor labels were built before trusting the full-feature result.")
    print()

print("=== Do the top features actually vary within the transition region? ===")
tr_mask_all = (y_raw > TRANSITION_LOW) & (y_raw < TRANSITION_HIGH)
for f in top_features:
    print(f"\n{f}:")
    print(f"  Overall dataset:        mean={X[f].mean():.4f}  std={X[f].std():.4f}")
    print(f"  Transition-region only: mean={X.loc[tr_mask_all, f].mean():.4f}  std={X.loc[tr_mask_all, f].std():.4f}")
    print(f"  Value counts in transition region (top 5):")
    print(X.loc[tr_mask_all, f].value_counts().head(5).to_string())
