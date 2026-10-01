"""
Builds the two still-missing meeting-prep deliverables:
  1. The Pearson correlation heat map, actually PLOTTED (previously only
     raw pearson_corr_*.csv matrices existed).
  2. A transition-region true-vs-predicted scatter plot, a SHAP analysis on
     the transition-region test rows (for physical interpretation of the
     top features), and a comparison table -- the rest of the "meeting-prep
     package".

Retrains the full pruned-feature model from scratch, with the exact same
split/seed/hyperparameters as retrain_physics_features.py, since no model
was persisted to disk from that run. The printed sanity-check R^2 values
should match that job's 0.9919 / 0.1583 -- if they don't, something about
the data or split has changed and the rest of this output shouldn't be
trusted.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
PRUNED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"
CORR_CSV = "/scratch/mma9420/committor_check/Q/physics_features/pearson_corr_all_frames.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/meeting_package/"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-4
N_WEIGHT_BINS = 20
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8
N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

# --- Load the pruned feature set, with frame_index pulled from the unpruned
#     merged file (same approach as retrain_physics_features.py) ---
pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
feature_cols = [c for c in pruned_cols if c != "Committor_prob"]
print(f"Pruned feature set: {len(feature_cols)} features")

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

X = df[feature_cols]
y_raw_test = y_raw[idx_test]
mask_test_tr = (y_raw_test > TRANSITION_LOW) & (y_raw_test < TRANSITION_HIGH)
print(f"Transition-region test rows: {mask_test_tr.sum():,}")


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


print("\nRetraining full pruned-feature model (same split as retrain_physics_features.py)...")
model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=2
)
model.fit(X.iloc[idx_train], y_logit[idx_train], sample_weight=weights[idx_train])
pred_test = inv_logit(model.predict(X.iloc[idx_test]))
r2_overall = r2_score(y_raw_test, pred_test)
r2_trans = r2_score(y_raw_test[mask_test_tr], pred_test[mask_test_tr])
print(f"Sanity check -- overall R^2 {r2_overall:.4f}, transition R^2 {r2_trans:.4f} "
      f"(should match retrain_physics_features.py's 0.9919 / 0.1583 -- if not, stop and investigate)")

# ------------------------------------------------------------------
# 1. Pearson correlation heat map, actually plotted. Restricted to the 101
#    surviving pruned features (the one that matters -- the full
#    166-candidate matrix is already saved as a raw CSV and is too large to
#    read as a single useful heatmap).
# ------------------------------------------------------------------
print("\nPlotting Pearson correlation heat map...")
corr_full = pd.read_csv(CORR_CSV, index_col=0)
pruned_feature_set = [c for c in feature_cols if c in corr_full.columns]
corr_pruned = corr_full.loc[pruned_feature_set, pruned_feature_set]

fig, ax = plt.subplots(figsize=(22, 20))
im = ax.imshow(corr_pruned.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(pruned_feature_set)))
ax.set_xticklabels(pruned_feature_set, rotation=90, fontsize=5)
ax.set_yticks(range(len(pruned_feature_set)))
ax.set_yticklabels(pruned_feature_set, fontsize=5)
ax.set_title(f"Pearson correlation -- final {len(pruned_feature_set)} pruned features (all frames)")
fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
plt.tight_layout()
plt.savefig(OUT_DIR + "pearson_heatmap_pruned_features.png", dpi=200)
plt.close(fig)
print(f"Saved {OUT_DIR}pearson_heatmap_pruned_features.png")

# ------------------------------------------------------------------
# 2. Transition-region scatter plot: true vs. predicted
# ------------------------------------------------------------------
print("\nPlotting transition-region scatter...")
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_raw_test[mask_test_tr], pred_test[mask_test_tr], alpha=0.5, s=12)
ax.plot([0.2, 0.8], [0.2, 0.8], "k--", lw=1, label="y = x")
ax.set_xlabel("True committor")
ax.set_ylabel("Predicted committor")
ax.set_xlim(0.2, 0.8)
ax.set_ylim(0, 1)
ax.set_title(f"Transition-region test predictions (n={mask_test_tr.sum()}, R^2={r2_trans:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + "transition_region_scatter.png", dpi=200)
plt.close(fig)
print(f"Saved {OUT_DIR}transition_region_scatter.png")

# ------------------------------------------------------------------
# 3. SHAP on transition-region test rows -- for physical interpretation of
#    the top features, not just a raw importance ranking.
# ------------------------------------------------------------------
print("\nComputing SHAP values on transition-region test rows...")
X_test_trans = X.iloc[idx_test][mask_test_tr]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_trans)

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
mean_abs_shap.to_csv(OUT_DIR + "shap_importance_transition_region.csv", header=["mean_abs_shap"])
print("\nTop 15 features by mean |SHAP| (transition-region test rows):")
print(mean_abs_shap.head(15).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
top15 = mean_abs_shap.head(15).iloc[::-1]
ax.barh(top15.index, top15.values, color="crimson")
ax.set_xlabel("Mean |SHAP value| (transition-region test rows)")
ax.set_title("Top 15 features by SHAP importance")
plt.tight_layout()
plt.savefig(OUT_DIR + "shap_summary_transition_region.png", dpi=200)
plt.close(fig)
print(f"Saved {OUT_DIR}shap_summary_transition_region.png")

# ------------------------------------------------------------------
# 4. Comparison table
# ------------------------------------------------------------------
comparison = pd.DataFrame([
    {"run": "Old baseline (64 features)", "overall_r2": 0.99, "transition_r2": 0.0305},
    {"run": "Old Check C (64 features, transition-only train)", "overall_r2": None, "transition_r2": 0.3224},
    {"run": "Old confidence-filtered (n_visits>=84)", "overall_r2": None, "transition_r2": 0.3440},
    {"run": "New physics features, pruned (101), full training", "overall_r2": 0.9919, "transition_r2": 0.1583},
    {"run": "New physics features, pruned (101), transition-only training", "overall_r2": 0.6446, "transition_r2": 0.3914},
    {"run": "New physics features, all kept (166), full training", "overall_r2": 0.9922, "transition_r2": 0.2030},
    {"run": "New physics features, all kept (166), transition-only training", "overall_r2": 0.6337, "transition_r2": 0.4054},
])
comparison.to_csv(OUT_DIR + "comparison_table.csv", index=False)
print(f"\nSaved {OUT_DIR}comparison_table.csv")

print(f"\nAll meeting-prep package outputs saved to {OUT_DIR}")
