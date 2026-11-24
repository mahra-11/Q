"""
Step 8 variant: the n_visits>=84 high-confidence subset, explicitly skipped
in retrain_physics_features.py because it needs an extra join back to
committor_grid_visits.csv via reconstructed Rg/Q bins not present in the
pruned/merged physics-feature dataset.

committor_grid_visits.csv only carries a `cell_id` string ("{rg_bin}_{q_bin}"),
not frame-level columns, and rg_bin/q_bin come from pd.cut(..., bins=60,
labels=False) -- a DATA-DRIVEN binning whose edges depend on exactly which
rows were in the merge at build time (run_committor_visits.sbatch). To get
matching bins, this script reproduces that exact source merge:
  chignolin_mdtraj_features_full_labeled.csv (frame_index, radius_gyration)
  inner-joined with Q_values_allframes_with_fate.csv (frame_index derived as
  round(Time_ps/200) -- NOT the file's own Frame column, see HANDOFF_v2.md),
  dropna(subset=["Fate"]) -- then the same pd.cut(bins=60) on radius_gyration
  and Q, same cell_id string concatenation. Any divergence in row population
  versus the original build would silently shift the bin edges and break the
  join, so row counts and match rates are printed at every step rather than
  assumed correct.

Historical reference (project working notes, Check A + Check C combined):
training on "transition-region AND n_visits>=84" = 2,204 rows (vs. Check
C's 4,325 with no confidence filter); own training R^2 0.6941; full
transition-region test R^2 0.3063; high-confidence test subset R^2 0.3441.
This script reproduces that same filter (transition-region AND n_visits>=84
on TRAINING rows only) on the new 101-feature pruned set, evaluated on this
project's own block-split test set -- not a re-run of the historical split,
so exact agreement isn't expected, only a sanity-check-scale comparison.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score

MDTRAJ_CSV = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
FATE_CSV = "/scratch/mma9420/committor_check/Q/Q_values_allframes_with_fate.csv"
VISITS_CSV = "/scratch/mma9420/committor_check/Q/committor_grid_visits.csv"
MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
PRUNED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_confidence_filtered/"
os.makedirs(OUT_DIR, exist_ok=True)

BINS = 60
EPS = 1e-4
N_WEIGHT_BINS = 20
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8
N_VISITS_THRESHOLD = 84

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

REF_BASELINE_TRANSITION = 0.0305
REF_CHECK_C_TRANSITION = 0.3224
REF_CONFIDENCE_FILTERED_FULL_TRANS = 0.3063   # old 64-feat: trained on transition+n_visits>=84, eval on full transition test
REF_CONFIDENCE_FILTERED_HICONF = 0.3441       # old 64-feat: same training, eval on high-confidence transition test subset only
REF_PRUNED_FULL_TRANSITION = 0.1583
REF_PRUNED_TRANSONLY_TRANSITION = 0.3914

# ------------------------------------------------------------------
# Step 1: reproduce the exact row population/binning used to build
# committor_grid_visits.csv, so rg_bin/q_bin/cell_id match it.
# ------------------------------------------------------------------
print("Reproducing the Rg/Q grid binning from the original source files...")
df_feat = pd.read_csv(MDTRAJ_CSV, usecols=["frame_index", "radius_gyration"])
df_fate = pd.read_csv(FATE_CSV, usecols=["Time_ps", "Q", "Fate"])
df_fate["frame_index"] = (df_fate["Time_ps"] / 200).round().astype(int)

merged = df_feat.merge(df_fate[["frame_index", "Q", "Fate"]], on="frame_index", how="inner")
merged = merged.sort_values("frame_index").dropna(subset=["Fate"])
print(f"Reconstructed grid population: {len(merged):,} rows (should match the row count "
      f"run_committor_visits.sbatch originally binned)")

merged["rg_bin"] = pd.cut(merged["radius_gyration"], bins=BINS, labels=False)
merged["q_bin"] = pd.cut(merged["Q"], bins=BINS, labels=False)
merged["cell_id"] = merged["rg_bin"].astype(str) + "_" + merged["q_bin"].astype(str)

frame_cell = merged[["frame_index", "cell_id"]].drop_duplicates(subset="frame_index")

# ------------------------------------------------------------------
# Step 2: join to committor_grid_visits.csv for n_visits per cell, then
# onto each frame.
# ------------------------------------------------------------------
visits = pd.read_csv(VISITS_CSV, usecols=["cell_id", "n_visits"])
visits["cell_id"] = visits["cell_id"].astype(str)
frame_cell = frame_cell.copy()
frame_cell["cell_id"] = frame_cell["cell_id"].astype(str)

frame_n_visits = frame_cell.merge(visits, on="cell_id", how="left")
n_missing = frame_n_visits["n_visits"].isna().sum()
print(f"Frame-level n_visits join: {len(frame_n_visits):,} frames, "
      f"{n_missing:,} with no matching cell_id in committor_grid_visits.csv "
      f"({100 * n_missing / len(frame_n_visits):.2f}%)")
if n_missing / len(frame_n_visits) > 0.01:
    print("WARNING: >1% of frames failed to match a grid cell -- binning may not "
          "match the original committor_grid_visits.csv population. Treat results "
          "with caution and report this number back before trusting anything below.")

frame_n_visits[["frame_index", "cell_id", "n_visits"]].to_csv(
    OUT_DIR + "frame_n_visits.csv", index=False
)
print(f"Saved {OUT_DIR}frame_n_visits.csv")

# ------------------------------------------------------------------
# Step 3: load the pruned physics-feature set, join n_visits (left join --
# frames with no match keep NaN n_visits and are excluded from the
# confidence filter but still counted in the full transition-region test
# set, consistent with how the historical Check A/C numbers worked).
# ------------------------------------------------------------------
pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
feature_cols = [c for c in pruned_cols if c != "Committor_prob"]
print(f"\nPruned feature set: {len(feature_cols)} features")

df = pd.read_csv(MERGED_CSV, usecols=["frame_index"] + feature_cols + ["Committor_prob"])
df = df.sort_values("frame_index").reset_index(drop=True)
n_before = len(df)
df = df.merge(frame_n_visits[["frame_index", "n_visits"]], on="frame_index", how="left")
assert len(df) == n_before, "join changed row count -- frame_index is not unique somewhere"
n_matched = df["n_visits"].notna().sum()
print(f"{len(df):,} rows total, {n_matched:,} ({100 * n_matched / len(df):.1f}%) matched to an n_visits value")

# ------------------------------------------------------------------
# Step 4: identical split and sample weighting to retrain_physics_features.py
# ------------------------------------------------------------------
y_raw = df["Committor_prob"].values
y_clipped = np.clip(y_raw, EPS, 1 - EPS)
y_logit = np.log(y_clipped / (1 - y_clipped))

bin_edges = np.linspace(0, 1, N_WEIGHT_BINS + 1)
bin_idx = np.clip(np.digitize(y_raw, bin_edges) - 1, 0, N_WEIGHT_BINS - 1)
bin_counts = np.bincount(bin_idx, minlength=N_WEIGHT_BINS)
weights = np.array([1.0 / bin_counts[b] for b in bin_idx])
weights = weights / weights.mean()

n = len(df)
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
n_visits_test = df["n_visits"].values[idx_test]
mask_test_tr = (y_raw_test > TRANSITION_LOW) & (y_raw_test < TRANSITION_HIGH)
mask_test_tr_hiconf = mask_test_tr & (n_visits_test >= N_VISITS_THRESHOLD)
print(f"Transition-region test rows: {mask_test_tr.sum():,} "
      f"(of which {mask_test_tr_hiconf.sum():,} also have n_visits>={N_VISITS_THRESHOLD})")

# ------------------------------------------------------------------
# Step 5: training subset = transition-region AND n_visits>=84, train rows
# only (mirrors the historical "combined" experiment)
# ------------------------------------------------------------------
y_raw_train = y_raw[idx_train]
n_visits_train = df["n_visits"].values[idx_train]
train_mask = (
    (y_raw_train > TRANSITION_LOW) & (y_raw_train < TRANSITION_HIGH) & (n_visits_train >= N_VISITS_THRESHOLD)
)
train_conf_idx = idx_train[train_mask]
print(f"\nTraining subset (transition-region AND n_visits>={N_VISITS_THRESHOLD}): "
      f"{len(train_conf_idx):,} rows (historical reference: 2,204 rows on the old 64-feature set)")


def inv_logit(y):
    return 1.0 / (1.0 + np.exp(-y))


model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=RANDOM_STATE, n_jobs=2
)
model.fit(X.iloc[train_conf_idx], y_logit[train_conf_idx], sample_weight=weights[train_conf_idx])

pred_train = inv_logit(model.predict(X.iloc[train_conf_idx]))
pred_test = inv_logit(model.predict(X.iloc[idx_test]))

r2_train = r2_score(y_raw_train[train_mask], pred_train)
r2_test_full_trans = r2_score(y_raw_test[mask_test_tr], pred_test[mask_test_tr])
r2_test_hiconf = (
    r2_score(y_raw_test[mask_test_tr_hiconf], pred_test[mask_test_tr_hiconf])
    if mask_test_tr_hiconf.sum() > 1 else float("nan")
)

print(f"\nOwn training data R^2: {r2_train:.4f} (historical reference: 0.6941)")
print(f"Full transition-region test R^2: {r2_test_full_trans:.4f} (historical reference: 0.3063)")
print(f"High-confidence transition-region test subset R^2: {r2_test_hiconf:.4f} "
      f"(historical reference: 0.3441; n={mask_test_tr_hiconf.sum()})")

print("\n=== Comparison ===")
print(f"{'run':65s} {'transition-region test R^2':>28s}")
print(f"{'Old baseline (64 features, full training)':65s} {REF_BASELINE_TRANSITION:>28.4f}")
print(f"{'Old Check C (64 features, transition-only train)':65s} {REF_CHECK_C_TRANSITION:>28.4f}")
print(f"{'Old confidence-filtered, full trans. test (64 feat)':65s} {REF_CONFIDENCE_FILTERED_FULL_TRANS:>28.4f}")
print(f"{'Old confidence-filtered, hi-conf test only (64 feat)':65s} {REF_CONFIDENCE_FILTERED_HICONF:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'New physics feat (101), full training':65s} {REF_PRUNED_FULL_TRANSITION:>28.4f}")
print(f"{'New physics feat (101), transition-only training':65s} {REF_PRUNED_TRANSONLY_TRANSITION:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'THIS: confidence-filtered train, full trans. test':65s} {r2_test_full_trans:>28.4f}")
print(f"{'THIS: confidence-filtered train, hi-conf test only':65s} {r2_test_hiconf:>28.4f}")

feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance.csv", header=["importance"])
print("\nTop 15 features by importance:")
print(feature_importance.head(15).to_string())

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Training subset: transition-region AND n_visits>={N_VISITS_THRESHOLD}, {len(train_conf_idx):,} rows\n")
    f.write(f"Own training R^2: {r2_train:.4f}\n")
    f.write(f"Full transition-region test R^2: {r2_test_full_trans:.4f} (n={mask_test_tr.sum()})\n")
    f.write(f"High-confidence transition-region test R^2: {r2_test_hiconf:.4f} (n={mask_test_tr_hiconf.sum()})\n\n")
    f.write("References (old 64-feature set):\n")
    f.write(f"  Baseline:                              {REF_BASELINE_TRANSITION:.4f}\n")
    f.write(f"  Check C:                                {REF_CHECK_C_TRANSITION:.4f}\n")
    f.write(f"  Confidence-filtered, full trans test:   {REF_CONFIDENCE_FILTERED_FULL_TRANS:.4f}\n")
    f.write(f"  Confidence-filtered, hi-conf test only: {REF_CONFIDENCE_FILTERED_HICONF:.4f}\n")
    f.write("References (new physics features, 101 pruned):\n")
    f.write(f"  Full training:            {REF_PRUNED_FULL_TRANSITION:.4f}\n")
    f.write(f"  Transition-only training: {REF_PRUNED_TRANSONLY_TRANSITION:.4f}\n")

print(f"\nSaved outputs to {OUT_DIR}")
