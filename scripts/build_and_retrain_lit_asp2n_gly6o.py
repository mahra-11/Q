"""
Computes literature priority #3 (Asp2 backbone N to Gly6 backbone O,
the mirror-image hydrogen bond partner of the already-rescued Asp2 O to
Gly6 N) as a direct heavy-atom distance -- NOT via baker_hubbard, since
the earlier check confirmed this exact pair never appears anywhere in
the 166-candidate unpruned set, meaning baker_hubbard never detected it
as a frequent enough hydrogen bond in this trajectory to register it at
all. A direct distance sidesteps that frequency-detection filter and
lets the model learn its own relationship to the committor, regardless
of how often it's "bonded" by baker_hubbard's criteria.

Physically: per the literature (the pairwise-interaction committor paper,
arXiv 2606.31832), this is described as part of a competing/misfolded
turn geometry (Asp2 N--Gly6 O), as opposed to the native-turn bond
already rescued (Asp2 O--Gly6 N) -- i.e. this may behave as a BAD-turn
indicator rather than a good-turn one. That's a real possibility, not
assumed away; the retrain below is what actually tests it.

Named with a `lit_` prefix (not `hb_`) to mark it as computed directly
for this literature check, distinct from the auto-detected baker_hubbard
features -- avoids implying it passed the same frequency-based detection
the hb_ columns did.

Reuses the same trajectory/topology config and sequence guard-rail as
build_physics_features.py, and the same frame_index scheme, so the join
back onto the existing merged dataset is exact.

Retrains on the 101 (pruned) + 3 (already-rescued) + 1 (this new one) =
105 feature set, both full and transition-region-only training, directly
comparable to the 104-feature results from the previous rescue
(0.1726 / 0.3969) to isolate whether this one additional feature helps,
hurts, or does nothing.
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml
import mdtraj as md
import lightgbm as lgb
from sklearn.metrics import r2_score

CONFIG_YAML = "/scratch/mma9420/q_pipeline/config/chignolin.yaml"
MERGED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_with_physics_features.csv"
PRUNED_CSV = "/scratch/mma9420/committor_check/Q/physics_features/regression_dataset_pruned.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_lit_asp2n_gly6o/"
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_SEQ = ["TYR", "TYR", "ASP", "PRO", "GLU", "THR", "GLY", "THR", "TRP", "TYR"]
EPS = 1e-4
N_WEIGHT_BINS = 20
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

RESCUED_FEATURES = ["hb_5THRN_2ASPOD1", "hb_5THRN_2ASPOD2", "hb_6GLYN_2ASPO"]
NEW_FEATURE = "lit_d_Asp2N_Gly6O"

REF_PRUNED_FULL_TRANSITION = 0.1583
REF_PRUNED_TRANSONLY_TRANSITION = 0.3914
REF_CONFIDENCE_FILTERED_NEW_HICONF = 0.4609
REF_RESCUED_FULL = 0.1726        # 101+3 rescued, full training -- isolating against
REF_RESCUED_TRANSONLY = 0.3969   # 101+3 rescued, transition-only -- isolating against

# ------------------------------------------------------------------
# Step 1: compute the new feature directly from the trajectory.
# ------------------------------------------------------------------
with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)
TRAJ_PATH = cfg["trajectory_xtc"]
TOP_PATH = cfg["topology_pdb"]

print(f"Trajectory: {TRAJ_PATH}")
print(f"Topology:   {TOP_PATH}")
print("Loading trajectory (full file, stride=1 -- expect this to take a while)...")
traj = md.load(TRAJ_PATH, top=TOP_PATH, stride=1)
n_frames = traj.n_frames
actual_seq = [r.name for r in traj.topology.residues]
print(f"{n_frames:,} frames, residues: {actual_seq}")

if actual_seq != EXPECTED_SEQ:
    sys.exit(f"ERROR: topology residues {actual_seq} don't match expected CLN025 "
              f"sequence {EXPECTED_SEQ} -- stop here.")

top = traj.topology
asp2_n = top.select("resid 2 and name N")
gly6_o = top.select("resid 6 and name O")
if len(asp2_n) == 0 or len(gly6_o) == 0:
    sys.exit(f"ERROR: couldn't find Asp2 N (found {len(asp2_n)}) or Gly6 O (found {len(gly6_o)}) "
              "-- check atom names in this topology.")
print(f"Asp2 N atom index: {asp2_n[0]}, Gly6 O atom index: {gly6_o[0]}")

dist = md.compute_distances(traj, [[asp2_n[0], gly6_o[0]]])[:, 0]
print(f"{NEW_FEATURE}: min={dist.min():.4f} nm, max={dist.max():.4f} nm, "
      f"mean={dist.mean():.4f} nm, fraction < 0.35nm (H-bond range): {(dist < 0.35).mean():.4f}")

frame_index = np.arange(n_frames)
new_feat_df = pd.DataFrame({"frame_index": frame_index, NEW_FEATURE: dist})
new_feat_df.to_csv(OUT_DIR + f"{NEW_FEATURE}.csv", index=False)
print(f"Saved {OUT_DIR}{NEW_FEATURE}.csv")

# ------------------------------------------------------------------
# Step 2: merge onto the existing feature set and retrain.
# ------------------------------------------------------------------
pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
base_feature_cols = [c for c in pruned_cols if c != "Committor_prob"]
feature_cols = base_feature_cols + RESCUED_FEATURES + [NEW_FEATURE]
print(f"\nTotal feature set: {len(feature_cols)} features "
      f"({len(base_feature_cols)} base + {len(RESCUED_FEATURES)} rescued + 1 new)")

df = pd.read_csv(MERGED_CSV, usecols=["frame_index"] + base_feature_cols + RESCUED_FEATURES + ["Committor_prob"])
n_before = len(df)
df = df.merge(new_feat_df, on="frame_index", how="left")
assert len(df) == n_before, "join changed row count -- frame_index is not unique somewhere"
n_matched = df[NEW_FEATURE].notna().sum()
print(f"New feature join: {n_matched:,} / {len(df):,} frames matched ({100 * n_matched / len(df):.2f}%)")
if n_matched / len(df) < 0.999:
    print("WARNING: join did not match almost all frames -- check frame_index alignment before trusting results.")

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


print("\n=== Full feature set (105), all training rows ===")
model_full, r2tr_full, r2o_full, r2tr_test_full = fit_and_score(idx_train, "full")

print("\n=== Full feature set (105), transition-region-only training ===")
train_tr_idx = idx_train[(y_raw[idx_train] > TRANSITION_LOW) & (y_raw[idx_train] < TRANSITION_HIGH)]
model_trans, r2tr_trans, r2o_trans, r2tr_test_trans = fit_and_score(train_tr_idx, "transition-region-only training")

print("\n=== Comparison ===")
print(f"{'run':65s} {'transition-region test R^2':>28s}")
print(f"{'New physics feat (101), full training':65s} {REF_PRUNED_FULL_TRANSITION:>28.4f}")
print(f"{'New physics feat (101), transition-only':65s} {REF_PRUNED_TRANSONLY_TRANSITION:>28.4f}")
print(f"{'New confidence-filtered, hi-conf test':65s} {REF_CONFIDENCE_FILTERED_NEW_HICONF:>28.4f}")
print(f"{'101+3 rescued, full training -- isolating against':65s} {REF_RESCUED_FULL:>28.4f}")
print(f"{'101+3 rescued, transition-only -- isolating against':65s} {REF_RESCUED_TRANSONLY:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'THIS: 101+3+1 (lit_d_Asp2N_Gly6O), full training':65s} {r2tr_test_full:>28.4f}")
print(f"{'THIS: 101+3+1 (lit_d_Asp2N_Gly6O), transition-only training':65s} {r2tr_test_trans:>28.4f}")

feature_importance = pd.Series(model_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance_full_model.csv", header=["importance"])
rank = feature_importance.reset_index()
rank.columns = ["feature", "importance"]
new_feat_rank = rank.index[rank["feature"] == NEW_FEATURE].tolist()
print(f"\n{NEW_FEATURE} rank: #{new_feat_rank[0] + 1} of {len(feature_cols)}" if new_feat_rank
      else f"\n{NEW_FEATURE}: not used in any split")

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"New feature: {NEW_FEATURE}\n")
    f.write(f"Total feature set: {len(feature_cols)} features\n\n")
    f.write(f"Full training: transition-region R^2 = {r2tr_test_full:.4f} (vs. {REF_RESCUED_FULL:.4f} without it)\n")
    f.write(f"Transition-only training: transition-region R^2 = {r2tr_test_trans:.4f} "
            f"(vs. {REF_RESCUED_TRANSONLY:.4f} without it)\n")

print(f"\nSaved outputs to {OUT_DIR}")
