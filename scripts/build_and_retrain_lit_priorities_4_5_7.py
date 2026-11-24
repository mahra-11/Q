"""
Combines two kinds of additions in one pass, since loading the full
trajectory is the expensive part and there's no reason to pay that cost
three separate times:

RESCUED (free, already computed, no new MDTraj work):
  hb_9TYRN_0TYRO  -- priority #6, Tyr0 O to Tyr9 N (the end-to-end
    hydrogen bond reported to rise 0%->14%->89% unfolded->TS->folded,
    Sobieraj & Setny 2022). Dropped at a VERY high correlation (0.985
    with saltbridge_Nterm_Cterm9) -- much higher than the earlier
    rescues (~0.83-0.89), so this one is plausibly a genuinely redundant
    drop rather than a wrongful one. Included anyway since re-adding an
    already-computed column costs nothing, and the retrain itself will
    show whether it adds anything on top of the salt bridge it's so
    correlated with.

FRESH COMPUTATION (never existed in the 166-candidate set):
  lit_d_Tyr0O_Trp8O   -- priority #4, largest-weight term in the
    pairwise-interaction committor paper's formula (arXiv 2606.31832).
    Backbone O to backbone O can't be a baker_hubbard hit (two oxygens,
    no shared hydrogen), so this was never going to appear automatically.
  lit_d_Tyr1N_Trp8O   -- priority #5, forms specifically on the TS-down
    route per the same paper.
  lit_d_Tyr0_Asp2_closest_heavy  -- priority #7, a top-4 "initiator"
    contact per Sobieraj & Setny 2022 (Granger causality). Residues 0
    and 2 are only 2 apart in sequence, below this project's
    MIN_SEQ_SEPARATION=3 filter in build_physics_features.py (meant to
    avoid trivially-close pairs) -- deliberately excluded by a general
    design rule that happens not to fit this specific, literature-
    flagged pair. Uses MDTraj's `closest-heavy` contact scheme (the same
    one already used for Gly fallbacks) rather than a single atom pair,
    matching "shortest heavy-atom distance" as specified.

Builds on top of the already-validated 101 (pruned) + 3 (previously
rescued: hb_5THRN_2ASPOD1, hb_5THRN_2ASPOD2, hb_6GLYN_2ASPO) feature set
-- NOT yet including priority #3 (lit_d_Asp2N_Gly6O), whose retrain job
was still pending when this script was written; that gets combined
separately once confirmed, to keep each addition independently verified
before stacking them.
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
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/retrain_lit_priorities_4_5_7/"
os.makedirs(OUT_DIR, exist_ok=True)

EXPECTED_SEQ = ["TYR", "TYR", "ASP", "PRO", "GLU", "THR", "GLY", "THR", "TRP", "TYR"]
EPS = 1e-4
N_WEIGHT_BINS = 20
RANDOM_STATE = 42
TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8

N_BLOCKS = 10
N_TEST_BLOCKS = 2
BUFFER_FRAMES = 2000

PREVIOUSLY_RESCUED = ["hb_5THRN_2ASPOD1", "hb_5THRN_2ASPOD2", "hb_6GLYN_2ASPO"]
NEWLY_RESCUED = ["hb_9TYRN_0TYRO"]
NEW_COMPUTED_FEATURES = ["lit_d_Tyr0O_Trp8O", "lit_d_Tyr1N_Trp8O", "lit_d_Tyr0_Asp2_closest_heavy"]

REF_PRUNED_FULL_TRANSITION = 0.1583
REF_PRUNED_TRANSONLY_TRANSITION = 0.3914
REF_CONFIDENCE_FILTERED_NEW_HICONF = 0.4609
REF_RESCUED_104_FULL = 0.1726        # 101+3 rescued, full training -- previous result
REF_RESCUED_104_TRANSONLY = 0.3969   # 101+3 rescued, transition-only -- previous result

# ------------------------------------------------------------------
# Step 1: load trajectory once, compute all 3 new distances.
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


def one_atom(residue_idx, name):
    sel = top.select(f"resid {residue_idx} and name {name}")
    if len(sel) == 0:
        sys.exit(f"ERROR: atom '{name}' not found in residue {residue_idx} "
                  f"({top.residue(residue_idx).name}) -- check topology atom names.")
    return int(sel[0])


new_features = {}

# Priority #4: Tyr0 O to Trp8 O
tyr0_o = one_atom(0, "O")
trp8_o = one_atom(8, "O")
new_features["lit_d_Tyr0O_Trp8O"] = md.compute_distances(traj, [[tyr0_o, trp8_o]])[:, 0]
print(f"lit_d_Tyr0O_Trp8O: atoms {tyr0_o}-{trp8_o}, "
      f"mean={new_features['lit_d_Tyr0O_Trp8O'].mean():.4f} nm")

# Priority #5: Tyr1 N to Trp8 O
tyr1_n = one_atom(1, "N")
new_features["lit_d_Tyr1N_Trp8O"] = md.compute_distances(traj, [[tyr1_n, trp8_o]])[:, 0]
print(f"lit_d_Tyr1N_Trp8O: atoms {tyr1_n}-{trp8_o}, "
      f"mean={new_features['lit_d_Tyr1N_Trp8O'].mean():.4f} nm")

# Priority #7: Tyr0 to Asp2, shortest heavy-atom distance (closest-heavy scheme,
# same as the Gly fallback elsewhere in this project) -- residues 0 and 2 are
# only 2 apart in sequence, below this project's usual MIN_SEQ_SEPARATION=3
# filter, deliberately overridden here for this literature-flagged pair only.
d_closest, _ = md.compute_contacts(traj, contacts=[[0, 2]], scheme="closest-heavy")
new_features["lit_d_Tyr0_Asp2_closest_heavy"] = d_closest[:, 0]
print(f"lit_d_Tyr0_Asp2_closest_heavy: mean={new_features['lit_d_Tyr0_Asp2_closest_heavy'].mean():.4f} nm")

frame_index = np.arange(n_frames)
new_feat_df = pd.DataFrame({"frame_index": frame_index, **new_features})
new_feat_df.to_csv(OUT_DIR + "new_features.csv", index=False)
print(f"Saved {OUT_DIR}new_features.csv")

# ------------------------------------------------------------------
# Step 2: merge onto the existing feature set and retrain.
# ------------------------------------------------------------------
pruned_cols = pd.read_csv(PRUNED_CSV, nrows=0).columns.tolist()
base_feature_cols = [c for c in pruned_cols if c != "Committor_prob"]
feature_cols = base_feature_cols + PREVIOUSLY_RESCUED + NEWLY_RESCUED + NEW_COMPUTED_FEATURES
print(f"\nTotal feature set: {len(feature_cols)} features "
      f"({len(base_feature_cols)} base + {len(PREVIOUSLY_RESCUED)} previously rescued + "
      f"{len(NEWLY_RESCUED)} newly rescued + {len(NEW_COMPUTED_FEATURES)} newly computed)")

df = pd.read_csv(
    MERGED_CSV,
    usecols=["frame_index"] + base_feature_cols + PREVIOUSLY_RESCUED + NEWLY_RESCUED + ["Committor_prob"],
)
n_before = len(df)
df = df.merge(new_feat_df, on="frame_index", how="left")
assert len(df) == n_before, "join changed row count -- frame_index is not unique somewhere"
for feat in NEW_COMPUTED_FEATURES:
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


print(f"\n=== Full feature set ({len(feature_cols)}), all training rows ===")
model_full, r2tr_full, r2o_full, r2tr_test_full = fit_and_score(idx_train, "full")

print(f"\n=== Full feature set ({len(feature_cols)}), transition-region-only training ===")
train_tr_idx = idx_train[(y_raw[idx_train] > TRANSITION_LOW) & (y_raw[idx_train] < TRANSITION_HIGH)]
model_trans, r2tr_trans, r2o_trans, r2tr_test_trans = fit_and_score(train_tr_idx, "transition-region-only training")

print("\n=== Comparison ===")
print(f"{'run':65s} {'transition-region test R^2':>28s}")
print(f"{'New physics feat (101), full training':65s} {REF_PRUNED_FULL_TRANSITION:>28.4f}")
print(f"{'New physics feat (101), transition-only':65s} {REF_PRUNED_TRANSONLY_TRANSITION:>28.4f}")
print(f"{'New confidence-filtered, hi-conf test':65s} {REF_CONFIDENCE_FILTERED_NEW_HICONF:>28.4f}")
print(f"{'101+3 rescued (104 total), full training':65s} {REF_RESCUED_104_FULL:>28.4f}")
print(f"{'101+3 rescued (104 total), transition-only':65s} {REF_RESCUED_104_TRANSONLY:>28.4f}")
print(f"{'-' * 65} {'-' * 28}")
print(f"{'THIS: %d features (+#6 rescued, +#4/#5/#7 new), full training' % len(feature_cols):65s} {r2tr_test_full:>28.4f}")
print(f"{'THIS: %d features (+#6 rescued, +#4/#5/#7 new), transition-only' % len(feature_cols):65s} {r2tr_test_trans:>28.4f}")

feature_importance = pd.Series(model_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.to_csv(OUT_DIR + "feature_importance_full_model.csv", header=["importance"])
rank = feature_importance.reset_index()
rank.columns = ["feature", "importance"]
print("\nNew feature ranks:")
for feat in NEWLY_RESCUED + NEW_COMPUTED_FEATURES:
    r = rank.index[rank["feature"] == feat].tolist()
    print(f"  {feat}: #{r[0] + 1} of {len(feature_cols)}" if r else f"  {feat}: not used in any split")

with open(OUT_DIR + "r2_summary.txt", "w") as f:
    f.write(f"Total feature set: {len(feature_cols)} features\n")
    f.write(f"Newly rescued: {NEWLY_RESCUED}\n")
    f.write(f"Newly computed: {NEW_COMPUTED_FEATURES}\n\n")
    f.write(f"Full training: transition-region R^2 = {r2tr_test_full:.4f} "
            f"(vs. {REF_RESCUED_104_FULL:.4f} with 104 features)\n")
    f.write(f"Transition-only training: transition-region R^2 = {r2tr_test_trans:.4f} "
            f"(vs. {REF_RESCUED_104_TRANSONLY:.4f} with 104 features)\n")

print(f"\nSaved outputs to {OUT_DIR}")
