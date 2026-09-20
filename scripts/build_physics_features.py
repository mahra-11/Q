"""
Physics-informed local features for the committor regression: backbone
torsions (phi/psi) + side-chain chi1 as cos/sin, side-chain-to-side-chain
distances, hydrogen bonds, and salt bridges. No collective variables --
everything here is a per-residue or per-pair local coordinate.

Merges the new features onto regression_dataset.csv via a *verified*
frame_index join (regression_dataset.csv itself has no frame_index column,
so alignment is checked against chignolin_mdtraj_features_full_labeled.csv,
which still has one, before trusting a positional merge -- row order was an
open question in the meeting handoff and is not assumed here). Then prunes
redundant features with a Pearson correlation map (>0.8), both over all
frames (as instructed) and, as a second look, over transition-region frames
only, since features that correlate globally can decouple locally.

Q, radius_gyration, rmsd_to_folded and frame_index are never included as
model inputs -- they exist here only to align rows.
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
import mdtraj as md

CONFIG_YAML = "/scratch/mma9420/q_pipeline/config/chignolin.yaml"
FULL_LABELED_CSV = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
REGRESSION_CSV = "/scratch/mma9420/committor_check/Q/regression_dataset.csv"
OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/"

CORR_THRESHOLD = 0.8
MIN_SEQ_SEPARATION = 3   # side-chain distance pairs must be >= 3 apart in sequence
HBOND_FREQ = 0.05        # keep H-bonds present in at least this fraction of frames
EXPECTED_SEQ = ["TYR", "TYR", "ASP", "PRO", "GLU", "THR", "GLY", "THR", "TRP", "TYR"]  # CLN025: YYDPETGTWY

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Step 1: load the same trajectory/topology used for the existing features
# ------------------------------------------------------------------
with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

TRAJ_PATH = cfg["trajectory_xtc"]
TOP_PATH = cfg["topology_pdb"]
STRIDE = 1  # matches run_native_contacts_allframes.sbatch's override -- every frame

print(f"Trajectory: {TRAJ_PATH}")
print(f"Topology:   {TOP_PATH}")
print(f"Stride:     {STRIDE}")
print("Loading trajectory (this is the full 500k+ frame file -- expect this to take a while)...")
traj = md.load(TRAJ_PATH, top=TOP_PATH, stride=STRIDE)
n_frames = traj.n_frames
residues = list(traj.topology.residues)
actual_seq = [r.name for r in residues]
print(f"{n_frames:,} frames, {len(residues)} residues: {actual_seq}")

if actual_seq != EXPECTED_SEQ:
    sys.exit(
        f"ERROR: topology residues {actual_seq} don't match expected CLN025 "
        f"sequence {EXPECTED_SEQ} (YYDPETGTWY). Wrong config/trajectory -- stop here."
    )

n_hydrogens = sum(1 for a in traj.topology.atoms if a.element.symbol == "H")
if n_hydrogens == 0:
    print("WARNING: topology has no hydrogens -- hydrogen-bond detection (Step 4) will find nothing.")

frame_index = np.arange(n_frames) * STRIDE
new_features = {}

# ------------------------------------------------------------------
# Step 2: backbone torsions (phi, psi) + side-chain chi1, as cos/sin
# ------------------------------------------------------------------
print("\nComputing torsions (phi, psi, chi1)...")
for name, fn in [("phi", md.compute_phi), ("psi", md.compute_psi), ("chi1", md.compute_chi1)]:
    idx, ang = fn(traj)
    print(f"  {name}: {ang.shape[1]} angles found")
    for k in range(ang.shape[1]):
        res = traj.topology.atom(idx[k][1]).residue.index
        new_features[f"cos_{name}_res{res}"] = np.cos(ang[:, k])
        new_features[f"sin_{name}_res{res}"] = np.sin(ang[:, k])

n_torsion_feats = len(new_features)
print(f"Total torsion features: {n_torsion_feats}")
print("(A rough 7-residue x 3-angle x 2 count would estimate ~42 -- MDTraj's actual "
      "phi/psi/chi1 counts above depend on which residues each angle is defined for, "
      "so a different total here is expected, not a bug.)")

# ------------------------------------------------------------------
# Step 3: side-chain-to-side-chain distances (|i-j| >= 3)
# ------------------------------------------------------------------
print("\nComputing side-chain-to-side-chain distances...")
gly_indices = {i for i, r in enumerate(residues) if r.name == "GLY"}
candidate_pairs = [
    (i, j) for i, j in combinations(range(len(residues)), 2) if j - i >= MIN_SEQ_SEPARATION
]

try:
    d, used_pairs = md.compute_contacts(traj, contacts=candidate_pairs, scheme="sidechain-heavy")
except Exception as e:
    print(f"  'sidechain-heavy' failed for the full pair set ({e}); falling back to "
          f"'closest-heavy' for any pair touching a Gly (residues {sorted(gly_indices)}), "
          f"'sidechain-heavy' for the rest")
    non_gly_pairs = [(i, j) for i, j in candidate_pairs if i not in gly_indices and j not in gly_indices]
    gly_pairs = [(i, j) for i, j in candidate_pairs if i in gly_indices or j in gly_indices]
    d_ng, used_ng = md.compute_contacts(traj, contacts=non_gly_pairs, scheme="sidechain-heavy")
    d_g, used_g = md.compute_contacts(traj, contacts=gly_pairs, scheme="closest-heavy")
    d = np.concatenate([d_ng, d_g], axis=1)
    used_pairs = list(used_ng) + list(used_g)

for k, (i, j) in enumerate(used_pairs):
    new_features[f"sc_dist_res{i}_res{j}"] = d[:, k]
print(f"Side-chain distance features: {len(used_pairs)}/{len(candidate_pairs)} candidate pairs "
      f"(many will overlap the existing contact_resI_resJ columns -- Step 7's pruning handles that)")

# ------------------------------------------------------------------
# Step 4: hydrogen bonds -- continuous H...acceptor distance, not a 0/1 flag
# ------------------------------------------------------------------
print(f"\nFinding hydrogen bonds (baker_hubbard, freq >= {HBOND_FREQ})...")
triplets = md.baker_hubbard(traj, freq=HBOND_FREQ, exclude_water=True)
print(f"  {len(triplets)} H-bonds present in >= {HBOND_FREQ:.0%} of frames")
if len(triplets) == 0:
    print("  WARNING: none found -- check topology hydrogens and/or lower HBOND_FREQ.")
else:
    hb_dist = md.compute_distances(traj, triplets[:, [1, 2]])
    for k, (dn, h, ac) in enumerate(triplets):
        a, b = traj.topology.atom(dn), traj.topology.atom(ac)
        col = f"hb_{a.residue.index}{a.residue.name}{a.name}_{b.residue.index}{b.residue.name}{b.name}"
        new_features[col] = hb_dist[:, k]

# ------------------------------------------------------------------
# Step 5: salt bridges -- N-terminal amine vs. each carboxylate group
#   (no Lys/Arg in CLN025's YYDPETGTWY, so the only positive charge is the N-term amine;
#   negatives are Asp2, Glu4, and the C-terminal carboxylate on residue 9)
# ------------------------------------------------------------------
print("\nComputing salt-bridge distances...")
top = traj.topology


def find_all_atoms(residue_idx, name_options):
    """All atoms in `residue_idx` (0-based topology index) whose name is any of
    `name_options` (not just the first match). Uses MDTraj's `resid` selection
    keyword (0-based residue.index), not `residue`/`resSeq` (PDB file numbering,
    usually 1-indexed) -- those are different keywords and silently match
    nothing if confused, which is what happened here."""
    found = []
    for name in name_options:
        found.extend(top.select(f"resid {residue_idx} and name {name}"))
    return np.array(sorted(set(found)), dtype=int)


nterm_n = find_all_atoms(0, ["N"])
if len(nterm_n) == 0:
    sys.exit("ERROR: couldn't find the N-terminal amine nitrogen (residue 0, atom name 'N') -- "
              "check atom naming in this topology before continuing.")
print(f"  N-terminal amine: residue 0, atom '{top.atom(nterm_n[0]).name}'")

carboxylate_groups = {
    "Asp2": (2, ["OD1", "OD2"]),
    "Glu4": (4, ["OE1", "OE2"]),
    "Cterm9": (9, ["OXT", "OT1", "OT2", "O"]),
}
for label, (res_idx, name_options) in carboxylate_groups.items():
    o_atoms = find_all_atoms(res_idx, name_options)
    if len(o_atoms) == 0:
        print(f"  WARNING: no carboxylate oxygens found for {label} (residue {res_idx}, "
              f"tried {name_options}) -- skipping. Check atom names in this topology.")
        continue
    pairs = np.array([[nterm_n[0], o] for o in o_atoms])
    dists = md.compute_distances(traj, pairs)
    new_features[f"saltbridge_Nterm_{label}"] = dists.min(axis=1)
    atom_names = [top.atom(o).name for o in o_atoms]
    print(f"  {label}: min distance over oxygens {atom_names}")

print(f"\nTotal new physics features: {len(new_features)}")

# ------------------------------------------------------------------
# Step 6: merge onto regression_dataset.csv via a verified frame_index join
# ------------------------------------------------------------------
print("\nMerging with the existing regression dataset...")
new_df = pd.DataFrame(new_features)
new_df["frame_index"] = frame_index

print(f"Loading {FULL_LABELED_CSV}")
old_df = pd.read_csv(FULL_LABELED_CSV)
print(f"Loading {REGRESSION_CSV}")
reg_df = pd.read_csv(REGRESSION_CSV)

if len(old_df) != len(reg_df):
    sys.exit(
        f"ERROR: {FULL_LABELED_CSV} has {len(old_df):,} rows but {REGRESSION_CSV} has "
        f"{len(reg_df):,} -- can't verify row alignment this way. Stop and check how "
        f"regression_dataset.csv was built before merging anything."
    )

shared_cols = [c for c in old_df.columns if c in reg_df.columns
               and c not in ("frame_index", "radius_gyration", "Q", "rmsd_to_folded")]
if not shared_cols:
    sys.exit("ERROR: no shared feature columns between the two CSVs to verify alignment with.")

check_col = shared_cols[0]
if not np.allclose(old_df[check_col].values, reg_df[check_col].values, equal_nan=True):
    sys.exit(
        f"ERROR: '{check_col}' differs between {FULL_LABELED_CSV} and {REGRESSION_CSV} at "
        f"the same row index -- regression_dataset.csv is NOT in the same frame order as the "
        f"labeled features file. (This was an open question in the meeting handoff.) Do not "
        f"assume positional alignment; fix the join before rerunning this script."
    )
print(f"  Alignment verified via '{check_col}' -- both files are in the same row order.")

reg_df = reg_df.copy()
reg_df["frame_index"] = old_df["frame_index"].values

if len(new_df) != len(reg_df):
    sys.exit(
        f"ERROR: trajectory has {len(new_df):,} frames but regression_dataset.csv has "
        f"{len(reg_df):,} rows -- TRAJ_PATH/STRIDE here doesn't match what built the "
        f"existing dataset."
    )

merged = reg_df.merge(new_df, on="frame_index", how="left")
if merged.shape[0] != reg_df.shape[0]:
    sys.exit("ERROR: merge changed row count -- duplicate frame_index values somewhere.")
n_missing = merged[list(new_features.keys())].isna().any(axis=1).sum()
if n_missing:
    sys.exit(f"ERROR: {n_missing} rows got no new features after the merge -- investigate before continuing.")

out_merged = OUT_DIR + "regression_dataset_with_physics_features.csv"
merged.to_csv(out_merged, index=False)
print(f"Saved merged dataset: {merged.shape[0]:,} rows x {merged.shape[1]} cols -> {out_merged}")

# ------------------------------------------------------------------
# Step 7: Pearson correlation pruning (>= 0.8), all frames + transition-region-only
# ------------------------------------------------------------------
LEAKY_AND_LABEL_COLS = ["frame_index", "radius_gyration", "rmsd_to_folded", "Q",
                         "rg_bin", "q_bin", "cell_id", "Committor_prob"]

# The new physics features (torsions, side-chain distances, H-bonds, salt
# bridges) all use these prefixes. When a new feature duplicates an old
# generic one (e.g. cos_phi_res2 vs. the old phi_cos_2), the new one should
# win the tie -- that's the whole point of adding physically-interpretable
# features in the first place.
NEW_FEATURE_PREFIXES = ("cos_", "sin_", "sc_dist_", "hb_", "saltbridge_")


def prune(df, label):
    X = df.drop(columns=[c for c in LEAKY_AND_LABEL_COLS if c in df.columns])
    X = X.loc[:, X.std() > 0]
    corr = X.corr(method="pearson")

    # Greedily keep columns in priority order (new physics features first,
    # then old ones, alphabetically within each group for determinism);
    # a column is dropped only if it's already redundant with something
    # already accepted into `keep`. This is what "prefer keeping the more
    # physically interpretable one" actually requires -- a plain "drop
    # whichever column comes later in the file" rule has no opinion on
    # which member of a pair is more interpretable.
    ordered = sorted(X.columns, key=lambda c: (not c.startswith(NEW_FEATURE_PREFIXES), c))
    keep, to_drop = [], []
    for c in ordered:
        if any(abs(corr.loc[c, k]) > CORR_THRESHOLD for k in keep):
            to_drop.append(c)
        else:
            keep.append(c)
    to_drop = sorted(to_drop)

    print(f"\n[{label}] {X.shape[1]} candidate features -> dropping {len(to_drop)} "
          f"as redundant (pairwise |corr| > {CORR_THRESHOLD}, new physics features "
          f"preferred over old generic ones when tied)")
    return corr, to_drop


print("\nPruning on all frames...")
corr_all, drop_all = prune(merged, "all frames")
corr_all.to_csv(OUT_DIR + "pearson_corr_all_frames.csv")
with open(OUT_DIR + "features_to_drop_all_frames.txt", "w") as f:
    f.write("\n".join(drop_all))

trans = merged[(merged["Committor_prob"] > 0.2) & (merged["Committor_prob"] < 0.8)]
print(f"\nPruning on transition-region frames only ({len(trans):,} rows)...")
corr_trans, drop_trans = prune(trans, "transition region")
corr_trans.to_csv(OUT_DIR + "pearson_corr_transition_region.csv")
with open(OUT_DIR + "features_to_drop_transition_region.txt", "w") as f:
    f.write("\n".join(drop_trans))

only_in_trans = sorted(set(drop_trans) - set(drop_all))
if only_in_trans:
    print(f"\n{len(only_in_trans)} features are redundant in the transition region but NOT "
          f"redundant over all frames -- worth a manual look before trusting the global list:")
    print(only_in_trans)

kept = [c for c in merged.columns if c not in LEAKY_AND_LABEL_COLS and c not in drop_all]
pruned_df = merged[kept + ["Committor_prob"]]
out_pruned = OUT_DIR + "regression_dataset_pruned.csv"
pruned_df.to_csv(out_pruned, index=False)
print(f"\nFinal pruned feature set: {len(kept)} features -> {out_pruned}")
print("(Pruning applied uses the all-frames correlation map; the transition-region-only "
      "drop list is saved separately for comparison, not applied.)")
print("\nNext: rerun check_feature_reliance.py on this pruned set (Step 9 -- make sure nothing "
      "new is trivially predictive), then retrain with the same LightGBM/logit/weighting setup "
      "used for the 64-feature baseline (Step 8).")
