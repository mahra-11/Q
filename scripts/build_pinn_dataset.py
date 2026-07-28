"""
Build the PINN training dataset: merge the MDTraj structural-feature table
with per-frame Q, keyed on frame_index (derived from Time_ps/200, NOT the
raw Frame column -- they diverge when stride>1, see README "environment
quirks").

Unlike build_regression_dataset.py (the earlier, label-based approach), this
does NOT compute or attach any committor label -- the physics-informed loss
in src/pinn_committor.py needs only feature vectors, frame ordering, and Q
(for the boundary condition), and derives everything else from the
martingale property directly. No binning, no label noise.

Keeps the same 64 `contact_resX_resY` / `phi_sin/cos_N` / `psi_sin/cos_N`
feature columns as the regression baseline (drops frame_index,
radius_gyration, Q, rmsd_to_folded as bookkeeping/non-features) so the two
approaches are comparable apples-to-apples.
"""
import sys
import pandas as pd

MDTRAJ_CSV = "/scratch/mma9420/committor_check/chignolin_mdtraj_features_full_labeled.csv"
Q_CSV      = "/scratch/mma9420/committor_check/Q/Q_values_allframes.csv"
OUT_CSV    = "/scratch/mma9420/committor_check/Q/pinn_dataset.csv"

NON_FEATURE_COLS = {"frame_index", "radius_gyration", "Q", "rmsd_to_folded"}

print(f"Loading MDTraj features: {MDTRAJ_CSV}")
df_feat = pd.read_csv(MDTRAJ_CSV)
print(f"Loading Q values:        {Q_CSV}")
df_q = pd.read_csv(Q_CSV)

if "frame_index" not in df_feat.columns:
    sys.exit(f"ERROR: 'frame_index' not found in {MDTRAJ_CSV}. Columns: {list(df_feat.columns)}")
for col in ["Time_ps", "Q"]:
    if col not in df_q.columns:
        sys.exit(f"ERROR: '{col}' not found in {Q_CSV}. Columns: {list(df_q.columns)}")

df_q = df_q.copy()
df_q["frame_index"] = (df_q["Time_ps"] / 200).round().astype(int)

merged = pd.merge(
    df_feat, df_q[["frame_index", "Q"]], on="frame_index", how="inner",
).sort_values("frame_index").reset_index(drop=True)
print(f"Merged: {len(merged):,} frames (of {len(df_feat):,} feature rows, {len(df_q):,} Q rows)")

feature_cols = [c for c in df_feat.columns if c not in NON_FEATURE_COLS]
print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")

out = merged[["frame_index", "Q"] + feature_cols].dropna().reset_index(drop=True)
print(f"After dropping rows with any NaN: {len(out):,}")

out.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
