"""
Re-run just the Pearson-correlation pruning (Step 7) on the already-computed
regression_dataset_with_physics_features.csv, without recomputing the
trajectory-derived features (Steps 1-6, the slow part). Use this whenever
only the pruning rule itself changes; rerun build_physics_features.py from
scratch only if the underlying features need to change.
"""
import pandas as pd

OUT_DIR = "/scratch/mma9420/committor_check/Q/physics_features/"
MERGED_CSV = OUT_DIR + "regression_dataset_with_physics_features.csv"

CORR_THRESHOLD = 0.8
LEAKY_AND_LABEL_COLS = ["frame_index", "radius_gyration", "rmsd_to_folded", "Q",
                         "rg_bin", "q_bin", "cell_id", "Committor_prob"]
NEW_FEATURE_PREFIXES = ("cos_", "sin_", "sc_dist_", "hb_", "saltbridge_")

print(f"Loading {MERGED_CSV}")
merged = pd.read_csv(MERGED_CSV)
print(f"{merged.shape[0]:,} rows x {merged.shape[1]} cols")


def prune(df, label):
    X = df.drop(columns=[c for c in LEAKY_AND_LABEL_COLS if c in df.columns])
    X = X.loc[:, X.std() > 0]
    corr = X.corr(method="pearson")

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
