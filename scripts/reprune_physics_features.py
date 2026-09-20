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
# Salt bridges are never auto-dropped -- see build_physics_features.py for why.
PROTECTED_PREFIXES = ("saltbridge_",)

print(f"Loading {MERGED_CSV}")
merged = pd.read_csv(MERGED_CSV)
print(f"{merged.shape[0]:,} rows x {merged.shape[1]} cols")


def prune(df, label):
    X = df.drop(columns=[c for c in LEAKY_AND_LABEL_COLS if c in df.columns])
    X = X.loc[:, X.std() > 0]
    corr = X.corr(method="pearson")

    # Greedily keep columns in priority order (new physics features first,
    # then old ones, alphabetically within each group for determinism);
    # a column is dropped only if it's already redundant with something
    # already accepted into `keep`. This is a naming-based tiebreaker, not
    # a real physics judgment -- it does NOT mean the dropped column was
    # less physically meaningful, just that it duplicated a column named
    # more consistently with the rest of the new feature set. `reasons`
    # records the exact correlation for every drop so this is auditable
    # rather than a black box -- a 0.99 duplicate and a 0.81 near-duplicate
    # are very different claims and both get logged, not collapsed.
    ordered = sorted(
        X.columns,
        key=lambda c: (not c.startswith(PROTECTED_PREFIXES), not c.startswith(NEW_FEATURE_PREFIXES), c),
    )
    keep, to_drop, reasons = [], [], []
    for c in ordered:
        if c.startswith(PROTECTED_PREFIXES):
            keep.append(c)
            continue
        redundant_with = sorted(
            ((k, corr.loc[c, k]) for k in keep if abs(corr.loc[c, k]) > CORR_THRESHOLD),
            key=lambda kv: -abs(kv[1]),
        )
        if redundant_with:
            to_drop.append(c)
            for kept_col, r in redundant_with:
                reasons.append({"dropped_feature": c, "kept_feature": kept_col, "correlation": r})
        else:
            keep.append(c)
    to_drop = sorted(to_drop)
    reasons_df = pd.DataFrame(reasons).sort_values(
        "correlation", key=lambda s: s.abs(), ascending=False
    )

    print(f"\n[{label}] {X.shape[1]} candidate features -> dropping {len(to_drop)} "
          f"as redundant (pairwise |corr| > {CORR_THRESHOLD}, new physics features "
          f"preferred over old generic ones when tied, salt bridges never dropped)")
    return corr, to_drop, reasons_df


print("\nPruning on all frames...")
corr_all, drop_all, reasons_all = prune(merged, "all frames")
corr_all.to_csv(OUT_DIR + "pearson_corr_all_frames.csv")
with open(OUT_DIR + "features_to_drop_all_frames.txt", "w") as f:
    f.write("\n".join(drop_all))
reasons_all.to_csv(OUT_DIR + "drop_reasons_all_frames.csv", index=False)

trans = merged[(merged["Committor_prob"] > 0.2) & (merged["Committor_prob"] < 0.8)]
print(f"\nPruning on transition-region frames only ({len(trans):,} rows)...")
corr_trans, drop_trans, reasons_trans = prune(trans, "transition region")
corr_trans.to_csv(OUT_DIR + "pearson_corr_transition_region.csv")
with open(OUT_DIR + "features_to_drop_transition_region.txt", "w") as f:
    f.write("\n".join(drop_trans))
reasons_trans.to_csv(OUT_DIR + "drop_reasons_transition_region.csv", index=False)

only_in_trans = sorted(set(drop_trans) - set(drop_all))
if only_in_trans:
    print(f"\n{len(only_in_trans)} features are redundant in the transition region but NOT "
          f"redundant over all frames -- worth a manual look before trusting the global list:")
    print(only_in_trans)

print(f"\nWeakest (lowest-|correlation|) drops on all frames -- worth a manual look, "
      f"since these are the least clearly redundant of the bunch:")
print(reasons_all.reindex(reasons_all["correlation"].abs().sort_values().index).head(10).to_string(index=False))

kept = [c for c in merged.columns if c not in LEAKY_AND_LABEL_COLS and c not in drop_all]
pruned_df = merged[kept + ["Committor_prob"]]
out_pruned = OUT_DIR + "regression_dataset_pruned.csv"
pruned_df.to_csv(out_pruned, index=False)
print(f"\nFinal pruned feature set: {len(kept)} features -> {out_pruned}")
