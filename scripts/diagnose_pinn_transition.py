"""
Diagnose *why* the PINN's frame-level R^2 in the transition region is so bad
(scripts/compare_pinn_empirical.py reported R^2 = -4.65, RMSE = 0.43 on the
real run -- despite the binned mean curve visually matching the empirical
curve closely).

A binned mean can match the true curve even if individual predictions are
garbage, as long as errors within a bin average out. The two most likely
explanations for that here:

 1. Bimodal collapse: q_pred clusters near 0 and near 1 even for frames
    truly in the transition region, rather than taking smooth intermediate
    values -- plausible because ~75%+ of frames satisfy the boundary
    condition and dominate training by sheer count, while the martingale
    loss gets little signal specifically *inside* the sparse transition
    region (especially at lag=1, where most adjacent-frame pairs already
    sit safely in one basin and contribute near-zero loss regardless of
    what happens in between).
 2. Genuine high-variance spread: predictions vary smoothly but with large
    amplitude, consistent with the network using other features to
    differentiate same-Q frames (real physics -- but the earlier
    correlation check found only ~0.36 max correlation between any single
    feature and committor within the transition region, so large swings
    driven by 63 weakly-correlated features would look more like noise
    than signal).

This script isolates frames whose *Q* falls in the transition-adjacent
range (not the interpolated target, to avoid circularity) and reports the
q_pred distribution: histogram counts, and fraction landing near 0 (<0.1),
near 1 (>0.9), vs genuinely intermediate (0.1-0.9). A bimodal shape
(most mass near 0/1) confirms explanation 1; a roughly unimodal spread
around the true value confirms explanation 2.
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

CONFIG_YAML = "config/chignolin.yaml"
with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

pinn_cfg = cfg.get("pinn", {})
OUT_DIR = pinn_cfg.get("output_dir", os.path.join(cfg["q_output_dir"], "pinn"))
PRED_CSV = os.path.join(OUT_DIR, "predictions.csv")
EMPIRICAL_CURVE_CSV = pinn_cfg.get(
    "empirical_curve_csv", os.path.join(cfg["q_output_dir"], "committor_1d_visits_sem.csv"))

# Q range corresponding to empirical committor in (0.2, 0.8) -- read off the
# empirical curve directly rather than hardcoding, so this stays correct if
# thresholds or the curve shape change.
emp = pd.read_csv(EMPIRICAL_CURVE_CSV).dropna(subset=["p_folded_visits"]).sort_values("q_bin_center")
in_transition = emp[(emp["p_folded_visits"] > 0.2) & (emp["p_folded_visits"] < 0.8)]
q_low, q_high = in_transition["q_bin_center"].min(), in_transition["q_bin_center"].max()
print(f"Transition-region Q range (empirical committor in (0.2, 0.8)): [{q_low:.3f}, {q_high:.3f}]")

pred = pd.read_csv(PRED_CSV)
sub = pred[(pred["Q"] >= q_low) & (pred["Q"] <= q_high)]
print(f"Frames with Q in this range: {len(sub):,}")

q_pred = sub["q_pred"].to_numpy()
near_0 = (q_pred < 0.1).mean()
near_1 = (q_pred > 0.9).mean()
intermediate = ((q_pred >= 0.1) & (q_pred <= 0.9)).mean()

print(f"\nq_pred distribution for these frames:")
print(f"  near 0 (<0.1):        {near_0:.1%}")
print(f"  near 1 (>0.9):        {near_1:.1%}")
print(f"  intermediate:         {intermediate:.1%}")
print(f"  mean={q_pred.mean():.3f}  std={q_pred.std():.3f}  "
      f"median={np.median(q_pred):.3f}")

verdict = "BIMODAL COLLAPSE" if (near_0 + near_1) > 0.5 else "SPREAD (not collapsed)"
print(f"\nVerdict: {verdict}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(q_pred, bins=40, range=(0, 1), color="steelblue", edgecolor="white")
ax.axvline(0.1, color="gray", linestyle=":", linewidth=1)
ax.axvline(0.9, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel(r"$q_\theta(x)$")
ax.set_ylabel("frame count")
ax.set_title(f"PINN predictions for Q in [{q_low:.2f}, {q_high:.2f}] (transition region)\n{verdict}")
plt.tight_layout()
out_png = os.path.join(OUT_DIR, "transition_region_prediction_histogram.png")
plt.savefig(out_png, dpi=200)
print(f"\nSaved: {out_png}")
