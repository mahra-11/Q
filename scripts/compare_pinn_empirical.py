"""
Quantify how well the PINN committor (scripts/train_pinn_committor.py's
predictions.csv) agrees with the empirical visit-based committor curve
(committor_1d_visits_sem.csv), as a direct, comparable-in-kind number to the
label-based regression baseline's R^2 (overall 0.9888, transition-region
(0.2, 0.8) only 0.0230 -- see README).

Per frame, the empirical curve is linearly interpolated in Q to give a
target committor value (clipped to the curve's own Q range, since the curve
has no data to extrapolate beyond it). R^2 and RMSE of q_pred against that
target are then reported overall and restricted to frames whose
interpolated target falls in the transition region (0.2, 0.8) -- the same
region the regression baseline was evaluated on.

This is necessarily an approximation (the empirical curve is itself a
40-bin average with its own SEM, not a per-frame ground truth), but it's the
same kind of comparison the regression phase used, just against a curve
built without needing PINN predictions to construct it.
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

CONFIG_YAML = "config/chignolin.yaml"
with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

pinn_cfg = cfg.get("pinn", {})
OUT_DIR = pinn_cfg.get("output_dir", os.path.join(cfg["q_output_dir"], "pinn"))
PRED_CSV = os.path.join(OUT_DIR, "predictions.csv")
EMPIRICAL_CURVE_CSV = pinn_cfg.get(
    "empirical_curve_csv", os.path.join(cfg["q_output_dir"], "committor_1d_visits_sem.csv"))
REPORT_TXT = os.path.join(OUT_DIR, "comparison_report.txt")

TRANSITION_LOW, TRANSITION_HIGH = 0.2, 0.8


def r2_rmse(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return r2, rmse


print(f"Loading PINN predictions: {PRED_CSV}")
pred = pd.read_csv(PRED_CSV)
print(f"Loading empirical curve:  {EMPIRICAL_CURVE_CSV}")
emp = pd.read_csv(EMPIRICAL_CURVE_CSV).dropna(subset=["p_folded_visits"]).sort_values("q_bin_center")

q_curve = emp["q_bin_center"].to_numpy()
p_curve = emp["p_folded_visits"].to_numpy()

# Only evaluate frames whose Q falls inside the empirical curve's own range --
# outside it, "target" would be extrapolation, not a real empirical estimate.
in_range = (pred["Q"] >= q_curve.min()) & (pred["Q"] <= q_curve.max())
n_dropped = (~in_range).sum()
if n_dropped:
    print(f"Dropping {n_dropped:,} frames outside empirical curve's Q range "
          f"[{q_curve.min():.3f}, {q_curve.max():.3f}]")
pred = pred[in_range].copy()

pred["target"] = np.interp(pred["Q"], q_curve, p_curve)

overall_r2, overall_rmse = r2_rmse(pred["target"].to_numpy(), pred["q_pred"].to_numpy())

transition = pred[(pred["target"] > TRANSITION_LOW) & (pred["target"] < TRANSITION_HIGH)]
if len(transition) > 0:
    trans_r2, trans_rmse = r2_rmse(transition["target"].to_numpy(), transition["q_pred"].to_numpy())
else:
    trans_r2, trans_rmse = float("nan"), float("nan")

lines = [
    f"Frames evaluated (in empirical curve's Q range): {len(pred):,}",
    f"Overall:            R^2 = {overall_r2:.4f}   RMSE = {overall_rmse:.4f}",
    f"Transition region ({TRANSITION_LOW}, {TRANSITION_HIGH}), n={len(transition):,}:"
    f"  R^2 = {trans_r2:.4f}   RMSE = {trans_rmse:.4f}",
    "",
    "For comparison, the label-based LightGBM regression baseline got:",
    "  Overall test R^2 = 0.9888 (misleading -- dominated by the easy majority near committor=1)",
    "  Transition-region test R^2 = 0.0230 (best achieved after diagnostics: ~0.31-0.34)",
]
report = "\n".join(lines)
print("\n" + report)

with open(REPORT_TXT, "w") as f:
    f.write(report + "\n")
print(f"\nSaved: {REPORT_TXT}")
