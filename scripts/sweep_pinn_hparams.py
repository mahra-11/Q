"""
Hyperparameter sweep to fix the bimodal-collapse problem found by
diagnose_pinn_transition.py: at lag=1, 86.6% of transition-region
predictions saturate near 0 or 1 instead of taking smooth intermediate
values (mean ~0.51 matches the true curve on average, but frame-level
R^2 = -4.65 -- see README).

Hypothesis: at lag=1, most (x_t, x_t+lag) pairs -- including pairs
starting in the transition region -- have nearly-identical feature
vectors (frames 200 ps apart are highly autocorrelated), so the
martingale loss is satisfied almost trivially regardless of what q_theta
does there, leaving the boundary loss (which sees ~75%+ of frames, all
with hard 0/1 targets) to dominate the gradient landscape and push the
network toward a sharp classifier-like decision surface instead of a
smooth committor. A longer lag should force more transition-region-
originating pairs to actually resolve into a basin by t+lag, giving the
martingale loss real, non-trivial signal specifically where it's needed.
A lower boundary_weight should reduce how hard the classifier-like pull
dominates regardless of lag.

Sweeps `lag` x `boundary_weight`, training a fresh (shorter-epoch, for
speed -- this ranks configs, it doesn't produce a final model) network per
combination, and reports both the R^2 metrics (compare_pinn_empirical.py's
methodology) and the bimodal-collapse fraction (diagnose_pinn_transition.py's
methodology) side by side, so an improved R^2 can be confirmed as a real
fix rather than a fluke.
"""
import os
import sys
import itertools
import yaml
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.pinn_committor import CommittorNet, FeatureScaler, train

CONFIG_YAML = "config/chignolin.yaml"
with open(CONFIG_YAML) as f:
    cfg = yaml.safe_load(f)

pinn_cfg = cfg.get("pinn", {})
DATASET_CSV = pinn_cfg.get("dataset_csv", os.path.join(cfg["q_output_dir"], "pinn_dataset.csv"))
OUT_DIR = pinn_cfg.get("output_dir", os.path.join(cfg["q_output_dir"], "pinn"))
EMPIRICAL_CURVE_CSV = pinn_cfg.get(
    "empirical_curve_csv", os.path.join(cfg["q_output_dir"], "committor_1d_visits_sem.csv"))
SWEEP_OUT_CSV = os.path.join(OUT_DIR, "sweep_results.csv")

HIDDEN_DIMS = tuple(pinn_cfg.get("hidden_dims", [128, 128, 64]))
BATCH_SIZE = pinn_cfg.get("batch_size", 4096)
LR = pinn_cfg.get("lr", 1e-3)
VAL_FRAC = pinn_cfg.get("val_frac", 0.1)
SEED = pinn_cfg.get("seed", 0)
Q_UNFOLDED_BOUNDARY = pinn_cfg.get("q_unfolded_boundary", 0.1)
Q_FOLDED_BOUNDARY = pinn_cfg.get("q_folded_boundary", 0.9)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SWEEP_EPOCHS = 60  # shorter than the full 200 -- ranking configs, not training a final model
LAGS = [1, 5, 25, 125]
BOUNDARY_WEIGHTS = [1.0, 0.3]

print(f"Device: {DEVICE}")
print(f"Loading: {DATASET_CSV}")
df = pd.read_csv(DATASET_CSV).sort_values("frame_index").reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in ("frame_index", "Q")]
X_raw = df[feature_cols].to_numpy(dtype=np.float32)
frame_index = df["frame_index"].to_numpy()
Q = df["Q"].to_numpy(dtype=np.float32)

n_train = int(len(df) * (1 - VAL_FRAC))
scaler = FeatureScaler.fit(X_raw[:n_train])
X = scaler.transform(X_raw).astype(np.float32)

emp = pd.read_csv(EMPIRICAL_CURVE_CSV).dropna(subset=["p_folded_visits"]).sort_values("q_bin_center")
q_curve = emp["q_bin_center"].to_numpy()
p_curve = emp["p_folded_visits"].to_numpy()
in_transition = emp[(emp["p_folded_visits"] > 0.2) & (emp["p_folded_visits"] < 0.8)]
q_low, q_high = in_transition["q_bin_center"].min(), in_transition["q_bin_center"].max()
print(f"Transition-region Q range: [{q_low:.3f}, {q_high:.3f}]")

target_all = np.interp(Q, q_curve, p_curve)
in_curve_range = (Q >= q_curve.min()) & (Q <= q_curve.max())
q_sub_mask = (Q >= q_low) & (Q <= q_high)


def r2_rmse(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, np.sqrt(np.mean((y_true - y_pred) ** 2))


results = []
for lag, boundary_weight in itertools.product(LAGS, BOUNDARY_WEIGHTS):
    print(f"\n=== lag={lag}  boundary_weight={boundary_weight} ===")
    torch.manual_seed(SEED)
    model = CommittorNet(n_features=X.shape[1], hidden_dims=HIDDEN_DIMS)
    try:
        model, history = train(
            model, X, frame_index, Q, lag=lag,
            q_unfolded_boundary=Q_UNFOLDED_BOUNDARY, q_folded_boundary=Q_FOLDED_BOUNDARY,
            epochs=SWEEP_EPOCHS, batch_size=BATCH_SIZE, lr=LR, boundary_weight=boundary_weight,
            val_frac=VAL_FRAC, device=DEVICE, seed=SEED, log_every=SWEEP_EPOCHS,
        )
    except ValueError as e:
        print(f"SKIPPED: {e}")
        continue

    model.eval()
    with torch.no_grad():
        q_pred = model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    overall_r2, overall_rmse = r2_rmse(target_all[in_curve_range], q_pred[in_curve_range])

    trans_mask = in_curve_range & (target_all > 0.2) & (target_all < 0.8)
    trans_r2, trans_rmse = r2_rmse(target_all[trans_mask], q_pred[trans_mask])

    pred_sub = q_pred[q_sub_mask]
    frac_near0 = (pred_sub < 0.1).mean()
    frac_near1 = (pred_sub > 0.9).mean()
    frac_intermediate = 1 - frac_near0 - frac_near1

    results.append({
        "lag": lag, "boundary_weight": boundary_weight,
        "overall_r2": overall_r2, "overall_rmse": overall_rmse,
        "transition_r2": trans_r2, "transition_rmse": trans_rmse,
        "frac_intermediate": frac_intermediate,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    })
    print(f"overall R^2={overall_r2:.4f}  transition R^2={trans_r2:.4f}  "
          f"frac_intermediate={frac_intermediate:.1%}")

results_df = pd.DataFrame(results).sort_values("transition_r2", ascending=False)
results_df.to_csv(SWEEP_OUT_CSV, index=False)
print(f"\nSaved: {SWEEP_OUT_CSV}")
print("\nRanked by transition-region R^2 (best first):")
print(results_df.to_string(index=False))
