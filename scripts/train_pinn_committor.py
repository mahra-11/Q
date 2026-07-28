"""
Train a physics-informed committor network q_theta(x) on the dataset built
by build_pinn_dataset.py.

Loss = martingale term (q_theta(x_t) - q_theta(x_{t+lag}))^2, averaged over
every frame pair `lag` steps apart in the real trajectory, PLUS a boundary
term anchoring q_theta ~= 0 / 1 in the confidently unfolded/folded basins.
No binning, no pre-computed committor label -- every frame in the dataset
contributes signal, not just the sparse transition-region subset that the
earlier label-based regression (build_regression_dataset.py / LightGBM)
couldn't get past R^2 ~ 0.3 on (see README).

After training, evaluates q_theta across all frames, bins by Q the same way
committor_1d_visits_sem.py does, and plots the result against that empirical
curve as a sanity check -- the two should agree if the network learned a
real committor and not something degenerate (e.g. a constant, which the
martingale loss alone can't rule out).
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

LAG = pinn_cfg.get("lag", 1)
HIDDEN_DIMS = tuple(pinn_cfg.get("hidden_dims", [128, 128, 64]))
EPOCHS = pinn_cfg.get("epochs", 200)
BATCH_SIZE = pinn_cfg.get("batch_size", 4096)
LR = pinn_cfg.get("lr", 1e-3)
BOUNDARY_WEIGHT = pinn_cfg.get("boundary_weight", 1.0)
VAL_FRAC = pinn_cfg.get("val_frac", 0.1)
SEED = pinn_cfg.get("seed", 0)
Q_UNFOLDED_BOUNDARY = pinn_cfg.get("q_unfolded_boundary", 0.1)
Q_FOLDED_BOUNDARY = pinn_cfg.get("q_folded_boundary", 0.9)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Lag: {LAG} frame(s) ({LAG * 200} ps)")

print(f"Loading: {DATASET_CSV}")
df = pd.read_csv(DATASET_CSV).sort_values("frame_index").reset_index(drop=True)
feature_cols = [c for c in df.columns if c not in ("frame_index", "Q")]
print(f"Frames: {len(df):,}  Features: {len(feature_cols)}")

X_raw = df[feature_cols].to_numpy(dtype=np.float32)
frame_index = df["frame_index"].to_numpy()
Q = df["Q"].to_numpy(dtype=np.float32)

# Fit the scaler on the training block only (matches the train()'s own
# contiguous train/val split) so val statistics never leak into it.
n_train = int(len(df) * (1 - VAL_FRAC))
scaler = FeatureScaler.fit(X_raw[:n_train])
X = scaler.transform(X_raw).astype(np.float32)

torch.manual_seed(SEED)
model = CommittorNet(n_features=X.shape[1], hidden_dims=HIDDEN_DIMS)

model, history = train(
    model, X, frame_index, Q, lag=LAG,
    q_unfolded_boundary=Q_UNFOLDED_BOUNDARY, q_folded_boundary=Q_FOLDED_BOUNDARY,
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, boundary_weight=BOUNDARY_WEIGHT,
    val_frac=VAL_FRAC, device=DEVICE, seed=SEED,
)

ckpt_path = os.path.join(OUT_DIR, "committor_net.pt")
torch.save({
    "model_state": model.state_dict(),
    "scaler": scaler.state_dict(),
    "feature_cols": feature_cols,
    "hidden_dims": HIDDEN_DIMS,
    "lag": LAG,
}, ckpt_path)
print(f"Saved model: {ckpt_path}")

history_df = pd.DataFrame(history)
history_csv = os.path.join(OUT_DIR, "training_history.csv")
history_df.to_csv(history_csv, index=False)
print(f"Saved: {history_csv}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(history_df["train_loss"], label="train")
axes[0].plot(history_df["val_loss"], label="val")
axes[0].set_title("Total loss")
axes[0].set_xlabel("epoch")
axes[0].legend()
axes[1].plot(history_df["train_martingale"], label="train martingale")
axes[1].plot(history_df["val_martingale"], label="val martingale")
axes[1].plot(history_df["train_boundary"], label="train boundary", linestyle="--")
axes[1].plot(history_df["val_boundary"], label="val boundary", linestyle="--")
axes[1].set_title("Loss components")
axes[1].set_xlabel("epoch")
axes[1].legend(fontsize=8)
plt.tight_layout()
loss_png = os.path.join(OUT_DIR, "training_history.png")
plt.savefig(loss_png, dpi=200)
print(f"Saved: {loss_png}")

# --- Evaluate q_theta on every frame, bin by Q, compare to the empirical curve ---
model.eval()
with torch.no_grad():
    X_t = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    q_pred = model(X_t).cpu().numpy()

df["q_pred"] = q_pred
pred_path = os.path.join(OUT_DIR, "predictions.csv")
df[["frame_index", "Q", "q_pred"]].to_csv(pred_path, index=False)
print(f"Saved: {pred_path}")

N_BINS = 40
df["q_bin"] = pd.cut(df["Q"], bins=N_BINS, labels=False)
_, q_edges = pd.cut(df["Q"], bins=N_BINS, retbins=True)
q_centers = (q_edges[:-1] + q_edges[1:]) / 2
binned = df.groupby("q_bin")["q_pred"].mean().reset_index()
binned["q_bin_center"] = q_centers[binned["q_bin"].astype(int)]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(binned["q_bin_center"], binned["q_pred"], "o-", color="steelblue",
        label=r"PINN $q_\theta(x)$ (mean per Q bin)")

if os.path.exists(EMPIRICAL_CURVE_CSV):
    emp = pd.read_csv(EMPIRICAL_CURVE_CSV)
    ax.errorbar(emp["q_bin_center"], emp["p_folded_visits"], yerr=emp["sem_visits"],
                fmt="o-", color="crimson", ecolor="crimson", elinewidth=1, capsize=3,
                markersize=4, alpha=0.8, label="Empirical (visit-based, mean +/- SEM)")
else:
    print(f"NOTE: empirical curve not found at {EMPIRICAL_CURVE_CSV}, skipping overlay")

ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Q (fraction of native contacts)")
ax.set_ylabel("Committor")
ax.set_ylim(-0.05, 1.05)
ax.set_title("PINN committor vs. Q, compared to empirical curve")
ax.legend()
plt.tight_layout()
compare_png = os.path.join(OUT_DIR, "pinn_vs_empirical_committor.png")
plt.savefig(compare_png, dpi=300)
print(f"Saved: {compare_png}")

print("\nDone.")
