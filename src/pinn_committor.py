"""
Physics-informed committor learning.

Instead of fitting a regressor to noisy, pre-binned empirical committor
labels (the approach in build_regression_dataset.py / run_q_regression.sbatch,
which hit a hard ceiling in the transition region -- see README), this trains
a neural network q_theta(x) directly against the committor's defining physics:
it is a martingale under the dynamics,

    q(x_t) = E[q(x_{t+tau}) | x_t]

so (q_theta(x_{t+tau}) - q_theta(x_t))^2, averaged over consecutive/short-lag
frame pairs from the real trajectory, is a valid loss requiring no binning
and no pre-computed label at all. Every frame pair is signal, not just the
handful of frames that sit in the sparse transition region.

The martingale loss alone is degenerate (any constant function scores zero),
so it's combined with boundary conditions anchoring q_theta ~= 0 in the
confidently-unfolded basin and q_theta ~= 1 in the confidently-folded basin.
"""
import numpy as np
import torch
from torch import nn


class CommittorNet(nn.Module):
    """MLP q_theta(x) -> [0, 1], sigmoid output."""

    def __init__(self, n_features, hidden_dims=(128, 128, 64)):
        super().__init__()
        dims = [n_features, *hidden_dims]
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        layers += [nn.Linear(dims[-1], 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FeatureScaler:
    """Standardize features to zero mean / unit std, fit on training data only."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, x):
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean, std)

    def transform(self, x):
        return (x - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["mean"], state["std"])


def make_lagged_pairs(frame_index, lag):
    """
    Find row-position pairs (i, j) such that frame_index[j] - frame_index[i]
    == lag exactly -- skips any gap left by an earlier inner-join (e.g. a
    frame dropped because it had no matching row in one of the merged CSVs),
    so every pair is a real `lag`-frame step in the underlying trajectory,
    not an artifact of row position.

    Parameters
    ----------
    frame_index : np.ndarray, shape (n,), sorted ascending
    lag : int

    Returns
    -------
    idx_t, idx_t_lag : np.ndarray, shape (n_pairs,)
        Row positions such that frame_index[idx_t_lag] - frame_index[idx_t] == lag.
    """
    pos_by_frame = {f: i for i, f in enumerate(frame_index)}
    idx_t, idx_t_lag = [], []
    for i, f in enumerate(frame_index):
        j = pos_by_frame.get(f + lag)
        if j is not None:
            idx_t.append(i)
            idx_t_lag.append(j)
    return np.array(idx_t, dtype=np.int64), np.array(idx_t_lag, dtype=np.int64)


def boundary_targets(q, q_unfolded_boundary, q_folded_boundary):
    """
    Boolean mask + target value for frames confidently in one basin.

    Returns
    -------
    mask : np.ndarray, bool, shape (n,)
    target : np.ndarray, float, shape (n,) -- only meaningful where mask is True
    """
    unfolded = q <= q_unfolded_boundary
    folded = q >= q_folded_boundary
    mask = unfolded | folded
    target = np.where(folded, 1.0, 0.0)
    return mask, target


def committor_loss(model, x_t, x_lag, x_bc, y_bc, boundary_weight):
    """
    Total physics-informed loss: martingale term + weighted boundary term.

    Parameters
    ----------
    model : CommittorNet
    x_t, x_lag : torch.Tensor, shape (n_pairs, n_features)
        Feature vectors at time t and t+lag for the same set of pairs.
    x_bc : torch.Tensor, shape (n_bc, n_features)
        Feature vectors for frames confidently in the folded/unfolded basins.
    y_bc : torch.Tensor, shape (n_bc,)
        Boundary targets (0.0 or 1.0) matching x_bc.
    boundary_weight : float

    Returns
    -------
    total, martingale_loss, boundary_loss : torch.Tensor (scalars)
    """
    q_t = model(x_t)
    q_lag = model(x_lag)
    martingale_loss = torch.mean((q_t - q_lag) ** 2)

    if x_bc.shape[0] > 0:
        q_bc = model(x_bc)
        boundary_loss = torch.mean((q_bc - y_bc) ** 2)
    else:
        boundary_loss = torch.zeros((), device=x_t.device)

    total = martingale_loss + boundary_weight * boundary_loss
    return total, martingale_loss, boundary_loss


def train(model, X, frame_index, Q, lag, q_unfolded_boundary, q_folded_boundary,
          epochs=200, batch_size=4096, lr=1e-3, boundary_weight=1.0,
          val_frac=0.1, device="cpu", seed=0, log_every=10):
    """
    Train a CommittorNet with the martingale + boundary loss.

    Splits by contiguous trajectory blocks (not randomly) into train/val,
    since consecutive frames are temporally correlated -- a random split
    would leak near-identical neighboring frames across the split.

    Parameters
    ----------
    model : CommittorNet
    X : np.ndarray, shape (n, n_features) -- already scaled
    frame_index : np.ndarray, shape (n,), sorted ascending
    Q : np.ndarray, shape (n,)
    lag : int
    q_unfolded_boundary, q_folded_boundary : float
    epochs, batch_size, lr, boundary_weight, val_frac : training hyperparameters
    device : "cpu" or "cuda"
    seed : int
    log_every : int -- print every N epochs

    Returns
    -------
    model : trained CommittorNet
    history : dict of per-epoch lists -- train/val loss, martingale, boundary
    """
    rng = np.random.default_rng(seed)
    n = len(frame_index)
    split = int(n * (1 - val_frac))
    train_slice = slice(0, split)
    val_slice = slice(split, n)

    def build_indices(sl):
        idx_t, idx_lag = make_lagged_pairs(frame_index[sl], lag)
        idx_t += sl.start
        idx_lag += sl.start
        mask_bc, y_bc = boundary_targets(Q[sl], q_unfolded_boundary, q_folded_boundary)
        bc_idx = np.nonzero(mask_bc)[0] + sl.start
        return idx_t, idx_lag, bc_idx, y_bc[mask_bc]

    train_idx_t, train_idx_lag, train_bc_idx, train_bc_y = build_indices(train_slice)
    val_idx_t, val_idx_lag, val_bc_idx, val_bc_y = build_indices(val_slice)

    print(f"Train pairs: {len(train_idx_t):,} | Train boundary frames: {len(train_bc_idx):,}")
    print(f"Val pairs:   {len(val_idx_t):,} | Val boundary frames:   {len(val_bc_idx):,}")
    if len(train_idx_t) == 0:
        raise ValueError(
            f"No training pairs found at lag={lag} -- the trajectory has gaps "
            "larger than this lag, or too few frames. Try a smaller lag."
        )

    model.to(device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_xt = X_t[val_idx_t]
    val_xlag = X_t[val_idx_lag]
    val_xbc = X_t[val_bc_idx]
    val_ybc = torch.as_tensor(val_bc_y, dtype=torch.float32, device=device)

    history = {k: [] for k in
               ["train_loss", "train_martingale", "train_boundary",
                "val_loss", "val_martingale", "val_boundary"]}

    n_train_pairs = len(train_idx_t)
    n_train_bc = len(train_bc_idx)
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_train_pairs)
        epoch_losses = []
        for start in range(0, n_train_pairs, batch_size):
            batch = perm[start:start + batch_size]
            xt = X_t[train_idx_t[batch]]
            xlag = X_t[train_idx_lag[batch]]

            if n_train_bc > 0:
                bc_batch = rng.choice(n_train_bc, size=min(n_train_bc, batch_size), replace=False)
                xbc = X_t[train_bc_idx[bc_batch]]
                ybc = torch.as_tensor(train_bc_y[bc_batch], dtype=torch.float32, device=device)
            else:
                xbc = X_t[:0]
                ybc = torch.zeros(0, device=device)

            optimizer.zero_grad()
            total, mart, bnd = committor_loss(model, xt, xlag, xbc, ybc, boundary_weight)
            total.backward()
            optimizer.step()
            epoch_losses.append((total.item(), mart.item(), bnd.item()))

        train_loss, train_mart, train_bnd = np.mean(epoch_losses, axis=0)
        history["train_loss"].append(train_loss)
        history["train_martingale"].append(train_mart)
        history["train_boundary"].append(train_bnd)

        model.eval()
        with torch.no_grad():
            vt, vm, vb = committor_loss(model, val_xt, val_xlag, val_xbc, val_ybc, boundary_weight)
        history["val_loss"].append(vt.item())
        history["val_martingale"].append(vm.item())
        history["val_boundary"].append(vb.item())

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.5f} "
                  f"(mart={train_mart:.5f} bnd={train_bnd:.5f})  "
                  f"val_loss={vt.item():.5f} (mart={vm.item():.5f} bnd={vb.item():.5f})")

    return model, history
