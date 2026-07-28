# Q — Fraction of Native Contacts & Committor Analysis

Companion pipeline to [chignolin-committor](https://github.com/mahra-11/chignolin-committor), built to add **Q**
(the fraction of native contacts, Best-Hummer-Eaton 2013 metric) as a reaction coordinate for Chignolin folding,
and to estimate an **empirical committor** as a function of Q from a single long MD trajectory — without
needing to launch new shooting simulations.

All scripts here were run on the NYU HPC cluster, driven from a `q_pipeline/` working directory
(`src/`, `config/`, `scripts/`, `logs/`) alongside the existing MDTraj-derived feature table for the
system (`chignolin_mdtraj_features_full_labeled.csv`).

## What's here

### Core module
- `src/native_contacts.py` — `compute_q()` / `compute_q_sample()`: loads a trajectory with MDTraj and
  computes Q per frame against a native/reference structure, using the Best-Hummer-Eaton smooth
  contact metric. Raises a clear error (instead of silently returning NaN) if no native contacts are
  found within the cutoff — small peptides like Chignolin may need a looser cutoff than the 0.45 nm default.

### Config
- `config/chignolin.yaml` — trajectory/topology paths, `q_folded_threshold` / `q_unfolded_threshold`
  (default 0.8 / 0.2), and `q_output_dir` for where Q results get written.

### Computing Q (SLURM jobs)
- `scripts/run_native_contacts.sbatch` — full trajectory, at the config's `stride`.
- `scripts/run_native_contacts_allframes.sbatch` — every single frame (stride override = 1).
- `scripts/run_native_contacts_trial.sbatch` — fast sanity-check run, samples only the first N frames
  via `mdtraj.iterload` instead of scanning the whole file.

### Q vs. radius of gyration
- `scripts/plot_q_vs_rg_allframes.py` — 2D histogram heatmap (linear color scale).
- `scripts/plot_q_vs_rg_logscale.py` — same, but log color scale. **Use this one** — frame counts span
  ~4 orders of magnitude, so a linear scale makes everything except the single most-populated cell look
  empty. This is the same reasoning src/contours.py in the main repo uses free energy (`-kT ln P`) instead
  of raw counts for exactly this kind of plot.
- `scripts/bin_counts_q_vs_rg.py` — full frame-count grid (including empty bins) plus a summary report,
  for inspecting the raw density before deciding how to visualize it.

### Committor estimation
The core idea: for every frame, scan **forward** along the one recorded trajectory to see whether Q
reaches the folded or unfolded threshold first — a deterministic 0/1 "Fate" label, not a probability by
itself. Averaging Fate over many frames with similar Q *is* the empirical committor estimate.

- `scripts/compute_frame_fate.py` — computes per-frame Fate (0.8/0.2 thresholds) and a naive binned
  committor-vs-Q curve (no autocorrelation correction).
- `scripts/compute_frame_fate_f09u01.py` — same, with folded=0.9/unfolded=0.1 thresholds instead.
- `scripts/plot_committor_vs_Q.py` — plots the naive curve, marks the 0.5 crossing (estimated transition
  state Q\*).

**Autocorrelation-corrected versions** — averaging raw frames overweights long, highly-correlated dwells
(e.g. one uninterrupted 40,000-frame stay in the folded state isn't 40,000 independent observations).
Two ways to correct for this, built to compare against each other:

- **Visit-based** (recommended; same "count once" logic `src/committor.py` in the main repo already uses
  for clusters, applied here to Q bins / grid cells instead): group consecutive frames that stay in the
  same bin/cell into one "visit," take the middle frame as its representative, average Fate over visits
  — not over raw frames. Sample size adapts automatically to the real dynamics: a cell visited once for
  40,000 frames = 1 sample; a cell visited 400 separate times = 400 samples.
  - `scripts/run_committor_visits.sbatch` — 2D (Rg x Q grid).
  - `scripts/committor_1d_both.py`, `scripts/committor_1d_visits_sem.py` — 1D (Q only).
- **Tiered step-sampling** (simpler alternative): sample every Nth frame within a bin, where N scales
  with how many frames are in it (`<=100`→step 1, `101-500`→step 2, `501-1000`→step 5, `1001-4000`→step 10,
  `4001-10000`→step 20, `>10000`→step 100). Doesn't distinguish "one long dwell" from "many short visits"
  the way the visit-based method does.
  - `scripts/run_committor_tiered.sbatch` — 2D grid.

- `scripts/compare_committor_methods.py` — merges the two 2D grid results and compares them (scatter +
  histogram of per-cell differences). In practice the two methods agree almost everywhere except the
  smallest, most ambiguous cells near the transition state — see `committor_1d_both.png` for the 1D
  version of this comparison, where they're nearly identical.

**Uncertainty**: raw standard deviation of Fate (0/1) is *always* large near committor=0.5 — that's the
maximum possible std for any bounded [0,1] variable, regardless of sample size, and mostly restates
"this is uncertain" rather than telling you how *confident* the estimate is. Use **standard error of
the mean** (`sem = std / sqrt(n_visits)`) instead — it actually shrinks with more samples.
`committor_1d_visits_sem.py` (and its `_f09u01` variant) plot this correctly.

### Visualizing the committor
- `scripts/plot_committor_visits.py` — 2D heatmap of the visit-based committor + a confidence panel
  (n_visits, log scale). Same blue/red/white convention as `src/contours.py`'s existing committor plots.
- `scripts/plot_committor_both.py` — visits vs. tiered heatmaps side by side + difference map.
- `scripts/plot_committor_scatter.py` — bubble-plot alternative to the filled-grid heatmap (one dot per
  occupied cell, sized by sample count) — generally more legible than the grid version, since only a
  fraction of the 60x60 grid cells are ever occupied.
- `scripts/plot_frames_vs_committor.py` (+ `_f09u01` variant) — histogram of how much of the trajectory's
  time is spent at each committor value. Expect two tall bars near committor≈0/1 (confidently
  unfolded/folded) and a much smaller one near 0.5 (the transition state — high free energy, rarely
  visited), visible only on a log y-axis.

## Result so far

For Chignolin (folded=0.8/unfolded=0.2 thresholds), Q behaves as a clean reaction coordinate: the
empirical committor vs. Q curve is a smooth, monotonic sigmoid with essentially no noise, crossing
P=0.5 at **Q\* ≈ 0.554**. The visit-based and tiered methods agree closely across the whole curve,
with only minor divergence in the sparsely-sampled transition region — a good sign that the estimate
is robust to the specific choice of autocorrelation-correction method.

### Regression (structural features → committor) hit a real ceiling

Predicting the visit-based committor label from the 64 MDTraj structural features (`contact_resX_resY`,
`phi/psi_sin/cos_N`) with LightGBM looked strong overall (test R² = 0.9888) but that's dominated by
the easy majority of frames sitting at committor ≈ 1. Restricted to the transition region
(committor in (0.2, 0.8)) — the part that actually matters for locating the transition state —
test R² was only **0.0230**. Diagnostics (label-noise splits, training on transition-region-only
frames, regularization sweeps — not yet in this repo, see gap note below) pushed that up to at best
~0.31–0.34, which looks like a real ceiling for *this* combination of binned labels and features, not
a tuning problem. This is a known, named difficulty in transition-path theory, not an artifact of this
pipeline: committor labels are inherently noisy near p=0.5 (any individual trajectory is either
crossing or not — the "signal" is stochastic), and data is scarce exactly where it's needed most.

**⚠️ Gap**: the regression-phase scripts (`build_regression_dataset.py`, `run_q_regression.sbatch`,
`check_*.py`/`run_check_*.sbatch` diagnostics) that produced the numbers above exist only on the HPC
filesystem (`/scratch/mma9420/q_pipeline/scripts/`) — this session didn't have HPC access, so they
still haven't been copied into this repo. Worth doing before they're lost.

## Physics-informed committor learning

Rather than continuing to fight noisy, pre-binned labels, this phase trains a neural network
`q_θ(x)` directly against the committor's defining physics instead of a label at all: it's a
**martingale** under the dynamics, `q(x_t) = E[q(x_{t+τ}) | x_t]`. Every consecutive (or short-lag)
frame pair in the 534,743-frame trajectory is a training signal — no Rg×Q binning, no
label-scarcity problem in the transition region, no separate "confidence" bookkeeping.

- `src/pinn_committor.py` — core module:
  - `CommittorNet` — MLP (`hidden_dims`, default `[128, 128, 64]`) with a sigmoid output, same 64
    structural features as input.
  - `FeatureScaler` — standardizes features, fit on the training split only.
  - `make_lagged_pairs()` — finds `(x_t, x_{t+lag})` row pairs by actual `frame_index` difference
    (not row position), so a pair is never accidentally formed across a gap left by an earlier
    inner-join.
  - `committor_loss()` — martingale term `(q_θ(x_t) - q_θ(x_{t+lag}))²` + a boundary term anchoring
    `q_θ ≈ 0` for confidently-unfolded frames (`Q ≤ 0.1`) and `q_θ ≈ 1` for confidently-folded frames
    (`Q ≥ 0.9`), weighted by `boundary_weight`. The boundary term isn't optional — the martingale loss
    alone is degenerate (any constant function scores zero), so it's what anchors the network to an
    actual committor instead of a trivial solution.
  - `train()` — training loop; splits into train/val by contiguous trajectory blocks (not randomly),
    since neighboring frames are highly autocorrelated and a random split would leak across it.
- `scripts/build_pinn_dataset.py` — merges the MDTraj feature table with `Q_values_allframes.csv`
  (same `frame_index = Time_ps / 200` merge key as the rest of the pipeline — **not** the raw `Frame`
  column) into `pinn_dataset.csv`. Unlike `build_regression_dataset.py`, this attaches no committor
  label at all — just features, `frame_index`, and `Q` for the boundary condition.
- `scripts/train_pinn_committor.py` — loads `pinn_dataset.csv`, trains `CommittorNet`, saves the model
  checkpoint (`committor_net.pt`), loss-curve plot, per-frame predictions, and — as a sanity check —
  a plot of the trained `q_θ` binned by Q overlaid on the empirical visit-based curve
  (`committor_1d_visits_sem.csv`). Good agreement between the two is evidence the network learned a
  real committor rather than something degenerate.
- `scripts/run_pinn_committor.sbatch` — SLURM job: installs CPU-only PyTorch into `committor_env` if
  missing (it wasn't previously needed — everything else in this repo is sklearn/LightGBM/MDTraj),
  then runs the two scripts above. CPU should be fine given how small Chignolin is; switch to a GPU
  partition (commented in the script) if training turns out to be slow in practice.
- `config/chignolin.yaml`'s `pinn:` section holds all the hyperparameters (`lag`, `hidden_dims`,
  `epochs`, `boundary_weight`, boundary thresholds, etc.) — see the inline comments there, especially
  around `lag`, which is the main open tuning knob (short lag risks single-step noise dominating the
  signal; long lag risks frames no longer being meaningfully correlated, weakening the constraint).

**Not yet run** — this was built without HPC filesystem access in this session, so it hasn't been
executed against the real data or compared to the regression baseline yet. Next step is running
`scripts/run_pinn_committor.sbatch` on the HPC and checking `pinn_vs_empirical_committor.png`.

