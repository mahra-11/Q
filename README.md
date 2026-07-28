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

