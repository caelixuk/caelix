# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# Exp01 — Baseline Field and Diagnostics

## Abstract

Exp01 demonstrates that a gravity-like \(1/r\) potential field emerges as the equilibrium state of a simple 3D discrete transport process.

A local neighbour-coupled relaxation (“traffic”) evolves a scalar field \(\phi\) on a cubic lattice. When the system reaches its cleanest far-field regime, the measured deoffset exponent is extremely close to \(-1\) (i.e. \(\phi(r)\propto 1/r\)).

In addition, the time-to-optimum scales with lattice size as a non-trivial power law:

\[
T(N) \approx A\,N^{\sqrt{3}}\,.
\]

This baseline is the calibration anchor for downstream experiments. If Exp01 drifts, later results should be treated as untrusted.

> **Key finding (hard boundary, N=512):** the auto-fit monitor reports \(T=220{,}512\) and a far-field exponent essentially equal to \(-1\).

---

## Methodology and constraints

This repository is intentionally structured so that the baseline result cannot be “smuggled in” by design.

What the evolution **does**:

- Updates \(\phi\) by local, neighbour-coupled transport on a 3D lattice (a relaxation / diffusion-like rule).
- Uses only local stencil information (6-neighbour coupling).

What the evolution **does not** do:

- It does **not** compute gravitational forces.
- It does **not** evaluate \(1/r\), \(1/r^2\), Euclidean distance, or any explicit “gravity” function as part of the update rule.

Diagnostics (analysis) **do** compute radius and distance-like quantities, but only to *measure* what emerged:

- `radial.py` bins \(\phi\) by shell radius to estimate the far-field exponent.
- The pipeline monitor samples intermediate snapshots and records the iteration where the exponent is closest to \(-1\).

This separation matters: the \(1/r\) profile is treated as an *observed property* of the equilibrium, not an input.

---

## Experiment 01A — Pipeline baseline (hard boundary)

### Canonical preset

Defined in `experiments.py`:

- `--n 512`
- `--steps 0` (no dynamic evolution after building the steady field)
- `--traffic-iters 220512`
- `--traffic-rate 0.10 --traffic-inject 1.0 --traffic-decay 0.0`
- `--k-index 1.0 --X0 8000 --ds 1.0 --eps 0.005`
- `--delta-load`

Outputs:

- `--dump-radial radial.csv`
- `--dump-radial-fit radial_fit.csv`
- `--dump-shapiro shapiro.csv`

### Measured best-fit age (N=512)

For **N=512 (hard boundary)**, the pipeline auto-fit monitor reports:

- `best_inv_r_age_iters = 220512`

This is the iteration where the **far-field deoffset power-law exponent** is closest to \(-1\), i.e. the cleanest measured \(1/r\) regime under the current baseline configuration.

Interpretation:

- `--traffic-iters` is the **run budget** (how long the field is allowed to settle).
- `best_inv_r_age_iters` is the **measured optimum** within that budget.

Small ±O(10–100) variation can occur if code paths or fit-window details change. Mechanically, the intended behaviour is stable.

---

## Results

### Emergent \(1/r\) profile

The central result of Exp01 is that the equilibrium field exhibits a far-field exponent consistent with \(\phi(r)\propto 1/r\) when measured by a deoffset log–log fit on the radial profile.

This is not asserted by the update rule; it is measured from the steady field.

> **Practical read-out:** the `best_inv_r_age` row in `radial_fit.csv` captures the snapshot where \(|p+1|\) is minimised (with \(p\) the fitted exponent).

### Dimensional scaling of convergence: \(T \propto N^{\sqrt{3}}\)

Across confirmed hard-boundary baselines, the iteration index of the cleanest \(1/r\) regime follows:

\[
T(N) \approx A\,N^{\sqrt{3}}\quad\text{with}\quad A \approx 4.47040\,.
\]

The coefficient \(A\) is fitted with the exponent fixed to \(\sqrt{3}\). This provides a compact “speed limit” constant for the baseline configuration (traffic rate, injection, boundary mode, fit window).

Confirmed dataset (hard boundary):

|  N  | best_inv_r_age_iters
|-----|---------------------
| 64  | 6,384
| 128 | 19,856
| 192 | 40,288
| 224 | 52,192
| 272 | 73,200
| 512 | 220,512

Model comparison (\(A=4.47040\), exponent \(\sqrt{3}\)):

|  N  | Actual T | Predicted T | Relative error
|---- |----------|-------------|---------------
| 64  | 6,384    | 6,008.2     | +6.26%
| 128 | 19,856   | 19,959.1    | -0.52%
| 192 | 40,288   | 40,284.7    | +0.01%
| 224 | 52,192   | 52,613.3    | -0.80%
| 272 | 73,200   | 73,645.0    | -0.60%
| 512 | 220,512  | 220,262.0   | +0.11%

Interpretation:

- The mid/large-N points cluster within sub-1% error.
- The \(N=64\) point deviates more (likely discretisation / boundary fraction / fit-window starvation). Treat the scaling law as an asymptotic guide unless/until a small-N correction is modelled.

A plausible geometric reading is that the characteristic relaxation timescale is governed by the cubic lattice's space-diagonal structure (hence \(\sqrt{3}\)), rather than the \(N^2\) scaling of textbook diffusion on simple domains.

---

## Experiment 01B — Sponge boundary baseline

A sponge boundary provides a damped alternative to the hard sink layer.

Configuration:

- `--traffic-boundary sponge`
- `--traffic-sponge-width <w>`
- `--traffic-sponge-strength <s>`

Scaled interior-sponge smoke tests used **w = N/16** (integers):

- N=64 → w=4
- N=128 → w=8
- N=192 → w=12
- N=512 → w=32

Observed `best_inv_r_age` results (sponge-on-interior, strength=1.0):

|  N  | sponge_width | best_inv_r_age_iters
|---- |------------- |---------------------
| 64  |     4        | 6,320
| 128 |     8        | 19,808
| 192 |    12        | 40,240
| 512 |    32        | 220,464

Notes:

- These are slightly earlier than the hard-boundary baselines at the same N (differences are small: tens of iterations at N≥128).
- At N=512, sponge baseline yields 220,464 (Δ = -48, ≈ -0.02% vs hard-boundary 220,512).
- This confirms that for large systems, the convergence dynamics are driven by internal transport limits (\(N^{\sqrt{3}}\)), independent of the boundary condition.
- The persistence of the interior \(1/r\) regime under an absorbing boundary supports the interpretation that the baseline profile is not a reflective “box artefact”.
- For sponge runs where N is padded (larger total lattice to preserve a fixed interior), the best-age clock responds to the **total** lattice size rather than the interior size.

---

## Output layout

Under a run bundle `<out>/<experiment>/<run_id>/`:

- `_csv/01A_pipeline_baseline_radial_n512_<run_id>.csv`
- `_csv/01A_pipeline_baseline_radial_fit_n512_<run_id>.csv`
- `_csv/01A_pipeline_baseline_shapiro_n512_<run_id>.csv`
- `_log/...` (run log)

`experiments.py` rewrites bare filenames (`radial.csv`, etc.) into run-unique bundle paths.

---

## Appendix A — CSV contracts

### `radial.csv`

Produced by `radial.py` from the final \(\phi\) field.

Header:

`r, mean_phi, count, std_phi`

Columns:

- `r`: integer shell index about the lattice centre.
- `mean_phi`: mean(\(\phi\)) over points in the shell.
- `count`: number of lattice points in the shell.
- `std_phi`: shell standard deviation.

Notes:

- `r=0` is the centre voxel.
- Fits typically exclude `r=0` (division-by-r and log-domain constraints).

### `radial_fit.csv`

Produced by `radial.py` from the final \(\phi\) field, plus (optionally) monitoring rows provided by the pipeline.

Base columns:

- `fit_kind`: fit identifier (e.g. `powerlaw_loglog`, `inv_r_linear`, `inv_r_then_loglog`, `scan_powerlaw`).
- `r_min`, `r_max`: fit window.
- `n_points`: number of samples used.
- `slope`, `intercept`, `r2`: primary fit outputs.

Optional/extra columns (depending on `fit_kind`):

- `A`, `B`, `r2_inv_r`: from \(\phi \approx A/r + B\).
- `p`, `log_a0`, `r2_loglog`: from \((\phi-B) \approx a_0\,r^p\).
- `scan_used`, `scan_median`, `scan_p16`, `scan_p84`: from slope scan summaries.

Monitoring/selection metadata (used on specific rows):

- `age_iters`: iteration index of the selected snapshot.
- `err_abs_p_plus_1`: absolute error against \(-1\).

`radial.py` is fail-fast on insufficient points (<8 usable samples).

---

## Appendix B — Provenance recommendations

For reproducibility, provenance headers should include:

- boundary mode + sponge settings
- monitoring cadence (`traffic.check_every` if set)
- fit window used for the far-field exponent
- any smoothing / sigma parameters (if present)