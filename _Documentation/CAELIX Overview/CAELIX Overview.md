# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

This document is a running audit as we walk the (flat) CAELIX codebase. It is written in **stages** so we can tighten the system without losing the plot.

Document correct as of 08/02/26

---

## Goals

1. **Sanity:** identify invariants, sign conventions, and failure modes across experiments.
2. **CSV logging:** maximise what we write while keeping it **consistent**, **append-safe**, and **cheap**.
3. **Single contract:** every CSV row should carry a minimal run-identity payload so later analysis doesn't require guesswork.

## Repo inventory (24 modules)

**Entry / wiring**
- `cli.py` — argparse + output path normalisation + args → `PipelineParams`
- `params.py` — frozen stdlib dataclasses (behavioural defaults)
- `load.py` — micro → load mapping

**Execution spine**
- `core.py` — main entry; orchestrates pipeline + experiments
- `pipeline.py` — micro → load → phi → index → rays + fits
- `plumbing.py` — common wiring helpers
- `experiments.py` — experiment dispatch / wrappers

**Field + geometry**
- `lattice.py`, `walker.py`, `radial.py`, `rays.py`, `isotropy.py`

**Experiments**
- `coulomb.py`, `collider.py`, `corral.py`, `double_slit.py`, `relativity.py`, `oscillator.py`, `ringdown.py`, `soliton.py`, `stability.py`, `traffic.py`

**Output / viz**
- `visualiser.py`, `utils.py`

---

# Stage 1 — Wiring truth (CLI + Params + Pipeline)

## 1. `params.py` (contracts + defaults)

**Design choice:** frozen stdlib dataclasses. Good: defaults are explicit, immutable, and central.

Key defaults that materially affect results:
- `PipelineParams.seed = 6174` (also appears in CLI defaults)
- `LatticeParams`: `n=64`, `init_mode="sparse"`, `steps=200_000`, `p_seed=0.0008`
- `TrafficParams`: `mode="diffuse"`, `rate_rise=rate_fall=0.12`, `c2=0.25`, `gamma=0.10`, `dt=1.0`, boundary defaults
- Phase-8 scaffold knobs: `traffic_k`, `traffic_lambda` default 0.0
- Ringdown + soliton sweep defaults exist and are runnable (notably SG defaults)

**CSV implication:** if we don't log these defaults into each run row, we'll never know later whether a run used the standard contract or a mutated variant.

### Suggested “run identity contract” columns (minimum)
These should appear in *every* CSV written by the codebase (experiment outputs + radial/ray fits):
- `run_id` (or equivalent), `timestamp_utc` (or ISO local, but be consistent)
- `seed`, `n`, `steps` (or experiment-specific steps)
- `lattice_init_mode`, `p_seed`
- `traffic_mode`, `traffic_rate_rise`, `traffic_rate_fall`, `traffic_c2`, `traffic_gamma`, `traffic_dt`, `traffic_boundary_mode`, `traffic_sponge_width`, `traffic_sponge_strength`, `traffic_inject`, `traffic_decay`, `traffic_k`, `traffic_lambda`
- `k_index`, `r_fit_min`, `r_fit_max`
- plus experiment selectors: `walker`, `coulomb`, `collider`, `double_slit`, `corral`, `isotropy`, `relativity`, `oscillator`, `ringdown`, `soliton`, `bench_stability`

Rule: everything else can be derived if this is present.

---

## 2. `cli.py` (argparse + normalisation)

### Observations
- Owns flag definitions and **path normalisation** via `resolve_out_dir` + `resolve_out_path`.
- Uses `_normalise_out_flags(...)` to turn relative out files into absolute or `out/filename` paths.
- Builds a full `PipelineParams(...)` instance at the end.

### Good properties
- Fail-fast argument validation exists for several experiment modes.
- Ensemble constraints are enforced (`--plot` not allowed with `--ensemble`).
- `--seed` default is 6174.

### CSV / output surface area (from CLI flags)
These are output-relevant flags we need to trace to writers:
- `--log`
- `--dump-radial`, `--dump-radial-fit`, `--dump-shapiro`
- `--dump-walker`, `--walker-sweep-out`
- `--coulomb-out`, `--ds-out`, `--corral-out`, `--collider-out`, `--collider-out-dir`, `--iso-out`, `--rel-out`, `--osc-out`, `--ringdown-out`, `--soliton-out`
- Ensemble: `--ensemble-write-csv`
- Stability: `--bench-sweep-k-write-csv`

### Immediate logging opportunity
CLI already knows the active experiment, output paths, and the full param object.

So: we should ensure there is a single **run header row** (or sidecar JSON) emitted once per run that captures the full contract.

---

## 3. `pipeline.py` (micro → phi → radial → fits)

### Primary entry: `build_index_from_micro(...)`
Pipeline steps:
1. RNG from `params.seed`.
2. Either:
   - `delta_load` mode: single-site delta load with optional jitter.
   - else: lattice init (`sparse` or `multiscale`) + optional anneal.
3. `compute_load(...)`
4. `evolve_traffic(load, params.traffic, ...)` → produces `phi`.
5. `radial_profile(phi)` → `r, phi_r, counts`.
6. Fit windows: uses `params.r_fit_min` + `params.r_fit_max` (auto if 0).
7. Fits:
   - `fit_powerlaw_inverse` (log-log)
   - `fit_linear_inv_r`
   - `slope_scan_powerlaw`
   - plus a de-offset powerlaw attempt
8. Optional CSV writes:
   - `dump_radial_csv(path, r, phi_r, counts, provenance=...)`
   - `dump_radial_fit_csv(path, rows, provenance=...)`

### Existing provenance hooks
`build_index_from_micro()` already accepts:
- `csv_provenance`, `radial_provenance`, `radial_fit_provenance`

We should standardise this across all CSV writers.

### Sanity anchors we can enforce in logs
- `phi` finite
- fit windows valid
- `n_fit_pts` counts
- de-offset fit validity

---

# Stage 1 deliverables (what we now know)

1) Canonical defaults live in `params.py`.
2) CLI defines the output surface; we must trace each to a writer.
3) `pipeline.build_index_from_micro()` already supports provenance strings.

---

# Stage 2 — CSV writer inventory

Plan:
1. Enumerate every CSV writer in `utils.py`, `radial.py`, `rays.py`, and each experiment module.
2. Build a single table: file → function → what it writes → row schema → append semantics → provenance.
3. Identify missing run-identity columns and decide where to inject them.

---

## `utils.py` (shared helpers; provenance + logging infrastructure)

### What lives here (and why it matters)

`utils.py` is deliberately “boring” and import-safe. It contains:
- Fail-fast helpers: `_assert_finite(arr, name)`, `_as_float(x, name)`.
- Deterministic RNG: `_make_rng(seed)`.
- Stable numeric formatting: `fmt_g`, `fmt_f`.
- Plot bounds helper: `percentile_bounds`.
- Output path normalisation: `norm_path`, `here_dir`, `resolve_out_dir`, `resolve_out_path`, `resolve_out_file`, `ensure_parent_dir`.
- Time helpers: `now_s()` (perf_counter), `wallclock_iso()` + `wallclock_compact()`.
- Progress UI: `ProgressBar`, `make_weighted_progress_callback`, `estimate_cycles_baseline`.
- Tee logging: `TeeStdStreams` + `tee_to_file()`.
- Host probes / thread policy: `probe_apple_metal()`, `apply_conservative_thread_defaults()`.
- Fail-fast deps: `require_dependency()`, `require_dependencies()`.

### CSV-related primitives discovered

#### 1) Comment-header provenance builder
`write_csv_provenance_header(...) -> str`
- Produces `# key=value` lines that remain CSV-comment-friendly.
- Intended to be written **at the top of CSV artefacts**.
- Captures: producer, when, experiment (optional), cwd, python exe, command, plus arbitrary extra mapping.

This is the first “real” foundation for cross-file provenance.

**Immediate recommendation:** every CSV writer in the repo should support (or be wrapped to support) writing this header once when creating a new file.

#### 2) Path normalisation is strict (good)
`resolve_out_dir/resolve_out_path` includes a fail-fast guard `_reject_project_prefixed_relative(...)` to prevent accidentally doubling `Projects/<name>/...`.

**CSV implication:** fewer “mystery duplicate” output trees.

### Sanity / operational notes

- `wallclock_iso()` is **local time**, not UTC. That's fine for humans, but if we want deterministic provenance across machines, we should also log an explicit UTC timestamp (or include offset).
- `TeeStdStreams` makes stdout/stderr capture cheap and clean; the `ProgressBar` writes to `sys.__stderr__` to avoid log pollution.
- `apply_conservative_thread_defaults()` is intentionally single-path and uses env vars; it prints a summary line. If this affects reproducibility, we should also log chosen values in the run contract.

### CSV contract action item (Stage 2)

Define a single helper used by all CSV writers:
- “create file if missing → write provenance header → write CSV header row once → append rows”

We will implement this once we've enumerated all existing CSV writers (to avoid duplicating row schemas).

---

## `radial.py` (radial diagnostics + regression fits)

### What it does

- Collapses the 3D scalar field `phi[x,y,z]` into **integer-radius shells** around the lattice centre:
  - `ri = floor(sqrt(dx^2 + dy^2 + dz^2))`
  - Accumulates `sum(phi)` and `count` per shell, then `mean = sum / count`.
- This is the core “far-field check” for the expected ~`1/r` behaviour.

Implementation note:
- Shell accumulation is **Numba-jitted** (`_nb_radial_sums_counts`) and uses an upper bound `r_max_ub = floor(sqrt(3)*c)+2`.
- `radial_profile()` casts `phi` to `float32` (no copy if already compatible), returns:
  - `r` as `0..max_r` (float32)
  - `means` (float32)
  - `counts` (int64)

### Fit functions (and the assumptions they bake in)

1) `fit_powerlaw_inverse(phi_r, r, r_min, r_max)`
- Log-log regression on `phi ≈ a * r^p`.
- Uses mask: `r in [r_min, r_max]` and `phi_r > 0`.
- Requires **≥ 8 points**, else `ValueError` (fail-fast).
- Returns `RadialFit(slope=p, intercept=b, r2)` where `intercept` is **log(a)**.

2) `fit_linear_inv_r(phi_r, r, r_min, r_max)`
- Linear regression on `phi ≈ A*(1/r) + B`.
- Uses mask: `r > 0` and `phi_r > 0`.
- Requires **≥ 8 points**, else `ValueError`.
- Returns `RadialFit(slope=A, intercept=B, r2)`.

3) `fit_powerlaw_after_inv_r(phi_r, r, r_min, r_max)`
- First fits `A/r + B`.
- Then fits log-log on `(phi - B) ≈ a0 * r^p`, using only **positive residuals**.
- Returns a compact dict for CSV:
  - `A`, `B`, `r2_inv_r`, `p`, `log_a0`, `r2_loglog`, `n_points`.

4) `slope_scan_powerlaw(...)`
- Slides a sub-window (width = `width_frac*(r_max-r_min)`) across the window.
- Collects slopes from successful `fit_powerlaw_inverse` calls.
- If fewer than 3 usable windows: returns NaNs with `used` count.

### CSV writers (current behaviour)

- `dump_radial_csv(path, r, mean_phi, count, provenance=None)`
  - Opens with mode **`"w"`** (overwrites).
  - If `provenance` present, writes it *verbatim* at the top (expected `# ...` comment lines).
  - Then writes header row: `r, mean_phi, count`.

- `dump_radial_fit_csv(path, rows, provenance=None)`
  - Also **overwrites**.
  - Writes a fixed wide header:
    - Base: `fit_kind, r_min, r_max, n_points, slope, intercept, r2`
    - Extras: `A, B, r2_inv_r, p, log_a0, r2_loglog, scan_*`
  - Writes each row as `row.get(col, "")`.

### Sanity / logging notes

- **Positive-only filtering** (`phi_r > 0`) is a strong assumption. If `phi` can legitimately go negative (or oscillate around 0), these fits will silently ignore that structure. That may be correct for the sink-driven steady state, but it should be logged explicitly:
  - `n_pos_points` vs `n_total_points` per window.
- `r=0` is included in `radial_profile` output; fit masks exclude `r=0` for the `1/r` fit.
- Overwrite-only CSV behaviour is fine for “artefact per run” outputs, but filenames must be run-unique (or shift to append semantics for multi-run aggregates).
- Provenance is accepted as a raw string; we should standardise on `utils.write_csv_provenance_header(...)` and pass its output consistently.

---

## `rays.py` (2D ray tracing over index map + Shapiro-style regression)

### What it does

- Treats a 2D slice `n_map[y,x]` as an **index-of-refraction** field and traces a family of rays parameterised by impact parameter `b`.
- Computes the “delay integral” per ray:
  - `D(b) = ∫ (n - 1) ds`
- Fits a finite-domain proxy for log-like Shapiro scaling:
  - `D ≈ C + K * asinh(X0 / b)`

### Core API

- `ray_trace_delay(n_map, b_list, rp: RayParams) -> np.ndarray`
  - Returns `D` for each `b` (float64).
  - Delegates to numba kernel `_nb_ray_trace_delay(n_map, b_list, X0, ds)`.

- `fit_asinh_delay(D, b, X0) -> (K, C, R^2)`
  - Computes `u = asinh(X0/b)` then least-squares fit `D = C + K*u`.

### Integrator + sampling details

- Uses bilinear interpolation for `n(x,y)` and a central-difference gradient where possible.
- Inside the map interior (strictly `1 <= x < W-2` and `1 <= y < H-2`) it uses `_nb_bilinear_and_grad_interior`.
- Near edges, it samples `n` but forces `∇n = 0`.
- Outside the map, it assumes **vacuum**: `n = 1`, `∇n = 0`.

Ray update (midpoint-ish):
- Maintains unit tangent `(tx, ty)`.
- Uses an eikonal-style direction update:
  - `dn_ds = ∇n · t`
  - `dt/ds = (∇n - (dn_ds) t) / n`
- Performs a half-step to compute `(x_mid, y_mid, t_mid)` and accumulates `D += (n_mid - 1)*ds`.

### Safety / assumptions to log

- Contract says `n_map` is expected `>= 1` everywhere, but the tracer will still run if violated. We should log:
  - `n_min`, `n_max`, and whether any `n < 1` occurred.
- Edge handling sets gradients to zero, which can bias rays for large `b` or small maps. Log:
  - `frac_samples_interior` vs `edge_or_vacuum` (approx).
- `max_steps` is a hard cap; if a ray doesn't reach `x >= X0` by then, it silently stops. Log:
  - `hit_x0` count / `b_list` count.

### CSV writers

None in `rays.py` itself — it returns arrays/fit scalars. CSV responsibility should live in the caller (`pipeline.py` / `core.py`).

---

## `isotropy.py` (directional wave-speed calibration for telegraph solver)

### Purpose (as stated in-file)

- Measure effective group speed along axes vs diagonals for the **telegraph** solver.
- Treat this as a calibration step *before* making any SR/time-dilation claims.

### Method

1) Inject a single band-limited pulse at centre (3D Gaussian) for one tick.
2) Evolve with `traffic.evolve_telegraph_traffic_steps(...)`.
3) Record detector intensity `I(t) = phi^2` at multiple detector points.
4) Define arrival time as the tick of peak intensity (with optional sub-tick refinement).

### Implementation notes

- Pulse generator: `_gaussian_pulse(n, cx, cy, cz, sigma, amp) -> float32[n,n,n]`.
- Detector set: `_default_detectors(R)` includes:
  - 6 axes (±x, ±y, ±z)
  - 12 face diagonals (xy/xz/yz with sign variants)
  - 8 body diagonals (xyz with sign variants)
- Bounds are fail-fast: `_check_in_bounds(...)`.

Peak timing:
- Uses `argmax` over the trace of `phi^2`.
- Refines peak location with a 3-point parabolic vertex: `_parabolic_peak(tr, idx)` returning `t_peak_f = t_int + delta`.

Noise floor:
- Uses the last `tail_window=20` samples of each trace to compute `tail_I_mean` and `tail_I_std`, then simple SNR proxies.

Axis reference:
- Computes `c_axis_mean` from the 6 axis detectors:
  - `c_eff = R / t_peak_f` per axis detector.
- Also computes an amplitude diagnostic:
  - `Ir2_axis_mean = mean(peak_I * R^2)` over axis detectors.

Per-detector row includes (high-value columns)

- Geometry: `dx,dy,dz`, unit vector `ux,uy,uz`, `dist`, `kind`.
- Timing: `t_peak` (int), `t_peak_f` (float), plus `arrival_frac`, `tail_len`.
- Speeds: `c_eff`, `c_eff_int` and ratios vs `c_axis_mean`.
- Intensity: `peak_I`, `peak_phi`, `peak_I_r2`, `peak_I_r2_ratio`.
- Noise: `tail_I_mean`, `tail_I_std`, `snr_peak_over_tail_mean`, `snr_peak_over_tail_std`.

### CSV writers (current behaviour)

Two public entry points write CSVs (both **overwrite**):

1) `run_isotropy_calibration(..., out_csv, provenance_header=None, ...)`
- Writes optional provenance header first (expects `# ...` comment lines; caller supplies).
- Then writes a bundle of module-local `# key=value` comment lines (R/steps/sigma/amp/detectors/tail_window + key traffic params + `c_theory` + `c_axis_mean`).
- Writes the data table via `csv.DictWriter` with a fixed field list.

2) `run_isotropy_sigma_sweep(..., sigmas, out_csv, provenance_header=None, ...)`
- Same pattern, but emits `# isotropy_sigmas=...` and writes one row per detector per sigma.
- Reuses large buffers (`phi_buf`, `vel_buf`, `src_buf`) across sigma points to avoid repeated allocations (good).

Compatibility wrapper:
- `run_isotropy_test(...)` is a thin alias kept for `core.py` naming stability.

### Logging / provenance notes

- This module already does the right thing structurally: it supports a caller-provided provenance header *and* adds experiment-specific comment metadata.
- To standardise across the repo:
  - callers should pass `utils.write_csv_provenance_header(...)` into `provenance_header`.
  - we should ensure the provenance header includes the *full run identity contract* once (seed, n, steps, traffic params, etc.).

### Sanity footnotes (worth logging centrally)

- Arrival time is defined by peak intensity; that's robust, but if reflections or boundary effects occur, the true first-arrival could differ.
  - This module partially mitigates by also recording tail-window statistics and `arrival_frac`.
- Theoretical wave speed is logged as `c_theory = sqrt(traffic.c2)`; this assumes the discrete solver matches the continuum form under the chosen discretisation.

---

## `traffic.py` (hot-path field evolution: diffuse + telegraph + nonlinear)

### Role

Evolves the scalar field `phi` on a hard-bounded 3D lattice, driven by a sustained source (`load` / `src`).

This module is the primary performance lever: it is already mostly Numba kernels with fused neighbour loops to avoid full-grid temporaries.

### Modes (TrafficParams.mode)

1) **diffuse**
- First-order relaxer: inject into `phi`, then mix towards 6-neighbour mean.
- Uses asymmetric rates:
  - `rate_rise` if `target - phi > 0`, else `rate_fall`.
- Optional global `decay` applies multiplicative bleed each tick.
- Steady state around compact sources tends towards ~`1/r` with hard boundaries.

2) **telegraph**
- Damped second-order surrogate:
  - state is `(phi, vel)`.
  - update is Jacobi-style: read old → write new buffers.
- Parameters:
  - `gamma` is per-step velocity damping via `vel *= (1 - gamma)`.
  - `c2` acts like wave-speed squared.
  - `dt` advances position via `phi += dt * vel`.
  - `decay` multiplicative bleed on `phi` (and sponge may also damp).

3) **nonlinear**
- Telegraph stepping + local potential terms:
  - acceleration adds `-k*phi - lambda*phi^3`.
  - `traffic_lambda >= 0` enforced; `traffic_k` may be negative for double-well scans.

4) **sine_gordon**
- Telegraph stepping + bounded restoring force:
  - acceleration adds `-k*sin(phi)`.

### Boundary semantics

Hard boundaries throughout (no wrap).

Two boundary modes:
- `boundary_mode="zero"`:
  - hard clamp faces to zero each tick (`_nb_clamp_zero_faces`).
- `boundary_mode="sponge"`:
  - smooth quadratic damping ramp near faces (`_nb_apply_sponge`), no hard clamp.
  - enforced: `sponge_width > 0` and `sponge_strength in (0,1]`.

Important: when `boundary_mode="zero"`, sponge params must be exactly zero (fail-fast).

### Stability guard (telegraph / nonlinear / sine_gordon)

Fail-fast CFL-ish constraint:
- `sqrt(c2) * dt < (1/sqrt(3)) * 0.98`.

This is a very useful invariant to log in run metadata (both the computed CFL and the limit), because it explains “why it refused to run”.

### Obstacles (masked telegraph)

Telegraph kernels support a `mask`:
- `mask!=0` is a wall: clamp `phi=0`, `vel=0`.
- Masked neighbours are treated as zero in the Laplacian (no leaking through walls).

Used by `double_slit` (hard wall + apertures).

### Chirality / parity hooks

There is a chiral injection weighting patch (labelled 1/3, 2/3):
- `_nb_chiral_weight(ch, select)` gates coupling of injection to `src` by local `chirality[i,j,k]`.
- Chiral variants exist for unmasked and masked telegraph updates.

This means the solver can apply *handedness-dependent* sourcing without changing the Laplacian term.

### Public API

Batch allocators:
- `evolve_diffusion_traffic(load, tp)`
- `evolve_telegraph_traffic(load, tp)`
- `evolve_nonlinear_traffic(load, tp)`
- `evolve_sine_gordon_traffic(load, tp)`
- `evolve_traffic(load, tp)` (dispatch)

Steppers (advance existing state without per-tick full-grid copies):
- diffusion: `_nb_diffuse_steps(...)` behind `evolve_diffusion_traffic_steps(...)`.
- telegraph: `_nb_telegraph_steps(...)`, plus masked/chiral variants behind `evolve_telegraph_traffic_steps(...)`.

Wrappers enforce:
- dtype/contiguity via `np.ascontiguousarray(..., dtype=np.float32)`.
- shape agreement.
- finite checks via `_assert_finite(phi, ...)`.

### CSV writers

None. This module produces fields; logging happens in callers.

However: this is where many “sanity invariants” live, so the caller should always include these in the run contract:
- `mode`, `iters`, `inject`, `decay`
- diffusion: `rate_rise`, `rate_fall`
- telegraph+ : `gamma`, `c2`, `dt`, plus `cfl = sqrt(c2)*dt`
- boundary: `boundary_mode`, `boundary_zero`, `sponge_width`, `sponge_strength`
- nonlinear/SG: `traffic_k`, `traffic_lambda`
- mask/chiral: whether `mask` or `chirality` were provided, and `chiral_select`.

---

## `lattice.py` (ternary microstate init + anneal + multiscale correlated init)

### Role

Provides microstate initialisation for pipeline runs when **not** in `--delta-load` mode.

It builds an `int8` ternary field `s[x,y,z] ∈ {-1,0,+1}` which later becomes a “load” through `load.py`.

### `lattice_init(params, rng)` — sparse ternary seeding

- Allocates `s = zeros(n,n,n)` (int8).
- Samples `mask ~ U[0,1) < p_seed`.
- For masked sites, samples signs from `{-1,+1}`.
- Forces the centre voxel to `+1`.

**Implication:** occupancy fraction is approximately `p_seed`, with unbiased ± excitations.

### `lattice_init_multiscale(params, rng)` — correlated “malloc/inflation proxy”

Fast correlated init without per-voxel Metropolis.

1) Build a float32 correlated field `g` by summing multiscale octaves:
- Divisors: `[16, 8, 4, 2, 1]` (only those dividing `n` are used).
- For each divisor `d`:
  - sample coarse `noise` at `(n/d)^3`.
  - upsample by `repeat(d)` per axis.
  - accumulate with amplitude `amp` (halves each octave).
  - apply light smoothing (2 passes) after each octave.
- Apply final gentle smoothing (4 passes).

2) Threshold to match target occupancy:
- Uses `thr = quantile(|g|, q=1-p_seed)`.
- Sets `s = 0` then `s[|g| >= thr] = 1`.
- Applies sign: sites with `g < 0` become `-1`.
- Forces centre voxel to `+1`.

**Sanity note:** this produces spatially coherent ternary “domains” with roughly the requested occupancy fraction, without an explicit Hamiltonian.

### `lattice_anneal(s, params, rng, progress_cb=None)` — strict non-increasing “Metropolis-like”

- Performs `steps = params.steps` single-site proposals.
- Proposal picks uniformly among the two values not equal to `cur` (via `_nb_pick_prop`).
- Accept rule is **fail-fast monotone**:
  - compute local energy `e0` at site
  - set proposed state, compute `e1`
  - if `e1 > e0`, revert

Energy surrogate (local defect counter):
- mismatch cost: `j_mismatch * Σ |s_i - s_j|` over valid 6-neighbours (hard boundaries)
- occupancy cost: add `j_nonzero` if `s_i != 0`

This is not a physical Hamiltonian; it is a purposeful “edge/defect minimiser”.

Progress:
- If `progress_cb` is supplied, anneal is chunked in blocks of 4096 updates, still keeping the hot loop inside numba.

### Numba kernels

- `_nb_local_energy`: unrolled neighbour checks for speed.
- `_nb_smooth_pass`: 6-neighbour averaging with hard boundaries.
- `_nb_anneal_apply`: performs the accept/reject loop using only local energy deltas.

### CSV writers

None.

### Logging implications

If a run uses lattice microstate initialisation, the run contract should capture:
- `init_mode` (`sparse` vs `multiscale`)
- `p_seed`
- if anneal used: `steps`, `j_mismatch`, `j_nonzero`
- for multiscale: the divisor list and smoothing pass counts are fixed (implicit), but we should still log `init_mode=multiscale` because it materially changes correlation structure.

---

## `load.py` (microstate → non-negative load field)

### Role

Maps ternary microstate `s ∈ {-1,0,+1}` onto a **non-negative** scalar field `load ≥ 0`.

Intended interpretation: a proxy for local “processing cost”.

### `compute_load(s, lp)`

Outputs `float32[n,n,n]`.

Load is the sum of:

1) **Absolute activity cost**
- `load += w_abs * |s|`

2) **Neighbour mismatch cost** (6-neighbour edge proxy; *centred*)
- For each axis `0,1,2` we form edge mismatches against the `+1` neighbour:
  - `rolled = roll(s, -1, axis)`
  - `diff = |s - rolled|`
- Hard-boundary semantics: after `np.roll` we explicitly zero the wrapped face (so edges do **not** wrap).
- **Symmetric edge-sharing:** each undirected edge mismatch is counted once and then shared **50/50** between the two incident voxels. Concretely, we add `0.5*diff` to the current voxel and `0.5*diff` (shifted back by one) to the neighbour voxel.

This produces a centred (unbiased) stencil and avoids the directional “wind” that results when edge costs are assigned to only one endpoint.

Safety:
- `_assert_finite(load, "load")` at end.

### CSV writers

None. This is a pure field transform.

### Logging implications

Any run using microstate loads should include these in the run contract:
- `w_abs`, `w_mismatch`
- `load_mode="micro"` vs `delta_load` mode (since `delta_load` bypasses this path entirely).
- Note: mismatch is symmetric edge-shared (centred), not forward-assigned; this matters for isotropy claims.

---

## `walker.py` (Heavy Walker — moving delta source + wake/asymmetry diagnostics)

### Role

A self-contained diagnostic runner for **lag / wake effects** in the traffic solver.

Unlike the pipeline index build, this module:
- does **not** construct `n_map` or perform ray fits;
- directly steps the traffic solver on a 3D lattice with a **moving point source**.

### Path modes

- `path="linear"`:
  - axis-aligned unit steps (±x, ±y, ±z), repeated for `walker_move_steps`.
- `path="circle"`:
  - 4-connected (Manhattan) approximation to a circle in the x–y plane.
  - circle parameterised by `walker_circle_radius` and `walker_circle_period`.

A “hold phase” can run after motion:
- `walker_hold_steps` ticks with a fixed source at the final position.
- injection during hold uses `walker_hold_inject` (default 1.0).

### Stepping model

- State buffers are allocated once:
  - `phi` and `vel` as float32 cubes.
  - `src` as float32 cube.
- Warm-up phase:
  - sets `src[pos]=1` and runs `traffic.iters` ticks to build a baseline field.
- Per move tick:
  - updates position
  - sets `src[pos]=1` (unit delta)
  - advances either:
    - `evolve_telegraph_traffic_steps(phi, vel, src, tp, tick_iters)` if telegraph
    - else `evolve_diffusion_traffic_steps(phi, src, tp, tick_iters)`

“Mach-ish” estimate:
- for telegraph mode only, uses `c_est = sqrt(c2) * dt`.
- estimates per-step speed as `v_est = step_len / tick_iters`, where `step_len` is L1 length of the lattice step vector.
- reports `mach = v_est / c_est`.

### Measured diagnostics

Front/back wake split:
- `_front_back_sums(phi, pos, axis, sign, r_local)` sums `phi` in a local cube of radius `r_local` around the source.
- “front” and “back” are defined by the motion direction (`axis`, `sign`), using the dominant component of the step vector.

Per tick it logs:
- `phi_max`
- `front`, `back`
- `asym = (front - back) / (front + back + 1e-12)`
- `total_field = sum(phi)`
- `hubble = (front - back) / (total_field + 1e-12)`
- plus `v_est`, `c_est`, `mach`

Optional probe dipole metrics (`walker_probe_r > 0`):
- samples four cardinal points around centre at radius `probe_r` in x–y plane.
- records `phi_p0..p3`, `vel_p0..p3` and dipole magnitude/angle for both `phi` and `vel`.


### CSV writers 

Walker produces **two** CSV artefacts, both currently **overwrite** files and both currently **missing provenance headers**.

#### 1) Per-tick trace CSV (`--dump-walker`)

Code path:
- `dump_path = str(params.dump_walker).strip()`
- if non-empty:
  - `os.makedirs(dirname(dump_path) or ".", exist_ok=True)`
  - opens `open(dump_path, "w", newline="")`
  - writes a fixed header row and then one row per tick.

Schema (exact column order):
- `phase, path, t, x, y, z,
  cx, cy, cz, circle_radius, circle_period,
  step_dx, step_dy, step_dz, axis, sign,
  phi_max, front, back, total_field, hubble, asym,
  tick_iters, v_est, c_est, mach, stop_event,
  probe_r,
  phi_p0, phi_p1, phi_p2, phi_p3,
  vel_p0, vel_p1, vel_p2, vel_p3,
  dipole_mag_phi, dipole_ang_phi,
  dipole_mag_vel, dipole_ang_vel`

Notes:
- `stop_event` is `1` on the first hold tick, else `0`.
- Probe columns are blank unless `--walker-probe-r > 0`.

#### 2) Tick-iter sweep summary CSV (`--walker-sweep-out`)

Code path:
- `run_heavy_walker_sweep(params, tick_list, out_csv)`
- opens `open(out_csv, "w", newline="")`
- one row per tick value, where each tick value runs a full walker simulation with:
  - `walker_tick_iters = walker_hold_tick_iters = tick`
  - `dump_walker = ""` (suppressed)

Schema (exact column order):
- `move_tick_iters, hold_tick_iters, mach, move_asym_min, move_asym_last, hold_asym_max, hold_asym_final, overtake_step`

Notes:
- `overtake_step` is the first hold tick `t` with `asym > 0` (else blank).
- `mach` is taken from the **first move row**.

### Provenance patch target (clear and local)

Both writers should support the standard comment header string (from `utils.write_csv_provenance_header`).

Minimal change:
- Add an optional `provenance_header: str | None = None` to:
  - `run_heavy_walker(...)` and/or the `dump_walker` block
  - `run_heavy_walker_sweep(...)`
- When writing each CSV in overwrite mode:
  - if `provenance_header` is non-empty: write it first (ensure it ends with `\n`)
  - then write the CSV header row.

Caller policy:
- `core.py` already builds a run-scoped provenance header; it should pass that string down into walker so the artefacts become self-describing.

---

## `core.py` (thin dispatcher: startup checks, run identity, provenance, mode routing)

### Role

`core.py` is the runtime throat:
- applies conservative thread defaults
- checks deps
- parses CLI args (from `cli.py`)
- constructs **run identity** and output bundle structure
- dispatches into the selected runner (pipeline, walker, stability, experiment modules)

It aims to stay “thin”, but it is also where many CSV/provenance decisions currently live.

### Startup and hard invariants

- `apply_conservative_thread_defaults()` runs before importing hot modules (important for NumPy/Numba thread behaviour).
- `require_dependencies()` checks `numba/numpy/matplotlib` with fail-fast install hints.

### Run identity + bundling

- Parses args via `parse_cli()`.
- Mints a `run_id` and a human-friendly `when_iso` (wallclock, local).
- Uses `plumbing._ensure_bundle_dirs(run_dir)` to ensure a stable output layout:
  - `<out>/_csv/` and `<out>/_logs/`.
- Derives `exp_name` from the run folder (e.g. `07A_*`) via `plumbing._derive_exp_name`.

### Provenance pattern (good and already present)

- Builds a run-scoped provenance header using `utils.write_csv_provenance_header(...)`.
- Extends it per artefact by appending `# artefact=...`.

Example (pipeline path):
- `csv_prov = write_csv_provenance_header(... extra={run_id, n, steps, traffic_iters, seed, lattice_init})`
- `radial_prov = csv_prov + "# artefact=radial\n"`
- `radial_fit_prov = csv_prov + "# artefact=radial_fit\n"`
- `shapiro_prov = csv_prov + "# artefact=shapiro\n"`

It then *introspects* downstream signatures with `inspect.signature(...)` and passes whichever provenance kwarg the callee supports:
- prefers `radial_provenance` / `radial_fit_provenance` / `shapiro_provenance`
- otherwise falls back to generic `csv_provenance` / `provenance` / `provenance_header` / `csv_header`.

This is a pragmatic compatibility bridge while the repo converges on one naming scheme.

### Pipeline “single-run” path

- Resolves optional dump paths:
  - `--dump-radial`, `--dump-radial-fit`, `--dump-shapiro`
  - ensures parent dirs exist (downstream writers don't mkdir).
- Runs:
  - `n_map, metrics = build_index_from_micro(params, dump_radial_path=..., dump_radial_fit_path=..., **prov_kw, **pb_kw)`
- Prints an explicit metrics line (phi_max, slopes, r2, r-fit window, inv-r fit, de-offset fit, slope scan quantiles).

**Extra safety behaviour:**
If `--dump-radial-fit` is enabled, `core.py` appends a single derived summary row (`fit_kind="inv_r_plus_offset_powerlaw"`) to the radial-fit CSV if not already present. That row mirrors:
- `phi ≈ A/r + B` into the legacy slope/intercept/r2 slots,
- plus the de-offset exponent on `(phi-B)`.

This guards against misleading raw log-log slopes when there's a background offset.

### Ensemble path

- Supports `--ensemble` with optional multiprocessing.
- Enforces non-degenerate delta-load ensembles (requires `--delta-jitter > 0`).
- Collects per-seed rows from `pipeline._ensemble_one`.
- Writes `ensemble_metrics.csv` (overwrite) when `--ensemble-write-csv` is set.

**Provenance note:** the ensemble CSV currently has no provenance header.

### Experiment dispatch patterns

For many experiments `core.py`:
- derives a default output CSV path when not provided
- constructs a per-experiment provenance header with `artefact=<name>` plus relevant knobs
- passes it to the runner as `provenance_header=...`

Example: Coulomb does this correctly and includes: run_id, experiment, artefact, n, traffic mode/iters, seed, and all coulomb knobs.

### Logging gaps discovered (worth fixing later)

- Some CSV outputs (notably walker CSVs and `ensemble_metrics.csv`) lack provenance headers.
- `when_iso` is local time; consider also logging `when_utc` (or ISO with offset) inside provenance.
- The “compatibility bridge” signature inspection is useful today, but we should converge on a single kwarg name (likely `provenance_header`).

---

## `plumbing.py` (output bundling + naming conventions + small run utilities)

### Role

Provides the glue that keeps outputs predictable:
- derives stable experiment names from the filesystem context
- constructs a consistent run directory layout
- supplies tiny helpers so `core.py` stays dispatch-focused

### Output bundle contract

Two helpers are central:

- `_ensure_bundle_dirs(out_dir)`
  - Ensures these exist (mkdir -p semantics):
    - `<out_dir>/_csv/`
    - `<out_dir>/_logs/`

- `_bundle_path(out_dir, kind, filename)` (or equivalent pattern)
  - Ensures all artefacts land under the bundle subfolders.

**Why it matters:** we can safely turn on more CSV outputs without creating a sprawl of half-related files.

### Experiment name derivation

- `_derive_exp_name(out_dir)`
  - Uses basename of the output folder, trimming obvious date/run suffixes.
  - Intended to keep provenance stable even when you re-run with different timestamps.

This name is fed into provenance headers as `experiment=<exp_name>`.

### Run directory naming

- `_make_run_id(...)` (or `make_run_id`)
  - Produces a compact run identifier used in filenames and provenance.
  - `core.py` treats this as the primary row key for later joins.

### Logging utilities

- `_open_log_tee(out_dir, run_id)` (or similar)
  - Opens a `TeeStdStreams` into `<out_dir>/_logs/<run_id>.log`.
  - Prevents “console only” metrics from being lost.

### CSV implications

`plumbing.py` is where the “default output path” policy should ultimately live.

When `core.py` sees an experiment CSV path unset, it currently derives a default; that logic belongs here so:
- it is consistent across experiments
- it is testable
- and it doesn't bloat the dispatch code

### Sanity gap to resolve later

`plumbing.py` is structural, but it doesn't currently enforce that artefact filenames are **run-unique**.

Given several writers use **overwrite** mode, we should ensure defaults include `run_id` (or a compact timestamp) in filenames.

---

## `experiments.py` (thin per-experiment runners + CSV schema boundaries)

### Role

This module gathers “one-file-per-experiment” routines into callable entry points used by `core.py`.

It's where we can enumerate, in one place:
- which experiments exist
- which ones write CSV
- what each CSV row schema looks like
- and whether provenance headers are supported.

### Pattern observed across experiment runners

Most runners follow a stable skeleton:
1) validate params (fail-fast)
2) allocate state buffers
3) build a source/mask/chirality field
4) run the traffic solver for `steps` (often chunked)
5) compute diagnostics
6) if `out_csv` is set: write CSV (overwrite)

### CSV writers (as surfaced through core/CLI)

From CLI flags wired into experiment runners, the repo currently produces these experiment CSV artefacts:

- **Coulomb:** `--coulomb-out` → `coulomb.py` writer
- **Double slit:** `--ds-out` → `double_slit.py` writer
- **Corral:** `--corral-out` → `corral.py` writer
- **Collider:** `--collider-out` and `--collider-out-dir` → `collider.py` writer(s)
- **Isotropy:** `--iso-out` → `isotropy.py` writer
- **Relativity:** `--rel-out` → `relativity.py` writer
- **Oscillator:** `--osc-out` → `oscillator.py` writer
- **Ringdown:** `--ringdown-out` → `ringdown.py` writer
- **Soliton:** `--soliton-out` → `soliton.py` writer
- **Walker:** `--dump-walker` and `--walker-sweep-out` → `walker.py` writer(s)
- **Stability sweep:** `--bench-sweep-k-write-csv` → `stability.py` writer
- **Ensemble metrics:** `--ensemble-write-csv` → `pipeline.py` writer

`experiments.py` itself should not “own” the row schema for all of these, but it should clearly mark the boundary:
- where a runner emits a per-tick trace
- where it emits a per-run summary
- and which columns are guaranteed stable.

### Provenance support (expected / to verify)

Given the existing patterns:
- modules like `isotropy.py` already accept `provenance_header`.
- `core.py` also has a compatibility bridge to pass provenance through different kwarg names.

Actionable inventory goal for Stage 2:
- For each runner in `experiments.py`, record whether it:
  - accepts `provenance_header` (preferred)
  - or accepts `provenance` / `csv_provenance`
  - or does not support provenance at all.

Any runner that currently lacks provenance support should be treated as “needs patch” so CSV artefacts become self-describing.

### Why this matters for “log everything”

If we increase logging density (more columns, more per-step rows), we must avoid two failure modes:
- **schema drift**: every run writes slightly different columns
- **unjoinable CSVs**: no `run_id` / seed / config identity in the file

So, the standard policy should become:
- every *file* gets a provenance header
- every *row* gets `run_id` and the handful of key scalar params

---

## `coulomb.py` (two-charge signed-source diagnostic; interaction energy vs separation)

### What it does

A controlled “Coulomb-like” experiment on the same lattice/traffic machinery:
- places two signed delta sources along the x-axis at `(x1,c,c)` and `(x2,c,c)`
- solves **steady diffusion** (Poisson surrogate) for each separation `d`
- measures energy and constructs interaction energy by subtracting self-solves.

Hard constraint:
- **fail-fast** unless `params.traffic.mode == "diffuse"`.

### Fields solved per separation

For each `d` it solves:
1) **Pair field** (`phi_pair`): +q at `x1` and either +q (like) or -q (opposite) at `x2`.
2) **Self field 1** (`phi_self1`): +q at `x1`.
3) **Self field 2** (`phi_self2`): +q at `x2`.

Self energies are computed at *both* placements to reduce boundary artefacts on finite domains.

### Energies

Two energy surrogates are tracked:
- `E_phi  = Σ phi^2` (computed in numba as `_nb_phi_energy`).
- `E_grad = Σ |∇phi|^2` computed as the **6-neighbour edge sum** of squared differences, counting each undirected edge once (`_nb_grad_energy_6`).

Interaction energies subtract self energies from the pair energy:
- `E_int_phi  = E_pair_phi  - (E_self1_phi  + E_self2_phi)`
- `E_int_grad = E_pair_grad - (E_self1_grad + E_self2_grad)`

Renormalised interaction energies are also emitted:
- subtracts the far-field value at `d_max` so interaction tends to ~0 at large separation:
  - `E_int_*_renorm = E_int_* - E_int_*[d_max]`

### Solver behaviour and convergence

- Uses diffusion stepping only: `evolve_diffusion_traffic_steps(phi, src, tp_fast, step_chunk)`.
- Convergence test is based on `max_abs_diff(phi, phi_prev)` checked every `check_every` steps.
- If `tol == 0`, convergence is never checked (runs full `max_iters`).

Performance/consistency choice:
- overrides the traffic relaxation rates locally:
  - `tp_fast = replace(params.traffic, rate_rise=1.0, rate_fall=1.0)`
  - this is *only* for the Coulomb diagnostic.

Symmetry optimisation:
- if placements are symmetric about the centre (even `d`), it solves `self1` once and reuses it for `self2`.

Placement constraints:
- uses `delta_margin` to keep charges away from absorbing faces (`safe = max(2, margin+2)`)
- rejects separations that can't be placed within `[safe, n-1-safe]`.

### Fit summary (Phase-1)

After collecting rows, it fits the interaction **gradient energy** vs distance:
- metric: `fit_metric = "E_int_grad"`
- fit window: uses `d >= max(d_min, 4)`; requires `fit_n >= 6`.

Two models:
1) Inverse distance + offset:
- `y ≈ A*(1/d) + B` → `invr_A, invr_B, invr_r2`

2) Yukawa form + offset (grid-search k):
- `y ≈ A*exp(-k d)/d + B` via grid over `k ∈ [1e-4, 1.0]` (400 steps)
- emits `yuk_A, yuk_B, yuk_k, yuk_r2`
- also emits `yuk_half_range = ln(2)/k` if `k>0`.

Fit scalars are written into **every** row so single-row slicing still preserves the fit context.

### CSV output

Single CSV file, **overwrite** mode.

Schema is explicit and stable (`cols = [...]`), including:
- placement and solver metadata: `sign,q,d,x1,x2,iters_*`, converged flags + last deltas
- energies: `E_pair_*`, `E_self*_*`, `E_int_*`, renorm variants
- fit scalars: `fit_*`, `invr_*`, `yuk_*`

Provenance:
- accepts `provenance_header` (expected to be the run header from `core.py`).
- if provided, it appends extra comment lines:
  - `# coulomb_tp_rate_rise=...`, `# coulomb_tp_rate_fall=...`, `# fit_*_used=...`
- if not provided, it writes a minimal producer header itself.

**Net:** this module already conforms well to the “header provenance + fixed schema + overwrite artefact” pattern.

---

## `double_slit.py` (Young-style interference: telegraph + hard-wall mask + moving point source)

### What it does

- Requires `traffic.mode == "telegraph"` (fail-fast otherwise).
- Builds a **Dirichlet wall mask** at `x = wall_x` with either:
  - one slit (single-slit control), or
  - two slits separated in `y`, spanning all `z`.
- Injects a moving point source (“gun”) at `(x(t), cy, cz)`.
- Evolves the telegraph field one tick at a time with `mask` (and optional chirality).
- Samples intensity along a detector line: `I(y) = mean_{last window}(phi^2)` at `(detector_x, y, cz)`.
- Writes a CSV: `y,intensity`.

### Key parameters (from CLI-like `args`)

Geometry:
- `ds_wall_x` (default: `n//2` if <=0)
- `ds_detector_x` (default: `n-10` if <=0)
- `ds_line_z` (default: centre if <=0)

Slits:
- `ds_single_slit` (bool)
- `ds_slit_sep` (y separation between the two slits)
- `ds_slit_width` (width of each slit)

Gun motion:
- `ds_gun_start`
- `ds_gun_speed` (lattice sites per tick; `x = gun_start + int(t*gun_speed)`)
- `ds_gun_stop` (default: `wall_x - 10` if <=0)

Sampling:
- `ds_steps` (total ticks)
- `ds_burn` (ignore early transient)
- `ds_sample_every`
- `ds_window` (average over last `window` samples)

Source waveform:
- `ds_source ∈ {dc, sine, square}`
- `ds_amp`
- `ds_omega` (for sine)
- `ds_half_period` (for square)

Chirality:
- `chiral_select ∈ {-1,0,1}`
- `chiral_field ∈ {none, split_x, split_y, split_z}`
- if `chiral_select != 0`, `chiral_field` is required (fail-fast).

### Wall / mask semantics

- `mask[wall_x,:,:] = 1` is a hard wall.
- Slits are created by clearing the mask slice:
  - single slit: `mask[wall_x, y1_start:y1_end, :] = 0`
  - double slit: plus `mask[wall_x, y2_start:y2_end, :] = 0`
- Bounds are strict: slit endpoints must stay within `[1, n-2]`.

### Stepping

For each tick `t`:
- Clears `src`, computes `x(t)`.
- If `x < gun_stop`, injects the waveform at `src[x, cy, cz]`.
- Evolves one tick:
  - `phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, tp, 1, mask=mask, chirality=..., chiral_select=...)`
- If `t >= burn` and `(t-burn) % sample_every == 0`, samples `phi^2` along the detector line.

### Output path policy

- If `ds_out` is **explicit**, it writes there.
- If `ds_out` is **empty**, it derives a suite-consistent filename under `<out_dir>/_csv/`:
  - infers a code from the parent folder name prefix (e.g. `02A_*` → `02A`, else `02`).
  - requires `run_id` to be present in `provenance_header` (fail-fast).
  - name: `"{code}_double_slit_n{n}_{run_id}.csv"`.

This is good: it prevents silent overwrites when you forget to set `--ds-out`.

### CSV output (current behaviour)

- Opens **overwrite** mode.
- If `provenance_header` is non-empty, writes it verbatim first.
- Writes experiment comment lines:
  - `# double-slit`
  - `# n=... steps=... wall_x=... detector_x=...`
  - slit geometry and `cz`
  - gun motion + sampling window
  - source waveform params
- Data table:
  - header: `y,intensity`
  - rows: one per `y` with `intensity = pattern[y]`.

### Logging notes

- This CSV is intentionally minimal (a final pattern), but the **comment header already carries the knobs** needed to interpret it.
- For joinability, it still benefits from the global contract in `provenance_header` (seed, traffic params, etc.).

---

## `corral.py`  — “Quantum Corral” frequency sweep (masked telegraph)

### What it actually does

A frequency-response experiment inside a hard-walled cavity:
- constructs a **Dirichlet mask** (walls where `mask==1`), either:
  - `geom="sphere"` (true 3D sphere), or
  - `geom="cylinder"` (2D disk extruded through z).
- drives a **single point source** at the (offset) centre with a sinusoid:
  - `src[cx,cy,cz] = amp * sin(omega * t)`
- evolves **telegraph only** using the **masked** stepper:
  - `evolve_telegraph_traffic_steps(phi, vel, src, tp, 1, mask=mask)`
- for each omega, measures average stored energy over a late window, and records peak amplitudes.

Hard constraints:
- **fail-fast** unless `traffic.mode == "telegraph"`.
- fail-fast if the source lies inside the masked wall.

### Geometry + mask

Mask builder `_build_mask(n, geom, radius)`:
- allocates `mask = zeros(int8)` then sets `mask[...] = 1` outside the chosen region.
- “inside” count is `inside_count = sum(mask == 0)`.

### Sweep + measurement window

- `omegas = linspace(omega_start, omega_stop, omega_steps)`.
- For each omega:
  1) zero `phi` and `vel`.
  2) optional **warm-up**:
     - run `warm_steps` ticks of the same sinusoid.
     - continues phase by using `tt = t + warm_steps` during main burn so there's no discontinuity.
  3) main run is `burn_in` ticks.
  4) measurement window begins at:
     - `meas_start = int(burn_in * burn_frac)`
     - `meas_len = burn_in - meas_start`.

During measurement ticks it computes energy metrics over **inside-only** voxels.

### Energy metrics

`_energy_metrics(phi, vel, mask)` (inside-only):
- `e_phi = Σ(phi^2)`
- `e_vel = Σ(vel^2)`
- `e_tot = e_phi + e_vel`
- `max_phi = max(|phi|)`
- `max_vel = max(|vel|)`

Accumulation:
- averages `e_*` over the `meas_len` window.
- tracks `peak_phi`, `peak_vel` over the same window.

### CSV output

Single CSV file, **overwrite** mode (`open(out_csv, "w")`).

Header:
- writes optional `provenance_header` first (verbatim, ensured newline).
- then module-local comment lines:
  - `# corral_geom=...`
  - `# corral_radius=...`
  - `# corral_inside_count=...`
  - `# corral_omega_start=...`, `# corral_omega_stop=...`, `# corral_omega_steps=...`
  - `# corral_burn_in=...`, `# corral_meas_start=...`, `# corral_meas_len=...`
  - `# corral_warm_steps=...`, `# corral_burn_frac=...`, `# corral_amp=...`
  - `# corral_center_offset=dx,dy,dz`

Data table schema (one row per omega):
- `omega, e_phi, e_vel, e_tot, peak_phi, peak_vel`

So: **per-omega summary**, not per-tick.

### Return value

Returns a small dict:
- `best_omega` and `best_e_tot` (argmax of `e_tot` across the sweep)
- `elapsed_s`, `inside_count`, `out_csv`.

### Logging implications

This module already matches the house pattern:
- provenance header supported,
- run-unique file naming must be handled by the caller (since it overwrites).

One small footnote:
- the CSV does not log `traffic` params itself; it relies on provenance header to carry those.

---

## `collider.py` (two-walker collision; helical “spin” phase + optional nucleus/halo/detectors)

### Intent

A repeatable diagnostic: two moving point sources (“walkers”) approach head-on, each with a transverse helical orbit that imprints a handedness (“spin”) into the wake.

Primary comparison:
- same-spin (B spin = +1) vs opposite-spin (B spin = -1)

Not a physics claim; a controlled behavioural probe.

### Trajectories

Each walker is defined by `WalkerSpec`:
- head-on axis: `x(t) = x0 + vx*t`
- helical orbit in (y,z):
  - `ang(t) = phase0 + spin*omega*t`
  - `y(t) = y0 + radius*cos(ang)`
  - `z(t) = z0 + radius*sin(ang)`

Injection is nearest-cell delta:
- `src[xi,yi,zi] += q`

### Default spec generation (`_default_specs`)

- Uses `params.collider_vx`, `params.collider_orbit_radius`, `params.collider_orbit_omega`.
- Chooses a separation `sep` that makes collision time **integer**:
  - snaps `sep` so `t_c = sep/(2*vx)` is an integer (fail-fast if snap fails).
- For opposite spin (B = -1) it also chooses `phase0B = (2*omega*t_c) mod 2π` so transverse positions match at collision time (isolates “spin” from impact parameter).
- Stage-4 optional transverse offsets:
  - `collider_impact_b` (y) and `collider_impact_bz` (z), default 0.
- Default run length computed to keep walkers in-bounds.

### Core loop (`run_collider`)

- Always evolves using the **telegraph stepper**: `evolve_telegraph_traffic_steps(phi, vel, src, tp, 1)`.
  - **Note:** this module does *not* explicitly assert `traffic.mode == "telegraph"`; it simply uses the telegraph solver. That's an invariant worth enforcing later.

Per step:
- Clear `src`.
- Compute positions for A and B, inject `q` at each if in-bounds.
- Optional “hold” mode after collision:
  - `collider_hold`, `collider_hold_grace`, `collider_hold_steps`.
  - Defines `cut_step = t_c_int + hold_grace_steps`.
  - After `cut_step`, injection is shuttered (in_hold=1) and the field decays/evolves without driving.

### Optional stage features (all default OFF)

- **Stage-0:** `collider_enable_b` — disable walker B for single-body baselines.

- **Stage-3:** back-reaction (`collider_backreact`, `collider_backreact_k`)
  - samples local gradients at each walker (`_nb_grad_phi_at`)
  - applies acceleration to velocities, with:
    - mode: `repel` / `attract`
    - axes: `x` or `xyz`
    - vmax clamp: `collider_backreact_vmax`

- **Stage-6:** central nucleus (`collider_nucleus`)
  - third “anchor” source at the collision centre.
  - drive modes: `dc` or `sin` with `nucleus_q`, `nucleus_omega`, `nucleus_phase`.
  - injection is via a *small ball* (`_nb_inject_ball_uniform`) to avoid stationary delta blow-ups.
  - DC nucleus uses soft pin/damp constants (`nucleus_dc_alpha`, `nucleus_dc_beta`).

- **Stage-7:** velocity damping halo (`collider_halo`)
  - applies radial damping to `vel` inside radius `halo_r` around either:
    - `halo_center = nucleus` or `collision`
  - strength `halo_strength`, profile `linear|quadratic|exp`.
  - implemented as `_nb_apply_halo_damping(vel, cx,cy,cz, r, strength, profile)`.

- **Detectors:** calorimetry shells (`collider_detectors`)
  - precomputes two spherical shells as offset lists (stride-subsampled):
    - shell1: `[inner_frac, outer_frac] * n`
    - shell2: `[inner_frac, outer_frac] * n`
  - fail-fast if shell voxel coverage at the collision centre is zero.
  - optional octant binning (`collider_octants`) for directional asymmetry of shell energy.

### Diagnostics recorded

Per-step it tracks a *lot* (this file is intentionally “kitchen sink”):
- positions, velocities, separations (x-only and yz)
- `phi_max`, `vel_max`, energies `phi_energy`, `vel_energy`, total energy
- finite-difference energy changes (overall and during hold)
- mid-slab energies around collision x
- local ball energies around each walker and around collision centre
- shell energies + shell fractions + shell2-shell1 transport proxy
- gradient/backreaction terms + induced accelerations
- (optional) 8-octant energy bins for shell regions
- halo diagnostics (`halo_touched`, `halo_touched_max`)

### CSV output

Single CSV file, **overwrite** mode.

Pathing:
- If `out_csv` is provided: writes exactly there.
- Else writes into bundle-style:
  - `<out_dir>/_csv/{exp_code}_{variant}_{same_spin|opp_spin}_n{n}_{ts}.csv`
  - where `ts` is the basename of `out_dir`.

Header:
- Writes `provenance_header` if provided.
- Otherwise writes its own `write_csv_provenance_header(...)` with a wide `extra={...}` knob dump.
- Then module-local comment lines describing collider settings.
- Then a single massive fixed column header row.

**Minor correctness note:** `halo_touched_max` is included in the provenance extra dict before any steps run, so that header value will always be `0` even if the run later touches halo voxels. The per-row `halo_touched_max` column is the truthful one.

### Logging implications

Collider already meets the “self-describing artefact” standard (provenance header + fixed schema), but:
- it should hard-assert `traffic.mode == "telegraph"` for fail-loud consistency;
- and `out_dir` defaults still risk silent overwrite if you rerun into the same bundle timestamp (the filename key uses only `{ts}`), so the higher-level run bundling must be run-unique.

---

## `relativity.py` (telegraph “light clock” with gated detectors; stationary vs moving mirrors)

### What it is

A practical twin-paradox-style diagnostic built on the **telegraph** solver and a simple mirror detector model.

- Two mirrors A and B separated along **y** by `mirror_sep` (one-way distance = `mirror_sep`).
- **Stationary clock:** mirrors fixed at `x=cx`.
- **Moving clock:** mirrors translate at constant speed `v` along +x, using rounded voxel coords.
- A Gaussian pulse is injected at the currently “firing” mirror; a **tick** is counted when the *expected* mirror registers an arrival.

The module is explicit about what it is *not*:
- not specular reflection physics,
- not a Lorentz-rigid cavity,
- not an isotropy proof (see `isotropy.py` for calibration).

### Parameters (`RelativityParams`)

Key knobs (defaults shown in-file):
- lattice: `n=256`, `steps=2000`, `margin=12`
- telegraph: `c2=0.31`, `gamma=0.0`, `dt=1.0`, `decay=0.0`
- geometry: `mirror_sep=48`, `slab_half_thickness=1`
- motion: `v=0.20`, `v_ref=0.0` (small reference drift is supported to avoid lattice mode-locking)
- pulse: `pulse_amp=50`, `pulse_sigma=2.5`
- detection: `detect_threshold=0.05`, `refractory=6`

### Confinement geometry

Builds a strict z-slab mask:
- `_build_slab_mask(n, cz, half_thickness)` returns `mask=1` everywhere, then clears `mask[:,:,z0:z1]=0`.
- Mask semantics follow the telegraph masked solver: outside the slab, field is clamped (Dirichlet), reducing 3D clutter.

### Detection model (robust, phase-agnostic)

Arrival is detected via local **energy** over a small patch around the mirror:
- `_patch_energy(phi, vel, x, y, z0, z1, rad=1)` sums `Σ(phi^2 + vel^2)` over `(2*rad+1)^2 × slab_thickness`.

Thresholds are calibrated from a reference 1-tick injected pulse at mirror A:
- computes `e_ref` from a Gaussian injected into `ref_phi`.
- `e_high = frac_high * e_ref` (where `frac_high = detect_threshold`).
- `e_low = 0.25 * e_high`.

Each mirror has a Schmitt-style debounced detector:
- `_SchmittDetector(high, low, refractory)`
  - fires on rising edge crossing `high` when “armed”,
  - applies a refractory cooldown,
  - auto re-arms after cooldown (time-based, so it won't deadlock if the signal never drops below `low`).

### Flight-time gating + expected-mirror state machine

This is the core anti-false-positive logic.

- Effective wave speed estimate: `c_eff ≈ sqrt(c2) * dt`.
- Minimum plausible one-way time is enforced via:
  - `min_leg_ticks = max(6, ceil(1.05 * one_way / c_eff))`.
- Two independent universes are tracked:
  - stationary: `expect_s ∈ {"A","B"}`, `next_ok_s`
  - moving: `expect_m ∈ {"A","B"}`, `next_ok_m`

Per tick:
- Detectors only update when `t >= next_ok_*`, and only for the **currently expected** mirror.
- At most one hit per universe per tick is accepted.
- On accept:
  - increments hit count,
  - flips expectation A↔B,
  - schedules the next gate: `next_ok_* = t + min_leg_ticks`,
  - re-fires from the hit mirror on the next tick.

This prevents “self hits”, near-field ringing, and early-time disarming.

### Stepping

- Evolves two telegraph states in parallel (stationary and moving): `(phi_s, vel_s)` and `(phi_m, vel_m)`.
- Source injection is a small isotropic Gaussian (`_gaussian_pulse3d`) placed at the firing mirror.
- Uses `evolve_telegraph_traffic_steps(..., mask=slab_mask)`.

### Online derived diagnostics (written per row)

The CSV includes both raw detector samples and derived scalars:
- `T0_est`, `Tp_est`: estimated path/cycle timings derived from successive accepted hits.
- `c_axis_est`, `c_diag_est`: inferred speeds for axis vs diagonal legs (from observed leg distances / times).
- `ratio_est`: measured moving/stationary ratio.
- `gamma_corr_est`: an empirically corrected gamma-like scalar.
- `gamma_pred_axis`: axis-based SR prediction using `c_axis_est`.
- `anisotropy_A`: anisotropy factor between axis and diagonal propagation.
- `ratio_pred`, `ratio_err`: predicted ratio and relative error.
- `gamma_sr`: the textbook gamma using `v` and `c_eff`.

### CSV output

- Writer: `run_light_clock(...) -> out_csv`.
- Uses `resolve_out_path` + `ensure_parent_dir`.
- Opens **overwrite** mode.
- If `provenance_header` is provided, it is written first (verbatim `# ...` lines).
- Fixed schema header:
  - time + positions (`t, x_s, x_s_f, x_m, v_ref`)
  - mirror coords (stationary and moving)
  - raw mirror samples `phi_*` and energies `e_*`
  - hit flags + cumulative hit counts
  - derived timing/speed/gamma fields listed above.

### Logging gaps / follow-ups

- This module hardcodes its own `RelativityParams` separate from `PipelineParams`; if we want unified run contracts, `core.py` should inject *all* relevant traffic/boundary params into provenance (it already does on the caller side).
- It uses the masked telegraph stepper; for fail-loud consistency, upstream dispatch should ensure `traffic.mode == "telegraph"` when calling.

---

## `oscillator.py`  — Phase drift + lensing diagnostics (06B)

This file is not a generic “forced oscillator sweep”. It contains two related analysis-heavy experiments that reuse the same steady ~`1/r` background potential:

1) **Gravity phase drift (06B)**
   - Build a steady background potential `phi_bg` using the **diffusion** relaxer.
   - Run a **telegraph** simulation with two sinusoidal “local oscillators” (LOs) at different radii.
   - Estimate effective frequency / phase drift via complex demodulation.

2) **Lensing**
   - Treat the same background potential as an index field `n = 1 + alpha*phi`.
   - March a bundle of 2D rays through `∇n` to produce a deflection curve.

It is intentionally “kernel-light”: the stepping is delegated to `traffic.*` and the analysis is pure NumPy.

### Config dataclasses

- `OscillatorConfig`
  - probes: `r_near`, `r_far`, `axis ∈ {x,y,z}`
  - drive: `omega`, `drive_amp`
  - runtime: `steps`, `burn`, `warm`
  - demod: `demod_window`
  - output cadence: `series_every`

- `LensingConfig`
  - index strength: `alpha`
  - ray bundle: `ray_count`, `ray_span`
  - marching: `march_steps`, `ds`
  - launch: `x0`, `y0`, `theta0`

### Phase drift runner (06B)

Entry uses:
- diffusion to build `phi_bg` from a point-mass load (`mass_amp` at centre), then
- telegraph stepping to evolve an LO-driven field.

Probe placement:
- `near = center + r_near * axis_vec(axis)`
- `far  = center + r_far  * axis_vec(axis)`

The two local time-series are analysed after burn:
- `phi_near_ac`, `phi_far_ac` are AC traces (mean removed).
- Demodulation returns `omega_eff_*`, `f_meas_*`, and a per-tick phase-drift estimate.

### CSV outputs (overwrite, provenance supported)

This module writes up to **three** CSVs, all in **overwrite** mode, all `#`-comment-header compatible.

#### 1) Phase drift summary CSV (`out_csv`)

Writer: `_run_gravity_phase_drift_with_bg(..., out_csv, out_series_csv="", provenance_header="")`

- Opens `open(out_csv, "w", newline="")`.
- If `provenance_header` is empty, it builds one via `utils.write_csv_provenance_header(...)`.
- Adds experiment-specific `extra` keys into the provenance header (not a separate sidecar):
  - `artefact=oscillator_phase_summary`
  - `n, axis, r_near, r_far, omega, drive_amp, steps, burn, warm`

Schema (fixed field list):
- `n, axis, r_near, r_far,
  omega_drive, f_drive, omega_use, f_use,
  drive_amp, steps, burn, warm, demod_window,
  phi_near, phi_far,
  omega_eff_near, omega_eff_far,
  f_meas_near, f_meas_far,
  delta_f, frac_delta_f,
  phase_drift_rad_per_tick,
  amp_med_near, amp_med_far,
  wall_s`

Notes:
- `omega_use` may differ slightly from `omega_drive` depending on the demod analysis choice.
- `wall_s` is a wall-clock elapsed seconds measurement.

#### 2) Phase drift series CSV (`out_series_csv`, optional)

- Overwrite mode.
- Writes a provenance header if available, with `# artefact=oscillator_phase_series` appended.

Schema:
- `t, phi_near_ac, phi_far_ac`

This is **per-sample**, not per-tick (it honours `series_every`).

#### 3) Lensing CSV (`out_csv` in lens runner)

Lensing runner writes an overwrite CSV with provenance header support.

High-level contents:
- comment header includes lens knobs (`alpha`, ray bundle, march steps, ds, launch).
- data table emits one row per sample along each ray (sufficient to reconstruct trajectories / deflection curves).

(We will pin the exact lens schema when we audit the lens writer block later in this file — it is present and self-contained.)

### Logging implications

- The phase summary CSV is already in the “ideal” format: provenance header + stable schema + elapsed wall time.
- Series CSV is explicitly tagged via `# artefact=...` which makes later parsers unambiguous.
- This module imports and uses `write_csv_provenance_header` directly, so it does not depend on `core.py` bridging.

---

## `ringdown.py` (passive ringdown; sigma sweep with constant injection norm)

### What it does

Experiment 06C: a **single impulse** at `t=0`, then free evolution.

- One Gaussian source packet injected into telegraph `src` at the lattice centre.
- No re-injection after `t=0`.
- Evolve telegraph for `steps`.
- Measure:
  - late-time total energy survival (`E_final / E_land`), and
  - dominant late-time frequency from an FFT of a probe trace.

Hard constraint:
- **fail-fast** unless `traffic.mode == "telegraph"`.

Boundary policy is caller-owned; module only warns if `boundary_mode` is not `'zero'`.

### Injection normalisation (stability-first)

To make sigma sweeps comparable, the injected packet is L2-normalised per sigma:

- First computes unscaled norms for each sigma:
  - `E_src_unscaled(sigma) = Σ(src^2)`
- Chooses a constant target across the sweep:
  - `E_src_target = min_sigma E_src_unscaled(sigma)`
  - (so we **never amplify** a packet; only scale down)
- Per sigma:
  - `src_scale = sqrt(E_src_target / E_src_unscaled)`
  - `src *= src_scale`
- Logs `src_peak = max(|src|)`.

This is explicitly called out in the file comments as “stability-first”.

### Energy metrics

Total energy surrogate:
- `_total_energy(phi, vel) = Σ(phi^2) + Σ(vel^2)` (accumulated in float64).

Defines:
- `E_land`: a single snapshot captured at `land_tick` (sigma-aware settle time)
- `E_final`: energy at the end of the run
- `Efinal_over_Eland = E_final / E_land`
- `Efinal_over_Esrc_target = E_final / E_src_target`

Land tick heuristic:
- `land_tick = max(128, int(8*sigma), steps//50)`
- clamped to `< steps`.

### Probe + dominant frequency

Probe is the centre voxel:
- `probe[t] = phi[cx,cy,cz]`.

Also records:
- `probe_peak_abs = max_t |probe[t]|`.

Dominant frequency is measured on the last `probe_window` samples:
- `tail = probe[steps - probe_window : steps]`.
- `_dominant_freq(tail, dt)`:
  - subtract mean,
  - `rfft`, magnitude spectrum,
  - pick argmax excluding DC,
  - `freq_peak = k / (n * dt)` and `amp_peak = |FFT[k]|`.

`probe_window` is clamped:
- at least 64
- at most `steps//2`.

### CSV output

Single CSV file, **overwrite** mode.

- Requires `out_csv` path (fail-fast if empty).
- Writes optional `provenance_header` first if provided.
- Fixed header:
  - `sigma,E_src_target,src_scale,src_peak,E_land,E_final,Efinal_over_Eland,Efinal_over_Esrc_target,probe_peak_abs,freq_peak,amp_peak`

One row per sigma.

### Logging implications

This module is already close to “ideal” for auditability:
- fixed schema,
- provenance header support,
- explicit normalisation contract,
- derived scalars that let you compare runs cheaply.

Two things to ensure upstream:
- include `traffic.dt`, `traffic.c2`, `traffic.gamma`, `traffic.decay`, and boundary settings in the provenance header (core already does for most paths).
- avoid overwriting by ensuring `out_csv` defaults include `run_id`.

---

## `soliton.py`  — Nonlinear / Sine‑Gordon “lump survival” sweep with heavy diagnostics

This module is a **parameter sweep harness** designed to answer: “given a localised injection, do we get a persistent, localised structure, or does it phase‑convert / diffuse / go unstable?”

It is not a travelling‑packet tracker; it is a **stability / survival** benchmark with deliberately wide logging.

### Entry point

`run_soliton_scan(...) -> str` writes a **single CSV** (overwrite) with **one row per sweep point**.

Two modes depending on `traffic.mode`:

1) `traffic.mode == "nonlinear"`
   - Sweeps **phi^4 interaction** via `lambda_start..lambda_stop` (`lambda_steps`).
   - Uses the `traffic_k` already present in `traffic` (but also logs the caller‑supplied `k` argument into the CSV row).
   - Evolution default is `evolve_nonlinear_traffic_steps` (inject + telegraph‑style state).

2) `traffic.mode == "sine_gordon"` (fail‑loud contract)
   - The phi^4 lambda sweep is **forbidden** in SG mode.
   - Requires `lambda_start=lambda_stop=0` and `lambda_steps=1`, else raises.
   - Sweeps `sg_k` and/or `sg_amp` using `--soliton-sg-k-*` / `--soliton-sg-amp-*` ranges.
   - Uses geometric spacing when endpoints are positive; otherwise linear spacing.

### Initial condition

- Builds a **Gaussian source packet** at the centre with width `sigma` and amplitude `amp`.
- Optional `init_vev` (“vacuum expectation value”) pre‑loads the field with a uniform offset of sign `vev_sign`.

To keep analysis tractable at large `n`:
- Uses a **cropped cube** radius `crop_r = max(16, int(6*sigma))` for several expensive diagnostics.
- Also defines a larger `crop_r_outer` to detect “grey goo” (global conversion) versus localised lumps.

### Measurements

The file is intentionally “kitchen sink”, but it's coherent:

Energy (simple + physical breakdown):
- `E_land`, `E_final`, `E_ratio` using `Σ(phi^2)+Σ(vel^2)`.
- “Physical” Klein–Gordon style terms over the cropped cube:
  - kinetic, gradient, stiff (`0.5*k*phi^2`), phi^4 (`0.25*λ*phi^4`), and SG sine term.
  - both raw and a “shifted” variant to compare around the expected vacuum.

Localisation / size proxy:
- `Rg_land`, `Rg_final`, `Rg_ratio` (radius‑of‑gyration style metric) computed on the cropped cube.

Phase / excursion metrics:
- peak values at land/final and ratios vs `π`.
- fractions of voxels exceeding thresholds:
  - `frac_abs_phi_gt_pi_over2_*`, `frac_abs_phi_gt_pi_*`, `frac_abs_phi_gt_2pi_*`.

Outer‑shell / conversion detection:
- `outer_shell_abs_ratio_*`, plus a battery of outer/shell mean/abs/STD/percentile stats that distinguish “local blob” from “domain shift everywhere”.

### CSV output (overwrite, provenance supported)

- Overwrite mode: `open(out_csv, "w", newline="")`.
- If `provenance_header` is non-empty, it is written first (verbatim, newline enforced).
- Then a fixed, explicit header row (stable schema), then one row per sweep point.

The header is long (by design). It begins with:
- `idx, lambda, k, sigma, amp, sg_k, sg_amp, init_vev, vev_sign, phi0, steps, traffic_mode, traffic_boundary, traffic_sponge_width, traffic_sponge_strength, traffic_inject, traffic_decay, traffic_gamma, traffic_dt, traffic_c2, crop_r, src_peak, ...`

…and continues with grouped diagnostics:
- energies + ratios,
- `Rg_*`,
- peak/threshold fractions,
- outer/shell statistics,
- physical energy term breakdown,
- `vev_theory` and final mean errors,
- `wall_s`.

### Logging implications

- This module is already “max logging”: it provides a **stable wide schema** and supports provenance headers.
- The one gap is **artefact tagging**: it would benefit from `# artefact=soliton_scan` appended by the caller (or inside this writer) to make multi‑CSV bundles unambiguous.

---

## `collidersg.py` — Sine‑Gordon collider harness (sprites, wires, junction hardware)

This module is the **fail‑loud interaction testbed** for CAELIX's Sine‑Gordon mode. It lets us initialise multiple “sprites” (localised patches or procedural kink walls), evolve the SG field under confinement hardware (wire + k‑grid), and log per‑tick tracker readouts to CSV with a provenance header.

### What it is for

- **Experiment 09**: controlled collisions, scattering, routing and confinement behaviour.
- A minimal harness for “device engineering” on the SG substrate:
  - collisions (head‑on / off‑axis),
  - confinement (wire boxes, k‑grid, masks),
  - junction primitives (T, Y, OR‑junction),
  - measurement (trackers + fixed probes).

### Sprite model

Sprites are provided via `--collidersg-sprites` JSON (or a `.json` file). Each entry becomes a `SpriteSpec` and a runtime `ActiveSprite`.

Supported `kind` values:

- **`gaussian`**: procedural localised blob.
- **`kink_wall`**: procedural planar kink/antikink wall; the wall normal is taken from the requested velocity direction (`u = v/|v|`), and velocity initialisation is done by a finite shift.

Sprite asset stamping is enabled separately via `--collidersg-sprite-asset` (HDF5 patch) and is **not** a `kind`. When present, the asset patch is stamped into the full grids (phi/vel/src/load) for each sprite unless `track_only=true`.

Tracking flags:

- `track_only`: “microphone” sprite; does not inject field content.
- `fixed_probe`: disables motion updates during tracking and samples a local window only (prevents “heat‑seeking” detector drift).

### Entry points

- `run_collidersg(...) -> str` — runs a single collider scenario and writes a per‑tick CSV (overwrite). Returns the output CSV path.
- `parse_sprites_json(payload) -> List[SpriteSpec]` — parses `--collidersg-sprites` JSON (or a `.json` file path) into immutable specs.
- `initialise_kink_walls(n, sprites, dt, c2, sg_k) -> (phi0, vel0)` — procedural planar kink/antikink walls with arbitrary normal `u=v/|v|`.
- `initialise_from_sprite_asset(n, sprites, asset_h5, dt, sg_k, c2) -> (phi0, vel0, src0, load0, pll_info)` — stamps a sprite asset patch per sprite, with optional finite‑shift translational kick.

Mask builders used by wire confinement:
- `_build_t_junction_mask(...)`
- `_build_y_junction_mask(...)`
- `_build_or_junction_mask(...)` (shared dump cavity)

### Initialisation paths

`run_collidersg` has two mutually exclusive initialisation routes:

1) **Procedural sprites** (no `--collidersg-sprite-asset`)
   - `gaussian` stamps seed localised blobs.
   - `kink_wall` builds planar kinks/antikinks using the requested velocity direction as the wall normal.

2) **Sprite asset stamping** (`--collidersg-sprite-asset <path>`)
   - Reads an extracted HDF5 patch and stamps it into the global `phi/vel/src/load` grids.
   - A translational kick is injected using a sub‑voxel finite‑shift estimate.
   - `track_only=true` skips stamping (detectors become passive microphones).

Tracking‑specific contracts:
- `fixed_probe=true` disables motion updates during tracking and samples only the local window (prevents detector drift).
- For `kink_wall`, the commanded `vel` encodes the wall normal; tracker noise must **not** collapse the commanded velocity.

### CSV output (per‑tick) — stable core fields

`collidersg.py` emits a per‑tick CSV intended for quick inspection and downstream scripting. The full schema is defined in‑file (header row), but the stable core you can rely on is:

- `t` (tick)
- per‑sprite identifiers: `sid`, `status`
- per‑sprite kinematics: `x`, `y`, `z`, `vx`, `vy`, `vz`
- per‑sprite signal: `peak_abs`

When wire confinement / junction diagnostics are enabled, additional boolean counters are logged (e.g. trunk/branch occupancy and exit hits). These are geometry‑dependent and should be treated as *secondary* diagnostics.

The provenance header includes the confinement hardware identity (wire bounds, `sg_k_outside`, `wire_geom`, `junction_x/branch_len/branch_thick`, and OR‑junction dump parameters).

### Confinement hardware

`collidersg.py` supports a **wire confinement layer** built from a spatial `k_grid` plus an optional hard `domain_mask`:

- Inside the wire: `sg_k` (soft material).
- Outside the wire: `sg_k_outside` (stiff material).
- Boundaries: typically **Neumann** for topological charge preservation.

Wire geometry (`--collidersg-wire-geom`):

- `straight`: rectangular wire box.
- `t_junction`: a simple T‑junction slab feeding into Y branches.
- `y_junction`: symmetric Y‑junction mask.
- `or_junction`: OR‑style impact junction with a **shared dump cavity** to absorb backwash.

OR‑junction dump parameters:

- `--collidersg-dump-len`, `--collidersg-dump-throat`, `--collidersg-dump-y-pad` (geometry only; dissipation can be layered later if desired).

### Output and logging

- Writes a per‑tick CSV in overwrite mode, with a provenance header (append‑safe comment block) and a stable header row.
- Records per‑sprite position estimates, peak amplitudes and status.
- Includes geometry/hardware provenance (wire bounds, `sg_k_outside`, `wire_geom`, junction/dump parameters).

### Key lessons / known pitfalls

- **Planar objects are hard to track**: kink walls produce plateaus; use dominant‑axis tracking + centreline locking and fixed probes for detectors.
- **Velocity feedback is dangerous**: for kink walls, the commanded velocity encodes the wall normal; tracker noise must not collapse the commanded velocity.
- **Geometry is the gate**: fan‑out works reliably (piston → splash → split). Fan‑in is kinematically hostile without engineered chambers or collision logic.

---

## `stability.py`  — face-lock stability benchmarks (microstate neighbour protocols)

This is **not** a telegraph PDE stability bench. It is a discrete, local 3D microstate benchmark testing a specific claim:

> A stable “particle” (centre excitation with clean faces) requires enforcing **all 6 face-neighbour constraints**.

It implements two controlled experiments:

1) `stability_benchmark(rng, sp) -> Dict[str,float]`
2) `stability_face_subset_sweep(rng, sp, k) -> FaceSweepSummary`

Both operate on a tiny local workspace (`n=9`) around the centre, using an explicit noise + repair model.

### Invariant (“particle exists”)

A tick counts as stable only if:
- centre voxel is restored to `+1`, and
- **all 6 face neighbours are 0**:
  - `(±1,0,0)`, `(0,±1,0)`, `(0,0,±1)`.

If any face is non-zero after maintenance, the particle has **decayed** and the trial stops.

### Noise model (per tick)

Noise is applied to the full 26-neighbour shell around the centre (all `dx,dy,dz ∈ {-1,0,1}` excluding (0,0,0)):
- each shell site is corrupted to `±1` with probability `sp.p_noise`.
- the centre may be knocked to `0` with probability `sp.p_center_flip`.

### Maintenance model

A protocol is defined by a chosen set of neighbour offsets `offsets`.
Each tick:
- the protocol scans only those offsets;
- any scanned neighbour found non-zero is repaired back to `0`.

After repairs, the strict face-invariant is checked (all 6 faces must be 0). If it passes, the centre is restored to `+1` and the next tick proceeds.

### `stability_benchmark()` (hand-picked protocols)

Defines four protocols:
- `A_planar4`: the 4 in-plane faces (±x, ±y)
- `B_faces6`: all 6 faces (±x, ±y, ±z)
- `C_corners8`: the 8 corners (±1,±1,±1)
- `D_faces6_plus2`: faces6 plus two corners (1,1,1) and (-1,-1,-1)

For each protocol it runs `sp.trials` independent trials of length up to `sp.ticks` and returns a flat dict containing:
- `{name}.mean_survival` (mean survived ticks)
- `{name}.p_full` (fraction surviving the full `sp.ticks`)
- `{name}.reads_per_tick` (mean neighbour reads per tick, normalised by mean survival)
- `{name}.repairs_per_tick` (mean repairs per tick, normalised likewise)

This explicitly trades off:
- stability (mean survival / p_full) versus
- protocol overhead (reads/repairs).

### `stability_face_subset_sweep()` (bulletproof k-of-6 sweep)

Sweeps **all combinations** of `k` offsets chosen from the 6 faces.
For each subset it computes:
- `mean_survival`
- `p_full`

Then returns a `FaceSweepSummary` dataclass capturing best and worst subsets:
- `k`
- `best_name`, `best_p_full`, `best_mean_survival`
- `worst_name`, `worst_p_full`, `worst_mean_survival`

Names are rendered with stable face labels and an explicit “missing=…” list, e.g.
- `[+x,-x,+y,-y] missing=[+z,-z]`.

### CSV output

None. This module returns metrics objects only.

If we want these results logged, the right place is `core.py` (or a small wrapper) to write a CSV using `utils.write_csv_provenance_header`.

### Logging implications

- This is a **microstate** stability diagnostic, not a field evolution one.
- The key knobs to log for comparability are: `sp.trials`, `sp.ticks`, `sp.p_noise`, `sp.p_center_flip`, and `k` (for the sweep).

---

## `pipeline.py` (micro/delta → load → phi → n_map → rays + Shapiro fit; ensemble runner)

### Role

This is the “physics spine” of the repo:
- defines how a run turns either a microstate or a delta-load into a steady scalar field `phi`
- converts `phi` into an index map `n_map`
- traces rays through `n_map` and fits a Shapiro-style asinh law
- optionally runs ensembles and emits per-seed summary CSV

It contains both:
- **single-run** logic (called by `core.py`), and
- **ensemble** logic (also called by `core.py`).

### Core pipeline stages (single run)

1) **Load construction**

Two mutually exclusive modes:

- Microstate mode (default)
  - `s = lattice_init(...)` or `lattice_init_multiscale(...)`
  - optional `lattice_anneal(...)` if anneal enabled
  - `load = compute_load(s, lp)`

- Delta-load mode (`--delta-load`)
  - `load = zeros` then `load[cx,cy,cz] = delta_amp`
  - optional jitter:
    - chooses `(dx,dy,dz)` uniformly in a small cube and applies offset
    - used for ensemble de-degeneracy (`core.py` enforces `delta_jitter > 0` for ensemble)

**Logging imperative:** all delta-load knobs must be in provenance because they bypass micro init.

2) **Traffic evolution**

- `phi = evolve_traffic(load, params.traffic)`
- traffic mode drives the stepping kernel:
  - diffuse → steady relax
  - telegraph/nonlinear/SG → wave-like with `vel` (but pipeline mainly assumes diffuse unless explicitly requested)

3) **Index map construction**

- Converts 3D `phi` into a 2D index map `n_map`.

Observed pattern in this repo family:
- take a central z-slab or z-slice, then map `phi` into `n` using:
  - `n = 1 + k_index * phi_slice` (clipped if requested)

Pipeline logs:
- `n_map.min/max`, `phi_max`, and whether any `n < 1` (important for rays).

4) **Radial diagnostics + fits**

- calls `radial_profile(phi)`
- calls fit helpers from `radial.py`:
  - log-log powerlaw slope (expects ~ -1)
  - linear inv-r fit `A/r + B`
  - de-offset fit attempt
  - slope scan quantiles

Optional CSV writers (overwrite):
- `dump_radial_csv(..., provenance=radial_provenance)`
- `dump_radial_fit_csv(..., provenance=radial_fit_provenance)`

These already accept provenance strings.

5) **Rays + Shapiro fit**

- builds `b_list` (impact parameters) with a mixed schedule:
  - dense near centre plus a log-spaced tail (exact recipe is fixed in-file)
- `D = ray_trace_delay(n_map, b_list, rp)` where `rp` includes:
  - `X0` (integration half-length)
  - `ds` (step size)
  - `max_steps`

- fits the asinh law:
  - `u = asinh(X0 / b)`
  - `D ≈ C + K*u`

Shapiro output includes:
- `K` (slope)
- `C` (intercept)
- `r2_line`

In addition, pipeline computes a “mean_fit_r2” across a small family of fits (e.g. different b windows) and logs:
- `mean_fit_r2`
- `min_fit_r2`

Optional CSV writer (overwrite):
- `dump_shapiro_csv(path, rows, provenance=shapiro_provenance)`

(Writer function lives in pipeline or a nearby helper module; `core.py` treats it as a pipeline surface.)

### Ensemble runner

- `_ensemble_one(seed_i, params, ...)` runs the same pipeline with:
  - seed changed per member
  - delta-load jitter (required by `core.py`)
  - returns a dict of scalar metrics (phi_max, radial slope/r2, shapiro K/r2, etc.)

- `run_ensemble(params, seeds, workers)` collects rows.

CSV writer surface:
- when `--ensemble-write-csv` is set, `core.py` writes `ensemble_metrics.csv` (overwrite) from the list of per-seed dict rows.

**Current gap (confirmed earlier):** ensemble CSV has **no provenance header**.

### CSV outputs owned by pipeline

- `dump_radial_csv` (via radial.py) — overwrite, provenance supported.
- `dump_radial_fit_csv` (via radial.py) — overwrite, provenance supported.
- `dump_shapiro_csv` (pipeline-owned) — overwrite, provenance supported via `core.py` bridge.
- `ensemble_metrics.csv` (written from core/pipeline metrics rows) — overwrite, **no provenance header**.

### Logging action items discovered here

1) **Standardise provenance kwarg**
Pipeline currently accepts several names (`csv_provenance`, `radial_provenance`, `radial_fit_provenance`, `shapiro_provenance`).

Repo should converge on:
- `provenance_header` everywhere.

2) **Ensemble CSV needs a header**
Add `write_csv_provenance_header(... artefact=ensemble_metrics, experiment=...)` plus:
- seed list or hash
- delta-load/jitter knobs
- and any per-ensemble invariants.

3) **Rays termination logging**
For honest diagnosis we should include in shapiro CSV (or metrics):
- `hit_x0_count`
- `max_steps_reached_count`
- `n_lt_1_count` (or at least min(n)).

4) **b-list identity**
Since b-schedules change analyses, log:
- `b_schedule_kind` + params (or the full b_list hash) in the shapiro provenance comment header.

---

## `visualiser.py` (plotting + CSV readers; establishes “consumed columns”)

### Role

This module is the “truth of usage” layer:
- it encodes which CSV columns are relied upon downstream
- which means it implicitly defines which columns are **stable API** and which are expendable.

If we're going to “log everything”, we still need to know what's actually *read*.

### Inputs it expects

`visualiser.py` is primarily a reader of:
- `radial.csv` and `radial_fit.csv`
- `shapiro.csv`
- `ensemble_metrics.csv`
- experiment outputs (walker, collider, isotropy, etc.) depending on which plot flags are wired.

It typically assumes:
- CSV comment headers start with `#` and should be ignored by the parser.
- first non-comment line is the header row.

If any writer emits non-`#` preambles, visualiser will misparse.

### Plot surfaces (high-level)

Based on the CLI design, this module likely provides:

1) **Radial plots**
- `r` vs `mean_phi` with log scales
- powerlaw fit overlays (`slope`, `r2`)
- inv-r fit overlay (`A`, `B`, `r2_inv_r`)

2) **Shapiro plots**
- `D(b)` vs `asinh(X0/b)` with best-fit line
- residual plots (if implemented)
- fit-quality annotations using `r2_line`, `mean_fit_r2`, `min_fit_r2`

3) **Ensemble plots**
- histograms / scatter for key metrics:
  - `phi_max`, `radial_slope`, `radial_r2`, `shapiro_K`, `r2_line`, etc.

4) **Experiment-specific plots**
- walker asymmetry time series
- collider energy transport vs time
- isotropy speed ratios

### “Consumed columns” policy (what we must not break)

Even before refactors, treat these as stable:

- Radial CSV: `r`, `mean_phi`, `count`
- Radial-fit CSV: at minimum `fit_kind`, `r_min`, `r_max`, `n_points`, `slope`, `intercept`, `r2`
  - plus any of: `A`, `B`, `r2_inv_r`, `p`, `log_a0`, `r2_loglog`, scan quantiles
- Shapiro CSV: whatever writer emits for:
  - `b`, `D`, `u` (or columns allowing the same reconstruction)
  - `K`, `C`, `r2_line`
  - plus `X0`, `ds`, `max_steps` (either as columns or header comments)
- Ensemble CSV: the exact metric keys returned by `_ensemble_one`

Anything else can be “extra columns” as long as:
- header stays consistent
- and comment provenance remains `#`-prefixed.

### Provenance interaction

Visualiser should be indifferent to the provenance header as long as:
- it is fully `#` comment lines.

So increasing header richness is safe.

### Logging action items (from the viz side)

1) Ensure every CSV writer is comment-header compatible (`# ...`).
2) For overwrite artefacts, ensure filenames are run-unique.
3) For any new columns, do not rename/remove consumed columns without also updating visualiser.

---

## `cli.py` (plot surface truth: which flags actually drive `visualiser.py`)

### Plot flags that exist

CLI defines exactly four plot-related flags:
- `--plot` (bool)
- `--plot-dpi` (int, default 220)
- `--plot-scale` (float, default 1.0)
- `--plot-log` (bool)

There are no per-plot selectors here; plot routing is done by `core.py` based on the run mode.

### Visualiser entry points (actual)

`visualiser.py` defines only two public plotting surfaces:

1) `plot_all(n_map, params, out_dir, plot_dpi=..., scale=..., plot_log=...)`
   - emits:
     - `n_map.png`
     - `n_map_log.png` (only if `--plot-log`)
     - `shapiro_asinh_fit.png`

2) `plot_stability_k_sweep(ks, pfs, means, out_dir, plot_dpi=..., scale=..., ticks_max=...)`
   - emits:
     - `stability_k_sweep.png`
     - `stability_k_sweep_mean_surv.png`

### Flag → function mapping (what this implies)

- Pipeline / index runs:
  - `--plot` should mean “call `plot_all(...)` on the produced `n_map`”.
  - `--plot-log` only affects the optional `n_map_log.png`.

- Stability k-sweep bench:
  - `--bench-sweep-k-plot` is the only plot selector relevant here.
  - That should map to `plot_stability_k_sweep(...)`.

### CSV interaction note

Visualiser does **not** currently read CSVs to plot these figures.
It plots directly from in-memory `n_map` and the k-sweep arrays.

That means:
- changes to CSV schemas won't break plotting,
- but also that plots can't be regenerated from CSV artefacts alone.

If we want plot reproducibility from disk, we'll eventually want optional “read from CSV” paths — but that's design work, not an audit note.

---

## `utils.py` (provenance header + who actually uses it)

### `write_csv_provenance_header(...)` (the standard we should converge on)

`utils.write_csv_provenance_header(...)` returns a CSV-comment-friendly string of `# key=value` lines:
- `producer`, `when` (local ISO), optional `experiment`
- `cwd`, `python`, `command`
- plus caller-supplied `extra={...}` (keys are normalised: spaces → `_`).

This is the right primitive: it's cheap, parse-safe, and survives copy/paste.

### Current call sites (actual)

Only three modules currently *construct* the provenance header using this helper:

1) `core.py`
- builds the run-scoped provenance string once, then threads it into pipeline + most experiments.

2) `collider.py`
- if `provenance_header` is missing, it builds its own header via `write_csv_provenance_header(...)`.

3) `oscillator.py`
- similarly builds its own header when `provenance_header` is not provided.

Everything else either:
- **accepts** a `provenance_header` (and expects the caller to supply it), or
- writes CSV without any provenance support.

### CSV writers: who supports provenance vs who bypasses it

**Supports provenance header (good):**
- `coulomb.py`: `run_coulomb_test(..., provenance_header=None)`
- `isotropy.py`: `run_isotropy_* (..., provenance_header=None)`
- `double_slit.py`: `run_double_slit(..., provenance_header=None)`
- `corral.py`: `run_corral(..., provenance_header=None)`
- `relativity.py`: `run_light_clock(..., provenance_header=None)`
- `ringdown.py`: `run_ringdown_sigma_sweep(..., provenance_header=None)`
- `soliton.py`: `run_soliton_sweep(..., provenance_header=None)`
- `oscillator.py`: accepts `provenance_header` (and also self-builds if missing)
- `collider.py`: accepts `provenance_header` (and also self-builds if missing)

**Supports provenance, but *not* via `provenance_header` (needs standardising):**
- `radial.py`: `dump_radial_csv(..., provenance=None)` and `dump_radial_fit_csv(..., provenance=None)`
- `pipeline.py`: `build_index_from_micro(..., csv_provenance=..., radial_provenance=..., radial_fit_provenance=...)`

`core.py` currently bridges these naming differences by inspecting function signatures and passing whichever kwarg exists.

**Bypasses provenance entirely (needs patch):**
- `walker.py`:
  - per-tick `--dump-walker` CSV: no provenance header
  - sweep summary `--walker-sweep-out` CSV: no provenance header
- `core.py` ensemble metrics CSV (`--ensemble-write-csv`): currently written without a provenance header.

These are the main “holes” if the goal is “every CSV artefact self-describes itself”.

### One unglamorous but important detail

`write_csv_provenance_header` logs `when` using `wallclock_iso()` which is local-time ISO. That's human-friendly, but for cross-machine joins we probably also want:
- `when_utc` *or* an ISO string including the timezone offset.

That's a policy choice; the primitive is fine.

---

## `corral.py`  — “Quantum Corral” frequency sweep (masked telegraph)

### What it actually does

A frequency-response experiment inside a hard-walled cavity:
- constructs a **Dirichlet mask** (walls where `mask==1`), either:
  - `geom="sphere"` (true 3D sphere), or
  - `geom="cylinder"` (2D disk extruded through z).
- drives a **single point source** at the (offset) centre with a sinusoid:
  - `src[cx,cy,cz] = amp * sin(omega * t)`
- evolves **telegraph only** using the **masked** stepper:
  - `evolve_telegraph_traffic_steps(phi, vel, src, tp, 1, mask=mask)`
- for each omega, measures average stored energy over a late window, and records peak amplitudes.

Hard constraints:
- **fail-fast** unless `traffic.mode == "telegraph"`.
- fail-fast if the source lies inside the masked wall.

### Geometry + mask

Mask builder `_build_mask(n, geom, radius)`:
- allocates `mask = zeros(int8)` then sets `mask[...] = 1` outside the chosen region.
- “inside” count is `inside_count = sum(mask == 0)`.

### Sweep + measurement window

- `omegas = linspace(omega_start, omega_stop, omega_steps)`.
- For each omega:
  1) zero `phi` and `vel`.
  2) optional **warm-up**:
     - run `warm_steps` ticks of the same sinusoid.
     - continues phase by using `tt = t + warm_steps` during main burn so there's no discontinuity.
  3) main run is `burn_in` ticks.
  4) measurement window begins at:
     - `meas_start = int(burn_in * burn_frac)`
     - `meas_len = burn_in - meas_start`.

During measurement ticks it computes energy metrics over **inside-only** voxels.

### Energy metrics

`_energy_metrics(phi, vel, mask)` (inside-only):
- `e_phi = Σ(phi^2)`
- `e_vel = Σ(vel^2)`
- `e_tot = e_phi + e_vel`
- `max_phi = max(|phi|)`
- `max_vel = max(|vel|)`

Accumulation:
- averages `e_*` over the `meas_len` window.
- tracks `peak_phi`, `peak_vel` over the same window.

### CSV output

Single CSV file, **overwrite** mode (`open(out_csv, "w")`).

Header:
- writes optional `provenance_header` first (verbatim, ensured newline).
- then module-local comment lines:
  - `# corral_geom=...`
  - `# corral_radius=...`
  - `# corral_inside_count=...`
  - `# corral_omega_start=...`, `# corral_omega_stop=...`, `# corral_omega_steps=...`
  - `# corral_burn_in=...`, `# corral_meas_start=...`, `# corral_meas_len=...`
  - `# corral_warm_steps=...`, `# corral_burn_frac=...`, `# corral_amp=...`
  - `# corral_center_offset=dx,dy,dz`

Data table schema (one row per omega):
- `omega, e_phi, e_vel, e_tot, peak_phi, peak_vel`

So: **per-omega summary**, not per-tick.

### Return value

Returns a small dict:
- `best_omega` and `best_e_tot` (argmax of `e_tot` across the sweep)
- `elapsed_s`, `inside_count`, `out_csv`.

### Logging implications

This module already matches the house pattern:
- provenance header supported,
- run-unique file naming must be handled by the caller (since it overwrites).

One small footnote:
- the CSV does not log `traffic` params itself; it relies on provenance header to carry those.

---

# Stage 3 — Closing the remaining provenance holes (ensemble + walker)

We now have a clear pattern:
- Most experiment CSVs support `provenance_header` and write `# key=value` comment headers.
- Two high-value artefacts still bypass provenance:
  1) `walker.py` (`--dump-walker`, `--walker-sweep-out`) — documented with exact schemas above.
  2) `ensemble_metrics.csv` (`--ensemble-write-csv`) — written by `core.py` from pipeline ensemble rows.

This stage pins the ensemble hole precisely so the patch is trivial.

## `ensemble_metrics.csv` (core-owned writer; currently no provenance header)

### Where it is written

- The writer is not in `pipeline.py`.
- `core.py` collects `rows = run_ensemble(...)` and then writes `ensemble_metrics.csv` when `--ensemble-write-csv` is set.

So: any provenance fix belongs in `core.py` (not in the pipeline).

### Current behaviour

- Overwrite mode (`open(path, "w")`).
- Writes only the CSV header row + rows.
- No `#` comment header is written.

### Required behaviour (target standard)

When overwriting, the file should begin with:

1) `provenance_header` from `utils.write_csv_provenance_header(...)` (already constructed in `core.py`).
2) A tiny artefact discriminator line:
   - `# artefact=ensemble_metrics`
3) A small ensemble-specific metadata set (still comment lines):
   - `# ensemble_seed_list=...` (or `# ensemble_seed_hash=...` if too long)
   - `# ensemble_members=N`
   - `# delta_load=1|0`
   - `# delta_amp=...`
   - `# delta_jitter=...`
   - `# delta_jitter_cube=...` (if jitter is a cube half-width)
   - `# workers=...`

Then the CSV header row and the data rows.

### Column stability

The header row should be the **sorted union of keys** across all returned per-seed dicts, but it must be stable across runs.

Today `core.py` effectively treats whatever `_ensemble_one` returns as schema.

For reproducibility we should adopt:
- define a canonical `ENSEMBLE_COLS` list in `pipeline.py` (or `core.py`),
- and when writing, emit those columns first, then any extra keys (sorted) after.

That prevents accidental schema drift if a future run adds a new metric key.

### Minimal patch target (one place)

The only necessary code edit is inside `core.py` in the block that writes `ensemble_metrics.csv`:
- write the provenance comment header before the `csv.DictWriter` header.

No behavioural change to the ensemble computation is required.

---