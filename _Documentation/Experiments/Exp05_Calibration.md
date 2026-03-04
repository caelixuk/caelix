# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# Exp05 — Calibration (Isotropy)

## Abstract

Exp05 quantifies **directional isotropy** of transport on the CAELIX lattice.

A discrete cubic grid is not rotationally symmetric. If left unmeasured, grid anisotropy can masquerade as “physics” (e.g. preferred directions, spurious lensing, asymmetric propagation speeds). Exp05 provides a repeatable calibration that measures the effective propagation speed and amplitude decay along a set of canonical directions (axes, face diagonals, and body diagonals), enabling regression testing and parameter tuning.

This experiment does not attempt to “fix” anisotropy by force; it measures it, reports it, and supplies compact summary statistics suitable for automated comparisons.

---

## Theoretical expectations

In an ideal isotropic continuum wave equation with constant wave speed \(c\), a compact pulse propagates with a direction-independent travel time:

\[
 t_{\text{peak}}(\hat{n}) \approx \frac{r}{c}
\]

where \(r\) is the detector distance and \(\hat{n}\) is the propagation direction.

On a cubic lattice, deviations are expected:

- **Axis directions** are typically “faster” (shorter stencil path) than diagonals.
- **Face diagonals** and **body diagonals** can split as wavelength approaches the grid scale.

Exp05 treats the axis-mean response as a reference and reports fractional deviations for all other detectors.

---

## Methodology and constraints

What the evolution **does**:

- Evolves a real-valued scalar field \(\phi\) (and, in telegraph mode, a velocity-like state \(v\)).
- Places a compact source at the lattice centre.
- Samples the field at a set of detectors at equal radius but different directions.
- Extracts timing/amplitude observables from detector traces.
- Reports both per-detector rows and compact summary footers.

What the evolution **does not** do:

- It does **not** assume an isotropy model.
- It does **not** rotate the grid or interpolate to a continuum sphere.
- It does **not** smooth results post-hoc to remove anisotropy.

---

## Implementation

Exp05 is implemented in `isotropy.py`.

Two operating modes are supported:

1) **Single calibration**: run one \(\sigma\) (source width) and emit one CSV.
2) **Sigma sweep**: iterate \(\sigma\) values and emit a CSV containing multiple \(\sigma\)-grouped detector rows.

The source is a Gaussian pulse centred at \((c,c,c)\). For sweep mode, the Gaussian is filled in-place to avoid per-sigma allocation overhead.

---

## Detectors

Detectors are placed at a fixed radius \(R\) with the following canonical direction families:

- `axis`: \((\pm R, 0, 0)\), \((0, \pm R, 0)\), \((0, 0, \pm R)\)
- `face`: \((\pm R/\sqrt{2}, \pm R/\sqrt{2}, 0)\) permutations (rounded to grid)
- `body`: \((\pm R/\sqrt{3}, \pm R/\sqrt{3}, \pm R/\sqrt{3})\) permutations (rounded to grid)

The experiment reports the actual realised detector offsets and distances (in lattice units) so rounding is always visible.

---

## Observables

For each detector trace, Exp05 computes:

- `t_peak_f`: sub-tick peak time (parabolic interpolation around the discrete maximum)
- `peak_phi`: peak \(|\phi|\)
- `c_eff`: effective speed \(c_{\text{eff}} = r/t_{\text{peak}}\) when a valid peak is found

Noise / stability metrics over a late-time tail window:

- `tail_mu`, `tail_sd`: mean and std of tail \(|\phi|\)
- `snr_peak_over_tail_mean`: \(\text{peak}/(\text{tail\_mu}+\epsilon)\)
- `snr_peak_over_tail_sd`: \(\text{peak}/(\text{tail\_sd}+\epsilon)\)

Isotropy deltas relative to the axis-mean reference:

- `ratio_vs_axis_mean`: \(c_{\text{eff}}/\overline{c}_{\text{axis}}\)
- `delta_c`: \(c_{\text{eff}}-\overline{c}_{\text{axis}}\)
- `delta_c_frac`: \((c_{\text{eff}}/\overline{c}_{\text{axis}})-1\)

A simple amplitude-distance invariant check:

- `peak_I_r2 = peak_phi * r^2`
- `peak_I_r2_ratio`: ratio vs axis-mean \(\overline{I r^2}_{\text{axis}}\)
- `delta_Ir2_frac`: \((\text{ratio})-1\)

Peak validity gate:

- `valid_peak`: true if `t_peak_f > 0` and `snr_peak_over_tail_mean >= 10`

---

## Outputs

Exp05 produces one CSV per run.

The CSV contains:

- A provenance/comment header describing lattice and transport parameters.
- A data table of per-detector rows (and per-\(\sigma\) rows in sweep mode).
- A footer block of `# summary_...` lines with compact anisotropy statistics suitable for regression testing.

---

## CSV structure

### Header fields

The header includes (at minimum):

- lattice: `n`, centre `c`
- source: `sigma`, `amp`, detector radius `R`
- transport parameters:
  - `traffic_mode`, `traffic_dt`, `traffic_c2`, `traffic_gamma`, `traffic_decay`, `traffic_inject`
  - and, when present: `traffic_boundary`, `traffic_sponge_width`, `traffic_sponge_strength`

### Data columns

Per-detector columns include:

- `sigma`
- `kind` (axis/face/body)
- `dx`, `dy`, `dz` (realised detector offsets)
- `dist`, `dist_over_R`
- `t_peak_i`, `t_peak_f`
- `peak_phi`
- `c_eff`, `ratio_vs_axis_mean`, `delta_c`, `delta_c_frac`
- `peak_I_r2`, `peak_I_r2_ratio`, `delta_Ir2_frac`
- `tail_mu`, `tail_sd`, `snr_peak_over_tail_mean`, `snr_peak_over_tail_sd`
- `valid_peak`

### Footer summary

At the end of the CSV, Exp05 writes one or more comment lines:

- `# summary_sigma=<...> count_valid=<...> max_abs_delta_c_frac=<...> rms_delta_c_frac=<...> max_abs_delta_Ir2_frac=<...> rms_delta_Ir2_frac=<...>`

These are computed over detectors with `valid_peak==True` excluding the `axis` family.

---

## How to read the result

Typical checks:

1) **Peak validity:** ensure most non-axis detectors are `valid_peak==True` for the \(\sigma\) regime of interest.
2) **Speed anisotropy:** inspect `delta_c_frac`; the maximum absolute value is the headline isotropy deviation.
3) **Amplitude invariant:** inspect `delta_Ir2_frac`; large deviations can indicate direction-dependent attenuation.
4) **Scale dependence:** in sweep mode, anisotropy typically worsens as \(\sigma\) shrinks toward the grid scale (shorter wavelengths).
5) **Regression testing:** use the footer `max_abs_delta_c_frac` / `rms_delta_c_frac` to compare builds and parameter changes.

---


---

## Results (Exp05A)

The first isotropy sweep (telegraph transport, \(n=512\), \(R=80\), \(c^2=0.31\), \(dt=1\), \(\gamma=10^{-3}\), \(\sigma\in[1,8]\)) establishes two practical calibration facts:

1) **Effective vacuum speed (long-wavelength limit).**

- Theoretical: \(c=\sqrt{0.31}\approx 0.5568\)
- Measured (smooth pulse / continuum limit): \(c_{\mathrm{eff}}\approx 0.5595\)

This is within \(\sim 0.5\%\) and is the most useful value to carry forward when converting distance to time-of-flight in subsequent experiments.

2) **Directional anisotropy decreases with pulse width.**

A sharp pulse (small \(\sigma\)) couples to grid-scale structure and exhibits measurable direction dependence. As \(\sigma\) increases, the lattice behaves closer to an isotropic continuum.

Representative values from the sweep:

- \(\sigma=1.0\): \(c_{\mathrm{axis}}\approx 0.5507\), \(c_{\mathrm{diag}}\approx 0.5574\) → anisotropy \(\approx 1.2\%\)
- \(\sigma=8.0\): \(c_{\mathrm{axis}}\approx 0.5589\), \(c_{\mathrm{diag}}\approx 0.5595\) → anisotropy \(\approx 0.12\%\)

These numbers are consistent with the expected trend that short-wavelength components are more sensitive to stencil anisotropy.

3) **Numerical dispersion is visible across \(\sigma\).**

Across the sweep, \(c_{\mathrm{eff}}\) increases slightly as \(\sigma\) increases (longer wavelength content), indicating a dispersive numerical group velocity on the lattice. For downstream timing calculations, the appropriate choice is the large-\(\sigma\) limit (long-wavelength) rather than the small-\(\sigma\) regime.

### Practical guidance

For experiments that depend on time-of-flight and small relative delays (e.g. lensing / Shapiro-style measurements):

- Prefer \(\sigma\ge 4\) to keep anisotropy below \(\sim 0.3\%\) in this configuration.
- Use \(c\approx 0.559\) (the measured long-wavelength limit) for baseline conversions unless a run-specific calibration is performed.

---