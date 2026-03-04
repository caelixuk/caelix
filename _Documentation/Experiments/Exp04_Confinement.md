# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# Exp04 — Confinement (Quantum Corral)

## Abstract

Exp04 demonstrates the emergence of **discrete resonant structure from geometric confinement**.

By placing a hard interior boundary (a “corral”) around a driven source, a continuous transport law is forced to satisfy boundary conditions on a finite domain. The result is a spectrum in which the field **accepts energy efficiently only at specific drive frequencies**, corresponding to standing-wave modes of the confined region.

Operationally, sweeping the drive frequency \(\omega\) in telegraph transport yields sharp peaks in interior energy proxies \(E_{\mathrm{tot}}(\omega)\), consistent with a cavity selecting a discrete set of eigenmodes. The simulation does not compute eigenvalues analytically or impose mode shapes; the spectral structure is measured empirically from local dynamics under confinement.

---

## Theoretical predictions

Exp04 has clear, falsifiable expectations.

In the continuum limit, confined wave equations admit discrete eigenmodes determined by geometry and boundary conditions. For an ideal spherical cavity with a Dirichlet wall, radial modes are described by spherical Bessel functions (e.g. \(j_\ell(kr)\)). For the simplest spherically symmetric family (\(\ell=0\)), the mode condition reduces to zeros of \(j_0(kr)\) at the cavity boundary \(r=R\):

\[
 j_0(kR) = 0
\]

In a cylindrical (2D) approximation, the analogous condition involves Bessel functions such as \(J_0(kr)\):

\[
 J_0(kR) = 0
\]

A practical prediction follows: **\(E_{\mathrm{tot}}(\omega)\) should exhibit peaks when the drive matches cavity eigenmodes**, with peak locations shifting systematically with corral radius \(R\) and with the effective wave speed set by transport parameters (notably \(c^2\) and \(dt\)).

This section is a guide for interpretation, not a post-hoc fit: the experiment reports the measured spectrum and can be compared against these continuum expectations as the lattice is refined.

---

## Methodology and constraints

What the evolution **does**:

- Evolves a real-valued scalar field \(\phi\) on a 3D cubic lattice.
- Uses **telegraph transport** (second-order time behaviour with a velocity-like state) to support wave propagation.
- Builds an interior “corral” by masking the lattice so that the boundary acts as a hard Dirichlet wall.
- Drives the system with a single-point harmonic source at the centre:

\[
\phi_{\text{src}}(t) \propto \sin(\omega t)
\]

- Sweeps \(\omega\) across a configured range and measures interior energy proxies over a fixed measurement window.

What the evolution **does not** do:

- It does **not** solve an eigenvalue problem.
- It does **not** impose a spectrum or mode shapes.
- It does **not** use complex numbers, phase accumulators, or analytic cavity solutions.

The output spectrum is an empirical property of the transport rule + boundary constraints.

---

## Implementation

Exp04 is implemented in `corral.py`.

For each drive frequency \(\omega\) in the sweep:

1) Initialise the corral mask (Dirichlet boundary) inside the lattice.
2) Warm-up the driven system for a configured number of steps (to settle transients).
3) Continue driving for a fixed burn-in / measurement duration.
4) Over a measurement window, compute energy-like observables and peak amplitudes.
5) Emit one CSV row per \(\omega\).

The corral mask is a hard interior wall. This is intentionally non-reversible and is used as a controlled confinement boundary.

---

## Observables

Exp04 reports two energy channels and a few stability/diagnostic measures.

- **E_phi:** \(\sum \phi^2\) over the interior (scalar intensity).
- **E_vel:** \(\sum v^2\) over the interior (velocity/inertia channel).
- **E_tot:** \(E_{\phi} + E_{\mathrm{vel}}\).

Because resonances can be sharp, Exp04 also records dispersion of the total energy during the measurement window:

- `e_tot_std`: standard deviation of \(E_{\text{tot}}\) across the window.
- `e_tot_max`: maximum instantaneous \(E_{\text{tot}}\) across the window.

And peak amplitudes:

- `peak_phi`: max \(|\phi|\) over the interior during measurement.
- `peak_vel`: max \(|v|\) over the interior during measurement.

These are useful for distinguishing steady resonant plateaus from transient pumping.

---

## Outputs

Exp04 produces a single CSV containing:

- A provenance/comment header describing transport and corral parameters.
- One row per swept \(\omega\) with energy and peak diagnostics.

The progress bar reports completion by omega index (`omega_steps`). A non-advancing “heartbeat” tick is used during long inner loops to keep time/ETA updates responsive without misrepresenting completion fraction.

---

## CSV structure

### Header fields

The header includes (at minimum):

- lattice: `n`, centre `c`
- corral geometry: radius/shape parameters (as written by `corral.py`)
- drive: `amp`, `omega_min`, `omega_max`, `omega_steps`
- timing: `warm_steps`, `burn_in`, `meas_window` (or equivalent names)
- transport (telegraph) parameters:
  - `traffic_mode`, `traffic_dt`, `traffic_c2`, `traffic_gamma`, `traffic_decay`, `traffic_inject`
  - and, when present, boundary metadata: `traffic_boundary`, `traffic_sponge_width`, `traffic_sponge_strength`
- mask note: `corral_mask=dirichlet`

### Data columns

Each data row contains:

- `i_omega`: sweep index (stable ordering)
- `omega`: drive frequency
- energy means over measurement window:
  - `e_phi`, `e_vel`, `e_tot`
- energy dispersion / peaks:
  - `e_tot_std`, `e_tot_max`
  - `e_ratio` (mean `e_vel` divided by mean `e_phi` + \(10^{-12}\))
  - `peak_phi`, `peak_vel`

---

## How to read the result

Typical checks:

1) **Spectrum shape:** \(E_{\text{tot}}(\omega)\) should show distinct peaks.
2) **Stability:** strong resonances often show elevated `e_tot_max` and a characteristic `e_tot_std` profile.
3) **Mode character:** `e_ratio` indicates whether a resonance is dominated by velocity/inertia or scalar intensity.
4) **Geometric scaling:** changing corral radius should shift the peak locations in \(\omega\) in a structured way.

---

## Notes

- Exp04 is a confinement/resonance diagnostic: a hard interior boundary is used intentionally to test spectral structure.
- Because the corral mask enforces a Dirichlet wall, it is expected to break time-reversal symmetry even in otherwise lossless telegraph settings.
- Later experiments can build on this by introducing moving probes/walkers or by mapping mode shapes directly (e.g. saving interior snapshots at selected \(\omega\)).