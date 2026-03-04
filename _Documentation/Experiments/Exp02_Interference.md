# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# Exp02 — Interference (Double-Slit)

## Abstract

Exp02 confirms that the CAELIX lattice supports **wave mechanics and superposition** when operated in a hyperbolic transport regime (telegraph mode).

By switching the transport kernel from the diffusive/relaxation behaviour used in Exp01 to a momentum-carrying update rule (telegraph), we observe a qualitative transition: from static potential-like equilibria to propagating waves.

A textbook double-slit interference pattern emerges from a real-valued scalar field \(\phi\) under purely local neighbour coupling, with no analytic diffraction model and no explicit interference logic.

---

## Methodology and constraints

What the evolution **does**:

- Evolves a scalar field \(\phi\) on a cubic lattice using local neighbour-coupled transport (the same core transport family as Exp01).
- Injects a time-varying source signal (mode-dependent) in a defined source region.
- Applies a geometric mask at a wall plane with two slits (or one) to constrain propagation.

What the evolution **does not** do:

- It does **not** compute diffraction integrals.
- It does **not** analytically evaluate interference fringes.
- It does **not** “render” a pattern.
 - It does **not** use complex numbers or phase accumulators. The field \(\phi\) is real-valued; interference arises solely from constructive/destructive superposition under local momentum transport.

Diagnostics (analysis) **do** compute intensity-like observables on a detector:

- The detector measures \(I(y) = \phi^2\) along a fixed line at \((x=\text{detector\_x}, z=\text{cz})\), optionally averaged over a sampling window.

---

## Results

Exp02 is deliberately structured as **control vs experiment**:

- **02A (single slit)** establishes the diffraction envelope.
- **02B (double slit)** demonstrates superposition and fringe formation, modulated by that same envelope.

### 02A — Single-slit control

Observed outcome:

- A single, broad diffraction envelope (no high-frequency fringe comb).
- The envelope width is the relevant control metric (e.g. FWHM in voxels, or an equivalent half-power measure).

Record in the CSV/log:

- `pattern_mean`: envelope shape (normalised intensity is useful for comparison across runs).
- `sample_metrics`: stability of centroid and total intensity prior to window averaging.

### 02B — Double-slit interference

Observed outcome:

- A high-frequency fringe pattern (multiple distinct maxima/minima) **modulated by** the single-slit envelope.
- Fringe spacing \(\Delta Y\) and fringe count within the interior detector band are the primary diagnostics.

Recommended measurements to report:

- Fringe spacing \(\Delta Y\) (voxels) measured near the central region.
- Number of distinct peaks above a chosen threshold within the interior detector band.
- Symmetry about `y_off=0`.

### High-fidelity operating regime

In practice, interference clarity improves when damping is minimal and the drive is in a sufficiently high-frequency regime.

For reference, the repo's “high-fidelity” settings are typically characterised by:

- telegraph transport (`--traffic-mode telegraph`)
- low damping (`--traffic-gamma` small)
- coherent drive (e.g. sine source with fixed \(\omega\))

(Exact values are experiment-specific and should be read from the provenance header for the run.)

---
## Implementation

Exp02 is implemented in `double_slit.py`.

The simulation proceeds as:

1) Configure the lattice, wall geometry, slits, and detector placement.
2) Run for `steps` timesteps, applying:
   - source injection
   - mask / wall constraint
   - one transport update step
3) After burn-in, sample the detector every `sample_every` timesteps.
4) Compute the averaged detector pattern from the last `window` sampled frames.

---

## Implications

Exp02 demonstrates that the CAELIX substrate supports **self-interference** under local transport: the field carries momentum, propagates, and superposes.

This is the minimal requirement for wave-based phenomena (and is the prerequisite layer for any later claims about matter waves or quantum analogues). The key point is methodological: interference is not inserted as a formula — it is observed as a consequence of the update rule and boundary/geometry.

---

## Glossary

- **Telegraph mode:** a transport regime with inertia (hyperbolic character), supporting wave propagation rather than pure diffusion.
- **Sponge layer:** an absorbing boundary condition used to suppress reflections by damping the field near the edges. Sponge voxels are excluded from detector sampling via `detector_y=[y0:y1)`.
- **Detector line:** a fixed \((x, z)\) line sampled along \(y\), reporting intensity \(I(y)=\phi^2\) and derived metrics.

---
## Canonical geometry

A typical configuration uses:

- A wall plane at `wall_x` between source and detector.
- Two slits in \(y\) with:
  - slit width: `slit_width`
  - slit separation: `slit_sep`
  - centres symmetric about the lattice centre `cy`.
- Detector line:
  - `x = detector_x`
  - `z = cz` (usually the centre plane)
  - `y` spans the non-sponge interior (see boundary handling below).

Control runs:

- `single_slit=1` to validate single-aperture diffraction envelope.
- `single_slit=0` for the two-slit interference pattern.

---

## Boundary-aware detector handling (sponge)

If `traffic.boundary == "sponge"`, the detector logic is automatically constrained to avoid sampling inside the absorbing layer:

- The detector line is cropped to `y ∈ [w : n-w)` where `w = traffic.sponge_width`.
- `detector_x` is clamped to remain in the non-sponge interior `x ∈ [w : n-w-1]`.
- The CSV records the detector crop explicitly as `detector_y=[y0:y1)` and the boundary mode.

This prevents artificial roll-off (“clipping”) at the detector edges when sponge damping is enabled.

---

## Outputs

Exp02 produces a single structured CSV containing:

- A provenance/comment header describing parameters and measurement window.
- A section for the averaged pattern.
- A section containing per-sample summary metrics for quick QC.

Optionally (for forensics), Exp02 can also record the raw sampled detector time series. This is disabled by default.

- If `--ds-dump-samples` is enabled, the CSV includes a `# section:samples` table.
- In the same mode, the full raw sample cube is saved as a compressed `.npz` under `_npz/`, and the CSV header records the `.npz` path.

The file is intentionally self-describing: plots and analyses can be reproduced from the CSV alone.

---

## CSV structure

The output CSV is a comment-headed, multi-section file.

### Header fields

The header includes (at minimum):

- lattice: `n`, `steps`, `cy`, `cz`
- geometry: `wall_x`, `slit_sep`, `slit_width`, `single_slit`, slit ranges
- detector: `detector_x`, `detector_y=[y0:y1)`, boundary mode and sponge parameters
- source: mode, amplitude, angular frequency, half-period (where applicable)
- sampling: `burn`, `sample_every`, total samples, `window`
- dump: `dump_samples` flag
- npz: `npz` path (present only when raw samples are written)
- window bounds: `t_first`, `t_last`, `span`
- basic stats: `pattern_max`, `pattern_sum`, `phi_max_last`

### Section: averaged pattern

```
# section:pattern_mean
y,y_off,intensity,intensity_norm
...
```

- `y` is the absolute lattice index.
- `y_off = y - cy` recentres the pattern for plotting.
- `intensity = phi^2` averaged over the final window.
- `intensity_norm = intensity / sum(intensity)`.

### Section: sampled frames (optional)

```
# section:samples
t,sample_idx,y,intensity
...
```

This section records the raw sampled detector values (after burn-in). It is written only when `--ds-dump-samples` is enabled; otherwise it is omitted to keep CSV size manageable.

### Section: per-sample QC metrics

```
# section:sample_metrics
t,sample_idx,max_intensity,sum_intensity,centroid_y,width_rms
...
```

- `centroid_y` and `width_rms` are intensity-weighted quantities computed over the detector crop.

---

## How to read the result

Typical checks:

1) **Stationarity**: in `sample_metrics`, verify `sum_intensity` and `centroid_y` stabilise before the window used for averaging.
2) **Single vs double slit**:
   - single slit produces a broad envelope with a dominant central maximum.
   - double slit produces a fringe pattern modulated by the same envelope.
3) **Symmetry**: with symmetric slit placement and a centred detector, the pattern should be symmetric about `y_off=0`.

---

## Provenance recommendations

For reproducibility, keep the provenance header consistent with Exp01 and include:

- boundary mode + sponge settings
- exact detector crop (`detector_y=[y0:y1)`)
- source mode and timing parameters
- burn/sampling/window parameters
- whether raw samples were dumped (`dump_samples`) and the `.npz` path (if present)
- any damping/decay parameters applied during transport

---

## Notes

- Exp02 is intended as a geometry-driven propagation test. It is not an analytic optics solver.
- The structured CSV format is designed to remain GitHub-friendly while still capturing the evidence needed for scientific review.