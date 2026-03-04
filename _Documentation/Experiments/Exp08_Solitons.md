# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# Exp08 — Solitons (Non-linear births → sprite extraction → SG colliders)

## Abstract

Exp08 introduces **non-linear localisation** to CAELIX.

Exp07 established the linear collider baseline: driven packets ghost through one another, show interference, and then disperse once driving stops. Exp08 is the first suite whose purpose is to create **self-supporting coherent lumps** (“solitons”) that can persist long enough to be treated as **objects** rather than ongoing forcing.

This document initially covers the **non-linear** side (08A–08D). The Sine–Gordon (SG) side (08E–08F) is documented later once we have the 08A–08D run data in hand and the non-linear suite is stable.

---

## Conceptual basis

### Linear vs non-linear localisation (why Exp08 exists)

- **Linear (Exp07):** packets are sustained by injection and disperse without it; overlap is mostly superposition (“ghosting”).
- **Non-linear (Exp08A–08D):** the evolution includes a **self-interaction** term. The intent is that dispersion is countered by a non-linear restoring mechanism, allowing a localised excitation to **hold shape** (or at least remain recognisably bounded) over long windows.

Operationally, the non-linear suite is a stepping-stone: we first prove we can birth bounded structures and measure them cleanly before moving to SG, where “kinks / anti-kinks” and topological constraints give a more canonical soliton family.

### What we mean by “soliton” in CAELIX terms

Within this codebase, a “soliton” is not assumed to be an analytic textbook solution. It is a **computational object** identified by diagnostics:

- localisation: energy concentrated in an inner region, not leaking monotonically outward
- persistence: bounded metrics remain stable across many steps (or across a repeat window)
- coherence: the object's centroid and peak do not exhibit violent drift unless intentionally launched

If a candidate only exists when a driver is active, or if it rapidly decoheres into a shell/halo, it is treated as a **failed birth**.

---

## Goals (non-linear stage)

1) **Birth stability:** demonstrate that 08A–08D can generate coherent, bounded lumps without immediate explosion or diffusion.
2) **Measurement discipline:** ensure the soliton CSV is rich enough to diagnose failure modes (phase, drift, kinetic agitation, leakage).
3) **Extraction feasibility:** confirm we can define a “sprite” snapshot protocol that captures a lump in a stable phase window.

The long-range objective is an automated **sprite library**: store candidate solitons as inject-able sprites, then drop them into collision harnesses (eventually SG collider work) to test binding/capture/scattering.

---

## Experiments (08A–08D): non-linear proof-of-concept

These four experiments are the pre-SG stepping stones. They were designed to validate non-linear localisation and basic handling before attempting SG births and sprite extraction.

(Exact parameters live in `experiments.py` and are referenced here descriptively; we will fill in the numeric tables once the new run outputs are back.)

### 08A — Baseline non-linear birth

- Intent: create a single localised lump under the non-linear evolution.
- Success looks like: bounded inner mass/energy and a stable peak with modest drift.

### 08B — Robustness / nudged birth

- Intent: test sensitivity to small nudges (amplitude/seed/launch) without destabilising.
- Success looks like: same family of object, not a different regime.

### 08C — Persistence window test

- Intent: run long enough to see whether the lump remains coherent or slowly leaks into a halo.
- Success looks like: no monotonic “grey-goo” conversion; energy loss should be explainable (sponge/halo) not uncontrolled.

### 08D — Interaction / perturbation probe

- Intent: apply a controlled perturbation (or two-lump proximity) to see whether the structure survives stress.
- Success looks like: deformation followed by recovery, not catastrophic blow-up.

---

## Key knobs and hygiene

### Grid size and sponge scaling

Sponge width is scaled with N (tested previously):

- N=64 → sponge≈4
- N=128 → sponge≈8
- (and so on, scaling proportionally)

For Exp08 non-linear births, the sponge must be strong enough to prevent reflections from contaminating measurements, but not so aggressive that it “wins” and artificially lowers leakage metrics by simply eating the candidate.

### Soliton sigma (`--soliton-sigma`)

`soliton_sigma` controls the width of the injected / initialised Gaussian structure.

Empirical note from SG exploration (recorded here because it matters to extraction discipline):

- sigma ≈ 4.0 produced tight knots (radius ~8)
- sigma ≈ 8.0 produced large, diffuse low-energy structures (radius ~80)

For the non-linear suite, sigma has historically been fixed (often 4.0). We will explicitly justify that choice after we re-check the 08A–08D runs and compare localisation/leakage trends.

---

## Measurement and outputs (non-linear)

Exp08 emits a per-step “flight recorder” CSV for soliton births, analogous in spirit to Exp07 but focused on **object integrity** and **extractability**.

### Why we need extra metrics

We have a repeat failure mode when trying to extract sprites for later injection: the soliton can appear stable late in a run yet be **out of phase** (high kinetic stress) at the moment of extraction. Injecting a sprite captured during a kinetic surge often leads to rapid blow-up.

Therefore the CSV must allow us to answer, per step:

- Is the object still localised?
- Is it drifting/accelerating?
- Is its internal velocity field calm?
- Is energy steadily bleeding (expected) or spiking (instability)?

### Additional diagnostics added (non-linear)

The following diagnostics are intended to be cheap but decisive:

- `cm_v` — centroid radial drift rate per step (inner crop cube)
- `peak_v` — peak-location drift rate per step
- `peak_final_crop`, `peak_ret_crop` — peak |phi| measured inside the extraction crop (final-step crop vs re-centred/returned crop); used to avoid “looks stable globally, is stressed locally” traps
- `vel_rms_land`, `vel_rms_final`, `vel_rms_ratio` — RMS velocity in the inner crop cube
- `dE_fast`, `dE_fast_per_step` — cheap energy delta using phi^2 + v^2
- `dE_phys_shifted`, `dE_phys_shifted_per_step` — shifted physical-energy delta (useful in symmetry-breaking scans)

Interpretation:

- **Extraction window candidate:** low `vel_rms_ratio`, low `cm_v` and `peak_v`, and `peak_final_crop/peak_ret_crop` staying bounded (no local blow-up), with no energy spikes.
- **Off-phase / stressed state:** elevated `vel_rms_ratio` and/or rapid drift even if peak looks intact.
- **Grey-goo conversion:** sustained drop in inner fractions with corresponding rise in outer/shell metrics.

### Run data to be added

Once we have the fresh 08A–08D outputs, we will add:

- a parameter table per experiment (N, steps, sigma, non-linear term knobs)
- stability summaries (windows where the object is extraction-safe)
- representative plots (radius, vel_rms_ratio, inner fraction, energy deltas)

### 08A instability note (λ scan, non-linear)

In early 08A scans, we hit a reproducible **non-finite blow-up** in a narrow λ region (e.g. idx≈26, λ≈4.2e-2) when running with an aggressive timestep / low stiffness.

Observed case (N=256):
- unstable: `dt=0.10`, `k=0.01` → non-finite during evolve around `t≈129..161/2000`
- stabilised: `dt=0.05`, `k=0.05` → scan completes without NaNs/inf

Working hypothesis: this is primarily a **numerical stability** issue (CFL-like constraint interacting with the non-linear term), not a pure “N too small” artefact — because N changes (e.g. 192→256) did not move the failure point, while dt/k changes did.

#### 08A empirical read (rerun with richer CSV metrics)

We re-ran 08A after the expanded soliton CSV logging landed.

Run: `08A_soliton_scan_n256_20260214_040734.csv` (N=256, dt=0.05, k=0.1, λ sweep 1e-3→1e-1, 33 points).

Findings (high level):

- **No extraction-worthy “object” emerges** in this non-linear scan. The injected peak is fixed at `peak_land≈2.5` but the final peak collapses to `peak_final≈0.083–0.194`, giving `peak_ret≈0.033–0.077`.
- The structure **spreads substantially**: `Rg_land≈4.90` grows to `Rg_final≈19–30` (`Rg_ratio≈3.9–6.2`). This is consistent with dispersion into a broad puff rather than a bounded lump.
- Kinetics do **ring down** strongly: `vel_rms_final` falls by orders of magnitude compared to `vel_rms_land`, but by the time the run is calm the candidate is already de-localised (the “calm because it died” failure mode).
- There is **no obvious grey-goo / global conversion signature**: outer/shell deltas remain small and shell means stay near zero. The dominant behaviour is dispersion + sponge loss, not runaway vacuum conversion.

Implication: 08A–08D remain proof-of-concept scaffolding. The SG side (08E–08F) is still the primary candidate source for sprite extraction.

#### 08B empirical read (symmetry-breaking baseline, k<0)

We ran 08B as the same basic birth/sweep harness as 08A, but with **negative stiffness** (`--soliton-k -0.1`) to enter the symmetry-breaking / double-well regime.

Run: `08B_symmetry_breaking_baseline_n256_20260214_104559.csv` (N=256, dt=0.05, k=-0.1, λ sweep 1e-3→1e-1, 33 points).

Findings (high level):

- The dominant behaviour is **global roll-to-VEV**, not a localised object birth. The outer/shell metrics move away from zero decisively: `mean_phi_shell_final≈0.71–7.11` and `delta_outer_shell_final≈0.28–2.81`.
- The shell mean tracks the expected scale for a double-well potential, where the nominal minima sit near `|phi|≈sqrt(|k|/λ)` (here `|k|=0.1`). As λ increases, the implied VEV drops and the observed shell mean drops with it.
- “Peak retention” becomes misleading in this regime: `peak_ret` can exceed 1 because the **background field level changes**, not because a bounded lump strengthens.
- Localisation still fails by the usual geometric measures: `Rg_final≈24–25` with `Rg_ratio≈5`, consistent with a broad excitation plus global phase conversion rather than a stable coherent lump.

Implication: 08B is valuable as a **diagnostic validation** case (the CSV correctly detects symmetry breaking and VEV drift), but it is not a sprite candidate source.


#### 08C empirical read (symmetry-breaking local probe, gentle k<0)

We ran 08C as a “gentle” symmetry-breaking probe: same soliton birth harness, but with a much smaller negative stiffness (`--soliton-k -0.001`) and a tight λ band intended to keep the implied VEV at **O(1)**.

Run: `08C_symmetry_breaking_local_probe_n256_20260214_110551.csv` (N=256, dt=0.05, k=-0.001, λ sweep 5e-4→2e-3, 17 points).

Findings (high level):

- Localisation still fails in the same geometric sense as 08A/08B: `Rg_ratio≈4.82–5.65` (broadening into a puff rather than a bounded object).
- Background drift is present but **milder than 08B** (consistent with |k| being 100× smaller): `mean_phi_shell_final≈-0.27..+0.53` and `delta_outer_shell_final≈-0.70..+0.96`.
- The run exhibits **well-choice sign ambiguity**: the shell mean flips sign across the λ sweep despite `vev_sign=+1` and `phi0=0`. This is consistent with spontaneous symmetry breaking from numerical noise when starting near the unstable origin.

Implication: 08C is still a useful diagnostic probe, but interpretation is muddied by the sign flip. To make it a true “local probe”, the background should be initialised into a chosen vacuum (init at ±VEV) so the shell metrics measure drift from a fixed well rather than which well was selected.


#### 08D empirical read (VEV-initialised local probe; comparison to prior N=512 test)

We ran 08D as a direct counterpart to 08C, keeping the same `(k, λ)` band but switching on **VEV initialisation** to remove the well-choice coin-flip and make shell drift interpretable.

Run: `08D_symmetry_breaking_local_probe_vev_init_n256_20260214_112743.csv` (N=256, dt=0.05, k=-0.001, λ sweep 5e-4→2e-3, 17 points, `init_vev=1`, `vev_sign=+1`).

Findings (high level):

- The **sign ambiguity is removed**: `mean_phi_shell_final` remains positive across the sweep (≈0.38–0.76), and `delta_outer_shell_final` stays consistently positive (≈0.11–0.55). This is the intended “fixed vacuum” behaviour.
- Under VEV init, `Rg_*` computed on `rho=phi^2` becomes **background-dominated** (the crop is filled with vacuum), so `Rg_ratio≈1` is not evidence of a stable lump. For VEV-initialised runs, localisation should be judged using **deviation-from-vacuum** measures (e.g. `(phi-phi0)^2`), not raw `phi^2` moments.
- Kinetic ring-down remains comparable to 08C (no magic calmness from VEV init); the main gain is **interpretability**, not improved localisation.

Implication: we can treat N=256 as the sensible baseline for the symmetry-breaking probes and move on. For any future VEV-init runs we will rely on the deviation-based localisation metrics (added in `soliton.py`) rather than raw `Rg_*`.

---

## SG stage status (08E): baseline sweeps for candidate mapping

The SG side is where we expect sprite candidates to come from, but the formation/extraction problem is phase-sensitive: a single snapshot can be a “one-step bomb” if captured during a kinetic surge.

Therefore 08E is treated as a **mapper**: it does not extract sprites; it sweeps parameters and writes diagnostics (including SG-only tail window aggregates) so we can locate pockets where a compact core persists and the tail window looks calm.

### 08E runs logged so far

#### 08E-01 — Breather baseline (N=192)

Run:
- Bundle: `_Output/08_solitons/08E_sine_gordon_breather_baseline/20260216_055330/`
- CSV: `08E_sine_gordon_breather_baseline_n192_20260216_055330.csv`
- Log: `08E_sine_gordon_breather_baseline_n192_20260216_055330.log`
- When: 2026-02-16 05:53 (run_id `20260216_055330`)

Setup (from CSV provenance):
- Grid: `N=192`, seed `6174`
- Integrator: `dt=0.05`, `c2=0.31`, `gamma=0.0`, `decay=0.0`
- Boundary: `sponge` (width `24`, strength `0.2`)
- Steps: `12000`
- SG sweep: `sg_k` from `0.42 → 0.50` (9 points), `sg_amp=5.0` (fixed)
- Sigma: `4.0` (fixed)
- Tail window: `512` (default)

Outcome (high level):
- 9/9 points completed with `status=ok`.
- No evidence of heavy wrapping: `tail_frac_abs_phi_gt_2pi_max = 0` for all points; `tail_frac_abs_phi_gt_pi_max` stays tiny (≈ `8.5e-05 → 1.44e-04`).
- Tail calmness is broadly consistent across the sweep:
  - `tail_vel_rms_mean ≈ 0.108 → 0.114` (mean ≈ `0.112`)
  - `tail_lap_abs_max ≈ 7.55 → 9.18` (mean ≈ `8.20`)
  - `tail_dE_phys_shifted_per_step_abs_max ≈ 52.0 → 57.2` (mean ≈ `54.9`)
- Core engagement varies materially despite similar tail statistics:
  - `peak_final_over_pi` spans `0.069 → 1.673`.
  - Strongest peak in this sweep: idx=3 (`sg_k≈0.4484`) with `peak_final_over_pi≈1.673`.
  - Weakest peak in this sweep: idx=7 (`sg_k≈0.4892`) with `peak_final_over_pi≈0.069` (effectively “didn't form”).


Interpretation:
- This looks like a clean *baseline mapper* run: stable numerics, sponge behaving, no runaway wrap, and enough spread in `peak_final_over_pi` to locate a pocket where a compact core actually forms.
- Next action for 08E is to expand around the pocket (around `sg_k≈0.44–0.46`) and then enable `--dump-sprite` to exercise besttail capture once the pocket is confirmed.

#### 08E-02 — Deeper mapper sweep (σ × sg_k × sg_amp) (N=192)

Run:
- Bundle: `_Output/08_solitons/08E_sine_gordon_breather_baseline/20260216_073040/`
- CSV: `08E_sine_gordon_breather_baseline_n192_20260216_073040.csv`
- Log: `08E_sine_gordon_breather_baseline_n192_20260216_073040.log`
- When: 2026-02-16 07:30 (run_id `20260216_073040`)
- Timing: `[timing] wall_s=9082` (~2.52h)

Setup (from CSV provenance):
- Grid: `N=192`, seed `6174`
- Integrator: `dt=0.05`, `c2=0.31`, `gamma=0.0`, `decay=0.0`
- Boundary: `sponge` (width `24`, strength `0.2`)
- Steps: `15000`
- SG sweeps (full grid, 99 points):
  - `sigma ∈ {3.8, 4.0, 4.2}`
  - `sg_k` from `0.42 → 0.48` (11 points)
  - `sg_amp ∈ {4.75, 4.9937, 5.25}`
- Tail window: `512` (default)

Outcome (high level):
- 99/99 points completed with `status=ok`.
- Wrapping remains clean:
  - `tail_frac_abs_phi_gt_2pi_max = 0` for all points.
  - `tail_frac_abs_phi_gt_pi_max` stays small (≈ `6.0e-05 → 1.76e-04`).
- Tail calmness stays within a tight band across the whole grid:
  - `tail_vel_rms_mean ≈ 0.098 → 0.137` (mean ≈ `0.114`)
  - `tail_lap_abs_max ≈ 6.10 → 9.03` (mean ≈ `7.73`)
  - `tail_dE_phys_shifted_per_step_abs_max ≈ 42.5 → 56.5` (mean ≈ `53.3`)
- The “did it actually form a compact core?” axis shows very large variance:
  - `peak_final_over_pi` spans `0.016 → 1.781` (mean ≈ `0.991`).
  - Best point in this grid: idx=60 (`sigma=4.0`, `sg_amp=4.75`, `sg_k≈0.4736`) with `peak_final_over_pi≈1.781`.
  - Worst point in this grid: idx=35 (`sigma=4.0`, `sg_amp=5.25`, `sg_k=0.42`) with `peak_final_over_pi≈0.016`.


Interpretation:
- This run does what 08E should do: it maps a large parameter grid while keeping the tail window well-behaved (no 2π wrap, no numerical drama).
- The grid strongly suggests *pockets* rather than a monotonic trend. Next step is to take the top pocket(s) (around `sg_k≈0.47–0.48` with `sg_amp≈4.75–5.0`, and `sigma≈4.0`) and rerun a smaller confirmation sweep with `--dump-sprite` enabled so we start exercising besttail export + sprite extraction.


### 08F verification runs (long windows; sprite readiness)

08F is the follow-on to 08E: we take the best mapper pocket and run a **long window** to confirm persistence and tail calmness before turning on sprite export.

#### 08F-01 — Long-run confirmation (50k) from 08E-01 pocket (N=192)

Run:
- Bundle: `_Output/08_solitons/08F_sine_gordon_breather_stable/20260216_073100/`
- CSV: `08F_sine_gordon_breather_stable_n192_20260216_073100.csv`
- Log: `08F_sine_gordon_breather_stable_n192_20260216_073100.log`
- When: 2026-02-16 07:31 (run_id `20260216_073100`)
- Timing: `[timing] wall_s=308.253` (~5.1 min)

Setup (from provenance):
- Grid: `N=192`, seed `6174`
- Integrator: `dt=0.05`, `c2=0.31`, `gamma=0.0`, `decay=0.0`
- Boundary: `sponge` (width `24`, strength `0.2`)
- Steps: `50000`
- Point: `sigma=4.0`, `sg_k=0.4484`, `sg_amp=5.0`
- Tail window: `512`

Outcome (single-point summary):
- `status=ok`.
- Compact core persists to the end: `peak_final_over_pi≈1.401`.
- Tail window is exceptionally calm (orders of magnitude quieter than the 12–15k mapper runs):
  - `tail_vel_rms_mean≈0.00294` (min≈0.00269, max≈0.00335)
  - `tail_lap_abs_max≈0.0444`
  - `tail_dE_phys_shifted_per_step_abs_max≈0.00497` (mean≈-5.90e-04)
- Wrapping is clean: `tail_frac_abs_phi_gt_pi_max=0` and `tail_frac_abs_phi_gt_2pi_max=0`.


Interpretation:
- This is a confirmed **sprite-ready** parameter point: it maintains a strong peak while the tail window is quiet enough that a besttail selector should have no trouble finding a safe phase.
- Next step: rerun this exact point with `--dump-sprite` enabled to exercise besttail selection + in-memory sprite export (no full-grid besttail snapshot) and validate that the extracted radius/support fraction looks like a compact breather rather than grey goo.


### 08G extraction runs (besttail gating → sprite asset)

08G is the first run where we turn on `--dump-sprite` for SG and attempt to materialise a reusable **sprite asset** for the collider harness.

#### 08G-01 — First extraction attempt (missing besttail)

Observed failure mode (core fail-loud):

- `--dump-sprite` was set but no snapshot was written.
- CSV shows `status=ok` but `besttail_ok=0` and `besttail_dumped=0`.

Root cause:

- The viability gate was still using **centre-crop** tail metrics. Once the breather drifted away from centre, the measurements were misleading: centre-crop looked calm while the object itself was elsewhere.

Fix:

- Implemented **peak-centred** tail window aggregates (`tailp_*`) and switched besttail scoring/gating to prefer them.

#### 08G-02 — Second extraction attempt (threshold too strict)

After peak-centred metrics landed, the diagnostic line made the failure unambiguous:

- `tail_vel_rms_mean≈0.003` (centre-crop)
- `tailp_vel_rms_mean≈0.106` (peak-centred)
- Default gate `SG_BESTTAIL_MAX_VEL_RMS_MEAN=0.03` ⇒ reject

Interpretation:

- This is a **threshold issue**, not an energy-drift issue. A breather can be stable while still having significant local oscillatory velocity near the core.

Fix:

- Changed peak-centred tail window (`tailp_*`) to use the larger probe cube (`crop_r_outer`) so RMS is not dominated by the tight core-only window.
- Relaxed the drift-energy gate as needed (kept conservative but no longer pathological).

#### 08G-03 — Sprite successfully produced

Run produced a sprite asset:

- `[dumpsprite] OK: [extractsprite] phase=begin in=... kind=besttail out=.../_sprites`

At this point we observed an important disk issue:

- The old pipeline wrote a **full-grid besttail HDF5** (~45MB) purely as a handoff to `extractsprite.py`, plus the small sprite asset (~60KB).

This is unacceptable for large sprite hunts.


Fix: in-memory sprite extraction

- `extractsprite.py` now exposes `extract_sprite_from_fields(...)`.
- `soliton.py` calls it directly on the in-memory `best_tail_phi/vel/src` fields.
- `--dump-sprite` now writes **only the sprite asset** under `_sprites/`.
- The soliton CSV ledger fields remain, but `besttail_h5` now points at the **sprite asset** path.
- Full-grid HDF5 snapshots are written only when `--dump-hdf5` is explicitly set.

### Sprite asset contract (what a “sprite” is)

A sprite asset is a **compact HDF5 patch** intended for re-injection into a fresh SG grid. It is deliberately *not* a full-grid state dump.

The sprite asset contains:

- **Patch fields**: `phi`, `vel` (and any auxiliary fields required by the integrator such as `src`, `load`) on a small cube.
- **Patch size**: `L` (half-extent) and derived inner/outer radii.
- **Provenance**: extraction step `t*`, run_id, and minimal physics settings used during extraction (e.g. `dt`, `c2`, and SG parameters such as `sg_k` when available).

The consumer contract is:

- The collider harness **stamps** the patch into a clean SG grid at a requested position (and optionally with a phase sign).
- Motion is applied by modifying the time-derivative field (`vel`) at injection time.
- Sprite assets should be treated as *library items*: small, portable, and reproducible.

Disk rule (hard):

- `--dump-sprite` writes **only** the sprite asset under `_sprites/`.
- Full-grid HDF5 snapshots are written **only** when `--dump-hdf5` is explicitly set.

This separation is critical: large sprite hunts must not saturate disk by emitting full-grid “handoff” H5 files.


Fix: sweep behaviour

- Core's sprite handling for sweeps is now non-fatal per point.
- If a sweep has points that do not meet thresholds, they record `besttail_ok=0/besttail_dumped=0` and the sweep continues.
- The run fails loud only if **no sprite assets** were produced at all.

### Collider injection conditioning (why “stable” is not always “kickable”)

Breather sprites are **standing waves**: internal kinetic energy sloshes between potential/kinetic phases. A sprite extracted from a stable tail window can still be a poor launch frame for translational kicks.

Therefore the collider harness performs a small, deterministic **injection conditioning** step when loading sprite assets:

- **Local phase select (PLL):** evolve the patch locally for a short window and select the frame that maximises `max(|phi|)` (the “tall” phase).
  - Tie-break on minimal core `|vel|`.
  - This avoids locking onto the flat/kinetic phase where `∇phi ≈ 0` and kicks have no traction.
- **Kick normalisation (gain):** estimate kick traction for the requested direction and apply a bounded gain so the realised translation is closer to the requested velocity.
- **Phase-π polarity handling:** for `phase=π` sprites (valleys), the translational kick term is phase-signed so the valley moves in the requested direction.
- **No drift removal for breathers:** LSQ “bulk drift” fitting is disabled for SG sprite assets because it can hallucinate drift on standing waves and sabotage kick leverage.

Diagnostics:

- Collider CSV provenance includes `pll_info_json` (per-sprite phase select and gain diagnostics) so failed collisions can be debugged from the ledger without reruns.

Lessons learned (from early collider trials):

- **Kinetic-phase lock is a dead end:** selecting a frame with large internal `|vel|` tends to produce `∇phi≈0` (“flat” breather) and the translational kick collapses to ~0 regardless of gain.
- **Over-clever leverage proxies mislead:** maximising a derived “kick leverage” scalar can select the wrong phase for breathers; peak amplitude (`max(|phi|)`) is the robust objective.
- **Drift removal sabotages standing waves:** least-squares bulk drift fits can hallucinate translation on breathers and distort the field; disable for SG sprite assets.
- **Phase-π must be sign-handled:** valleys (`phase=π`, `s=-1`) require the translational kick to be phase-signed or they will move in the wrong direction even if the requested velocity is correct.

## Appendix — SpriteFactory candidate search logic (hardcoded; no external AI calls)

We intentionally avoid external AI/API calls for candidate selection so the repository remains self-contained and reproducible. The sprite factory must therefore use deterministic, hardcoded selection logic based on the soliton CSV metrics.

### A. High-level pipeline

1) **Mapper run (08E-style):** sweep `(sigma, sg_k, sg_amp)` on a fixed grid and write CSV only.
2) **Rank points:** compute a stable score from tail-window diagnostics; shortlist top-N points.
3) **Verification rerun:** rerun shortlisted points at higher steps (and fixed dt/c2/boundary) to confirm persistence.
4) **Extraction run:** for a verified point, run long enough to provide a history window, then extract a sprite from a **stable *and* kickable phase** (not necessarily the final step).
5) **Sprite asset write:** emit a compact sprite patch HDF5 (and optional meta/quicklook) under `_sprites/`. Full-grid HDF5 snapshots are written only when `--dump-hdf5` is explicitly requested.

### B. Deterministic ranking (SG mode)

Use SG-specific signals first; ignore metrics that are known to mislead in SG (e.g. `peak_ret` when `peak_land` is a single-step injection artefact).

**Hard filters (fail-fast):**

- `status != ok` → reject.
- Non-finite diagnostics → reject.
- Reject points that never meaningfully engage nonlinearity (optional): `peak_final_over_pi < 0.5`.

**Core-exists gate:**

- Require `peak_final_over_pi >= 1.0` *or* a small but non-zero wrap footprint (`frac_abs_phi_gt_pi2_final > 0`).

**Tail calmness gate:**

- Prefer small `tail_vel_rms_mean` and `tail_vel_rms_max`.
- Prefer small `tail_dE_phys_shifted_per_step_abs_max`.
- Penalise large `tail_lap_abs_max` (curvature spikes).

**Leakage / grey-goo gate:**

- Penalise large `abs(delta_outer_shell_final)` and large `abs(mean_phi_shell_final)`.
- Penalise large `outer_shell_abs_ratio_final`.

**Tie-breakers:**

- Prefer low drift: `cm_v`, `peak_v`.
- Prefer compactness: low `Rg_final` at fixed peak (but treat VEV-init cases separately).

A minimal, deterministic SG score to start with:

- `score = peak_final_over_pi / (1 + tail_vel_rms_mean) / (1 + tail_dE_abs_max) / (1 + tail_lap_abs_max)`
- then apply leakage penalties by dividing by `(1 + abs(delta_outer_shell_final))` and `(1 + abs(mean_phi_shell_final))`.

### C. Stable/kickable phase extraction (avoid single-snapshot bombs)

Given a verified parameter point, the extraction run must select an extraction step t* from a tail history window. For breathers, “kickable” correlates with the tall phase (high |phi|, low core |vel|), not merely low tail RMS:

- Scan the last `W` steps (e.g. 512–2048) and find a local minimum of `vel_rms` (or the minimum over the window).
- Require that `dE_phys_shifted_per_step` is small in the neighbourhood of `t*`.
- Prefer that wrap fractions are stable across the window (no sudden π→2π surges).

Export `(phi[t*], vel[t*])` (and any auxiliary fields required by the integrator) as a compact sprite asset. In the current implementation this export is performed in-memory (no intermediate full-grid HDF5 handoff).

### D. Verification in collider harness

Before using a sprite in a full collision run, perform a short, deterministic “settle test” in the collider harness:

- Stamp sprite into a clean SG grid.
- Evolve for a short window (e.g. 128–256 steps) with **no additional forcing**.
- Reject if peak/curvature/velocity metrics spike beyond configured thresholds.

This distinguishes intrinsically unstable births from stable sprites that only fail under overlap/collision.
