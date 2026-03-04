# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# Exp00 — Heavy Walker (dynamic source lag / wake diagnostics)

## Abstract

Exp00 documents the **Heavy Walker** harness (`--walker`): a diagnostic experiment that translates a **unit delta source** through the traffic solver and measures **wake / lag asymmetry**.

This suite is intentionally **field-driven** (prescribed kinematics). It does **not** use the microstate→load pipeline and does **not** implement backreaction. Its purpose is to calibrate and characterise the behaviour of the traffic update (telegraph / diffusion) under controlled forcing so that later experiments can choose regimes deliberately rather than by accident.

### Key findings (from 00A–00L)

- In **telegraph mode**, wake strength is primarily governed by **pacing** (`tick_iters`) and by the pair (**damping** `gamma`, **memory** `decay`).
- Increasing **damping** (`gamma=0.010`, 00F) suppresses ringing and collapses wake magnitude.
- Adding **telegraph decay** (`decay=0.010`, 00I/00K) bounds the field mass, removes the large early `total_field` transient seen in the zero-decay runs, and strongly suppresses wake while preserving telegraph kinematics.
- Apparent **directional modulation** on a circular path (00E) vanishes once decay is enabled (00K), indicating that most of the earlier k=4 signal was dominated by long-lived memory/ringing rather than unavoidable stencil geometry.
- **Diffusion mode** behaves qualitatively differently: with `decay=0.0` it accumulates mass and saturates into an extreme trailing wake (00G); with decay enabled it becomes bounded but remains strongly lag-dominated (00H).
- The tick sweep (00L) provides a compact “phase map” for selecting pacing without adding further lettered runs.

---

## Goals (completed)

1) Establish that the wake metric is **well-posed** and stable at large `n` (no NaNs, bounded `phi_max`).
2) Quantify how wake scales with pacing (`tick_iters`) and identify practical **quasi-steady** settings.
3) Separate the roles of **damping** (`gamma`) versus **memory** (`decay`) in telegraph mode.
4) Provide diffusion controls to avoid misattributing generic relaxer lag to telegraph-specific dynamics.
5) Detect and then eliminate spurious **directional artefacts** under motion by bounding memory (00E → 00K).
6) Replace further alphabetic runs with a compact **tick sweep** summary (00L).

---

## Conceptual basis

### What a “walker” is in CAELIX

A walker run evolves a scalar field `phi` (and, in telegraph mode, its auxiliary `vel`) under the traffic dynamics while a compact forcing term `src` is translated through the lattice.

At each walker step:

- `src` is reset to a single-voxel delta at the current position.
- The traffic solver is advanced for a fixed number of **tick iterations**.
- We compare the field in front of the source versus behind it to quantify lag.

### Why this matters

For telegraph-style dynamics, a moving source can leave a trailing wake due to finite propagation speed, damping, and numerical dispersion. Exp00 measures this effect directly and provides a reproducible way to:

- validate solver stability at large `n` (memory/throughput)
- study ringing vs damping
- calibrate “how many solver iterations per voxel move” are required for quasi-steady motion

---

## Harness overview

### Primary entry points

- CLI flag: `--walker`
- Path selector: `--walker-path <linear|circle>`
- Steps: `--walker-steps <int>`
- Pacing: `--walker-tick-iters <int>`
- Local metric radius: `--walker-r-local <int>`

The walker harness allocates its own arrays (`phi`, `vel`, `src`) and does not depend on microstates.

### Kinematics (prescribed)

#### Linear path

For `walker_path=linear`, the delta source position is advanced by integer offsets each step:

- `dx, dy, dz` (defaults: `dx=1, dy=0, dz=0`)

There is no inertia and no force law; the source is a controlled driver used to probe the field.

#### Circle path

For `walker_path=circle`, the source traces a circle of radius `walker_circle_radius` about the chosen centre. If `walker_circle_period=0`, one full revolution occurs over `walker_steps`.

### Warm start

Before motion begins, the harness performs a short warm start by advancing the traffic solver for `traffic.iters` iterations with the delta source held fixed. (In Exp00A this is deliberately minimal.)

---

## Measurement definitions

### Local front/back sums

At each step we compute front/back field mass in a local cube centred on the source:

- cube half-extent: `r_local = walker_r_local`
- cube size: `(2r_local+1)^3`

For linear motion, the cube is split along the motion axis into:

- **front**: voxels ahead of the source
- **back**: voxels behind the source

We record:

- `front_sum`, `back_sum`
- `asym = (front_sum - back_sum) / (front_sum + back_sum)` (signed wake indicator)
- `hubble = (front_sum - back_sum) / total_field` (global-normalised wake indicator)

### Global scalars

We also record:

- `phi_max` (global max of `phi`)
- `total_field = sum(phi)`

Optional (off by default): probe samples on a radius `walker_probe_r`.

---

## Parameters and regimes

### Telegraph vs diffusion

- **Telegraph**: finite propagation speed (controlled by `c2`, `dt`) with optional damping (`gamma`) and decay.
- **Diffuse**: relaxer / smoother; useful as a baseline for “how quickly the field catches up” under a moving source.

### Interpreting pacing

For a linear walker with unit voxel moves:

- implied driver speed: `v_est = 1 / walker_tick_iters` (voxels per solver iteration)
- telegraph wave speed scale: `c_est ≈ sqrt(c2) * dt`

Exp00 is designed to compare wake strength across pacing choices, not to claim a physical Mach number without a defined unit mapping.

---

## Known failure modes

- **Boundary interaction:** if the walker approaches the boundary, reflections and sponge/clamp effects can dominate the asymmetry metrics.
- **Underdamped ringing:** very low `gamma` in telegraph mode can preserve voxel-scale ringing; metrics may reflect grid dispersion rather than coherent wake.
- **Too-fast pacing:** small `tick_iters` can push into shock-like regimes where the wake saturates and becomes highly non-linear.

---

## Experiment 00A — Heavy Walker test (moving source dynamics)

### Purpose

A minimal, large-grid sanity test of the telegraph solver under a slowly translated delta source.

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=telegraph`
- `traffic-c2=0.31`, `traffic-dt=1.0`
- `traffic-gamma=0.001`, `traffic-decay=0.0`
- warm start: `traffic-iters=1`
- `walker-steps=64`
- `walker-tick-iters=200`

### Fully specified defaults (unless overridden)

- `walker_path=linear`
- `dx,dy,dz = (1,0,0)` (moves along +x)
- `walker_r_local=6` → local cube size `13×13×13`
- `walker_hold_steps=0` (no hold phase)
- `walker_probe_r=0` (no probe dipoles)

### Observed results (00A, n=512, run_id=20260224_174834)

Artefact:

- `_Output/00_heavy_walker/00A_heavy_walker_test/20260224_174834/_csv/00A_heavy_walker_test_n512_20260224_174834.csv`

Key scalar stability:

- `phi_max` is tightly bounded: min `0.812644`, max `0.815058`, last `0.814140`.
- `total_field = sum(phi)` exhibits a short switch-on transient (max `103048` at `t=3`), then settles to a narrower band (for `t≥10`: min `27767`, max `61549`, mean `46068`).

Wake / lag diagnostics:

- `asym` range: min `-9.050e-4`, max `+2.323e-3`, mean `-9.170e-5`, last `-1.524e-4`.
- The wake sign is predominantly trailing for +x motion: `asym < 0` on `45/64` steps (≈`70%`), consistent with a weak backward wake.
- `hubble` is effectively zero at this pacing: mean `-2.208e-7` (range `-1.877e-6 … +2.611e-6`).

Interpretation:

This run is in an extremely sub-sonic regime (unit voxel moves with `tick_iters=200`), so the expected wake is small. The measured asymmetry is correspondingly tiny and predominantly trailing, while the primary field scalars remain stable at `n=512`. The early `total_field` spike is attributed to the intentionally minimal warm start (`traffic-iters=1`) and should not be over-interpreted.

---

## Experiment 00B — Heavy Walker warm-started test (moving source dynamics)

### Purpose

A direct follow-up to 00A that increases the warm start (`traffic-iters`) so the moving phase begins closer to the solver's quasi-steady regime.

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=telegraph`
- `traffic-c2=0.31`, `traffic-dt=1.0`
- `traffic-gamma=0.001`, `traffic-decay=0.0`
- warm start: `traffic-iters=200`
- `walker-steps=64`
- `walker-tick-iters=200`

### Observed results (00B, n=512, run_id=20260224_180928)

Artefact:

- `_Output/00_heavy_walker/00B_heavy_walker_warmstart/20260224_180928/_csv/00B_heavy_walker_warmstart_n512_20260224_180928.csv`

Key scalar stability:

- `phi_max` remains tightly bounded: min `0.812726`, max `0.815017`, last `0.814144`.
- `total_field = sum(phi)` shows a short transient early in the move phase (max `103237` at `t=2`), then settles to a narrower band (for `t≥10`: min `33525`, max `61661`, mean `46422`).

Wake / lag diagnostics:

- `asym` range: min `-9.424e-4`, max `+2.465e-3`, mean `-9.684e-5`, last `-1.541e-4`.
- The wake sign is predominantly trailing for +x motion: `asym < 0` on `46/64` steps (≈`72%`).
- `hubble` remains effectively zero: mean `-2.211e-7` (range `-1.542e-6 … +2.769e-6`).

Interpretation:

00B is statistically very similar to 00A: the field scalars remain stable at `n=512` and the measured wake is tiny and predominantly trailing at this extremely sub-sonic pacing (`tick_iters=200`). Increasing the warm start to `traffic-iters=200` does not eliminate the early `total_field` spike, suggesting that the initial transient is dominated by the onset of motion / delta reinjection rather than insufficient pre-relaxation.

---

## Experiment 00C — Heavy Walker pacing contrast (warm-started; faster move pacing)

### Purpose

A pacing contrast run: identical to 00B except `walker-tick-iters` is reduced to amplify wake/lag signatures while keeping the run short.

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=telegraph`
- `traffic-c2=0.31`, `traffic-dt=1.0`
- `traffic-gamma=0.001`, `traffic-decay=0.0`
- warm start: `traffic-iters=200`
- `walker-steps=64`
- `walker-tick-iters=100`

### Observed results (00C, n=512, run_id=20260224_182529)

Artefact:

- `_Output/00_heavy_walker/00C_heavy_walker_pacing_fast/20260224_182529/_csv/00C_heavy_walker_pacing_fast_n512_20260224_182529.csv`

Key scalar stability:

- `phi_max` remains tightly bounded: min `0.812085`, max `0.814710`, last `0.813101`.
- `total_field = sum(phi)` again shows an early transient (max `103237` at `t=4`), then settles into a broader band than 00A/00B (for `t≥10`: min `26722`, max `80254`, mean `46686`).

Wake / lag diagnostics:

- `asym` range: min `-1.974e-3`, max `+2.233e-3`, mean `-6.041e-4`, last `-1.038e-3`.
- The wake sign is strongly trailing for +x motion: `asym < 0` on `50/64` steps (≈`78%`).
- `hubble` remains small but no longer negligible at the 1e-7 level: mean `-1.308e-6` (range `-7.525e-6 … +2.522e-6`).

Interpretation:

Halving `tick_iters` (200 → 100) increases the effective driver speed and produces a visibly stronger trailing wake: the mean asymmetry magnitude increases by ~6× relative to 00A/00B, and the final step remains noticeably negative. Scalar stability remains good (`phi_max` bounded), and the persistent early `total_field` spike reinforces the conclusion from 00B: the dominant transient is tied to motion / delta reinjection rather than warm-start depth.

---

## Experiment 00D — Heavy Walker hold/decay (fast move phase then stop)

### Purpose

A two-phase run that makes the wake visible (fast pacing move phase) and then stops the source to measure how quickly the wake relaxes back toward symmetry.

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=telegraph`
- `traffic-c2=0.31`, `traffic-dt=1.0`
- `traffic-gamma=0.001`, `traffic-decay=0.0`
- warm start: `traffic-iters=200`
- move phase: `walker-steps=64`, `walker-tick_iters=100`
- hold phase: `walker-hold-steps=64`, `walker-hold-tick-iters=200`

### Observed results (00D, n=512, run_id=20260224_184042)

Artefact:

- `_Output/00_heavy_walker/00D_heavy_walker_hold_decay/20260224_184042/_csv/00D_heavy_walker_hold_decay_n512_20260224_184042.csv`

Key scalar stability:

- Move phase `phi_max` is bounded: min `0.812085`, max `0.814710`, last `0.813101`.
- Hold phase `phi_max` remains bounded and slightly higher on average: min `0.814003`, max `0.814792`, last `0.814336`.
- Move phase `total_field` shows the familiar early transient (max `103237`), then averages `47758`.
- Hold phase `total_field` stabilises into a tight band: min `43584`, max `46291`, mean `45087`, last `45063`.

Wake / lag diagnostics:

Move phase:

- `asym` mean `-6.041e-4`, last `-1.038e-3` (strong trailing wake at `tick_iters=100`).

Hold phase:

- `asym` range: min `-1.010e-3`, max `+7.194e-4`, mean `-5.153e-5`, last `-6.590e-5`.
- Wake relaxation is clear: `asym` at the start of hold is `-4.009e-4` and decays to `-6.590e-5` by the final step (≈6× reduction in magnitude).
- `hubble` also relaxes toward ~0: mean `-1.019e-7` during hold.

Interpretation:

00D confirms that the wake produced by faster pacing is not “stuck” to the lattice: once the source stops, the asymmetry decays rapidly toward symmetry on the hold timescale (`hold_tick_iters=200`). This makes Exp00 a useful calibration tool: it can both amplify wake signatures (move phase) and measure decay/relaxation behaviour under the chosen telegraph damping.

---

## Experiment 00E — Heavy Walker circle-path anisotropy probe (directional wake)

### Purpose

A directional anisotropy probe: move a delta source on a full circle so the motion direction rotates continuously, then measure whether wake metrics and probe dipoles vary with direction (axis vs diagonal segments).

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=telegraph`
- `traffic-c2=0.31`, `traffic-dt=1.0`
- `traffic-gamma=0.001`, `traffic-decay=0.0`
- warm start: `traffic-iters=200`
- `walker-path=circle`, `walker-circle-radius=8`, `walker-circle-period=0`
- `walker-probe-r=8`
- `walker-steps=256`
- `walker-tick-iters=100`

### Observed results (00E, n=512, run_id=20260224_191136)

Artefact:

- `_Output/00_heavy_walker/00E_heavy_walker_circle_anisotropy/20260224_191136/_csv/00E_heavy_walker_circle_anisotropy_n512_20260224_191136.csv`

Key scalar stability:

- `phi_max` remains bounded across the orbit: min `0.811786`, max `0.814759`, mean `0.813429`.
- `total_field = sum(phi)` shows the familiar early transient (min `18906`, max `103051`). In the post-transient window (`t≥20`): min `33526`, max `66446`, mean `47329`.

Wake / lag diagnostics:

- `asym` (all steps): min `-1.157e-3`, max `+2.355e-3`, mean `-1.995e-4` (≈`70%` of steps negative).
- `asym` (post-transient, `t≥20`): mean `-2.307e-4`.
- Directional modulation is present: binning `asym` by orbit angle shows a peak-to-peak variation of ≈`6.9e-4` (post-transient).
- A simple angular Fourier check shows the dominant modulation at **4×** the orbit angle (k=4 amplitude ≈`2.23e-4`), consistent with residual cubic-lattice anisotropy (axis vs diagonal segments).

Probe dipoles:

- With `probe_r=8`, the `dipole_ang_phi` tracks the orbit direction closely: the wrapped angle error has a 95th percentile of ≈`0.253 rad` (≈`14.5°`).

Interpretation:

00E provides direct evidence that, while scalar stability remains good, the wake diagnostics retain a measurable **direction-dependent** component in telegraph mode. The dominant 4-fold modulation is exactly the kind of residual anisotropy expected from a 6-neighbour cubic stencil and is therefore a useful calibration target (rather than a bug).

---

## Experiment 00F — Heavy Walker damping contrast (linear, fast pacing, strong damping)

### Purpose

A damping contrast run: identical to the fast-pacing linear walker (00C) but with a much stronger telegraph damping term (`gamma=0.010`) to suppress ringing and reduce wake/lag signatures.

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=telegraph`
- `traffic-c2=0.31`, `traffic-dt=1.0`
- `traffic-gamma=0.010`, `traffic-decay=0.0`
- warm start: `traffic-iters=200`
- `walker-steps=64`
- `walker-tick-iters=100`

### Observed results (00F, n=512, run_id=20260224_194217)

Artefact:

- `_Output/00_heavy_walker/00F_heavy_walker_damping_strong/20260224_194217/_csv/00F_heavy_walker_damping_strong_n512_20260224_194217.csv`

Key scalar stability:

- `phi_max` is extremely stable: min `0.813160`, max `0.813777`, mean `0.813710`, last `0.813697`.
- `total_field = sum(phi)` is also stable and shows no large switch-on spike: min `20586`, max `47660`, mean `45879`, last `45345`.
- Post-transient (`t≥10`) band is tight: min `45345`, max `47606`, mean `46618`.

Wake / lag diagnostics:

- `asym` is strongly suppressed relative to 00C: min `-1.298e-5`, max `+8.412e-5`, mean `+4.568e-5`, last `+1.140e-5`.
- Wake sign is no longer predominantly trailing: `asym < 0` on `3/64` steps (≈`5%`).
- `hubble` remains tiny: mean `9.254e-8` (range `-2.566e-8 … +3.638e-7`).

Interpretation:

00F demonstrates that increasing telegraph damping by 10× largely eliminates both (i) the early large `total_field` transient seen in the weakly damped runs and (ii) the amplified trailing wake observed at fast pacing (00C/00D). This strongly suggests that, in the weak-damping regime, the measured wake magnitude is dominated by underdamped ringing / dispersive memory rather than an unavoidable geometric lag.

---

## Experiment 00G — Heavy Walker diffusion baseline (linear, fast pacing)

### Purpose

A diffusion-mode control run: keep the walker forcing and pacing identical to the fast linear telegraph case (00C) but switch `traffic-mode=diffuse` to separate generic relaxer lag from telegraph-specific ringing/wake dynamics.

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=diffuse`
- warm start: `traffic-iters=200`
- `traffic-decay=0.0`
- `walker-steps=64`
- `walker-tick-iters=100`

### Observed results (00G, n=512, run_id=20260224_201847)

Artefact:

- `_Output/00_heavy_walker/00G_heavy_walker_diffuse_baseline/20260224_201847/_csv/00G_heavy_walker_diffuse_baseline_n512_20260224_201847.csv`

Key scalar stability:

- `phi_max`: min `10.654843`, max `10.901070`, mean `10.886095`, last `10.901070`.
- `total_field = sum(phi)` grows monotonically (no decay): min `300.000021`, max `6600.000710`, mean `3450.000315`, last `6600.000710`.

Wake / lag diagnostics:

- `asym`: min `-4.887845e-1`, max `-2.298754e-1`, mean `-4.747503e-1`, last `-4.887845e-1`.
- `asym < 0` on `64/64` steps (uniformly trailing for +x motion).
- `hubble` mean `-1.017583e-1`.

Interpretation:

In diffusion mode (with `traffic-decay=0.0`), the injected field mass accumulates and the relaxer cannot “keep up” with a moving delta source at this pacing. The result is an extreme trailing asymmetry (nearly all local mass behind the source). This is a useful control: it shows that the modest wakes in telegraph mode (00C/00D) are not merely a generic artefact of “moving a source with finite tick budget”, and that the telegraph dynamics (and damping) materially change the lag regime.

---

## Experiment 00H — Heavy Walker diffusion control with decay (linear, fast pacing)

### Purpose

A diffusion-mode follow-up to 00G that adds nonzero decay (`traffic-decay=0.010`) so the injected field mass is bounded. This allows a fairer comparison of *steady* diffusion lag versus telegraph wake.

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=diffuse`
- warm start: `traffic-iters=200`
- `traffic-decay=0.010`
- `walker-steps=64`
- `walker-tick-iters=100`

### Observed results (00H, n=512, run_id=20260224_225041)

Artefact:

- `_Output/00_heavy_walker/00H_heavy_walker_diffuse_decay/20260224_225041/_csv/00H_heavy_walker_diffuse_decay_n512_20260224_225041.csv`

Key scalar stability:

- `phi_max` is extremely stable: min `8.757954`, max `8.758883`, mean `8.758863`, last `8.758883`.
- `total_field = sum(phi)` is now bounded (no accumulation runaway): min `94.145`, max `99.000`, mean `98.880`, last `99.000`.

Wake / lag diagnostics:

- `asym` remains uniformly trailing for +x motion (`asym < 0` on `64/64` steps), but at a much smaller magnitude than 00G: min `-1.889e-1`, max `-1.472e-1`, mean `-1.881e-1`, last `-1.889e-1`.
- `hubble` remains large and negative but is less extreme than 00G: mean `-1.211e-1` (range `-1.216e-1 … -9.542e-2`).

Interpretation:

Adding decay bounds the diffusion field mass and reduces the saturated trailing wake seen in 00G, but the lag remains qualitatively severe (uniformly negative `asym`). This supports the conclusion that diffusion is a poor model for translating a compact moving source at this pacing: even when bounded, the relaxer behaves like an “accumulation behind the source” field rather than a quasi-comoving profile.

---

## Experiment 00I — Heavy Walker telegraph decay contrast (linear, fast pacing)

### Purpose

A telegraph-mode follow-up to 00C that adds nonzero decay (`traffic-decay=0.010`) while keeping weak damping (`gamma=0.001`). This probes whether the early transient and trailing wake are driven primarily by field memory (decay) versus oscillatory ringing (damping).

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=telegraph`
- `traffic-c2=0.31`, `traffic-dt=1.0`
- `traffic-gamma=0.001`, `traffic-decay=0.010`
- warm start: `traffic-iters=200`
- `walker-steps=64`
- `walker-tick-iters=100`

### Observed results (00I, n=512, run_id=20260224_230023)

Artefact:

- `_Output/00_heavy_walker/00I_heavy_walker_telegraph_decay/20260224_230023/_csv/00I_heavy_walker_telegraph_decay_n512_20260224_230023.csv`

Key scalar stability:

- `phi_max` is extremely stable: min `0.812774`, max `0.813023`, mean `0.812980`, last `0.812970`.
- `total_field = sum(phi)` is bounded and substantially lower than the zero-decay telegraph runs: min `18138`, max `35760`, mean `34352`, last `33893`.
- Post-transient (`t≥10`) band is tight: min `33893`, max `35306`, mean `34724`.

Wake / lag diagnostics:

- `asym` is strongly suppressed relative to 00C/00D: min `-4.886e-5`, max `+7.424e-6`, mean `-1.213e-5`, last `-4.684e-5`.
- Wake sign remains predominantly trailing for +x motion: `asym < 0` on `53/64` steps (≈`83%`).
- `hubble` remains negligible: mean `-3.150e-8`.

Interpretation:

Adding telegraph decay (`0.010`) largely removes the large early `total_field` transient and collapses the wake magnitude by more than an order of magnitude versus the weakly damped, zero-decay fast-pacing case (00C). This indicates that a significant component of the observed wake/transient in the zero-decay regime is driven by **field memory accumulation**, not purely by oscillatory ringing. Decay therefore provides a distinct calibration knob from damping: it bounds the steady field mass while preserving telegraph kinematics.

---

## Experiment 00J — Heavy Walker near-Mach stress test with relaxation hold (telegraph)

### Purpose

A near-Mach pacing stress test in telegraph mode. The move phase uses `walker-tick-iters=2` (very fast, ≈Mach~0.9 for `c2=0.31, dt=1.0`) with bounded memory (`traffic-decay=0.010`). The run then switches to a slow hold (`hold_tick_iters=200`) to measure how quickly the wake relaxes once the source stops.

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=telegraph`
- `traffic-c2=0.31`, `traffic-dt=1.0`
- `traffic-gamma=0.001`, `traffic-decay=0.010`
- warm start: `traffic-iters=200`
- move phase: `walker-steps=64`, `walker-tick-iters=2`
- hold phase: `walker-hold-steps=64`, `walker-hold-tick-iters=200`

### Observed results (00J, n=512, run_id=20260224_231907)

Artefact:

- `_Output/00_heavy_walker/00J_heavy_walker_near_mach_hold/20260224_231907/_csv/00J_heavy_walker_near_mach_hold_n512_20260224_231907.csv`

Key scalar stability:

Move phase (t=1..64):

- `phi_max` is elevated and highly non-linear: min `1.075925`, max `1.227722`, mean `1.203151`, last `1.200321`.
- `total_field = sum(phi)` is bounded but rises strongly across the move phase: min `10648.517`, max/last `20255.359`, mean `15457.941`.

Hold phase (t=65..128):

- `phi_max` collapses immediately into the steady band: min `0.813630`, max `0.813700`, mean `0.813661`, last `0.813661`.
- `total_field` stabilises into a tight post-stop band: min `33296.246`, max `35050.998`, mean `33739.429`, last `33720.636`.

Wake / lag diagnostics:

Move phase:

- `asym` is large and oscillatory: min `-3.8128e-1`, max `+1.2595e-1`, mean `-4.9728e-2`, last `+1.2555e-1`.
- Wake sign is mixed (fast regime): `asym < 0` on `30/64` steps.

Hold phase:

- `asym` relaxes to a small, consistently trailing value: min `-1.0496e-4`, max `-1.1522e-5`, mean `-3.7121e-5`, last `-3.6516e-5` (`asym < 0` on `64/64`).
- `asym` magnitude reduces by ≈`2.87×` over the hold window (start `-1.0496e-4` → end `-3.6516e-5`).
- `hubble` remains negligible during hold: mean `-9.7516e-8`.

Interpretation:

00J demonstrates the expected behaviour of the near-Mach regime: the local wake metric becomes large and sign-flipping (strong non-linear, dispersive behaviour) during the fast move phase, even with decay enabled. Once the source stops and the solver is given a slow hold, the field rapidly returns to the same stable band seen in the lower-speed telegraph runs, with a small trailing asymmetry.

---

## Experiment 00K — Heavy Walker circle anisotropy probe with telegraph decay (bounded memory)

### Purpose

A decay-bounded repeat of 00E: move the delta source on a full circle in telegraph mode while enabling `traffic-decay=0.010` to test whether the previously observed k=4 directional modulation persists when field memory is bounded.

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=telegraph`
- `traffic-c2=0.31`, `traffic-dt=1.0`
- `traffic-gamma=0.001`, `traffic-decay=0.010`
- warm start: `traffic-iters=200`
- `walker-path=circle`, `walker-circle-radius=8`, `walker-circle-period=0`
- `walker-probe-r=8`
- `walker-steps=256`
- `walker-tick-iters=100`

### Observed results (00K, n=512, run_id=20260224_234007)

Artefact:

- `_Output/00_heavy_walker/00K_heavy_walker_circle_decay_anisotropy/20260224_234007/_csv/00K_heavy_walker_circle_decay_anisotropy_n512_20260224_234007.csv`

Key scalar stability:

- `phi_max` is extremely stable across the orbit: min `0.812774`, max `0.813097`, mean `0.813033`.
- `total_field = sum(phi)` is bounded and quickly reaches a tight steady band: min `18138`, max `35734`, mean `35077`.
- Post-transient (`t≥20`) band is exceptionally tight: min `35179.48`, max `35191.96`, mean `35186.00`.

Wake / lag diagnostics:

- `asym` is near zero throughout: min `-9.553e-6`, max `+3.611e-5`, mean `+7.311e-7` (≈`26%` of steps negative).
- Directional modulation is effectively absent at this scale: binned peak-to-peak variation ≈`8.82e-6` (post-transient).
- A Fourier check shows the k=4 component is tiny: amplitude ≈`1.54e-6` (post-transient), orders of magnitude below 00E.

Probe dipoles:

- With wake strongly suppressed, `dipole_ang_phi` becomes noisy and is not a reliable direction tracker in this regime; interpret probe angles only when accompanied by a clear dipole magnitude and a non-negligible asymmetry signal.

Interpretation:

00K shows that enabling telegraph decay (`0.010`) collapses the circle-path wake and removes the previously measurable k=4 directional modulation seen in 00E. This strongly suggests that most of the 00E modulation was driven by long-lived field memory / ringing rather than an unavoidable geometric anisotropy of the stencil. In practice, decay is therefore a powerful “cleaning” knob when the goal is near-isotropic behaviour under moving-source diagnostics.

---

## Experiment 00L — Heavy Walker tick sweep (clean telegraph regime)

### Purpose

A compact pacing map that replaces further lettered experiments: sweep `walker-tick-iters` across a range (near-Mach → quasi-static) in the **clean telegraph** regime (`gamma=0.001`, `decay=0.010`) and record summary wake metrics per tick.

### Canonical invocation (from `experiments.py`)

- `n=512`
- `traffic-mode=telegraph`
- `traffic-c2=0.31`, `traffic-dt=1.0`
- `traffic-gamma=0.001`, `traffic-decay=0.010`
- warm start: `traffic-iters=200`
- `walker-steps=64`
- `walker-sweep-ticks=2,4,8,16,32,64,100,200`

### Observed results (00L, n=512, run_id=20260225_002811)

Artefact:

- `_Output/00_heavy_walker/00L_heavy_walker_tick_sweep_clean_telegraph/20260225_002811/_csv/walker_sweep.csv`

Summary table (per tick)

| tick | Mach | move_asym_min | move_asym_last |
|---:|---:|---:|---:|
| 2 | 0.898 | -3.81284e-1 | +1.25553e-1 |
| 4 | 0.449 | -5.67950e-2 | +6.81220e-2 |
| 8 | 0.225 | +2.15500e-2 | +5.44150e-2 |
| 16 | 0.112 | +2.07730e-2 | +2.26740e-2 |
| 32 | 0.056 | +2.08700e-3 | +2.34800e-3 |
| 64 | 0.028 | +3.64000e-4 | +3.66000e-4 |
| 100 | 0.018 | -4.90000e-5 | -4.70000e-5 |
| 200 | 0.009 | -5.10000e-5 | -5.10000e-5 |

Interpretation

- **Near-Mach regime (tick=2):** highly non-linear, sign-flipping wake (large negative minimum with positive final), consistent with the fast dispersive behaviour seen in 00J.
- **Intermediate regime (tick=4–16):** wake remains appreciable, with a tendency toward positive final asymmetry in this bounded-memory telegraph setting.
- **Quasi-static regime (tick≥32):** wake magnitude collapses rapidly (≈`2e-3` at tick=32, ≈`3.6e-4` at tick=64).
- **Slow regime (tick≥100):** wake is effectively negligible and slightly trailing (≈`-5e-5`), matching the low-wake behaviour observed in 00I/00K.

Operationally, this sweep provides a single “phase map” for choosing walker pacing: use `tick≥64` for clean, quasi-steady diagnostics; use `tick≤4` to intentionally enter the strongly non-linear wake/shock regime.

---

## Conclusions

Exp00 establishes Heavy Walker as a reliable **traffic-solver calibration harness** and provides a compact regime map for later CAELIX experiments.

### Practical regime guidance

- **Clean, quasi-steady motion diagnostics:** use telegraph with **bounded memory** (`decay≈0.010`) and modest damping (`gamma≈0.001`), and choose `tick_iters ≥ 64` (00L). This yields near-zero wake and stable global scalars.
- **Intentional non-linear / shocky behaviour:** use `tick_iters ≤ 4` (00L) or the explicit near-Mach move (00J). Expect sign-flipping asymmetry and strongly non-linear transients.
- **Avoid diffusion for moving-source dynamics:** diffusion lags severely even when bounded by decay (00H). It is useful as a control but not as a comoving driver.

### Interpretation discipline

- Most apparent “anisotropy” seen under motion in the zero-decay regime (00E) disappears once decay bounds field memory (00K). Treat directional artefacts as a **memory/ringing diagnostic first**, and only as stencil geometry second.
- `gamma` and `decay` are distinct knobs: damping suppresses ringing (00F), while decay bounds memory and removes large early transients without requiring heavy damping (00I/00K).
