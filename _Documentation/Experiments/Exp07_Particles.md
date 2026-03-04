# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# Exp07 — Particles (Linear collider, walkers, nucleus + halo)

## Abstract

Exp07 establishes the **Linear Interaction Baseline** for particle dynamics in CAELIX.

Before introducing non-linear binding mechanisms (Exp08), we first characterise what a driven, coherent wave-packet (“Walker”) does in a **linear telegraph vacuum**. Exp07 therefore builds a collider-style setup: one or two walkers approach the interaction region, overlap or fly past, and then separate.

Primary objectives:

1) **Quantify superposition (“ghosting”).** In the linear limit, two walkers pass through one another without hard scattering. Overlap produces interference in the interaction zone, but the outgoing wakes remain wave-like and unbound.
2) **Spin & chirality.** Walkers can be driven with a helical injection (“spin” analogue). Same-spin vs opposite-spin collisions test whether phase structure leaves a measurable signature in local energy deposition even when the substrate remains linear.
3) **Back-reaction probe.** A perturbative coupling can feed field gradients back into the walker trajectory. Any systematic deflection is a first hint of an emergent “proto-force” in an otherwise linear medium.

Exp07 is the control group for later particle claims: **any deviation from this baseline** (persistent localisation, capture, true scattering) becomes meaningful only because Exp07 pins down what “non-interaction” looks like in the same apparatus.

---

## Theoretical basis

### The superposition principle (the physics of ghosts)

In Exp07 the carrier evolution is **linear** in \(\phi\) and \(v\) (telegraph mode with optional damping). For two sources A and B in a linear field, the total response is simply:

\[
\phi_{A+B} = \phi_A + \phi_B
\]

Prediction (linear baseline):

- The **energy density** scales like \(E \propto (\phi_A + \phi_B)^2\), so the interaction zone can show **interference fringes**.
- But the **wavefronts emerge unaltered** in the sense that there is no hard scattering event, no bound state, and no persistent composite without extra mechanisms.

A useful physical interpretation is “ghost matter”: coherent excitations that couple to the carrier, but (in the linear limit) do not couple to each other strongly enough to scatter.

### Chirality (“spin”) as helical injection

A Walker is a moving, localised driver (a controlled source term). Spin in Exp07 is implemented by orbiting the injection point around its translational centre (a helical path). This imprints a handed phase structure onto the wake.

Same-spin \((+,+)\) vs opposite-spin \((+,-)\) collisions therefore probe whether the vacuum is sensitive to phase/chirality through **measurable deposition differences**, even when true field–field scattering is absent.

### Back-reaction as a perturbative force law

Back-reaction adds a controlled feedback: field gradients produce incremental momentum updates on the walker. In a linear substrate this is not “non-linear scattering of the field”, but it can produce trajectory-level effects:

- **Flyby deflection** (impact parameter changes)
- **Early-time repulsion / attraction** signatures

Any repeatable deviation from the straight-line baseline is a candidate proto-force signal.

---

## Experimental setup

Exp07 is implemented in `collider.py` and executed via `core.py`, but the physical picture is a lattice collider with controlled drivers and calorimetry.

### Geometry

- **Two-body collider:** Walker A and Walker B are initialised at \(x=\pm d\) and translated toward \(x=0\) with speed \(v\). The collision tick is snapped to an integer step (`t_c_int`) so that same-spin vs opposite-spin comparisons align in time.
- **Flyby control:** An impact parameter \(b\) (and optional \(b_z\)) offsets the trajectories to create near-misses.
- **Single-body (hydrogen) mode:** Walker B can be disabled; Walker A then probes a central nucleus/halo configuration.

### Injection (“walkers”)

- A Walker is a moving, localised driver that continuously injects a coherent source term into the carrier.
- **Spin / chirality analogue:** the injection point can orbit its translational centre (helix), imprinting handed phase structure onto the wake.
- **Shutter / decay control:** driving can be disabled after a chosen tick to test whether localisation persists without forcing.

### Back-reaction

Back-reaction feeds a controlled field-gradient signal back into walker momentum. In a linear substrate this is a perturbative trajectory coupling: it can produce deflection without implying true field–field scattering.

### Detectors (calorimetry)

The readout is structured as collider instrumentation:

- **Interaction slab:** energy fractions in a central region (`mid_frac_*`) act as the primary “collision detector”.
- **Shell calorimeters:** concentric shells (`shell1_*`, `shell2_*`) capture how energy redistributes as the wake expands.
- **Octants:** coarse angular energy bins test anisotropy and asymmetry.

### Nucleus and halo (hydrogen regime)

- **Nucleus:** a central bias/pin (DC) or driven centre, used as an attractor/interaction target.
- **Halo:** an explicit damping region (cooling proxy) used to bleed energy and test capture/spiral-in behaviour.

### Suites and intent

The Exp07 preset family (07A–07Q) is arranged as escalating probes:

- **07A–07D:** superposition baseline (same vs opposite spin, head-on). Expect ghosting with interference signatures.
- **07E–07F:** decay test (turn off driving; measure dissolution).
- **07G–07H:** flyby probe with back-reaction (glancing impact).
- **07I–07J:** binding attempt (slow approach; look for capture vs escape).
- **07K–07L:** nucleus-on probe (centre bias without halo).
- **07M–07Q:** nucleus + halo “hydrogen” regime (capture/spiral-in vs flyby).

(Exact preset names and parameters are defined in `experiments.py`.)

---

## Measurement and outputs

Exp07 emits a single per-step CSV per run (written under the canonical `_csv/` folder when passed as a bare filename) plus a standard log under `_log/`.

### Provenance

Each CSV begins with provenance headers including:

- run id, timestamp
- grid size and telegraph parameters
- collider configuration (enable_b, nucleus, halo, detectors)
- critical numeric knobs (charges, speeds, orbit params, backreaction, hold/shutter)

These headers are treated as part of the experimental record.

### CSV structure (per step)

The collider CSV is intentionally “flight-recorder” heavy. Representative column groups:

#### Kinematics

- `step`, `t_c_int`, `t_rel`
- positions: `xA,yA,zA`, `xB,yB,zB` (B may be disabled)
- separation: `sep, sep_x, sep_yz` (NaN when B disabled)
- min separation summary: recorded in the log/footer and (when B enabled) tracked during the run

#### Driver bookkeeping (research-grade)

These columns exist to prevent “hidden driver” artefacts:

- injected magnitudes: `src_qA, src_qB, src_qN` (post shutter / enable / nucleus mode)
- injection diagnostics: `src_max, src_E`
- injection cells: `cellAx,cellAy,cellAz`, `cellBx,cellBy,cellBz`, `cellNx,cellNy,cellNz`
- helix phase: `angA, angB, dphi` (dphi wrapped to \([-\pi,\pi]\))

#### Field diagnostics

- global energies and maxima: `phi_energy, phi_max`, `vel_energy, vel_max`, `E_tot`, `maxabs_tot`
- centre-point samples: `phi_c, vel_c` (sampled after DC nucleus/halo post-processing)

#### Calorimetry / deposition

- mid-slab energies/fractions: `phi_midE, vel_midE, mid_frac_*`
- local balls around A/B/centre: `phi_A_E, vel_A_E, localA_frac_*` and B equivalents (NaN when B disabled)
- shells around interaction centre: `shell1_*`, `shell2_*` fractions and derived ring metrics
- octant energies for anisotropy checks: `octE0..octE7` (present; may be zero if octants disabled)

#### Back-reaction

When enabled:

- `dVAx,dVAy,dVAz`, `dVBx,dVBy,dVBz`
- momenta: `PAx,PAy,PAz`, `PBx,PBy,PBz`

#### Nucleus and halo

- nucleus mode and parameters are in provenance
- halo activity is recorded as “actually damped voxels” per step:
  - `halo_touched_vel`, `halo_touched_phi`
  - running maxima: `halo_touched_vel_max`, `halo_touched_phi_max`

---

## How to read the collider

Exp07 is a control experiment: the goal is to make “non-interaction” concrete and measurable.

### Success criteria (the linear baseline)

1) **Ghost signature (superposition):**
   - Head-on overlap gives `sep_min≈0` and `phi_max≈(A+B)` (≈2.0 for unit drivers).
   - Outgoing wakes remain wave-like: no hard trajectory reversal and no bound separation without additional mechanisms.

2) **Interference signature (chirality):**
   - Same-spin vs opposite-spin differences should appear primarily in **interaction-zone deposition** (e.g. `mid_frac_tot`, shell fractions), not in gross kinematics.

3) **Proto-force probe (back-reaction):**
   - With non-zero \(b\), compare `sep_min` to the nominal impact parameter.
   - Any repeatable deflection, early-time kick, or anisotropic calorimetry shift is evidence of trajectory-level coupling.

### Hygiene / interpretation discipline

- Use the driver bookkeeping (`src_q*`, shutter state) to rule out hidden forcing.
- If the log reports late-time out-of-bounds injection/clamping, prioritise the interaction window around `t_c_int` for physics interpretation.
- Halo touched counts measure actual damping events (a real cooling proxy), not halo mask size.

## Baseline results (07A / 07B, 2026-02-13)

These are the first re-run baselines after the Exp07 collider CSV/logging fixes.

**Shared setup (both runs):** telegraph carrier (`c2=0.31`, `dt=1.0`, `gamma=0.001`, `decay=0`), `n=512`, sponge boundary (`width=32`, `strength=1.0`), `vx=0.15`, orbit radius `80`, omega `0.12`, steps `2000`. Back-reaction, detectors, nucleus, and halo are **off**.

### Key measurements

| Metric | 07A (opp spin, `spin_b=-1`) | 07B (same spin, `spin_b=+1`) | Notes |
|---|---:|---:|---|
| `t_c_int` | 427 | 427 | collision time is snapped to an integer step |
| `sep_min` | ~7.79e-12 (step 427) | ~7.33e-12 (step 427) | effectively a centred hit (numerical zero) |
| `phi_max` peak | 2.015 (step 427) | 2.011 (step 424) | close to A+B ≈ 2.0 (superposition) |
| `vel_max` peak | 1.984 | 2.009 | peak velocity envelope |
| `E_tot` peak | 1936.46 | 1942.21 | comparable energy budget |
| `mid_frac_tot` peak | 0.0935 (step 427) | 0.1139 (step 435) | same-spin deposits more energy in the interaction slab |
| `mid_frac_tot` mean (t_c±25) | 0.0872 | 0.0968 | local deposition signature persists over a short window |

### Conclusions

1) **Ghosting / superposition confirmed.** A head-on overlap yields `phi_max` ≈ 2.0 with no evidence of hard scattering or “thud” behaviour. This is the expected linear-baseline outcome.

2) **Spin signature is measurable.** Same-spin (07B) shows a larger interaction-zone deposition (`mid_frac_tot`) than opposite-spin (07A), consistent with a chirality-dependent interference pattern.

3) **Clean run invariants.** No warnings, no out-of-bounds events, halo metrics remain zero (disabled), and there is no hidden driving beyond the two walkers (`src_qA=src_qB=1`, `src_qN=0`).

These two runs pass as the canonical Exp07 baseline pair to compare against detector-enabled variants (07C/07D) and later back-reaction / nucleus / halo regimes.

---

## Detector-enabled baseline (07C / 07D, 2026-02-13)

07C/07D repeat the 07A/07B baseline pair with the collider detector rig enabled (shell calorimetry and derived ring metrics). Back-reaction, nucleus and halo remain **off**.

### Key measurements

| Metric | 07C (opp spin) | 07D (same spin) | Notes |
|---|---:|---:|---|
| `t_c_int` | 427 | 427 | integer-snapped collision step |
| `sep_min` | ~7.79e-12 (step 427) | ~7.33e-12 (step 427) | centred overlap (numerical zero) |
| `phi_max` peak | 2.015 (step 427) | 2.011 (step 424) | A+B ≈ 2.0 (superposition) |
| `E_tot` peak | 1936.46 | 1942.21 | comparable energy budget |
| `mid_frac_tot` peak | 0.0935 (step 427) | 0.1139 (step 435) | same-spin deposits more in the interaction slab |
| `shell1_frac_tot` mean (t_c±25) | 0.0140 | 0.0138 | shell deposition near collision is similar |
| `shell2_frac_tot` mean (t_c±25) | 0.0150 | 0.0149 | shell deposition near collision is similar |

### Conclusions

1) **Detector wiring is live and stable.** Shell metrics are non-zero, well-behaved, and consistent with the expected post-collision wake expansion.

2) **Spin signature remains localised to the interaction zone.** The mid-slab deposition (`mid_frac_tot`) shows the same-spin enhancement seen in 07A/07B, while shell deposition averaged near the collision window is broadly similar between spins.

3) **Still a clean linear baseline.** As with 07A/07B, there is no evidence of hard scattering; the main measurable difference is interference structure, not trajectory disruption.

---

## Decay / shutter test (07E / 07F, 2026-02-13)

07E/07F are the “stop singing, does the song persist?” controls. The walkers are driven until the collision tick, then the injection shutter disables driving (both sources off) and the remaining field is allowed to disperse.

### Key measurements

| Metric | 07E (opp spin) | 07F (same spin) | Notes |
|---|---:|---:|---|
| Injection cut step (`inj_scale→0`) | 427 | 427 | shutter engages exactly at the snapped collision step |
| `src_qA/src_qB` after cut | 0 / 0 | 0 / 0 | confirms no hidden driving post-cut |
| `E_tot` at cut | 1644.76 | 1647.11 | comparable energy at shutdown |
| `E_tot` final (step 2999) | 23.04 | 23.10 | small residual energy in the box |
| Survival fraction (`E_final/E_cut`) | 0.0140 | 0.0140 | ≈ 1.4% of energy remains by end |
| `mid_frac_tot` mean (cut-25..cut-1) | 0.0861 | 0.0836 | interaction-zone deposition while driven |
| `mid_frac_tot` mean (cut..cut+499) | 0.0225 | 0.0340 | collapse toward low-level residuals |

### Conclusions

1) **Driven localisation does not persist.** Once injection shuts off, the “particles” rapidly lose identity and the field energy disperses, with only ~1.4% of the pre-cut energy remaining by the end of the run.

2) **Shutter wiring is correct and fail-obvious.** `inj_scale` drops to 0 and both `src_qA/src_qB` go to 0 at the cut tick, so post-cut behaviour is not contaminated by continued forcing.

3) **Motivation for Exp08 remains intact.** These runs cleanly demonstrate the limitation of linear driven walkers: without a non-linear localisation mechanism, packets disperse.

---

## Scatter / flyby with back-reaction (07G / 07H, 2026-02-13)

07G/07H are the first “do they notice each other?” probes: a non-zero impact parameter (`b=4.0`) with **full-vector back-reaction** enabled (`axes=xyz`, `k=0.020`, `mode=repel`, `vmax=0.250`). Detectors and octant flux are **on**; nucleus and halo are **off**.

**Important log note (both runs):** each run reports a late-time out-of-bounds injection event (07G: step 2171, 07H: step 2174) and clamps injection thereafter. This occurs well after the interaction window (`t_c≈427`) and does not affect the near-collision metrics below, but it means late-time trajectory / `sep_final` should not be over-interpreted.

### Key measurements

| Metric | 07G (opp spin, `spin_b=-1`) | 07H (same spin, `spin_b=+1`) | Notes |
|---|---:|---:|---|
| `t_c_int` | 427 | 427 | collision time snapped to integer step |
| `sep_min` | 4.229 (step 427) | 4.016 (step 431) | near-miss: compare to nominal `b=4.0` |
| `phi_max` peak (t_c±25) | 1.041 (step 431) | 1.030 (step 414) | no doubled peak: no direct overlap |
| `E_tot` peak (t_c±25) | 1722.56 (step 452) | 1725.05 (step 452) | comparable energy budget |
| `mid_frac_tot` peak (t_c±25) | 0.0914 | 0.0917 | interaction-zone deposition |
| `mid_frac_tot` mean (t_c±25) | 0.0869 | 0.0874 | small difference, within the same band |
| `shell1_frac_tot` mean (t_c±25) | 0.0140 | 0.0141 | shell calorimetry near collision |
| `shell2_frac_tot` mean (t_c±25) | 0.01509 | 0.01510 | shell calorimetry near collision |

### Conclusions

1) **Weak deflection, not scattering.** Both runs achieve a closest approach very near the nominal impact parameter (`b=4.0`). Opposite spin shows a slightly larger `sep_min` (4.229), consistent with a weakly repulsive flyby in this configuration.

2) **No hard interaction signature.** As expected for a linear carrier with modest back-reaction, there is no “collision peak” (`phi_max` stays ≈1.03–1.04, not ≈2.0), and the interaction-zone deposition (`mid_frac_tot`) remains in the same band for both spins.

3) **Detector rig remains stable under back-reaction.** Shell fractions are non-zero and well-behaved around the interaction window, indicating that enabling back-reaction + octants does not destabilise the measurement plumbing.

Operational note: if we want to avoid late-time clamping in future flyby runs, either shorten `--collider-steps` so the run ends before the walkers exit the box, or add an explicit early-stop condition in the experiment wiring (stop once both walkers are out-of-bounds and the wake has cleared the detector shells).

---

## Binding attempt (07I / 07J, 2026-02-13)

07I/07J attempt a “capture” regime by slowing the approach (`vx=0.05`) and increasing the back-reaction coupling (`k=0.20`, `axes=xyz`, `mode=repel`, `vmax=0.250`) with a modest impact parameter (`b=4.0`). Detectors and octants are **on**; nucleus and halo are **off**.

### Key measurements

| Metric | 07I (opp spin, `spin_b=-1`) | 07J (same spin, `spin_b=+1`) | Notes |
|---|---:|---:|---|
| `t_c_int` | 1280 | 1280 | slow approach → later interaction window |
| `sep_min` | 6.081 (step 1700) | 19.941 (step 1247) | closest approach differs strongly by spin |
| `sep_final` (step 4999) | 187.42 | 308.44 | both runs escape (no binding) |
| `phi_max` peak (t_c±25) | 1.0278 | 1.0261 | no overlap peak (still a near-miss) |
| `E_tot` peak | 2093.58 | 2113.75 | comparable energy budget |
| `E_tot` final | 1358.48 | 1243.27 | still driven; energy remains high |
| `mid_frac_tot` mean (t_c±25) | 0.0446 | 0.0513 | interaction-zone deposition band |
| `PB` peak | 6.46e-4 | 1.68e-3 | momentum proxy shows stronger kick in 07J |

### Conclusions

1) **No binding / no capture.** Both runs end with large separation; in this configuration, back-reaction produces a deflected flyby rather than a closed orbit.

2) **Spin changes the interaction strength.** Same-spin (07J) stays much farther at closest approach (`sep_min≈19.9`) yet shows a larger momentum proxy (`PB` peak), consistent with a stronger early-time deflection that prevents a close pass.

3) **Still consistent with a linear substrate + feedback.** There is no sign of a hard collision event (`phi_max≈1.026–1.028` around the interaction window). The main effect is trajectory shaping via the feedback term, not field-field scattering.

---

07K/07L add a third body: a **central DC nucleus** at the lattice centre (`nucleus=1`, `nucleus_q=1`, `mode=dc`). Back-reaction remains enabled (`k=0.20`, `axes=xyz`, `mode=repel`, `vmax=0.250`), with a slow approach (`vx=0.05`) and `b=4.0`. Detectors and octants are **on**; halo is **off**.

Interpretation note: in this mode the nucleus is a DC centre bias/pin (reflected in `phi_c`), not an additional “walker source” (so `src_qN` remains 0 and `src_E` stays at 2.0).

### Key measurements

| Metric | 07K (opp spin) | 07L (same spin) | Notes |
|---|---:|---:|---|
| `t_c_int` | 1280 | 1280 | slow approach → later interaction window |
| `sep_min` | 6.569 (step 1568) | 4.677 (step 1225) | closest approach; both remain flybys |
| `sep_final` (step 4999) | 285.23 | 460.83 | both runs escape (no binding) |
| `phi_max` peak (t_c±25) | 1.0306 (step 1298) | 1.0314 (step 1288) | no overlap peak |
| `E_tot` peak | 2094.61 | 2107.27 | comparable energy budget |
| `E_tot` final | 899.94 | 532.77 | still driven; residual field remains |
| `mid_frac_tot` mean (t_c±25) | 0.0451 | 0.0617 | nucleus + spin changes local deposition |
| `mid_frac_tot` peak (t_c±25) | 0.0464 | 0.0689 | same-spin deposits more in the mid slab |
| `phi_c` mean | 0.0134 | 0.0129 | DC nucleus leaves a non-zero centre bias |

**Log note (07L):** a late-time out-of-bounds injection event occurs at step 4361 (B leaves the box) and clamps injection thereafter. This is far after the interaction window and does not affect the near-window metrics, but late-time separation should be treated cautiously.

### Conclusions

1) **No nucleus-assisted capture in this regime.** Both runs end with large separation; the nucleus bias is not sufficient to produce binding without additional dissipation (halo) or a genuinely non-linear localisation mechanism.

2) **Nucleus changes the interaction-zone signature.** With the DC nucleus active, same-spin (07L) shows a noticeably higher `mid_frac_tot` in the interaction window than opposite-spin (07K), indicating the centre bias modifies the interference/deposition structure.

3) **Bookkeeping remains consistent.** Halo remains inactive (`halo_touched_*_max = 0`), driver sources remain A/B only (`src_qA=src_qB=1`, `src_qN=0` by design in DC nucleus mode), and detector outputs remain stable.

---

## Hydrogen: halo capture probe (07M / 07N, 2026-02-13)

07M/07N switch to **single-walker hydrogen mode** (B disabled) and add a **central DC nucleus** plus a **local damping halo** intended to bleed energy and encourage capture. The approach is a near-miss (`b=8.0`) with strong back-reaction enabled (`k=0.20`, `axes=xyz`, `mode=repel`, `vmax=0.250`).

- **07M:** halo strength = 0.010 (soft)
- **07N:** halo strength = 0.030 (strong)

**Important log note (both runs):** the walker exits the box and injection is clamped at step **5013** (`inA=0`, `enable_b=0`). Treat the physics primarily up to (and slightly beyond) the closest-approach window.

### Key measurements

Distances below are measured to the lattice centre \((n/2,n/2,n/2)\), i.e. the nucleus location.

| Metric | 07M (soft halo) | 07N (strong halo) | Notes |
|---|---:|---:|---|
| `b` (impact) | 8.0 | 8.0 | near-miss trajectory |
| closest approach to nucleus (`r_min`) | 10.58 (step 1643) | 9.45 (step 1695) | both pass deep inside the halo radius (72) |
| `phi_max` peak (±25 around `r_min`) | 1.018 | 1.011 | no overlap-style doubling (single walker) |
| `E_tot` peak (±25 around `r_min`) | 857.44 | 752.50 | stronger halo reduces the energy budget |
| `mid_frac_tot` mean (±25 around `r_min`) | 0.0730 | 0.0764 | interaction-slab deposition (nucleus + halo active) |
| out-of-bounds clamp step | 5013 | 5013 | after this, injection is clamped and late-time values are not comparable |

### Conclusions

1) **No capture in the halo-only regime.** Despite passing well inside the halo radius, the walker does not bind; it ultimately exits the box and triggers injection clamping.

2) **Stronger halo damps more energy but still does not bind.** 07N shows a lower near-pass `E_tot` peak than 07M, consistent with increased damping strength, but the qualitative outcome remains escape.

3) **These runs bracket the binding threshold.** The fact that both strengths fail to capture motivates the later hydrogen variants (07O–07Q) where the geometry, nucleus coupling, and halo parameters are pushed into a clearly “capture-prone” regime.

---

## Hydrogen: deep-orbit + nucleus charge sweep (07O / 07P, 2026-02-13)

07O/07P push the hydrogen regime into a deliberately capture-prone configuration:

- **Head-on approach** (`b=0.0`) with slow translational speed (`vx=0.05`).
- **Deep orbit geometry** (`orbit_radius=32`).
- **Aggressive halo** (`halo_r=96`, `halo_strength=0.25`, exp profile).
- **DC nucleus** with a charge sweep:
  - **07O:** `nucleus_q=1.0`
  - **07P:** `nucleus_q=0.5`

These runs are single-walker (B disabled) and are long (`steps=8000`) specifically to see whether the trajectory remains bounded inside the box rather than becoming an eventual escape.

### Key measurements

Distances below are measured to the lattice centre \((n/2,n/2,n/2)\), i.e. the nucleus location.

| Metric | 07O (full q) | 07P (half q) | Notes |
|---|---:|---:|---|
| `b` (impact) | 0.0 | 0.0 | head-on geometry |
| closest approach (`r_min`) | 31.7529 (step 1264) | 31.7529 (step 1264) | essentially the orbit radius; deep pass |
| max radius (`r_max`) | 174.7807 | 174.7808 | remains well inside the box |
| final radius (`r_final`) | 142.0451 | 142.0454 | still in-bounds at end of run (`inA=1`) |
| `E_tot` mean (last 1000) | 778.813 | 778.812 | approaches a steady-state balance (drive vs damping) |
| `phi_max` mean (last 1000) | 1.00272 | 1.00272 | single-walker envelope |
| `mid_frac_tot` mean (last 1000) | 0.08641 | 0.08641 | deposition in the interaction slab |
| `phi_c` mean (last 1000) | 0.01441 | 0.00721 | centre bias scales with nucleus charge |

### Conclusions

1) **Bounded motion (no escape within 8000 steps).** Unlike the halo-soft/halo-strong flybys (07M/07N), these deep-orbit runs stay fully in-bounds for the full duration (`inA=1` throughout). That is the first clean indication of a capture-prone regime.

2) **Charge affects the centre bias, not the gross trajectory (in this configuration).** Halving `nucleus_q` halves the measured centre bias (`phi_c` and `vel_c`), while the large-scale trajectory statistics (`r_min/r_max/r_final`) remain nearly identical. In other words, the nucleus is “visible” in the field probe even when the orbit geometry dominates the kinematics.

3) **Still not a hard bound state in the conservative sense.** The walker does not settle at a fixed separation; it continues to roam at radii \(\sim 140–175\) late in the run. The correct reading is “bounded / captured into an extended orbit-like regime”, not “merged into the nucleus”.


These two runs pass as the charge-sweep baseline feeding into the long-duration verification in 07Q.

## Hydrogen: long-duration verification (07Q, 2026-02-14)

07Q is the long-run verification of the “Hydrogen Goldilocks” configuration (50k steps, `n=512`). The goal is simple: once captured, does the particle remain bounded **without** late-time clamping or boundary escape, and does the system settle into a stable orbit-like regime?

### Summary

Verdict: **bounded, metastable orbit-like regime** (a high-orbit / “Rydberg-like” state).

The particle is captured early, is ejected from the inner halo/core region, and then settles into a large, slowly expanding orbit centred well outside the interaction shells but still safely inside the simulation boundary. Total energy is stable with a slight downward trend, indicating a robust composite that relaxes rather than destabilises.

### Metrics (windowed)

| Metric | Early window (0–2k) | Mid window (~25k) | Late window (48–50k) | Notes |
|---|---:|---:|---:|---|
| `inA` (confinement) | 1 | 1 | 1 | in-bounds for the full 50k steps; no clamps/boundary exits |
| `r` (distance to nucleus) | `r_min=31.75`, mean ~135.4 | mean ~135.4 | mean ~157.2 | positive secular drift (mid→late +21.7) |
| `E_tot` | mean ~6.7 (initial transient) | 783.2 ± 1.7 | 771.6 ± 0.8 | highly stable late; slight dissipation/relaxation |
| interaction region | core-dominated | diffuse / outer | diffuse / outer | shell 1/2 usage drops to near-zero mid/late |
| halo status | active | exited (`r > 96`) | inactive (counter stalled) | confirms orbiting in the quiet zone between halo and sponge |

### Key events

- **Closest approach:** `r_min = 31.75` (early; matches the deep-orbit injection geometry).
- **Halo exit:** particle moves beyond the halo radius (`r = 96`), after which `halo_touched` counters stop increasing.
- **Drift:** orbit remains bounded but slowly expands outward (mean radius increases from ~135 to ~157).

### Interpretation

07Q provides the first clean long-duration evidence that the Goldilocks hydrogen configuration can support a persistent, bounded “orbit” without late-time numerical failure or boundary interactions. The particle is neither pinned to the centre nor scattering away; it loiters in the outer field with low interaction-shell deposition, consistent with a delocalised orbit rather than a tight bound core state.

Operationally: this is the regime to use as the **baseline capture state** before introducing Exp08 non-linear binding mechanisms.

---

## Constraints and anti-fudge notes

- Exp07 uses a linear carrier with explicit forcing. It is not a self-contained particle theory.
- Any claim about “interaction” must first rule out driver artefacts using the injection bookkeeping columns.
- When B is disabled, B-local metrics and separation are NaN by design (fail-obvious).
- Halo touched counts measure **actual damping events**, not mask size.

---

## Next steps

Before interpreting Exp07 results as physics-like behaviour:

1) Rerun the full Exp07 preset suite after any changes to telegraph kernel, injection, or detector geometry.
2) Confirm same/opposite spin differences persist across seeds and minor parameter nudges.
3) Use Exp08 solitons (or another non-linear localisation mechanism) for any claim of persistent particle identity without continuous driving.