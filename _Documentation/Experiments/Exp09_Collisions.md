# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# Non-Linear Collisions (SG sprite collider harness)

## Abstract

Here we introduce the first **non-linear collision harness** in CAELIX: the Sine–Gordon (SG) sprite collider (`--collidersg`).

The Soliton suite established the production pipeline for **reusable SG sprite assets** (breathers / kinks as compact HDF5 patches) and the notion that “stable” is not necessarily “kickable”. Here we consume those assets and test **interaction physics**: scattering, binding, capture, annihilation-like events, and robustness under overlap.

This document focuses on the collider harness itself: how sprites are stamped into a fresh SG grid, how translation is applied, how boundary conditions shape outcomes, and what diagnostics are written so collisions can be reproduced and compared.

---

## Conceptual basis

### What a “collision” means in CAELIX

In this codebase, a collision is a **controlled interaction experiment**:

- Two (or more) compact SG sprites are injected into an initially clean grid.
- Each sprite is assigned a translational velocity and optional phase (`0` or `π`).
- The system evolves under SG dynamics with fixed integrator and boundary conditions.
- We measure whether the encounter produces: pass-through, scattering, capture, breakup, or long-lived bound states.

We treat “collision” as a *diagnostic primitive*: a standardised way to test whether sprites behave like reusable objects, not a promise of textbook soliton integrability.

### Why SG collisions are the next milestone

The SG regime is the first place in CAELIX where we have:

- **Topological structure** (kinks / anti-kinks) and phase polarity.
- **Breather objects** that persist without ongoing forcing.
- A practical route to a sprite library for repeatable experiments.

Collisions are therefore the first meaningful “object-level physics” test: if sprites cannot be launched and made to interact predictably, the sprite library is not yet operational.

---

## Goals

1) **Reliable sprite injection:** stamp an asset into a clean SG grid without immediate blow-up.
2) **Reliable translation:** apply requested translational velocity with correct directionality for both phase 0 and phase π.
3) **Repeatable scenarios:** define a small set of canonical collision scenarios (A, B, …) with stable defaults.
4) **Ledger-first diagnostics:** every run writes enough provenance + per-step metrics to reproduce outcomes from the CSV alone.
5) **Boundary discipline:** make boundary choice an explicit knob (sponge vs clamp etc) and record it.

---

## Collider harness overview

### Primary entry points

- CLI flag: `--collidersg`
- Scenario selector: `--collidersg-scenario <A|B|…>`
- Sprite roster: `--collidersg-sprites <json>`
- Sprite asset: `--collidersg-sprite-asset <path/to/sprite.h5>`

The collider harness is designed to be run from `experiments.py` with canonical output layout under `_Output/09_collisions/…`.

### Sprite roster format

A roster describes *instances* stamped from the same asset:

- `sid` : unique integer instance id
- `pos` : `[x,y,z]` position in grid coordinates
- `vel` : `[vx,vy,vz]` requested translational velocity in grid units / step
- `phase` : `0` or `π` (polarity)

The current harness clones a single sprite asset for all instances. (Future: allow per-sprite asset selection once the library grows.)

### Asset contract

The collider expects a **compact sprite patch** HDF5 produced by Exp08:

- patch fields (`phi`, `vel`, and any required auxiliaries)
- patch half-extent (`L`) and extraction provenance
- minimal physics meta (at least `dt`, `c2`, and SG parameters such as `sg_k` if available)

Disk rule: collider consumes the compact sprite patch; it does not require full-grid state dumps.

---

## Translation (the “kick” problem)

### Why translation is non-trivial for breathers

Breathers are standing waves. Their internal motion can be large even when the object is stable.

A naïve translational kick can be swallowed into internal breathing unless applied at a **kickable phase** where the field has strong spatial gradients (the “tall” phase).

### Injection conditioning (PLL + gain)

When loading a sprite asset, the collider performs a small deterministic conditioning step:

1) **Local phase select (PLL):** locally evolve the sprite patch for a short window and select the frame that maximises `max(|phi|)` (tall phase), tie-breaking on minimal core `|vel|`.
2) **Kick normalisation (gain):** estimate kick traction for the requested direction and apply a bounded gain to better match realised translation to the requested velocity.
3) **Polarity handling:** for `phase=π` instances, the translational kick must be phase-signed so valleys move in the requested direction.

Important constraint: drift-removal fits are *disabled* for SG breathers; fitting “bulk translation” to a standing wave hallucinates drift and sabotages kick leverage.

### Directionality invariant

For each sprite instance:

- Requested `vx>0` must move +x (right), requested `vx<0` must move -x (left).
- This must hold for both phase 0 and phase π.

Any inversion here is treated as a harness bug, not a physics result.

---

## Scenarios (canonical experiments)

### Scenario A — head-on, anti-phase

Two identical instances stamped from the same asset:

- equal and opposite velocities along one axis
- `phase=0` vs `phase=π`

Purpose:

- validate polarity handling
- test whether anti-phase encounters lead to stronger interaction than in-phase

### Scenario B — head-on, in-phase

As A but `phase=0` for both.

Purpose:

- separate polarity effects from pure overlap/energy effects

### Scenario C — grazing pass

Offset impact parameter; velocities still opposed.

Purpose:

- test scattering angle and survival under near-miss

(Additional scenarios will be added once A–C are stable and reproducible.)

---

## Measurements and outputs

### Output layout

Each collision run writes:

- collider CSV (per-step ledger)
- optional quicklook plots (if enabled)
- run header provenance sufficient to replay

### Minimum per-step metrics

Collision runs must record, at minimum:

- per-sprite centroid / peak positions (and velocities inferred)
- per-sprite peak amplitude `max(|phi|)`
- per-sprite core agitation (RMS `|vel|` in a small probe cube)
- inter-sprite separation and minimum separation over time

These are the core “did it collide?” signals.

### Provenance (run header)

The run header must include:

- integrator settings (`dt`, `c2`, `gamma`, `decay`)
- SG parameters (`sg_k` and any others required)
- boundary settings (type, sponge width/strength)
- sprite asset identity (path + extraction provenance)
- roster JSON (positions, requested velocities, phases)
- injection conditioning diagnostics (`pll_info_json`)

This ensures collisions are reproducible without reading code.

---

## Failure modes (known)

- **Sprite overlap at t=0:** if the stamped patches overlap too strongly, `phi_abs_max` can exceed π and the run aborts. This is a setup error (spacing too tight / L too large / too many sprites).
- **Kinetic-phase injection:** selecting a flat/kinetic frame leads to near-zero translational traction (`∇phi≈0`) and kicks collapse.
- **Phase-π sign error:** valleys move in the wrong direction unless the kick term is phase-signed.
- **Gain explosion:** if traction estimate is noisy/near-zero, gain can blow up; gains must be bounded and diagnostics recorded.

---

## Results

### 09A — Head-on anti-phase (v=0.2 request, no contact)

**Claim:** At low requested speed (`|v|=0.2`) the encounter does not reach core-contact. A repulsive / wake-gap forms and the pair separate.

**Evidence (CSV kinematics):**

- Initial separation: **~72.0** voxels.
- Closest approach: **23.44** voxels at **t=12407**.
- End separation: **~30.38** voxels (run end).

With a sprite radius of **R≈8**, core-contact would require separation ≲**2R≈16**. The closest approach remains **~7.4 voxels** short of contact.

**Repulsion / “stall” signature:**

- The approach is monotonic until roughly **t≈10k**, after which the closing rate collapses.
- The minimum occurs at **t≈12407**, followed by renewed separation, consistent with an effective repulsive gap forming between the objects' outer fields.

**Interpretation:**

- At this requested speed the interaction appears dominated by a **non-linear cushion** (vacuum/wake/outer-field pressure) that prevents the cores from touching.
- Measured bulk translation is much smaller than the requested `v=0.2` in this run; collision-class outcomes likely require either higher approach speed or different injection/boundary tuning.

### 09A — Head-on anti-phase (v=0.4 request, permanent merger)

**Claim:** Doubling the requested approach speed (`|v|=0.4`) crosses the repulsive barrier and produces a **stable fused bound state** (“permanent merger”).

**Evidence (tracker + longevity):**

- Fusion event at **t≈7522**.
- Post-merger (`t>8000`), the tracked separation flatlines at **0.0000** (the two peak trackers lock onto the same voxel peak / perfect overlap).
- The merged object persists for the remaining **~22.5k** steps of a **30k** continuation with no breakup.

**Kinematics (post-merger):**

- Drift velocity: **~0.0006** units/step (effectively stationary).
- Settled x-position: **~103.09** (grid centre = 96).

**Dynamics:**

- Peak amplitude traces overlap and show a sustained rhythmic “heartbeat” (oscillation without decay).
- The centre-of-mass trajectory is essentially flat: the bound state is a stopped, high-energy particle.

**Interpretation:**

- The SG system exhibits a **threshold behaviour**: below a certain realised approach speed the outer fields repel and prevent core-contact (09A v=0.2); above it, the encounter can **fuse** into a stationary oscillating entity.
- This is a genuine collider-class outcome: two moving breathers behave like “atoms” that can merge into a heavier, stable oscillating object.

### 09B — Off-axis near-miss (v=0.4 request, hyperbolic scattering)

**Claim:** With an impact parameter (off-axis Y offset), `|v|=0.4` produces **hyperbolic scattering** (a gravity-assist style deflection) rather than merger.

**Setup:**

- Requested speeds: **±0.4** along x.
- Impact parameter via Y lanes:
  - S1: **y=86**
  - S2: **y=106**
  - Δy = **20**

**Evidence (kinematics):**

- Periapsis (closest approach): **18.85** voxels at **t=10716**.
- Separation does not collapse to zero: the encounter is a clean near-miss.

**Deflection (qualitative):**

- Sprite 1 deflects **upward (+Y)** toward the centre line.
- Sprite 2 deflects **downward (-Y)** toward the centre line.

**Interpretation:**

- The interaction is effectively **attractive** (a “Sine–Gordon gravity” analogue): trajectories curve inward during fly-by.
- High approach speed penetrates deeper than the low-speed repulsion barrier case, but the offset introduces sufficient angular momentum to prevent merger.


### 09B — Off-axis capture (v=0.2 request, orbital decay → merger)

**Claim:** At lower speed with a smaller impact parameter, the pair are **captured into a long-lived binary** that undergoes **orbital decay** and finally **merges** into a single breather.

**Lifecycle (run id 154904):**

- **Capture (t≈0 → 4000):** sprites approach at `|v|=0.2` with an offset **Δy=10** and are captured (attractive interaction dominates scattering).
- **Orbit (t≈4000 → 11000):** sustained mutual orbit for ~7k steps.
  - Separation remains roughly **~35–50** voxels.
  - Separation oscillates rapidly with period **≈23 steps** (interpreted as the **breathing period**, not the orbital period).
- **Decay (t≈11000 → 12400):** gradual tightening consistent with **radiative loss / phonon shedding** during repeated passes.
  - Typical tightening sequence: **40 → 30 → 25** voxels.
- **Merger (t≈12476):** crossing the critical distance **≈21** voxels triggers collapse.
  - Separation drops to **0.00** and the system fuses into a single massive breather.

**Interpretation:**

This is a full “binary lifecycle” in SG: capture → long orbit → orbital decay → merger, suggesting a real dissipative channel that can bleed energy and angular momentum from bound pairs until the repulsive barrier fails.


### 09C — Static initialisation (v=0, cold fusion / molecule formation)

**Claim:** A close anti-phase pair initialised at rest can **self-assemble** and spontaneously **merge** into a bound state (“cold fusion”).

**Setup:**

- Positions: **x=86** and **x=106** (Δx = **20**), y=z=96.
- Requested velocities: **0.0** (static initialisation).
- Phases: **0** and **π** (anti-phase).

**Outcome:**

- Initial separation: **20.0**.
- Collision / merger time: **t=7425**.
- Post-merger: distance **→ 0** and remains merged (final distance = 0).

**Interpretation:**

- Confirms the attractive nature of the anti-phase interaction: even with zero initial kinetic energy, the field gradient produces a net pull that collapses the gap (“vacuum collapse”).
- The timescale is slow: ~7400 steps to close an effective **~10-unit** half-gap per sprite, consistent with a gentle long-range attraction rather than an impulsive impact.

(Report source: FG qualitative + kinematic interpretation of run id 161135.)

**Horizon test (run id 164143):**

- Wider initial separation: **x=86** and **x=112** (Δx = **26**).
- Outcome: still **merges**.
- Merger time: **t=7846**.

**Comparison:**

- Baseline (Δx=20) merged at **t=7425**.
- Adding **+6** units of separation delays merger by only **~421** steps.

**Interpretation:**

At Δx≈26 (≈**3.25R** for R≈8), the pair remain deep inside the interaction well: attraction is still strong at >3R and does not appear exponentially weak over this range.


**Horizon test:**

- Much wider initial separation: **x=76** and **x=116** (Δx = **40**).
- Outcome: **no merger**; separation increases.
- Final separation: **~47.0** (started at 40.0).

**Asymmetry / drift:**

- Sprite 1 (phase 0): drifts **~1.0** unit left.
- Sprite 2 (phase π): drifts **~6.0** units right.

**Interpretation:**

- This suggests we have crossed the interaction horizon somewhere between **Δx≈26** (merger) and **Δx≈40** (drift).
- The large phase-dependent mobility is unexpected under the SG `φ→-φ` symmetry and likely reflects discrete-lattice effects (e.g. Peierls–Nabarro pinning) or a small residual asymmetry in the sprite patch that couples differently under sign flip.

**Provisional bound:**

A practical “event horizon” for spontaneous attraction in this setup appears to be **~Δx≈30** (≈**3.75R** for R≈8): inside this range, pairs fuse; outside, attraction collapses and drift dominates.


### 09D — Kink wall collision (Neumann boundaries, k=0.05, v=0.2): pass-through with tracker aliasing

**Claim:** With Neumann boundaries (topological charge preserved) and a softened SG parameter (`sg_k=0.05`), two **+2π kink walls** can be launched at `|v|=0.2` and collide without sticking or annihilating. The apparent “permanent merger” in the tracker output is a **measurement artefact** caused by tracker window aliasing during overlap.

**Observed (what the CSV appears to say):**

- The tracked separation drops to **0** at **t≈2740** and remains at **0** thereafter.
- Both tracked instances drift together to the right at approximately **v≈0.05**.

**Reality check (field invariant):**

- `phi_abs_max` at late time remains **≈4π** (e.g. **~12.52** at t=10k), which is inconsistent with any true fusion into a single +2π wall or annihilation.
- Therefore both kink walls **survive** the encounter; the topological charge does not unwind.

**Cause: tracker teleportation / aliasing during overlap**

At the moment of closest approach, the two gradient peaks enter each other's tracking window:

- With `--collidersg-track-r 14`, once separation falls to ~16, the other wall's peak becomes eligible.
- Numerical noise makes one peak fractionally sharper; the losing tracker **jumps ship** and latches onto the other wall.
- From that point onward, both `sid=1` and `sid=2` are tracking the *same* wall, producing a false “distance=0” flatline.

A representative signature around impact:

- t≈2700: S1 at x≈87, S2 at x≈103
- t≈2740: S1 jumps to x≈102 while S2 remains x≈103 (single-frame teleport)

**Interpretation:**

- The physical interaction is consistent with the SG expectation for identical kinks: **pass-through with a phase delay**, not a sticky merger.
- This is the first clear evidence that kink-class objects are viable collision primitives in CAELIX (unlike breathers, which readily form bound states).

---


### 09E — Binary transistor baseline (wire-confined kink walls, sg_k=0.05): elastic reflection in a waveguide

**Purpose:** establish a *non-explosive* “wire” confinement model and a repeatable baseline interaction for transistor-style geometry experiments.

**Setup:**

- Two **+2π kink walls** (scenario E / kink_wall init).
- Requested velocities: **±0.2** along x.
- SG stiffness: **sg_k=0.05** (soft / “fat tyres”).
- Spatial confinement: a central wire box with **high outside stiffness** (`sg_k_outside=5.0`) and a hard active-domain mask.
- Boundary mode: **Neumann** (topological charge preserved; no sponge / no zero clamp).
- Tracker: **YZ locked to wire centreline** (prevents edge-ripple distraction).

**Outcome:** stable propagation + clean repulsive interaction (no sticking, no annihilation).

- Start positions: **x≈59.20** and **x≈131.20** (initial separation **≈72.00**).
- Turning point (closest approach): **min separation ≈10.80** at **t=1167** (x≈90.10 and x≈100.90).
- Post-turn, both walls reverse direction and separate — consistent with **elastic reflection** of like-charge kinks under confinement.

**Realised speed (from CSV kinematics):**

Although the request was `|v|=0.2`, the realised translation in the waveguide is much smaller:

- Approach window **t=300…900**:
  - S1: **vx≈+0.0283** (R²≈0.999)
  - S2: **vx≈-0.0279** (R²≈0.999)
- After interaction **t=1500…2500**:
  - S1: **vx≈-0.0279** (R²≈0.999)
  - S2: **vx≈+0.0279** (R²≈0.999)

So the system behaves like a stiff “waveguide piston”: motion is highly damped relative to the request, but it is **stable, symmetric, and directionally correct**.

**Field invariant / stability:**

- During the interaction phase (t≲3000), `phi_abs_max` stays near **≈4π** (mean ≈12.60; min ≈12.57; max ≈12.62), consistent with two surviving +2π kinks.
- A later dip toward **≈2π** occurs when the walls approach the grid edges (around t≈4300; x≈2.6 and x≈187.5) and then recover as they reflect under Neumann boundaries.
- Both sprites remain **ALIVE** throughout the 5k-step baseline.

**Interpretation:**

09E is the first successful “hardware-style” confinement test: a spatial `k` grid + domain mask can guide topological walls, preserve charge, and produce a repeatable repulsive interaction suitable as the baseline for **binary transistor geometry** experiments. The next practical task is to improve the realised speed (or accept it as a property of the waveguide) before attempting junction routing.


### 09F — Binary transistor phase test (wire-confined kink walls, sg_k=0.05): overlap → partial unwind

**Purpose:** test phase/polarity handling in the waveguide harness by running the same wire-confined kink-wall setup as 09E but with `phase=π` on sprite 2.

**Setup:**

- Two kink walls in the same central wire box used for 09E (domain mask active; `sg_k_outside` confinement enabled).
- Requested velocities: **±0.4** along x.
- SG stiffness: **sg_k=0.05**.
- Boundary mode: **Neumann**.
- Tracker: **YZ locked to the wire centreline**.
- Phase: S1 **0**, S2 **π**.

**Observed outcome (from CSV):**

- Initial positions: **x≈59.46** and **x≈131.46** (initial separation **≈72.0**).
- The pair reach **full overlap**: minimum separation **0.00** at **t=1137** (both trackers report the same x thereafter).
- Immediately after overlap, the global field invariant collapses from **~4π** to **~2π**:
  - `phi_abs_max` **t=0:** **12.57** (≈4π)
  - `phi_abs_max` **t=1200:** **6.46** (≈2π)
  - Post-event window **t=2000…5000:** mean `phi_abs_max` **≈6.28** (≈2π)
- Both sprites remain **ALIVE** for the full 5k steps.

**Interpretation:**

Unlike 09E (like-charge repulsion / reflection), the phase-π configuration produces an **annihilation-like unwind** at first overlap: the system rapidly transitions from a +4π plateau to a +2π plateau and stays there. This is consistent with the harness successfully exercising a polarity channel (kink vs anti-kink / phase-signed wall) rather than merely suffering tracker aliasing.

A later dip in `phi_abs_max` (minimum **≈4.88** around **t≈4400**) coincides with the remnants approaching the grid edges under Neumann reflection and does not change the qualitative outcome.


**Implication for “binary transistor” geometry work:**

09F demonstrates that **phase/polarity is dynamically meaningful** in the waveguide regime: like-charge walls can reflect cleanly (09E), while the phase-π configuration can unwind into a lower-charge state after overlap. This is the first concrete hint that the wire-confined harness can support **balanced ternary primitives** (+1, 0, -1) once tracking/identity is upgraded.


### 09G — Wire calibration (single kink wall): speed curves and stable operating point

**Purpose:** convert the 09E/09F wire into an *engineerable* transport medium by measuring realised wall speed vs key knobs, using a **single +2π kink wall** (no collisions) and stopping before boundary effects.

**Baseline harness:** N=192, Neumann boundaries, wire mask active, bevel=0, `sg_k_outside=5.0`, single `kink_wall` launched from x≈60 along +x. Default run length: 3600 (shortened for high-v to avoid edge interaction).

**Result 1 — `sg_k_outside` is inert under the mask:** sweeping `sg_k_outside` (0.5 → 5.0) produced **no measurable change** in x(t) or `vx_fit` (all runs `vx_fit≈0.0279`), consistent with the hard domain mask removing the exterior “universe” from the dynamics.

**Result 2 — wire width is inert for a YZ-planar wall:** sweeping wire width (8 → 20 vox, centred at 96) likewise produced **no measurable change** in transport speed (`vx_fit≈0.0279` across all widths). This is expected for the `kink_wall` primitive, which is uniform across the wire cross-section.

**Result 3 — requested velocity controls realised velocity, but with a clamp + release:** with `sg_k=0.05`, realised speeds were strongly suppressed for moderate requests and then rose sharply near the relativistic ceiling.

- The harness enforces `|v|<c` with `c=sqrt(c2)`; for `c2=0.31`, `c≈0.5568`.
- Measured (clean pre-bounce fits):
  - `v_req=0.10 → vx≈0.0050`
  - `v_req=0.20 → vx≈0.0106`
  - `v_req=0.40 → vx≈0.0279`
  - `v_req=0.50 → vx≈0.0518`
  - `v_req=0.53 → vx≈0.0720`
  - `v_req=0.55 → vx≈0.0903` (requires shorter run / earlier start to avoid boundary)
  - Driving at `v≈c` produces large `phi_abs_max` excursions (not suitable for clean logic transport).

**Result 4 — `sg_k` sweep identifies a stable plateau:** with `v_req=0.50`, sweeping `sg_k` upward shows a clean transport plateau around `sg_k≈0.02–0.08` (stable, linear x(t), `phi_abs_max≈2π`). Extremely low `sg_k` (≈0.001–0.005) appears faster but shows poorer linearity / proxy dips.

**Recommended “production wire” settings (for junction work):**

- `sg_k_outside=5.0` (baseline-consistent; largely inert under mask)
- `sg_k=0.08`
- `v_req=0.52` (realised `vx≈0.063`, stable, low ringing; `v_req=0.53` is faster `vx≈0.069` and still clean)


### 09H — First T-junction fan-out (wire-confined kink wall, sg_k=0.08, v=0.52): symmetric splitter + mirror bounce

**Purpose:** demonstrate a first functional *junction primitive* in the wire geometry: one injected +2π kink-class signal entering a T-junction and producing **two symmetric outputs**.

**Setup (run id 20260218_205912):**

- Geometry: `wire_geom=t_junction` with `junction_x=120`, `branch_len=48`, `branch_thick=2`.
- Wire box: `wire_y0=90..wire_y1=102`, `wire_z0=90..wire_z1=102` (centreline **y=95.5**).
- Boundaries: **Neumann**.
- Parameters: `c2=0.31` → `c≈0.5568`, `dt=0.05`, `sg_k=0.08`, `sg_k_outside=5.0`.
- Drive: single injector `sid=1` with `v_req=0.52` (highly relativistic, `v/c≈0.934`, `γ≈2.80`).
- Measurement: add a passive detector `sid=2` with `track_only=true` (tracks but does not stamp any second wall) in the -Y branch to avoid the single-tracker ambiguity.

**Outcome:** resounding success — the junction behaves as a **FAN-OUT** primitive.

**Evidence (CSV, simultaneous symmetry):**

- **Junction entry / split:** at **t≈865**, the injected signal reaches the junction slab (`x≈121`) and transitions from trunk-tracking to branch tracking.
- **Simultaneous branch presence:** from **t≈1023 onward**, both trackers are in opposite branches at the same ticks with identical `peak_abs` values (to the logged precision), confirming true fan-out rather than single-peak reassociation.
- **Two outputs with matched amplitude:** in the post-split window, the tracked peak amplitudes are identical to 6 decimal places in both branches, e.g.
  - `t=1030`: `peak_abs` **6.400164** (sid=1) and **6.400164** (sid=2)
  - `t=1050`: `peak_abs` **6.405404** (sid=1) and **6.405404** (sid=2)
- **Perfect geometric mirror about the centreline:** once both trackers lock (from **t≈1023 onward**), their Y positions satisfy

  `y_pos(t) + y_neg(t) = 191.000  ⇒  (y_pos + y_neg)/2 = 95.5`.

  This is exact centreline symmetry for the wire box (90..101).

- **Simultaneous dead-end reflection:** at **t=1129**,
  - +Y branch reaches **y=148.013** (sid=1)
  - -Y branch reaches **y=42.987** (sid=2)

  giving `(148.013 + 42.987)/2 = 95.5`.

- **Return to junction and interaction:** by **t≈2100**, both signals have returned to the junction region with matched amplitude (`peak_abs` **6.554250** for both), consistent with a symmetric crossroads encounter (repulsive stall / re-entry).

**Interpretation:** a planar +2π wall in this confined SG wire can be driven hard enough to overcome the junction impedance and produce two symmetric, charge-preserving outputs. With Neumann boundaries, the branch ends behave as clean mirrors, enabling repeatable return-to-junction interactions.

**Next:** move from “fan-out exists” to “logic exists”: add exit counters / sink zones (gamma regions) and build a two-input truth table at the junction.
