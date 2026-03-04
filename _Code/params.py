# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""params.py — shared dataclasses (stdlib-only contracts)

Role
----
Centralises configuration dataclasses used across the CAELIX flat module set.
These objects are passed through the pipeline and experiments to keep CLI wiring
and defaults consistent.

Import style
------------
All modules live in the same folder and are imported flat:
  `from params import PipelineParams, TrafficParams, ...`

Design constraints
------------------
- stdlib-only: this file must not import numpy or any project modules.
- Frozen dataclasses: treat params as immutable value objects.
- Defaults are part of the behavioural contract: changing them changes runs.

Key dataclasses
---------------
- LatticeParams     : microstate anneal / lattice size
- LoadParams        : microstate -> load weights
- TrafficParams     : diffusion/telegraph solver settings
- RayParams         : ray trace step size / domain parameters
- StabilityParams   : face-lock benchmark settings
- PipelineParams    : top-level wiring (includes sub-params + experiment knobs)

Flat-module layout note
-----------------------
This module is intentionally "boring". Computation lives elsewhere:
  - lattice.py / load.py / traffic.py / radial.py / rays.py
  - pipeline.py composes them; visualiser.py handles plots.
"""

from __future__ import annotations


from dataclasses import dataclass



@dataclass(frozen=True)
class LatticeParams:
    n: int = 64                 # cubic lattice size (n x n x n)
    init_mode: str = "sparse"   # "sparse" (existing) | "multiscale" (fast correlated init)
    steps: int = 200_000         # anneal proposals
    p_seed: float = 0.0008       # probability of seeding an excitation (±1) per site
    j_mismatch: float = 1.0      # mismatch weight in energy
    j_nonzero: float = 0.25      # bias towards vacuum (penalise non-zero occupancy)


@dataclass(frozen=True)
class LoadParams:
    w_abs: float = 1.0          # direct cost for non-zero state
    w_mismatch: float = 0.15    # extra cost for local disagreement


@dataclass(frozen=True)
class StabilityParams:
    trials: int = 256
    ticks: int = 600
    p_noise: float = 0.0020      # probability each shell site is corrupted per tick
    p_center_flip: float = 0.0005  # probability center is zeroed per tick


@dataclass(frozen=True)
class FaceSweepSummary:
    k: int
    best_name: str
    best_p_full: float
    best_mean_survival: float
    worst_name: str
    worst_p_full: float
    worst_mean_survival: float


@dataclass(frozen=True)
class TrafficParams:
    iters: int = 30_000          # time steps

    # Dynamics mode:
    #   - "diffuse"      : first-order diffusion (heat equation surrogate)
    #   - "telegraph"    : damped second-order dynamics (wave/telegraph surrogate)
    #   - "nonlinear"    : telegraph + local potential + phi^4 stiffening (Phase 8 scaffold)
    #   - "sine_gordon"  : telegraph + bounded local potential (Phase 8.3; breather/oscillon)
    mode: str = "diffuse"

    # Diffusion relaxation (used in mode="diffuse").
    # Default 0.12 is a pragmatic midpoint: stable neighbour-mix (well below aggressive/overshoot regimes)
    # while converging in practical wall-time for n≈128–512; rise/fall are kept equal to avoid bias.
    rate_rise: float = 0.12      # mix used when target > phi (field rising)
    rate_fall: float = 0.12      # mix used when target < phi (field falling)

    # Telegraph dynamics (used in mode="telegraph").
    # Update is: v = (1-gamma)*v + c2*lap(phi) + inject*src ; phi += dt*v
    c2: float = 0.25             # wave-speed squared (grid units / tick^2)
    gamma: float = 0.10          # velocity damping in [0,1)
    dt: float = 1.0              # integration step (keep 1.0 unless you know why)

    # Shared source/sink terms.
    inject: float = 1.0          # source injection strength per tick
    decay: float = 0.0           # per-tick decay in [0,1)
    boundary_zero: bool = True   # Dirichlet boundary at 0 (legacy; used when boundary_mode="zero")

    # Boundary handling:
    #   - "open"    : no hard face clamp; sponge may still apply if width/strength > 0
    #   - "zero"    : hard clamp faces to 0 each tick (Dirichlet)
    #   - "neumann" : hard clamp faces to copy inward neighbour (∂phi/∂n = 0)
    #   - "sponge"  : absorbing sponge layer near faces (no hard clamp)
    #
    # Note: sponge_* parameters control the absorbing layer behaviour.
    boundary_mode: str = "zero"  # "open"|"zero"|"neumann"|"sponge"
    sponge_width: int = 0         # cells from each face where damping ramps in (>=0)
    sponge_strength: float = 0.0  # per-tick damping at the face in [0,1]
    sponge_axes: str = "xyz"  # which axes receive sponge damping (e.g. "xy" for side-walls only)

    # Phase 8 (scaffold): non-linear local terms for oscillon/soliton scans.
    # These are properties of the vacuum material model (not particle-specific constants).
    #
    # traffic_k meaning depends on `mode`:
    #   - nonlinear   : linear restoring term,        F_k = -k * phi
    #   - sine_gordon : bounded restoring term,       F_k = -k * sin(phi)
    # traffic_lambda is only used by `mode="nonlinear"`.
    traffic_k: float = 0.0        # vacuum stiffness coefficient (see mode notes above)
    traffic_lambda: float = 0.0   # phi^4 stiffening strength: F_nl = -λ * phi^3 (nonlinear only)


@dataclass(frozen=True)
class RingdownParams:
    # Passive ringdown / resonance sweep (06B)
    steps: int = 1000
    pulse_amp: float = 50.0

    sigma_start: float = 1.2
    sigma_stop: float = 6.0
    sigma_step: float = 0.2

    probe_window: int = 500


# Soliton / oscillon scan (08A)
@dataclass(frozen=True)
class SolitonParams:
    # Soliton / oscillon scan (08A)
    steps: int = 2000
    sigma: float = 4.0
    sigma_start: float = 0.0
    sigma_stop: float = 0.0
    sigma_steps: int = 1
    amp: float = 100.0

    k: float = 0.0  # vacuum stiffness

    lambda_start: float = 0.0
    lambda_stop: float = 0.0
    lambda_steps: int = 20

    # 08B/08C: symmetry-breaking scaffold controls
    init_vev: bool = False       # if True, initialise background to ±v (rather than 0)
    vev_sign: int = +1           # +1 or -1; selects which vacuum branch when init_vev=True

    # 08E (sine-gordon): dedicated sweep controls.
    # Sine-Gordon uses traffic_k as the non-linear restoring strength (acc += -k*sin(phi)).
    # These sweeps are separate from the phi^4 scaffold (k/lambda) above.
    #
    # Defaults chosen to be immediately runnable for 08E-style scans.
    # Keep these conservative (sine-gordon is bounded, but large k makes the field very stiff).
    sg_k_start: float = 0.01
    sg_k_stop: float = 0.10
    sg_k_steps: int = 10

    # Optional amplitude sweep for Sine-Gordon.
    # Default range targets the breather regime (roughly O(1) to a few radians).
    sg_amp_start: float = 1.0
    sg_amp_stop: float = 6.0
    sg_amp_steps: int = 6


@dataclass(frozen=True)
class RadialFit:
    slope: float
    intercept: float
    r2: float


@dataclass(frozen=True)
class RayParams:
    X0: float = 2000.0
    ds: float = 1.0
    r_min: float = 1.0


@dataclass(frozen=True)
class PipelineParams:
    seed: int = 6174
    lattice: LatticeParams = LatticeParams()
    load: LoadParams = LoadParams()
    traffic: TrafficParams = TrafficParams()
    ringdown: RingdownParams = RingdownParams()
    soliton: SolitonParams = SolitonParams()

    # Convert potential -> index
    k_index: float = 1.0

    # Radial diagnostics fit window
    r_fit_min: float = 3.0
    r_fit_max: float = 0.0  # 0 => auto (based on n)

    # Ray tracing / Shapiro regression
    X0: float = 2000.0
    ds: float = 1.0
    eps: float = 0.005  # deep weak-field recommended
    shapiro_b_max: float = 0.0  # 0 => auto (based on lens half-size)
    delta_load: bool = False
    delta_jitter: int = 0
    delta_margin: int = 6

    # LiveView (shared-memory viewer)
    liveview: bool = False

    # Heavy-walker (dynamic source) experiment
    walker: bool = False
    walker_path: str = "linear"  # linear|circle
    walker_mach1: bool = False  # convenience: set tick-iters to ~Mach 1 (telegraph only)
    walker_circle_radius: int = 8
    walker_circle_period: int = 0  # 0 => one revolution over walker_steps
    walker_center_x: int = -1  # -1 => center (n//2)
    walker_center_y: int = -1  # -1 => center (n//2)
    walker_center_z: int = -1  # -1 => center (n//2)
    walker_probe_r: int = 0  # 0 => off; else sample phi/vel at 4 probes on this radius
    walker_steps: int = 64
    walker_tick_iters: int = 200
    walker_hold_steps: int = 0
    walker_hold_tick_iters: int = 200
    walker_dx: int = 1
    walker_dy: int = 0
    walker_dz: int = 0
    walker_r_local: int = 6
    walker_hold_inject: float = 1.0
    dump_walker: str = ""

    # Collider (two-walker) experiment
    collider_spin_b: int = -1  # -1|+1 (chirality for walker B)
    collider_vx: float = 0.15
    collider_orbit_radius: float = 80.0
    collider_orbit_omega: float = 0.12
    collider_steps: int = 2000

    # Collider participant toggles
    # Default keeps legacy behaviour (two walkers). Disable B for single-body “Hydrogen” runs.
    collider_enable_b: bool = True

    # Collider local damping halo (Stage 7; optional)
    # Phenomenological proxy for radiative loss: extra damping to velocity field near center.
    # OFF by default to preserve prior experiments.
    collider_halo: bool = False
    collider_halo_r: int = 60                 # radius in voxels (must be >0 when halo enabled)
    collider_halo_strength: float = 0.02      # per-step damping strength in [0,1]
    collider_halo_profile: str = "quadratic"  # "linear"|"quadratic"|"exp"
    collider_halo_center: str = "nucleus"     # "nucleus"|"collision"

    # Collider back-reaction (Stage 3; optional)
    # When enabled, walkers can respond to the local field gradient along the head-on axis.
    collider_backreact: bool = False
    collider_backreact_k: float = 0.0     # 0 => off even if collider_backreact True
    collider_backreact_mode: str = "repel"  # "repel"|"attract"
    collider_backreact_vmax: float = 0.50  # clamp |vx| to avoid instability

    # Collider detectors / calorimetry (optional)
    collider_detectors: bool = False
    collider_shell_stride: int = 2
    collider_shell1_inner_frac: float = 0.18
    collider_shell1_outer_frac: float = 0.22
    collider_shell2_inner_frac: float = 0.22
    collider_shell2_outer_frac: float = 0.26

    # Collider decay/hold mode (optional)
    # After the snapped collision step t_c_int, stop injecting the walkers and keep evolving.
    collider_hold: bool = False
    collider_hold_grace_steps: int = 50   # steps after t_c_int before shutter closes
    collider_hold_steps: int = 800        # additional steps to run after shutter (0 => off)
    collider_center_ball_r: int = 8       # radius for center energy probe during hold

    # Collider scattering (Stage 4; optional)
    # Impact parameter offsets apply to walker B's helix center.
    collider_impact_b: float = 0.0        # transverse +y offset (voxels)
    collider_impact_bz: float = 0.0       # transverse +z offset (voxels)

    # Back-reaction axes (Stage 4; optional)
    # "x" preserves Stage 3 behaviour; "xyz" enables induced drift in y/z as well.
    collider_backreact_axes: str = "x"   # "x"|"xyz"

    # Octant calorimetry (Stage 4; optional)
    collider_octants: bool = False

    # Third-body nucleus (Stage 6; optional)
    # Stationary central source used as a momentum/energy sink for capture tests.
    # By default this is OFF and does not affect existing collider runs.
    collider_nucleus: bool = False
    collider_nucleus_q: float = 1.0         # strength multiplier (relative to walker injection scale)
    collider_nucleus_mode: str = "dc"      # "dc"|"sin"
    collider_nucleus_omega: float = 0.12    # only used when mode="sin"
    collider_nucleus_phase: float = 0.0     # radians; only used when mode="sin"
