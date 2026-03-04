# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""relativity.py

Special Relativity experiments for CAELIX.

Current focus: a practical “light clock” (twin paradox style) implemented on the existing
telegraph solver (traffic.py) and mask system.

What this *is*
- Two “mirrors” A and B separated along y by `mirror_sep` (one-way distance = mirror_sep).
- Slow clock: mirrors drift at constant speed `v_ref` along +x (rounded to voxel coords).
- Fast clock: mirrors drift at constant speed `v` along +x (rounded to voxel coords).
- A pulse is launched from A, travels to B, then A, etc. A “tick” is recorded when an
  arrival at the expected mirror is accepted.

What this *is not*
- Not specular reflection / boundary reflection physics.
- Not a Lorentz-transformed rigid cavity.
- Not a claim of isotropy: diagonal/axis anisotropy exists on the 6-neighbour stencil and
  is handled separately (see isotropy calibration).

- Each mirror has an energy trigger computed over a small patch around the mirror (sum(phi^2 + vel^2)), which is phase-agnostic and robust to voxel rounding.
- To prevent “self hits” and near-field ringing, we apply:
  1) A flight-time gate (`min_leg_ticks`) after each accepted hit before the next hit
     can be accepted in that universe.
  2) An expected-mirror state machine (A↔B): at most one hit is accepted per tick, and
     only at the currently expected mirror.
  3) A short detector refractory window to de-bounce peaks.

Confinement geometry
- A z-slab mask is used to clamp everything outside a thin slab around z=cz (Dirichlet),
  making propagation closer to 2D in the x–y plane and reducing boundary clutter.

Outputs
- CSV time-series with mirror positions, detector samples, accepted hit flags, and
  cumulative hit counts.
- Summary (tick counts, ratios, sanity checks like v < c_eff) should be computed
  and printed by the caller (core.py). This module does not own the global run timer or progress UI.

Notes
- For clean SR-style behaviour you must keep v < c_eff, where c_eff ≈ sqrt(c2) * dt for
  the telegraph update. If v exceeds c_eff, you're in “Mach cone / shock” territory and
  the clock becomes dominated by turbulence-triggered detections rather than bounces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import math

import numpy as np

from params import TrafficParams
from traffic import evolve_telegraph_traffic_steps
from utils import ensure_parent_dir, resolve_out_path


@dataclass(frozen=True)
class RelativityParams:
    n: int = 256
    steps: int = 2000
    c2: float = 0.31
    gamma: float = 0.0
    dt: float = 1.0
    decay: float = 0.0

    mirror_sep: int = 48
    v: float = 0.20
    v_ref: float = 0.0  # small reference drift to avoid v=0 mode-locking

    slab_half_thickness: int = 1
    margin: int = 12

    pulse_amp: float = 50.0
    pulse_sigma: float = 2.5

    detect_threshold: float = 0.05
    refractory: int = 6

    detect_mode: str = "first_cross"  # first_cross | window_peak
    start_threshold: float = 0.01  # fraction of e_ref to start peak-capture (window_peak)
    peak_window: int = 12  # ticks to observe after start_threshold crossing (window_peak)
    accept_threshold: float = -1.0  # fraction of e_ref required for acceptance; <0 disables (window_peak)

    out_csv: str = "_Output/relativity_lightclock.csv"


def _gaussian_pulse3d(
    n: int,
    cx: int,
    cy: int,
    cz: int,
    sigma: float,
    amp: float,
    out: np.ndarray,
    *,
    clear: bool = True,
) -> None:
    """Write a small isotropic Gaussian pulse into `out` (float32), in-place.

    This intentionally touches only a small cube around the center to keep it cheap.

    When `clear` is True, `out` is zeroed before writing. When False, the pulse is
    *added* into the existing contents (used for sub-voxel injection accumulation).
    """
    if bool(clear):
        out.fill(0.0)
    if sigma <= 0:
        out[cx, cy, cz] = float(amp)
        return

    r = int(max(2, math.ceil(3.0 * sigma)))
    x0 = max(0, cx - r)
    x1 = min(n, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(n, cy + r + 1)
    z0 = max(0, cz - r)
    z1 = min(n, cz + r + 1)

    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    for x in range(x0, x1):
        dx = float(x - cx)
        dx2 = dx * dx
        for y in range(y0, y1):
            dy = float(y - cy)
            dxy2 = dx2 + dy * dy
            for z in range(z0, z1):
                dz = float(z - cz)
                d2 = dxy2 + dz * dz
                out[x, y, z] = float(amp) * math.exp(-d2 * inv2s2)


def _build_slab_mask(n: int, cz: int, half_thickness: int) -> np.ndarray:
    """Mask everything outside a z-slab around cz. 1=wall, 0=vacuum."""
    mask = np.ones((n, n, n), dtype=np.int8)
    z0 = max(0, cz - half_thickness)
    z1 = min(n, cz + half_thickness + 1)
    mask[:, :, z0:z1] = 0
    return mask


@dataclass
class _SchmittDetector:
    high: float
    low: float
    refractory: int

    _armed: bool = True
    _cooldown: int = 0

    def step(self, e: float) -> bool:
        """Return True on a rising-edge crossing of `high` while armed.

        - Fires when armed and `e` rises above `high`.
        - Disarms for a short refractory window to de-bounce.
        - Re-arms automatically once cooldown expires (time-based), so local ringing
          that never drops below `low` cannot permanently deadlock the detector.
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            return False

        if not self._armed:
            self._armed = True

        if float(e) >= float(self.high):
            self._armed = False
            self._cooldown = int(self.refractory)
            return True

        return False


# --- Peak-capture helper for window_peak detection mode ---

@dataclass
class _PeakCapture:
    window: int
    start_e: float
    accept_e: float

    active: bool = False
    remaining: int = 0
    t_peak: int = -1
    e_peak: float = 0.0

    def reset(self) -> None:
        self.active = False
        self.remaining = 0
        self.t_peak = -1
        self.e_peak = 0.0

    def step(self, t: int, e: float) -> tuple[bool, int, float]:
        """Advance capture state.

        Returns (accepted, t_peak, e_peak). Acceptance occurs when:
        - we have started a capture (triggered by e >= start_e),
        - we have observed `window` ticks (inclusive of the start tick),
        - and (optionally) the peak exceeds accept_e (when accept_e > 0).

        Note: acceptance is *committed* at the tick the window ends, but reports the
        peak tick for timing.
        """
        ee = float(e)
        if not self.active:
            if ee < float(self.start_e):
                return (False, -1, 0.0)
            self.active = True
            self.remaining = int(max(1, self.window))
            self.t_peak = int(t)
            self.e_peak = ee
        else:
            if ee > float(self.e_peak):
                self.e_peak = ee
                self.t_peak = int(t)

        self.remaining -= 1
        if self.remaining > 0:
            return (False, -1, 0.0)

        tpk = int(self.t_peak)
        epk = float(self.e_peak)
        self.reset()

        if float(self.accept_e) > 0.0 and epk < float(self.accept_e):
            return (False, -1, 0.0)

        return (True, tpk, epk)


def _patch_energy(phi: np.ndarray, vel: np.ndarray, x: int, y: int, z0: int, z1: int, *, rad: int = 1) -> float:
    """Sum local energy in a small (2*rad+1)^2 × (z1-z0) patch.

    Energy is phase-agnostic: E = Σ(phi^2 + vel^2). This avoids missed beats when the
    mirror sample flips sign, drifts off-voxel, or carries a DC/envelope floor.
    """
    n0, n1, _ = phi.shape
    x0 = x - int(rad)
    x1 = x + int(rad) + 1
    y0 = y - int(rad)
    y1 = y + int(rad) + 1
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x1 > n0:
        x1 = n0
    if y1 > n1:
        y1 = n1
    if z0 < 0:
        z0 = 0
    if z1 > phi.shape[2]:
        z1 = phi.shape[2]

    e = 0.0
    for xi in range(int(x0), int(x1)):
        for yi in range(int(y0), int(y1)):
            for zi in range(int(z0), int(z1)):
                pv = float(phi[xi, yi, zi])
                vv = float(vel[xi, yi, zi])
                e += pv * pv + vv * vv
    return float(e)


def run_light_clock(
    p: RelativityParams,
    traffic: Optional[TrafficParams] = None,
    provenance_header: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    *,
    log_path: str = "",
    log_line: Optional[Callable[[str], None]] = None,
    quiet: bool = False,
    verbose: bool = False,
) -> str:
    """Run stationary vs moving light-clock and write a CSV.

    Returns the absolute output CSV path.

    The CSV is a time-series; summary (tick counts, ratios) should be computed/printed
    by the caller (core.py) so the output format remains stable.
    """
    def _emit(msg: str) -> None:
        s = str(msg).rstrip("\n")
        if log_line is not None:
            try:
                log_line(s)
                return
            except Exception:
                pass
        if isinstance(log_path, str) and log_path.strip() != "":
            try:
                lp = str(log_path).strip()
                ensure_parent_dir(lp)
                with open(lp, "a", encoding="utf-8", newline="") as lf:
                    lf.write(s + "\n")
                return
            except Exception:
                pass
        if (not quiet) and bool(verbose):
            # Only print when explicitly requested.
            print(s)

    n = int(p.n)
    if n < 32:
        raise ValueError("relativity: n too small")

    steps = int(p.steps)
    if steps <= 0:
        raise ValueError("relativity: steps must be > 0")

    if p.mirror_sep <= 4:
        raise ValueError("relativity: mirror_sep too small")

    if p.v <= 0.0:
        raise ValueError("relativity: v must be > 0")

    cx = n // 2
    cy = n // 2
    cz = n // 2

    half_sep = int(p.mirror_sep // 2)
    y_a = cy - half_sep
    y_b = cy + half_sep

    if y_a <= p.margin or y_b >= n - p.margin:
        raise ValueError("relativity: mirror_sep/margin out of bounds")

    x0 = p.margin
    x1 = n - p.margin - 1

    # Start both clocks from the same known-good lane (original fast-clock start).
    start_x = int(x0) + 2
    if start_x <= x0 or start_x >= x1:
        raise ValueError("relativity: invalid start_x for bounds")

    x_ref0_f = float(start_x)
    x_m0_f = float(start_x)

    # Ensure moving clock stays inside bounds.
    # Both clocks start from start_x and must remain within [x0, x1].
    max_dx = (x1 - start_x) - 2
    if max_dx <= 0:
        raise ValueError("relativity: n/margin leaves no room for motion")

    max_steps_motion = int(math.floor(max_dx / p.v))
    if steps > max_steps_motion:
        raise ValueError(
            "relativity: steps too large for motion bounds "
            f"(steps={int(steps)} max_steps_motion={int(max_steps_motion)} v={float(p.v):.6g} n={int(n)} margin={int(p.margin)})"
        )

    _emit(
        "[relativity] motion_preflight: start_x=%d max_dx=%d v=%.6g max_steps=%d (n=%d margin=%d)" % (
            int(start_x), int(max_dx), float(p.v), int(max_steps_motion), int(n), int(p.margin)
        )
    )

    # Traffic params (telegraph)
    if traffic is None:
        traffic = TrafficParams(
            mode="telegraph",
            c2=float(p.c2),
            gamma=float(p.gamma),
            dt=float(p.dt),
            inject=1.0,
            decay=float(p.decay),
        )

    # Fields
    phi_s = np.zeros((n, n, n), dtype=np.float32)
    vel_s = np.zeros((n, n, n), dtype=np.float32)

    phi_m = np.zeros((n, n, n), dtype=np.float32)
    vel_m = np.zeros((n, n, n), dtype=np.float32)

    src_s = np.zeros((n, n, n), dtype=np.float32)
    src_m = np.zeros((n, n, n), dtype=np.float32)

    mask = _build_slab_mask(n, cz, int(p.slab_half_thickness))
    z0_slab = max(0, cz - int(p.slab_half_thickness))
    z1_slab = min(n, cz + int(p.slab_half_thickness) + 1)

    # Detectors: energy Schmitt triggers over a local patch.
    # Calibration: interpret `detect_threshold` as a FRACTION of the injected pulse energy
    # captured by the detector patch. This keeps thresholds stable across sigma/slab and
    # avoids “never re-arm” failures when the local energy floor is above a tiny absolute threshold.
    rad = 2  # 5×5 patch in x–y over the slab thickness

    frac_high = float(p.detect_threshold)
    if not (0.0 < frac_high < 1.0):
        raise ValueError("relativity: detect_threshold must be a fraction in (0,1)")

    detect_mode = str(p.detect_mode).strip() if isinstance(p.detect_mode, str) else "first_cross"
    if detect_mode not in ("first_cross", "window_peak"):
        raise ValueError("relativity: detect_mode must be 'first_cross' or 'window_peak'")

    start_frac = float(p.start_threshold)
    if not (0.0 < start_frac < 1.0):
        raise ValueError("relativity: start_threshold must be a fraction in (0,1)")

    peak_window = int(p.peak_window)
    if peak_window < 1 or peak_window > 200:
        raise ValueError("relativity: peak_window must be in [1,200]")

    accept_frac = float(p.accept_threshold)
    if accept_frac >= 1.0:
        raise ValueError("relativity: accept_threshold must be < 1.0 (or <0 to disable)")

    # Reference energy for a 1-tick injected pulse at mirror A (vel=0 initially).
    ref_phi = np.zeros((n, n, n), dtype=np.float32)
    ref_vel = np.zeros((n, n, n), dtype=np.float32)
    _gaussian_pulse3d(n, int(cx), int(y_a), int(cz), p.pulse_sigma, p.pulse_amp, ref_phi)
    e_ref = _patch_energy(ref_phi, ref_vel, cx, y_a, z0_slab, z1_slab, rad=rad)

    e_high = frac_high * float(e_ref)
    e_low = 0.25 * e_high

    e_start = start_frac * float(e_ref)
    e_accept = float(accept_frac) * float(e_ref) if float(accept_frac) > 0.0 else -1.0

    _emit(
        "[relativity] detect_cal: mode=%s rad=%d frac=%.6g e_ref=%.6e e_high=%.6e e_low=%.6e start_frac=%.6g e_start=%.6e peak_window=%d accept_frac=%.6g e_accept=%.6e" % (
            str(detect_mode),
            int(rad),
            float(frac_high),
            float(e_ref),
            float(e_high),
            float(e_low),
            float(start_frac),
            float(e_start),
            int(peak_window),
            float(accept_frac),
            float(e_accept),
        )
    )

    det_sa = _SchmittDetector(e_high, e_low, int(p.refractory))
    det_sb = _SchmittDetector(e_high, e_low, int(p.refractory))
    det_ma = _SchmittDetector(e_high, e_low, int(p.refractory))
    det_mb = _SchmittDetector(e_high, e_low, int(p.refractory))

    cap_s = _PeakCapture(window=peak_window, start_e=float(e_start), accept_e=float(e_accept))
    cap_m = _PeakCapture(window=peak_window, start_e=float(e_start), accept_e=float(e_accept))
    last_peak_s_t: int = -1
    last_peak_s_e: float = float("nan")
    last_peak_m_t: int = -1
    last_peak_m_e: float = float("nan")

    # Flight-time gate and expected-mirror state.
    # Without this, a mirror can immediately "detect" the pulse it just emitted.
    one_way = float(abs(y_b - y_a))
    c_eff = math.sqrt(float(traffic.c2)) * float(traffic.dt)
    if c_eff <= 0.0:
        raise ValueError("relativity: invalid traffic params")

    # Conservative one-way transit gate. Too-small gates admit near-field ringing and break cadence.
    min_leg_ticks = int(max(6, math.ceil(0.95 * one_way / c_eff)))

    expect_s = "B"  # first arrival should be at B
    expect_m = "B"

    # Start gated so we cannot accept a hit before a physically plausible first arrival.
    next_ok_s = int(min_leg_ticks)
    next_ok_m = int(min_leg_ticks)

    # State: start by firing from mirror A toward mirror B in both cases.
    # Reference x is computed per-tick as a float drift (x_s_f).
    x_m = int(start_x)

    # Re-injection schedule: when a hit occurs, fire at that mirror for 1 tick.
    fire_s_a = True
    fire_s_b = False

    fire_m_a = True
    fire_m_b = False

    hits_s_a = 0
    hits_s_b = 0
    hits_m_a = 0
    hits_m_b = 0

    # --- Derived diagnostics (estimated online, written into the CSV) ---
    # We estimate the stationary round-trip period T0 from successive A-hits (A→B→A).
    # We estimate the moving round-trip period T' the same way. For the moving clock we
    # also estimate the actual travelled path-length per cycle by summing leg distances
    # between successive accepted hits (A↔B). This lets us compute an anisotropy-corrected
    # dilation estimate:
    #   gamma_corr = (T'/T0) * (c_diag/c_axis)
    # where c_axis = (2*L)/T0 and c_diag = D_cycle/T'.
    # These quantities are *apparatus diagnostics* (bandwidth + stencil texture), not claims
    # about continuum SR by themselves.
    last_sa_t: Optional[int] = None
    last_ma_t: Optional[int] = None

    # Stationary cycle tracking (A->B->A path length should be 2*mirror_sep).
    s_cycle_t0: Optional[int] = None
    s_cycle_len: float = 0.0
    s_last_hit_xy: Optional[tuple[int, int]] = None

    # Moving cycle tracking (path length is measured from mirror positions at hit times).
    m_cycle_t0: Optional[int] = None
    m_cycle_len: float = 0.0
    m_last_hit_xy: Optional[tuple[int, int]] = None

    # Online estimates (NaN until first full cycle completes).
    t0_est: float = float("nan")
    tp_est: float = float("nan")
    c_axis_est: float = float("nan")
    c_diag_est: float = float("nan")
    ratio_est: float = float("nan")
    gamma_corr_est: float = float("nan")
    gamma_pred_axis: float = float("nan")
    anisotropy_A: float = float("nan")
    ratio_pred: float = float("nan")
    ratio_err: float = float("nan")

    # SR reference gamma using the configured vacuum limit c_eff.
    v_run = float(p.v)
    gamma_sr = 1.0 / math.sqrt(max(1e-12, 1.0 - (v_run / float(c_eff)) ** 2))

    out_csv = resolve_out_path(__file__, p.out_csv)
    ensure_parent_dir(out_csv)

    _emit(f"[relativity] out_csv={out_csv}")
    _emit(
        "[relativity] n=%d steps=%d L=%d v=%.6g mirror_sep=%d slab=%d c_eff=%.6g" % (
            int(n), int(steps), int(p.mirror_sep), float(p.v), int(p.mirror_sep), int(p.slab_half_thickness), float(c_eff)
        )
    )
    _emit(
        "[relativity] pulse: amp=%.6g sigma=%.6g threshold=%.6g refractory=%d detect=%s start_thr=%.6g peak_window=%d accept_thr=%.6g" % (
            float(p.pulse_amp),
            float(p.pulse_sigma),
            float(p.detect_threshold),
            int(p.refractory),
            str(detect_mode),
            float(p.start_threshold),
            int(p.peak_window),
            float(p.accept_threshold),
        )
    )
    _emit("[relativity] ref_drift: v_ref=%.6g x_ref0=%.3f" % (float(p.v_ref), float(x_ref0_f)))
    _emit("[relativity] mov_drift: v=%.6g start_x=%d" % (float(p.v), int(start_x)))

    with open(out_csv, "w", newline="") as f:
        if provenance_header is not None and str(provenance_header).strip() != "":
            ph = str(provenance_header)
            f.write(ph)
            if not ph.endswith("\n"):
                f.write("\n")

        header_cols = [
            "t",
            "x_s",
            "x_s_f",
            "x_m",
            "v_ref",
            "a_sx",
            "a_sy",
            "b_sx",
            "b_sy",
            "a_mx",
            "a_my",
            "b_mx",
            "b_my",
            "phi_sa",
            "phi_sb",
            "phi_ma",
            "phi_mb",
            "e_sa",
            "e_sb",
            "e_ma",
            "e_mb",
            "hit_sa",
            "hit_sb",
            "hit_ma",
            "hit_mb",
            "hits_sa",
            "hits_sb",
            "hits_ma",
            "hits_mb",
            "T0_est",
            "Tp_est",
            "c_axis_est",
            "c_diag_est",
            "ratio_est",
            "gamma_corr_est",
            "gamma_pred_axis",
            "anisotropy_A",
            "ratio_pred",
            "ratio_err",
            "gamma_sr",
            "cap_s",
            "cap_m",
            "t_peak_s",
            "t_peak_m",
            "e_peak_s",
            "e_peak_m",
            "hit_t_s",
            "hit_t_m",
            "accept_t_s",
            "accept_t_m",
        ]
        f.write(",".join(header_cols) + "\n")

        for t in range(steps):
            # Reference mirrors (slow drift to break lattice mode-lock at v_ref=0)
            x_s_f = float(x_ref0_f) + float(p.v_ref) * float(t)
            ax_s = int(round(x_s_f))
            bx_s = ax_s
            ay_s = y_a
            by_s = y_b

            if ax_s <= x0 or ax_s >= x1:
                raise ValueError(
                    "relativity: reference drift left bounds during run "
                    f"(t={int(t)} ax_s={int(ax_s)} x0={int(x0)} x1={int(x1)} cx={int(cx)} v_ref={float(p.v_ref):.6g} steps={int(steps)} n={int(n)} margin={int(p.margin)})"
                )

            # Moving mirrors
            x_m_f = float(x_m0_f) + float(p.v) * float(t)
            ax_m = int(round(x_m_f))
            bx_m = ax_m
            ay_m = y_a
            by_m = y_b

            if ax_m <= x0 or ax_m >= x1:
                raise ValueError(
                    "relativity: motion left bounds during run "
                    f"(t={int(t)} ax_m={int(ax_m)} x0={int(x0)} x1={int(x1)} start_x={int(start_x)} v={float(p.v):.6g} steps={int(steps)} n={int(n)} margin={int(p.margin)})"
                )

            if progress_cb is not None:
                progress_cb(1)

            # Build sources (1 tick pulse when armed)
            if fire_s_a:
                _gaussian_pulse3d(n, int(ax_s), int(ay_s), int(cz), p.pulse_sigma, p.pulse_amp, src_s)
                fire_s_a = False
            elif fire_s_b:
                _gaussian_pulse3d(n, int(bx_s), int(by_s), int(cz), p.pulse_sigma, p.pulse_amp, src_s)
                fire_s_b = False
            else:
                src_s.fill(0.0)

            if fire_m_a:
                _gaussian_pulse3d(n, ax_m, ay_m, cz, p.pulse_sigma, p.pulse_amp, src_m)
                fire_m_a = False
            elif fire_m_b:
                _gaussian_pulse3d(n, bx_m, by_m, cz, p.pulse_sigma, p.pulse_amp, src_m)
                fire_m_b = False
            else:
                src_m.fill(0.0)

            # Evolve both universes
            phi_s, vel_s = evolve_telegraph_traffic_steps(phi_s, vel_s, src_s, traffic, 1, mask=mask)
            phi_m, vel_m = evolve_telegraph_traffic_steps(phi_m, vel_m, src_m, traffic, 1, mask=mask)

            # Point samples (phi) for logging.
            v_sa = float(phi_s[ax_s, ay_s, cz])
            v_sb = float(phi_s[bx_s, by_s, cz])
            v_ma = float(phi_m[ax_m, ay_m, cz])
            v_mb = float(phi_m[bx_m, by_m, cz])

            # Patch energy for detection (phase-agnostic).
            e_sa = _patch_energy(phi_s, vel_s, ax_s, ay_s, z0_slab, z1_slab, rad=rad)
            e_sb = _patch_energy(phi_s, vel_s, bx_s, by_s, z0_slab, z1_slab, rad=rad)
            e_ma = _patch_energy(phi_m, vel_m, ax_m, ay_m, z0_slab, z1_slab, rad=rad)
            e_mb = _patch_energy(phi_m, vel_m, bx_m, by_m, z0_slab, z1_slab, rad=rad)

            hit_sa = False
            hit_sb = False
            hit_ma = False
            hit_mb = False
            hit_t_s = int(t)
            hit_t_m = int(t)
            peak_t_s = -1
            peak_t_m = -1
            peak_e_s = float("nan")
            peak_e_m = float("nan")

            # Only allow detectors to update state when an arrival is physically plausible,
            # and only for the currently expected mirror. This prevents t=0 injection and
            # local ringing from disarming detectors during gated time.
            if t >= next_ok_s:
                if detect_mode == "first_cross":
                    if expect_s == "A":
                        hit_sa = det_sa.step(e_sa)
                    else:
                        hit_sb = det_sb.step(e_sb)
                else:
                    if expect_s == "A":
                        ok, tpk, epk = cap_s.step(int(t), float(e_sa))
                        if ok:
                            hit_sa = True
                            hit_t_s = int(tpk)
                            peak_t_s = int(tpk)
                            peak_e_s = float(epk)
                    else:
                        ok, tpk, epk = cap_s.step(int(t), float(e_sb))
                        if ok:
                            hit_sb = True
                            hit_t_s = int(tpk)
                            peak_t_s = int(tpk)
                            peak_e_s = float(epk)
            else:
                if detect_mode == "window_peak":
                    cap_s.reset()

            if t >= next_ok_m:
                if detect_mode == "first_cross":
                    if expect_m == "A":
                        hit_ma = det_ma.step(e_ma)
                    else:
                        hit_mb = det_mb.step(e_mb)
                else:
                    if expect_m == "A":
                        ok, tpk, epk = cap_m.step(int(t), float(e_ma))
                        if ok:
                            hit_ma = True
                            hit_t_m = int(tpk)
                            peak_t_m = int(tpk)
                            peak_e_m = float(epk)
                    else:
                        ok, tpk, epk = cap_m.step(int(t), float(e_mb))
                        if ok:
                            hit_mb = True
                            hit_t_m = int(tpk)
                            peak_t_m = int(tpk)
                            peak_e_m = float(epk)
            else:
                if detect_mode == "window_peak":
                    cap_m.reset()
            if bool(verbose) and (t % 50 == 0 or hit_sa or hit_sb or hit_ma or hit_mb):
                _emit(
                    "[relativity] t=%d expect_s=%s expect_m=%s e_sa=%.3e e_sb=%.3e e_ma=%.3e e_mb=%.3e next_ok_s=%d next_ok_m=%d" % (
                        int(t), str(expect_s), str(expect_m), float(e_sa), float(e_sb), float(e_ma), float(e_mb), int(next_ok_s), int(next_ok_m)
                    )
                )



            # Expected-mirror state machine: accept at most one hit per universe per tick.
            accept_sa = False
            accept_sb = False
            accept_ma = False
            accept_mb = False

            if expect_s == "A":
                accept_sa = hit_sa
            else:
                accept_sb = hit_sb

            if expect_m == "A":
                accept_ma = hit_ma
            else:
                accept_mb = hit_mb

            # When a mirror is accepted as hit, re-fire from that same mirror next tick,
            # flip expectation, and arm the next flight-time gate.
            if accept_sb:
                hits_s_b += 1
                fire_s_b = True
                fire_s_a = False
                expect_s = "A"
                next_ok_s = t + min_leg_ticks
                if detect_mode == "window_peak":
                    cap_s.reset()
            elif accept_sa:
                hits_s_a += 1
                fire_s_a = True
                fire_s_b = False
                expect_s = "B"
                next_ok_s = t + min_leg_ticks
                if detect_mode == "window_peak":
                    cap_s.reset()

            if accept_mb:
                hits_m_b += 1
                fire_m_b = True
                fire_m_a = False
                expect_m = "A"
                next_ok_m = t + min_leg_ticks
                if detect_mode == "window_peak":
                    cap_m.reset()
            elif accept_ma:
                hits_m_a += 1
                fire_m_a = True
                fire_m_b = False
                expect_m = "B"
                next_ok_m = t + min_leg_ticks
                if detect_mode == "window_peak":
                    cap_m.reset()

            # Use accepted hits for logging.
            hit_sa = accept_sa
            hit_sb = accept_sb
            hit_ma = accept_ma
            hit_mb = accept_mb
            if detect_mode == "window_peak":
                if hit_sa or hit_sb:
                    if peak_t_s >= 0:
                        last_peak_s_t = int(peak_t_s)
                        last_peak_s_e = float(peak_e_s)
                if hit_ma or hit_mb:
                    if peak_t_m >= 0:
                        last_peak_m_t = int(peak_t_m)
                        last_peak_m_e = float(peak_e_m)

            # --- Online derived diagnostics update ---
            # Update stationary and moving cycle path-lengths on each accepted hit.
            # For both universes we accumulate leg distances between successive accepted hits.
            if hit_sa or hit_sb:
                sx = int(ax_s)
                sy = int(ay_s) if hit_sa else int(by_s)
                if hit_sb:
                    sx = int(bx_s)
                    sy = int(by_s)

                if s_last_hit_xy is None:
                    s_last_hit_xy = (sx, sy)
                else:
                    dx = float(sx - s_last_hit_xy[0])
                    dy = float(sy - s_last_hit_xy[1])
                    s_cycle_len += math.hypot(dx, dy)
                    s_last_hit_xy = (sx, sy)

                # Start a new stationary cycle on first accepted A-hit.
                if hit_sa and s_cycle_t0 is None:
                    s_cycle_t0 = int(hit_t_s)
                    s_cycle_len = 0.0
                    s_last_hit_xy = (int(ax_s), int(ay_s))

                # Complete stationary cycle on next accepted A-hit.
                elif hit_sa and s_cycle_t0 is not None:
                    t0 = int(hit_t_s) - int(s_cycle_t0)
                    if t0 > 0:
                        t0_est = float(t0)
                        # For stationary, use the known geometric round-trip length.
                        c_axis_est = (2.0 * float(p.mirror_sep)) / float(t0_est)
                    s_cycle_t0 = int(hit_t_s)
                    s_cycle_len = 0.0
                    s_last_hit_xy = (int(ax_s), int(ay_s))

            if hit_ma or hit_mb:
                mx = int(ax_m)
                my = int(ay_m) if hit_ma else int(by_m)
                if hit_mb:
                    mx = int(bx_m)
                    my = int(by_m)

                if m_last_hit_xy is None:
                    m_last_hit_xy = (mx, my)
                else:
                    dx = float(mx - m_last_hit_xy[0])
                    dy = float(my - m_last_hit_xy[1])
                    m_cycle_len += math.hypot(dx, dy)
                    m_last_hit_xy = (mx, my)

                # Start a new moving cycle on first accepted A-hit.
                if hit_ma and m_cycle_t0 is None:
                    m_cycle_t0 = int(hit_t_m)
                    m_cycle_len = 0.0
                    m_last_hit_xy = (int(ax_m), int(ay_m))

                # Complete moving cycle on next accepted A-hit.
                elif hit_ma and m_cycle_t0 is not None:
                    tp = int(hit_t_m) - int(m_cycle_t0)
                    if tp > 0:
                        tp_est = float(tp)
                        # Measured path-length per cycle (A->B->A) from hit positions.
                        d_cycle = float(m_cycle_len)
                        c_diag_est = d_cycle / float(tp_est) if d_cycle > 0.0 else float("nan")
                    m_cycle_t0 = int(hit_t_m)
                    m_cycle_len = 0.0
                    m_last_hit_xy = (int(ax_m), int(ay_m))

            # Derived ratios once we have both periods.
            if (not math.isnan(t0_est)) and (not math.isnan(tp_est)) and t0_est > 0.0 and tp_est > 0.0:
                ratio_est = float(tp_est) / float(t0_est)
                # SR prediction using the measured axial group speed (apparatus-aware).
                if (not math.isnan(c_axis_est)) and c_axis_est > 0.0 and v_run < float(c_axis_est):
                    gamma_pred_axis = 1.0 / math.sqrt(max(1e-12, 1.0 - (v_run / float(c_axis_est)) ** 2))
                # Anisotropy factor A = c_axis / c_diag (NaN until both are known).
                if (not math.isnan(c_axis_est)) and (not math.isnan(c_diag_est)) and c_axis_est > 0.0 and c_diag_est > 0.0:
                    anisotropy_A = float(c_axis_est) / float(c_diag_est)
                    # Existing corrected gamma estimate (ratio * c_diag/c_axis).
                    gamma_corr_est = float(ratio_est) * (float(c_diag_est) / float(c_axis_est))
                # Predicted observed ratio using SR * anisotropy.
                if (not math.isnan(gamma_pred_axis)) and (not math.isnan(anisotropy_A)):
                    ratio_pred = float(gamma_pred_axis) * float(anisotropy_A)
                    if ratio_pred != 0.0:
                        ratio_err = (float(ratio_est) - float(ratio_pred)) / float(ratio_pred)

            cap_s_on = 1 if (detect_mode == "window_peak" and cap_s.active) else 0
            cap_m_on = 1 if (detect_mode == "window_peak" and cap_m.active) else 0
            t_peak_s = int(last_peak_s_t)
            t_peak_m = int(last_peak_m_t)
            e_peak_s = float(last_peak_s_e)
            e_peak_m = float(last_peak_m_e)
            accept_t_s = int(t) if (hit_sa or hit_sb) else -1
            accept_t_m = int(t) if (hit_ma or hit_mb) else -1
            row_hit_t_s = int(hit_t_s) if (hit_sa or hit_sb) else -1
            row_hit_t_m = int(hit_t_m) if (hit_ma or hit_mb) else -1

            f.write(
                f"{t},"
                f"{ax_s},{x_s_f:.6f},{ax_m},"
                f"{float(p.v_ref):.6g},"
                f"{ax_s},{ay_s},{bx_s},{by_s},"
                f"{ax_m},{ay_m},{bx_m},{by_m},"
                f"{v_sa:.9e},{v_sb:.9e},{v_ma:.9e},{v_mb:.9e},"
                f"{e_sa:.9e},{e_sb:.9e},{e_ma:.9e},{e_mb:.9e},"
                f"{1 if hit_sa else 0},{1 if hit_sb else 0},{1 if hit_ma else 0},{1 if hit_mb else 0},"
                f"{hits_s_a},{hits_s_b},{hits_m_a},{hits_m_b},"
                f"{t0_est:.6f},{tp_est:.6f},{c_axis_est:.9e},{c_diag_est:.9e},{ratio_est:.9e},{gamma_corr_est:.9e},{gamma_pred_axis:.9e},{anisotropy_A:.9e},{ratio_pred:.9e},{ratio_err:.9e},{gamma_sr:.9e},"
                f"{cap_s_on},{cap_m_on},{t_peak_s},{t_peak_m},{e_peak_s:.9e},{e_peak_m:.9e},{row_hit_t_s},{row_hit_t_m},{accept_t_s},{accept_t_m}\n"
            )

    _emit(f"[relativity] wrote {out_csv}")

    return out_csv
