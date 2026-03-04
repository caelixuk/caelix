# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""oscillator.py — Phase-based gravitational time dilation + lensing diagnostics.

Purpose
-------
Experiment 06A - Phase / frequency estimation.

Two parts live here (intentionally):

1) Phase drift (numeric proof):
   - Build a steady ~1/r potential using the diffusion traffic relaxer (01A style).
   - Run a telegraph simulation with two sinusoidal "local oscillators" (LOs) at
     different radii.
   - Record local time series and estimate phase / frequency via complex demodulation.

2) Lensing (money-shot data):
   - Use the same steady potential as an index field n(r)=1+alpha*phi.
   - Ray-march a bundle of 2D rays through the index gradient to produce a clean
     deflection curve that can be plotted/animated by visualiser.py.

Notes
-----
- This module is intentionally analysis-heavy and kernel-light.
- It uses only NumPy (no SciPy dependency).
- Telegraph stepping comes from `traffic.evolve_telegraph_traffic_steps`.

"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from params import TrafficParams
from traffic import evolve_diffusion_traffic_steps, evolve_telegraph_traffic_steps
from utils import ensure_parent_dir, wallclock_iso, write_csv_provenance_header


@dataclass(frozen=True)
class OscillatorConfig:
    # Gravity / potential
    mass_amp: float = 1.0  # amplitude of point-mass load used to build steady phi
    mass_soften: float = 1.0  # (reserved) softening for index build (not used in diffusion build)

    # Probe placement
    r_near: int = 40
    r_far: int = 160
    axis: str = "x"  # x|y|z  (probes placed on +axis from center)

    # Drive signal
    omega: float = 0.20
    drive_amp: float = 1.0

    # Runtime
    steps: int = 6000
    burn: int = 600
    warm: int = 0

    # Demodulation / analysis
    # Demod window is a simple moving average on the complex baseband.
    demod_window: int = 256

    # Output options
    series_every: int = 1  # record every k ticks


@dataclass(frozen=True)
class LensingConfig:
    # Reuse same steady potential, treat as index n = 1 + alpha*phi
    alpha: float = 0.005

    # Ray bundle
    ray_count: int = 21
    ray_span: float = 80.0  # initial y span (rays launched across this range)

    # Marching
    march_steps: int = 2000
    ds: float = 0.50  # step in lattice units

    # Launch
    x0: float = -200.0
    y0: float = 0.0
    theta0: float = 0.0  # radians, 0 = +x


# ---------------------------
# Utilities
# ---------------------------


def _center(n: int) -> Tuple[int, int, int]:
    c = int(n // 2)
    return c, c, c


def _axis_vec(axis: str) -> Tuple[int, int, int]:
    a = str(axis).strip().lower()
    if a == "x":
        return 1, 0, 0
    if a == "y":
        return 0, 1, 0
    if a == "z":
        return 0, 0, 1
    raise ValueError("axis must be one of: x, y, z")


def _probe_pos(n: int, axis: str, r: int) -> Tuple[int, int, int]:
    if int(r) <= 0:
        raise ValueError("probe radius must be > 0")
    cx, cy, cz = _center(n)
    ax, ay, az = _axis_vec(axis)
    x = int(cx + ax * int(r))
    y = int(cy + ay * int(r))
    z = int(cz + az * int(r))
    if not (0 <= x < n and 0 <= y < n and 0 <= z < n):
        raise ValueError("probe lies out of bounds (adjust r or n)")
    return x, y, z


def _build_point_mass_load(n: int, amp: float) -> np.ndarray:
    load = np.zeros((n, n, n), dtype=np.float32)
    cx, cy, cz = _center(n)
    load[cx, cy, cz] = float(amp)
    return load


def _build_steady_potential(
    n: int,
    load: np.ndarray,
    tp_diffuse: TrafficParams,
    iters: int,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Build a steady potential by diffusing an injected load field.

    Uses `evolve_diffusion_traffic_steps` in chunky steps to keep the shared
    progress bar alive.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")

    phi = np.zeros((n, n, n), dtype=np.float32)
    src = np.ascontiguousarray(load.astype(np.float32))

    chunk = 512
    done = 0
    while done < iters:
        steps = chunk
        if done + steps > iters:
            steps = iters - done
        phi = evolve_diffusion_traffic_steps(phi, src, tp_diffuse, int(steps))
        done += int(steps)
        if progress_cb is not None:
            progress_cb(int(steps))

    return phi


def _complex_demod(series: np.ndarray, omega: float, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (phase, amp) using complex demodulation at frequency omega.

    This is a light-weight alternative to Hilbert:
      z(t) = series(t) * exp(-i*omega*t)
      z_lp = moving_average(z)
      phase = unwrap(angle(z_lp))
      amp = abs(z_lp)
    """
    if series.ndim != 1:
        raise ValueError("series must be 1D")
    if window < 4:
        raise ValueError("demod window must be >= 4")
    n = int(series.shape[0])
    t = np.arange(n, dtype=np.float64)
    z = series.astype(np.float64) * np.exp(-1j * float(omega) * t)

    # Moving average low-pass on the complex baseband.
    w = int(min(int(window), n))
    if w <= 1:
        z_lp = z
    else:
        k = np.ones(w, dtype=np.float64) / float(w)
        z_lp = np.convolve(z, k, mode="same")

    phase = np.unwrap(np.angle(z_lp))
    amp = np.abs(z_lp)
    return phase.astype(np.float64), amp.astype(np.float64)


def _phase_slope(phase: np.ndarray, dt: float = 1.0) -> float:
    """Estimate d(phase)/dt using a robust linear fit."""
    if phase.ndim != 1:
        raise ValueError("phase must be 1D")
    n = int(phase.shape[0])
    if n < 8:
        return float("nan")
    t = np.arange(n, dtype=np.float64) * float(dt)

    # Fit phase = a*t + b
    tt = t - float(np.mean(t))
    pp = phase.astype(np.float64) - float(np.mean(phase))
    denom = float(np.dot(tt, tt))
    if denom <= 0.0:
        return float("nan")
    a = float(np.dot(tt, pp) / denom)
    return float(a)


# ---------------------------
# Phase drift experiment (06A)
# ---------------------------


# Expose public helper to build steady background potential (for reuse in core.py)
def build_steady_phi_bg(
    n: int,
    tp_diffuse: TrafficParams,
    cfg: OscillatorConfig,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """Build and return the steady background potential used by the oscillator suite.

    This is the expensive diffusion relax step (01A-style). Exposing it publicly allows
    callers (core.py) to compute phi_bg once and reuse it for both:
      - phase drift (run_gravity_phase_drift_with_bg)
      - lensing rays (run_lensing_rays)

    No behavioural changes: this mirrors the internal build done by `run_gravity_phase_drift`.
    """
    if int(n) < 16:
        raise ValueError("n too small")
    iters = int(getattr(tp_diffuse, "iters", 0))
    if iters < 1:
        raise ValueError("tp_diffuse.iters must be >= 1")
    load = _build_point_mass_load(n=int(n), amp=float(cfg.mass_amp))
    phi_bg = _build_steady_potential(
        n=int(n),
        load=load,
        tp_diffuse=tp_diffuse,
        iters=iters,
        progress_cb=progress_cb,
    )
    return phi_bg

def _run_gravity_phase_drift_with_bg(
    n: int,
    tp_telegraph: TrafficParams,
    cfg: OscillatorConfig,
    phi_bg: np.ndarray,
    out_csv: str,
    out_series_csv: str = "",
    provenance_header: str = "",
    progress_cb: Optional[Callable[[int], None]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Same as `run_gravity_phase_drift` but reuses a precomputed steady background potential."""
    if phi_bg.shape != (int(n), int(n), int(n)):
        raise ValueError("phi_bg shape mismatch")

    steps = int(cfg.steps)
    burn = int(cfg.burn)
    warm = int(cfg.warm)
    if steps < 8:
        raise ValueError("steps must be >= 8")
    if burn < 0 or burn >= steps:
        raise ValueError("burn must satisfy 0 <= burn < steps")

    omega = float(cfg.omega)
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and > 0")

    # Nyquist safety (same as public runner)
    c2 = float(tp_telegraph.c2)
    dt_tele = float(getattr(tp_telegraph, "dt", 1.0))
    c_max = math.sqrt(c2) * dt_tele
    lam_min = 10.0
    omega_max = (2.0 * math.pi * c_max) / lam_min
    if omega > omega_max:
        raise ValueError(
            f"Drive frequency omega={omega:.6g} exceeds safety maximum omega_max={omega_max:.6g} "
            f"(min wavelength {lam_min} voxels, c_max={c_max:.6g})"
        )

    t0 = time.perf_counter()

    phi = np.array(phi_bg, copy=True)
    vel = np.zeros((n, n, n), dtype=np.float32)
    src = np.zeros((n, n, n), dtype=np.float32)

    pA = _probe_pos(n, cfg.axis, int(cfg.r_near))
    pB = _probe_pos(n, cfg.axis, int(cfg.r_far))
    x_near, y_near, z_near = int(pA[0]), int(pA[1]), int(pA[2])
    x_far, y_far, z_far = int(pB[0]), int(pB[1]), int(pB[2])

    rec_every = int(max(1, int(cfg.series_every)))
    n_rec = int((steps + (rec_every - 1)) // rec_every)
    sA = np.zeros(n_rec, dtype=np.float64)
    sB = np.zeros(n_rec, dtype=np.float64)
    tt = np.zeros(n_rec, dtype=np.int32)

    if warm > 0:
        for t in range(int(warm)):
            src.fill(0.0)
            # Warm-up advances the drive phase; main loop must continue from this phase.
            a = float(cfg.drive_amp) * float(math.sin(float(omega) * float(t)))
            src[pA[0], pA[1], pA[2]] = float(a)
            src[pB[0], pB[1], pB[2]] = float(a)
            phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, tp_telegraph, 1)
            if progress_cb is not None:
                progress_cb(1)
                if (t % 200) == 0:
                    progress_cb(0)

    t_phase0 = int(max(0, int(warm)))

    rec_i = 0
    for t in range(int(steps)):
        src.fill(0.0)
        a = float(cfg.drive_amp) * float(math.sin(float(omega) * float(t + t_phase0)))
        src[pA[0], pA[1], pA[2]] = float(a)
        src[pB[0], pB[1], pB[2]] = float(a)

        phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, tp_telegraph, 1)

        if progress_cb is not None:
            progress_cb(1)
            if (t % 200) == 0:
                progress_cb(0)

        if (t % rec_every) == 0:
            tt[rec_i] = int(t + t_phase0)
            sA[rec_i] = float(phi[pA[0], pA[1], pA[2]] - phi_bg[pA[0], pA[1], pA[2]])
            sB[rec_i] = float(phi[pB[0], pB[1], pB[2]] - phi_bg[pB[0], pB[1], pB[2]])
            rec_i += 1

    tt = tt[:rec_i]
    sA = sA[:rec_i]
    sB = sB[:rec_i]

    burn_idx = int(min(int(burn) // rec_every, len(tt))) if burn > 0 else 0
    sA_w = sA[burn_idx:]
    sB_w = sB[burn_idx:]

    omega_peak_A = _fft_peak_omega(sA_w, dt=float(rec_every), omega_hint=float(omega))
    omega_peak_B = _fft_peak_omega(sB_w, dt=float(rec_every), omega_hint=float(omega))
    finite_peaks = [w for w in [omega_peak_A, omega_peak_B] if math.isfinite(w) and w > 0]
    omega_use = float(np.median(finite_peaks)) if finite_peaks else float(omega)

    phA, ampA = _complex_demod(sA_w, omega=omega_use * float(rec_every), window=int(cfg.demod_window))
    phB, ampB = _complex_demod(sB_w, omega=omega_use * float(rec_every), window=int(cfg.demod_window))

    dphA = _phase_slope_weighted(phA, ampA * ampA, dt=float(rec_every))
    dphB = _phase_slope_weighted(phB, ampB * ampB, dt=float(rec_every))

    # dphA/dphB are residual angular frequency offsets (rad/tick) because dt=rec_every.
    omega_eff_A = float(omega_use + dphA)
    omega_eff_B = float(omega_use + dphB)
    fA = float(omega_eff_A / (2.0 * math.pi))
    fB = float(omega_eff_B / (2.0 * math.pi))

    dph = np.unwrap((phA - phB).astype(np.float64))
    drift = _phase_slope_weighted(dph, (ampA * ampA + ampB * ampB) * 0.5, dt=float(rec_every))

    phiA = float(phi_bg[pA[0], pA[1], pA[2]])
    phiB = float(phi_bg[pB[0], pB[1], pB[2]])

    ampA_med = float(np.median(ampA)) if ampA.size else 0.0
    ampB_med = float(np.median(ampB)) if ampB.size else 0.0

    prov = str(provenance_header).strip()
    if prov == "":
        prov = write_csv_provenance_header(
            producer="CAELIX",
            command="",
            cwd="",
            python_exe="",
            when_iso=str(wallclock_iso()),
            extra={
                "artefact": "oscillator_phase_drift",
                "n": str(int(n)),
                "axis": str(cfg.axis),
                "r_near": str(int(cfg.r_near)),
                "r_far": str(int(cfg.r_far)),
                "probe_x_near": str(int(x_near)),
                "probe_y_near": str(int(y_near)),
                "probe_z_near": str(int(z_near)),
                "probe_x_far": str(int(x_far)),
                "probe_y_far": str(int(y_far)),
                "probe_z_far": str(int(z_far)),
                "omega": str(float(cfg.omega)),
                "drive_amp": str(float(cfg.drive_amp)),
                "steps": str(int(cfg.steps)),
                "burn": str(int(cfg.burn)),
                "warm": str(int(cfg.warm)),
                "demod_window": str(int(cfg.demod_window)),
                "series_every": str(int(rec_every)),
                "tele_c2": str(float(getattr(tp_telegraph, "c2", 0.0))),
                "tele_dt": str(float(getattr(tp_telegraph, "dt", 1.0))),
                "tele_gamma": str(float(getattr(tp_telegraph, "gamma", 0.0))),
                "tele_decay": str(float(getattr(tp_telegraph, "decay", 0.0))),
                "omega_max": str(float(omega_max)),
                "lam_min": str(float(lam_min)),
            },
        )

    ensure_parent_dir(out_csv)
    with open(out_csv, "w", newline="") as f:
        if prov != "":
            f.write(prov)
            if not prov.endswith("\n"):
                f.write("\n")
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n","axis","r_near","r_far",
                "probe_x_near","probe_y_near","probe_z_near",
                "probe_x_far","probe_y_far","probe_z_far",
                "rec_every",
                "tele_c2","tele_dt","tele_gamma","tele_decay",
                "omega_max","lam_min",
                "omega_drive","f_drive","omega_use","f_use",
                "drive_amp","steps","burn","warm","demod_window",
                "phi_near","phi_far",
                "omega_eff_near","omega_eff_far",
                "f_meas_near","f_meas_far",
                "delta_f",
                "frac_delta_f",
                "frac_delta_f_drive",
                "frac_delta_f_use",
                "phase_drift_rad_per_tick",
                "amp_med_near","amp_med_far",
                "wall_s",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "n": int(n),
                "axis": str(cfg.axis),
                "r_near": int(cfg.r_near),
                "r_far": int(cfg.r_far),
                "probe_x_near": int(x_near),
                "probe_y_near": int(y_near),
                "probe_z_near": int(z_near),
                "probe_x_far": int(x_far),
                "probe_y_far": int(y_far),
                "probe_z_far": int(z_far),
                "rec_every": int(rec_every),
                "tele_c2": float(getattr(tp_telegraph, "c2", 0.0)),
                "tele_dt": float(getattr(tp_telegraph, "dt", 1.0)),
                "tele_gamma": float(getattr(tp_telegraph, "gamma", 0.0)),
                "tele_decay": float(getattr(tp_telegraph, "decay", 0.0)),
                "omega_max": float(omega_max),
                "lam_min": float(lam_min),
                "omega_drive": float(omega),
                "f_drive": float(omega / (2.0 * math.pi)),
                "omega_use": float(omega_use),
                "f_use": float(omega_use / (2.0 * math.pi)),
                "drive_amp": float(cfg.drive_amp),
                "steps": int(steps),
                "burn": int(burn),
                "warm": int(warm),
                "demod_window": int(cfg.demod_window),
                "phi_near": float(phiA),
                "phi_far": float(phiB),
                "omega_eff_near": float(omega_eff_A),
                "omega_eff_far": float(omega_eff_B),
                "f_meas_near": float(fA),
                "f_meas_far": float(fB),
                "delta_f": float(fA - fB),
                "frac_delta_f": float((fA - fB) / (float(omega) / (2.0 * math.pi)) if omega != 0.0 else float("nan")),
                "frac_delta_f_drive": float((fA - fB) / (float(omega) / (2.0 * math.pi)) if omega != 0.0 else float("nan")),
                "frac_delta_f_use": float((fA - fB) / (float(omega_use) / (2.0 * math.pi)) if omega_use != 0.0 else float("nan")),
                "phase_drift_rad_per_tick": float(drift),
                "amp_med_near": float(ampA_med),
                "amp_med_far": float(ampB_med),
                "wall_s": float(time.perf_counter() - t0),
            }
        )

    if out_series_csv:
        ensure_parent_dir(out_series_csv)
        with open(out_series_csv, "w", newline="") as f:
            if prov != "":
                prov0 = str(prov)
                if prov0 != "" and (not prov0.endswith("\n")):
                    prov0 = prov0 + "\n"
                sprov = prov0 + "# artefact=oscillator_phase_series\n"
                f.write(sprov)
            w = csv.DictWriter(f, fieldnames=["t", "phi_near_ac", "phi_far_ac"])
            w.writeheader()
            for i in range(int(tt.shape[0])):
                w.writerow({"t": int(tt[i]), "phi_near_ac": float(sA[i]), "phi_far_ac": float(sB[i])})

    return {
        "phi_near": float(phiA),
        "phi_far": float(phiB),
        "f_meas_near": float(fA),
        "f_meas_far": float(fB),
        "delta_f": float(fA - fB),
        "phase_drift": float(drift),
        "omega_use": float(omega_use),
        "f_use": float(omega_use / (2.0 * math.pi)),
    }


# Public runner for phase drift that reuses a provided background
def run_gravity_phase_drift_with_bg(
    n: int,
    tp_telegraph: TrafficParams,
    cfg: OscillatorConfig,
    phi_bg: np.ndarray,
    out_csv: str,
    out_series_csv: str = "",
    provenance_header: str = "",
    progress_cb: Optional[Callable[[int], None]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Public wrapper for phase drift that reuses a precomputed steady background potential."""
    return _run_gravity_phase_drift_with_bg(
        n=n,
        tp_telegraph=tp_telegraph,
        cfg=cfg,
        phi_bg=phi_bg,
        out_csv=out_csv,
        out_series_csv=out_series_csv,
        provenance_header=provenance_header,
        progress_cb=progress_cb,
        verbose=verbose,
    )

def run_gravity_phase_drift(
    n: int,
    tp_diffuse: TrafficParams,
    tp_telegraph: TrafficParams,
    cfg: OscillatorConfig,
    out_csv: str,
    out_series_csv: str = "",
    provenance_header: str = "",
    progress_cb: Optional[Callable[[int], None]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Run the oscillator drift test and write results to CSV.

    Outputs:
      - `out_csv`: one-row summary (plus metadata columns for plotting).
      - `out_series_csv` (optional): time series for debugging/plotting.

    Returns a dict of key metrics.
    """

    if int(n) < 16:
        raise ValueError("n too small")

    steps = int(cfg.steps)
    burn = int(cfg.burn)
    warm = int(cfg.warm)
    if steps < 8:
        raise ValueError("steps must be >= 8")
    if burn < 0 or burn >= steps:
        raise ValueError("burn must satisfy 0 <= burn < steps")

    omega = float(cfg.omega)
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and > 0")
    # --- Hard safety check for drive frequency (Nyquist / wavelength floor) ---
    c2 = float(tp_telegraph.c2)
    dt_tele = float(getattr(tp_telegraph, "dt", 1.0))
    c_max = math.sqrt(c2) * dt_tele
    lam_min = 10.0  # minimum wavelength in voxels
    omega_max = (2.0 * math.pi * c_max) / lam_min
    if omega > omega_max:
        raise ValueError(
            f"Drive frequency omega={omega:.6g} exceeds safety maximum omega_max={omega_max:.6g} (min wavelength {lam_min} voxels, c_max={c_max:.6g})"
        )

    # --- Build steady potential (gravity field) ---
    t0 = time.perf_counter()

    load = _build_point_mass_load(n=n, amp=float(cfg.mass_amp))

    # Use tp_diffuse.iters as the default steady builder budget.
    # Caller can choose large iters; we still accept an explicit iters override by
    # encoding it into tp_diffuse.iters.
    iters = int(tp_diffuse.iters)
    if iters < 1:
        raise ValueError("tp_diffuse.iters must be >= 1")

    if verbose and progress_cb is None:
        print(
            "[osc] build steady phi: n=%d iters=%d inject=%.4g rise=%.4g fall=%.4g decay=%.4g"
            % (
                int(n),
                int(iters),
                float(tp_diffuse.inject),
                float(getattr(tp_diffuse, "rate_rise", 0.0)),
                float(getattr(tp_diffuse, "rate_fall", 0.0)),
                float(tp_diffuse.decay),
            )
        )

    phi_bg = _build_steady_potential(n=n, load=load, tp_diffuse=tp_diffuse, iters=iters, progress_cb=progress_cb)

    # --- Dynamic telegraph run on top of background ---
    phi = np.array(phi_bg, copy=True)
    vel = np.zeros((n, n, n), dtype=np.float32)
    src = np.zeros((n, n, n), dtype=np.float32)

    cx, cy, cz = _center(n)
    pA = _probe_pos(n, cfg.axis, int(cfg.r_near))
    pB = _probe_pos(n, cfg.axis, int(cfg.r_far))
    x_near, y_near, z_near = int(pA[0]), int(pA[1]), int(pA[2])
    x_far, y_far, z_far = int(pB[0]), int(pB[1]), int(pB[2])

    # Record series as delta from background (clean AC component).
    rec_every = int(max(1, int(cfg.series_every)))
    n_rec = int((steps + (rec_every - 1)) // rec_every)
    sA = np.zeros(n_rec, dtype=np.float64)
    sB = np.zeros(n_rec, dtype=np.float64)
    tt = np.zeros(n_rec, dtype=np.int32)

    # Optional warm-up to let any background ringing settle.
    if warm > 0:
        for t in range(int(warm)):
            src.fill(0.0)
            # Inject drive at both probes, same as main loop, but do NOT record.
            a = float(cfg.drive_amp) * float(math.sin(float(omega) * float(t)))
            src[pA[0], pA[1], pA[2]] = float(a)
            src[pB[0], pB[1], pB[2]] = float(a)
            phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, tp_telegraph, 1)
            if progress_cb is not None:
                progress_cb(1)
                if (t % 200) == 0:
                    progress_cb(0)

    t_phase0 = int(max(0, int(warm)))

    rec_i = 0

    for t in range(int(steps)):
        src.fill(0.0)

        a = float(cfg.drive_amp) * float(math.sin(float(omega) * float(t + t_phase0)))
        src[pA[0], pA[1], pA[2]] = float(a)
        src[pB[0], pB[1], pB[2]] = float(a)

        phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, tp_telegraph, 1)

        if progress_cb is not None:
            progress_cb(1)

        if (t % 200) == 0 and progress_cb is not None:
            # Heartbeat for very slow kernels / initial compilation.
            progress_cb(0)

        if (t % rec_every) == 0:
            tt[rec_i] = int(t + t_phase0)
            sA[rec_i] = float(phi[pA[0], pA[1], pA[2]] - phi_bg[pA[0], pA[1], pA[2]])
            sB[rec_i] = float(phi[pB[0], pB[1], pB[2]] - phi_bg[pB[0], pB[1], pB[2]])
            rec_i += 1

    # Trim series (in case of integer division mismatch)
    tt = tt[:rec_i]
    sA = sA[:rec_i]
    sB = sB[:rec_i]

    # Analysis window (skip burn-in)
    if burn > 0:
        burn_idx = int(min(int(burn) // rec_every, len(tt)))
    else:
        burn_idx = 0
    sA_w = sA[burn_idx:]
    sB_w = sB[burn_idx:]
    t_w = tt[burn_idx:].astype(np.float64)

    # --- Improved frequency/phase estimation ---
    # Estimate measured carrier omega for each series using FFT peak
    omega_peak_A = _fft_peak_omega(sA_w, dt=float(rec_every), omega_hint=float(omega))
    omega_peak_B = _fft_peak_omega(sB_w, dt=float(rec_every), omega_hint=float(omega))
    finite_peaks = [w for w in [omega_peak_A, omega_peak_B] if math.isfinite(w) and w > 0]
    if finite_peaks:
        omega_use = float(np.median(finite_peaks))
    else:
        omega_use = float(omega)
    # Demodulate both series using measured omega
    phA, ampA = _complex_demod(sA_w, omega=omega_use * float(rec_every), window=int(cfg.demod_window))
    phB, ampB = _complex_demod(sB_w, omega=omega_use * float(rec_every), window=int(cfg.demod_window))
    # Weighted phase slope estimation
    dphA = _phase_slope_weighted(phA, ampA * ampA, dt=float(rec_every))
    dphB = _phase_slope_weighted(phB, ampB * ampB, dt=float(rec_every))
    # dphA/dphB are residual angular frequency offsets (rad/tick) because dt=rec_every.
    omega_eff_A = float(omega_use + dphA)
    omega_eff_B = float(omega_use + dphB)
    fA = float(omega_eff_A / (2.0 * math.pi))
    fB = float(omega_eff_B / (2.0 * math.pi))
    # Phase drift between probes (unwrap then slope)
    dph = np.unwrap((phA - phB).astype(np.float64))
    drift = _phase_slope_weighted(dph, (ampA * ampA + ampB * ampB) * 0.5, dt=float(rec_every))

    # Local background potential values
    phiA = float(phi_bg[pA[0], pA[1], pA[2]])
    phiB = float(phi_bg[pB[0], pB[1], pB[2]])

    # Simple SNR proxies
    ampA_med = float(np.median(ampA)) if ampA.size else 0.0
    ampB_med = float(np.median(ampB)) if ampB.size else 0.0

    # Write summary CSV
    prov = str(provenance_header).strip()
    if prov == "":
        prov = write_csv_provenance_header(
            producer="CAELIX",
            command="",
            cwd="",
            python_exe="",
            when_iso=str(wallclock_iso()),
            extra={
                "artefact": "oscillator_phase_drift",
                "n": str(int(n)),
                "axis": str(cfg.axis),
                "r_near": str(int(cfg.r_near)),
                "r_far": str(int(cfg.r_far)),
                "probe_x_near": str(int(x_near)),
                "probe_y_near": str(int(y_near)),
                "probe_z_near": str(int(z_near)),
                "probe_x_far": str(int(x_far)),
                "probe_y_far": str(int(y_far)),
                "probe_z_far": str(int(z_far)),
                "omega": str(float(cfg.omega)),
                "drive_amp": str(float(cfg.drive_amp)),
                "steps": str(int(cfg.steps)),
                "burn": str(int(cfg.burn)),
                "warm": str(int(cfg.warm)),
                "demod_window": str(int(cfg.demod_window)),
                "series_every": str(int(rec_every)),
                "diffuse_iters": str(int(getattr(tp_diffuse, "iters", 0))),
                "diffuse_inject": str(float(getattr(tp_diffuse, "inject", 0.0))),
                "diffuse_rate_rise": str(float(getattr(tp_diffuse, "rate_rise", 0.0))),
                "diffuse_rate_fall": str(float(getattr(tp_diffuse, "rate_fall", 0.0))),
                "diffuse_decay": str(float(getattr(tp_diffuse, "decay", 0.0))),
                "tele_c2": str(float(getattr(tp_telegraph, "c2", 0.0))),
                "tele_dt": str(float(getattr(tp_telegraph, "dt", 1.0))),
                "tele_gamma": str(float(getattr(tp_telegraph, "gamma", 0.0))),
                "tele_decay": str(float(getattr(tp_telegraph, "decay", 0.0))),
                "omega_max": str(float(omega_max)),
                "lam_min": str(float(lam_min)),
            },
        )
    ensure_parent_dir(out_csv)
    with open(out_csv, "w", newline="") as f:
        if prov != "":
            f.write(prov)
            if not prov.endswith("\n"):
                f.write("\n")
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n","axis","r_near","r_far",
                "probe_x_near","probe_y_near","probe_z_near",
                "probe_x_far","probe_y_far","probe_z_far",
                "rec_every",
                "tele_c2","tele_dt","tele_gamma","tele_decay",
                "omega_max","lam_min",
                "omega_drive","f_drive","omega_use","f_use",
                "drive_amp","steps","burn","warm","demod_window",
                "phi_near","phi_far",
                "omega_eff_near","omega_eff_far",
                "f_meas_near","f_meas_far",
                "delta_f",
                "frac_delta_f",
                "frac_delta_f_drive",
                "frac_delta_f_use",
                "phase_drift_rad_per_tick",
                "amp_med_near","amp_med_far",
                "wall_s",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "n": int(n),
                "axis": str(cfg.axis),
                "r_near": int(cfg.r_near),
                "r_far": int(cfg.r_far),
                "probe_x_near": int(x_near),
                "probe_y_near": int(y_near),
                "probe_z_near": int(z_near),
                "probe_x_far": int(x_far),
                "probe_y_far": int(y_far),
                "probe_z_far": int(z_far),
                "rec_every": int(rec_every),
                "tele_c2": float(getattr(tp_telegraph, "c2", 0.0)),
                "tele_dt": float(getattr(tp_telegraph, "dt", 1.0)),
                "tele_gamma": float(getattr(tp_telegraph, "gamma", 0.0)),
                "tele_decay": float(getattr(tp_telegraph, "decay", 0.0)),
                "omega_max": float(omega_max),
                "lam_min": float(lam_min),
                "omega_drive": float(omega),
                "f_drive": float(omega / (2.0 * math.pi)),
                "omega_use": float(omega_use),
                "f_use": float(omega_use / (2.0 * math.pi)),
                "drive_amp": float(cfg.drive_amp),
                "steps": int(steps),
                "burn": int(burn),
                "warm": int(warm),
                "demod_window": int(cfg.demod_window),
                "phi_near": float(phiA),
                "phi_far": float(phiB),
                "omega_eff_near": float(omega_eff_A),
                "omega_eff_far": float(omega_eff_B),
                "f_meas_near": float(fA),
                "f_meas_far": float(fB),
                "delta_f": float(fA - fB),
                "frac_delta_f": float((fA - fB) / (float(omega) / (2.0 * math.pi)) if omega != 0.0 else float("nan")),
                "frac_delta_f_drive": float((fA - fB) / (float(omega) / (2.0 * math.pi)) if omega != 0.0 else float("nan")),
                "frac_delta_f_use": float((fA - fB) / (float(omega_use) / (2.0 * math.pi)) if omega_use != 0.0 else float("nan")),
                "phase_drift_rad_per_tick": float(drift),
                "amp_med_near": float(ampA_med),
                "amp_med_far": float(ampB_med),
                "wall_s": float(time.perf_counter() - t0),
            }
        )

    # Optional series CSV
    if out_series_csv:
        ensure_parent_dir(out_series_csv)
        with open(out_series_csv, "w", newline="") as f:
            if prov != "":
                sprov = prov + "# artefact=oscillator_phase_series\n"
                f.write(sprov)
                if not sprov.endswith("\n"):
                    f.write("\n")
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "t",
                    "phi_near_ac",
                    "phi_far_ac",
                ],
            )
            w.writeheader()
            for i in range(int(tt.shape[0])):
                w.writerow({"t": int(tt[i]), "phi_near_ac": float(sA[i]), "phi_far_ac": float(sB[i])})

    return {
        "phi_near": float(phiA),
        "phi_far": float(phiB),
        "f_meas_near": float(fA),
        "f_meas_far": float(fB),
        "delta_f": float(fA - fB),
        "phase_drift": float(drift),
        "omega_use": float(omega_use),
        "f_use": float(omega_use / (2.0 * math.pi)),
    }

def oscillator_work_units(
    tp_diffuse: TrafficParams,
    cfg: OscillatorConfig,
    do_lensing: bool = False,
    lens_cfg: Optional[LensingConfig] = None,
) -> int:
    """Compute total work units for the shared progress bar.

    - diffusion steady build: tp_diffuse.iters
    - warm-up ticks: cfg.warm
    - telegraph ticks: cfg.steps
    - lensing ray-march (optional): lens_cfg.ray_count * lens_cfg.march_steps
    """
    iters = int(getattr(tp_diffuse, "iters", 0))
    if iters < 0:
        iters = 0
    total = iters + int(max(0, int(cfg.warm))) + int(max(0, int(cfg.steps)))
    if do_lensing and (lens_cfg is not None):
        total += int(max(0, int(lens_cfg.ray_count))) * int(max(0, int(lens_cfg.march_steps)))
    return int(max(1, total))

# ---------------------------
# Lensing diagnostics
# ---------------------------


def _sample_phi_trilinear(phi: np.ndarray, x: float, y: float, z: float, oob: float = 0.0) -> float:
    n = int(phi.shape[0])
    if x < 0.0 or y < 0.0 or z < 0.0 or x > float(n - 1) or y > float(n - 1) or z > float(n - 1):
        return float(oob)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    z0 = int(math.floor(z))
    x1 = min(x0 + 1, n - 1)
    y1 = min(y0 + 1, n - 1)
    z1 = min(z0 + 1, n - 1)
    fx = float(x - float(x0))
    fy = float(y - float(y0))
    fz = float(z - float(z0))

    c000 = float(phi[x0, y0, z0])
    c100 = float(phi[x1, y0, z0])
    c010 = float(phi[x0, y1, z0])
    c110 = float(phi[x1, y1, z0])
    c001 = float(phi[x0, y0, z1])
    c101 = float(phi[x1, y0, z1])
    c011 = float(phi[x0, y1, z1])
    c111 = float(phi[x1, y1, z1])

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return float(c0 * (1.0 - fz) + c1 * fz)


def _grad_phi(phi: np.ndarray, x: float, y: float, z: float, eps: float = 1.0, oob: float = 0.0) -> Tuple[float, float, float]:
    # Finite differences on the trilinear sampler.
    gx = (_sample_phi_trilinear(phi, x + eps, y, z, oob=oob) - _sample_phi_trilinear(phi, x - eps, y, z, oob=oob)) / (2.0 * eps)
    gy = (_sample_phi_trilinear(phi, x, y + eps, z, oob=oob) - _sample_phi_trilinear(phi, x, y - eps, z, oob=oob)) / (2.0 * eps)
    gz = (_sample_phi_trilinear(phi, x, y, z + eps, oob=oob) - _sample_phi_trilinear(phi, x, y, z - eps, oob=oob)) / (2.0 * eps)
    return float(gx), float(gy), float(gz)


def run_lensing_rays(
    n: int,
    phi_bg: np.ndarray,
    cfg: LensingConfig,
    out_csv: str,
    provenance_header: str = "",
    progress_cb: Optional[Callable[[int], None]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Ray-march a 2D bundle through an index field derived from `phi_bg`.

    This does NOT touch the telegraph kernel. It produces a clean deflection dataset
    suitable for a headline plot ("money shot").

    The ray equation used is a simple paraxial approximation:
      d(theta)/ds ≈ -∂_⊥ ln n
    where n = 1 + alpha*phi.

    We march rays in the x-y plane at z=cz.
    """

    if phi_bg.shape != (n, n, n):
        raise ValueError("phi_bg shape mismatch")

    alpha = float(cfg.alpha)
    if not math.isfinite(alpha) or alpha == 0.0:
        raise ValueError("alpha must be finite and non-zero")

    ray_count = int(cfg.ray_count)
    if ray_count < 3:
        raise ValueError("ray_count must be >= 3")

    t0 = time.perf_counter()

    cx, cy, cz = _center(n)

    y_span = float(cfg.ray_span)
    ys = np.linspace(float(cy) - y_span, float(cy) + y_span, int(ray_count), dtype=np.float64)

    # Launch x in lattice coordinates
    x0 = float(cx) + float(cfg.x0)
    theta0 = float(cfg.theta0)

    ds = float(cfg.ds)
    if not math.isfinite(ds) or ds <= 0.0:
        raise ValueError("ds must be finite and > 0")

    # Compute background potential value for OOB
    phi_oob = float(np.median(phi_bg[0, :, :]))
    rows: List[Dict[str, float]] = []

    for ri, y0 in enumerate(ys.tolist()):
        x = float(x0)
        y = float(y0)
        th = float(theta0)

        # Track final deflection at end of march
        th0 = float(theta0)

        for si in range(int(cfg.march_steps)):
            # Index field
            ph = _sample_phi_trilinear(phi_bg, x, y, float(cz), oob=phi_oob)
            n_idx = float(1.0 + alpha * float(ph))
            if n_idx <= 1e-6:
                n_idx = 1e-6

            # Gradient
            gx, gy, _gz = _grad_phi(phi_bg, x, y, float(cz), eps=1.0, oob=phi_oob)

            # d/ds ln n = (alpha * grad(phi)) / n
            dlnn_dx = (alpha * float(gx)) / float(n_idx)
            dlnn_dy = (alpha * float(gy)) / float(n_idx)

            # Perpendicular gradient component (relative to ray direction)
            # dir = (cos th, sin th), perp = (-sin th, cos th)
            perp = (-math.sin(th), math.cos(th))
            dlnn_perp = float(dlnn_dx) * float(perp[0]) + float(dlnn_dy) * float(perp[1])

            # Sign convention: rays bend toward higher refractive index (higher n).
            th = float(th + float(dlnn_perp) * float(ds))
            x = float(x + math.cos(th) * float(ds))
            y = float(y + math.sin(th) * float(ds))

            if progress_cb is not None:
                progress_cb(1)
                if (si % 200) == 0:
                    progress_cb(0)

            # Save sparse trajectory samples (for plotting) every 20 steps
            if (si % 20) == 0:
                rows.append(
                    {
                        "ray_i": float(ri),
                        "step": float(si),
                        "x": float(x - float(cx)),
                        "y": float(y - float(cy)),
                        "theta": float(th),
                        "phi": float(ph),
                        "n": float(n_idx),
                    }
                )

        # Final summary point
        final_phi = float(_sample_phi_trilinear(phi_bg, x, y, float(cz), oob=phi_oob))
        rows.append(
            {
                "ray_i": float(ri),
                "step": float(cfg.march_steps),
                "x": float(x - float(cx)),
                "y": float(y - float(cy)),
                "theta": float(th),
                "phi": final_phi,
                "n": float(1.0 + alpha * final_phi),
            }
        )

    prov = str(provenance_header).strip()
    if prov == "":
        prov = write_csv_provenance_header(
            producer="CAELIX",
            command="",
            cwd="",
            python_exe="",
            when_iso=str(wallclock_iso()),
            extra={
                "artefact": "oscillator_lensing",
                "n": str(int(n)),
                "alpha": str(float(cfg.alpha)),
                "ray_count": str(int(cfg.ray_count)),
                "march_steps": str(int(cfg.march_steps)),
                "ds": str(float(cfg.ds)),
            },
        )
    ensure_parent_dir(out_csv)
    with open(out_csv, "w", newline="") as f:
        if prov != "":
            f.write(prov)
            if not prov.endswith("\n"):
                f.write("\n")
        w = csv.DictWriter(f, fieldnames=["ray_i", "step", "x", "y", "theta", "phi", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return {
        "wall_s": float(time.perf_counter() - t0),
        "ray_count": float(ray_count),
        "alpha": float(alpha),
    }

def run_oscillator_bundle(
    *,
    out_root: str,
    exp_code: str,
    n: int,
    when: str,
    tp_diffuse: TrafficParams,
    tp_telegraph: TrafficParams,
    osc: OscillatorConfig,
    do_lensing: bool = False,
    lens: Optional[LensingConfig] = None,
    provenance_header: str = "",
    progress_cb: Optional[Callable[[int], None]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Run 06A oscillator phase drift and (optionally) lensing under a single timestamped bundle."""
    tag = str(exp_code).strip()
    _bundle_dir, summary_csv, series_csv, lens_csv = resolve_oscillator_paths_full(
        out_root=out_root, tag=tag, n=int(n), when=str(when)
    )

    out_series = series_csv if int(getattr(osc, "series_every", 1)) > 0 else ""

    # Build steady phi once (shared by phase + lensing)
    load = _build_point_mass_load(n=int(n), amp=float(osc.mass_amp))
    iters = int(getattr(tp_diffuse, "iters", 0))
    if iters < 1:
        raise ValueError("tp_diffuse.iters must be >= 1")
    phi_bg = _build_steady_potential(
        n=int(n), load=load, tp_diffuse=tp_diffuse, iters=iters, progress_cb=progress_cb
    )

    metrics = _run_gravity_phase_drift_with_bg(
        n=int(n),
        tp_telegraph=tp_telegraph,
        cfg=osc,
        phi_bg=phi_bg,
        out_csv=summary_csv,
        out_series_csv=out_series,
        provenance_header=provenance_header,
        progress_cb=progress_cb,
        verbose=verbose,
    )

    if do_lensing:
        if lens is None:
            lens = LensingConfig()
        m2 = run_lensing_rays(
            n=int(n),
            phi_bg=phi_bg,
            cfg=lens,
            out_csv=lens_csv,
            provenance_header=provenance_header,
            progress_cb=progress_cb,
            verbose=verbose,
        )
        for k, v in m2.items():
            metrics[f"lensing_{k}"] = float(v)

    return metrics

# ---------------------------
# Path resolver helpers (for core.py wiring)
# ---------------------------


def resolve_oscillator_paths(out_root: str, tag: str, n: int, when: str) -> Tuple[str, str, str]:
    """Return (bundle_dir, summary_csv, series_csv) paths under the shared output convention.

    This module is wired later from `core.py`. For now we mirror the established layout:

      <out_root>/<tag>/YYYYMMDD_HHMMSS/
        _csv/
        _logs/

    The caller owns log naming; we provide the CSV targets.
    """
    when_s = str(when)
    bundle_dir = f"{str(out_root).rstrip('/')}/{str(tag).strip('/')}/{when_s}"
    csv_dir = f"{bundle_dir}/_csv"
    base = f"{str(tag)}_n{int(n)}_{when_s}"
    summary_csv = f"{csv_dir}/{base}.csv"
    series_csv = f"{csv_dir}/{base}_series.csv"
    return bundle_dir, summary_csv, series_csv

def resolve_oscillator_paths_full(out_root: str, tag: str, n: int, when: str) -> Tuple[str, str, str, str]:
    """Return (bundle_dir, summary_csv, series_csv, lens_csv) under the shared output convention."""
    bundle_dir, summary_csv, series_csv = resolve_oscillator_paths(out_root=out_root, tag=tag, n=n, when=when)
    when_s = str(when)
    csv_dir = f"{bundle_dir}/_csv"
    base = f"{str(tag)}_n{int(n)}_{when_s}"
    lens_csv = f"{csv_dir}/{base}_lensing.csv"
    return bundle_dir, summary_csv, series_csv, lens_csv

# ---------------------------
# Analysis helpers (extended)
# ---------------------------

def _fft_peak_omega(series: np.ndarray, dt: float, omega_hint: float, frac_band: float = 0.5) -> float:
    """
    Estimate the carrier omega (rad/s) in a real-valued series using FFT peak search near omega_hint.
    Returns the omega at the largest magnitude within a band centered on omega_hint.
    If not found, returns nan.
    """
    if series.ndim != 1 or len(series) < 8 or not math.isfinite(omega_hint) or omega_hint <= 0.0:
        return float("nan")
    n = int(series.shape[0])
    window = np.hanning(n)
    y = np.asarray(series, dtype=np.float64) * window
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=dt)
    omega_bins = 2.0 * math.pi * freqs
    band_width = abs(frac_band) * float(omega_hint)
    band_lo = max(0.0, float(omega_hint) - band_width)
    band_hi = float(omega_hint) + band_width
    # Find bins within band
    idx_band = np.where((omega_bins >= band_lo) & (omega_bins <= band_hi))[0]
    if idx_band.size < 1:
        return float("nan")
    mag = np.abs(spec[idx_band])
    if not np.any(np.isfinite(mag)):
        return float("nan")
    i_peak = int(np.argmax(mag))
    return float(omega_bins[idx_band[i_peak]])

def _phase_slope_weighted(phase: np.ndarray, weights: np.ndarray, dt: float = 1.0) -> float:
    """
    Weighted least squares slope estimation for phase = a*t + b.
    Weights must be non-negative and same shape as phase.
    Falls back to unweighted fit if invalid.
    """
    if phase.ndim != 1 or weights.ndim != 1 or phase.shape != weights.shape:
        return _phase_slope(phase, dt=dt)
    n = int(phase.shape[0])
    if n < 8:
        return float("nan")
    w = np.maximum(0, np.asarray(weights, dtype=np.float64))
    if not np.any(np.isfinite(w)) or np.sum(w) <= 0.0:
        return _phase_slope(phase, dt=dt)
    t = np.arange(n, dtype=np.float64) * float(dt)
    tt = t - float(np.sum(w * t) / np.sum(w))
    pp = phase.astype(np.float64) - float(np.sum(w * phase) / np.sum(w))
    denom = float(np.sum(w * tt * tt))
    if denom <= 0.0:
        return _phase_slope(phase, dt=dt)
    a = float(np.sum(w * tt * pp) / denom)
    return float(a)


def main() -> None:
    print("[oscillator] This module is intended to be run via core.py wiring (06A).")


if __name__ == "__main__":
    main()