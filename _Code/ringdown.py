# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""ringdown.py

Passive ringdown / resonance sweep (Experiment 06B).

- Single Gaussian injection at t=0 (no re-injection).
- Evolve under telegraph solver for T steps.
- Record late-time total energy and dominant frequency from a probe FFT; E_peak is sampled at a fixed stride.

Caller owns boundary policy (recommended: Dirichlet/zero at edges via traffic boundary).

Notes:
- Source injection uses the telegraph `src` impulse (acceleration-like), not a physical energy deposit.
- To make sigma sweeps meaningful, we L2-normalise the injected packet per sigma to a constant target
  (chosen as the minimum unscaled Σ(src^2) across the sweep so we never amplify; stability-first).
- We measure E_land after a short free-evolution window (src already zero) and use E_final/E_land as
  the primary survival metric.
"""

from __future__ import annotations

from typing import Callable, Optional

import math
import numpy as np

from params import TrafficParams
from traffic import evolve_telegraph_traffic_steps
from utils import ensure_parent_dir, resolve_out_path


def _gaussian_pulse3d(n: int, cx: int, cy: int, cz: int, sigma: float, amp: float, out: np.ndarray) -> None:
    out.fill(0.0)
    if sigma <= 0.0:
        out[cx, cy, cz] = float(amp)
        return

    r = int(max(2, math.ceil(3.0 * float(sigma))))
    x0 = max(0, cx - r)
    x1 = min(n, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(n, cy + r + 1)
    z0 = max(0, cz - r)
    z1 = min(n, cz + r + 1)

    inv2s2 = 1.0 / (2.0 * float(sigma) * float(sigma))

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


def _total_energy(phi: np.ndarray, vel: np.ndarray) -> float:
    return float(np.sum(phi.astype(np.float64) ** 2) + np.sum(vel.astype(np.float64) ** 2))


def _dominant_freq(x: np.ndarray, dt: float) -> tuple[float, float, int]:
    n = int(x.shape[0])
    if n < 8:
        return (float("nan"), float("nan"), int(-1))
    y = x.astype(np.float64) - float(np.mean(x.astype(np.float64)))
    spec = np.fft.rfft(y)
    mag = np.abs(spec)
    if mag.shape[0] <= 2:
        return (float("nan"), float("nan"), int(-1))
    k = int(np.argmax(mag[1:])) + 1
    amp_peak = float(mag[k])
    freq_peak = float(k) / (float(n) * float(dt))
    return (freq_peak, amp_peak, int(k))


def ringdown_work_units(steps: int, n_sigmas: int) -> int:
    return int(steps) * int(n_sigmas)


def run_ringdown_sweep_sigma(
    *,
    n: int,
    steps: int,
    sigmas: list[float],
    pulse_amp: float,
    probe_window: int,
    out_csv: str,
    traffic: TrafficParams,
    provenance_header: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    log_path: str = "",
    log_line: Optional[Callable[[str], None]] = None,
    quiet: bool = False,
    verbose: bool = False,
) -> str:
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
            print(s)

    n = int(n)
    if n < 32:
        raise ValueError("ringdown: n too small")

    steps = int(steps)
    if steps <= 0:
        raise ValueError("ringdown: steps must be > 0")

    if not isinstance(sigmas, list) or len(sigmas) < 1:
        raise ValueError("ringdown: sigmas must be a non-empty list")

    sigmas_f: list[float] = []
    for s in sigmas:
        sf = float(s)
        if not np.isfinite(sf) or sf <= 0.0:
            raise ValueError("ringdown: sigma values must be finite and > 0")
        sigmas_f.append(sf)

    pulse_amp = float(pulse_amp)
    if not np.isfinite(pulse_amp) or pulse_amp <= 0.0:
        raise ValueError("ringdown: pulse_amp must be finite and > 0")

    probe_window = int(probe_window)
    if probe_window < 64:
        probe_window = 64
    if probe_window >= steps:
        probe_window = max(64, steps // 2)

    out_csv = str(out_csv)
    if out_csv.strip() == "":
        raise ValueError("ringdown: out_csv must be a non-empty path")

    # Validate traffic
    if str(getattr(traffic, "mode", "")).strip().lower() != "telegraph":
        raise ValueError("ringdown: traffic.mode must be 'telegraph'")

    # Validate boundary configuration when present (caller owns the policy).
    if hasattr(traffic, "boundary_mode"):
        bm = getattr(traffic, "boundary_mode")
        if bm not in (None, "", "zero"):
            _emit(f"[ringdown] warning: traffic.boundary_mode={bm!r} (recommended: 'zero')")

    if hasattr(traffic, "boundary_zero") and hasattr(traffic, "boundary_mode"):
        try:
            bz = bool(getattr(traffic, "boundary_zero"))
            bm2 = getattr(traffic, "boundary_mode")
            if (bm2 == "zero") and (not bz):
                _emit("[ringdown] warning: traffic.boundary_mode='zero' but boundary_zero=False (legacy flag mismatch)")
        except Exception:
            pass

    cx = n // 2
    cy = n // 2
    cz = n // 2

    probe2_dx = int(min(64, max(1, n // 4)))
    px = int(min(n - 2, max(1, cx + probe2_dx)))
    py = int(cy)
    pz = int(cz)
    arrive_frac = 0.25

    phi = np.zeros((n, n, n), dtype=np.float32)
    vel = np.zeros((n, n, n), dtype=np.float32)
    src = np.zeros((n, n, n), dtype=np.float32)
    probe = np.zeros((steps,), dtype=np.float32)
    probe2 = np.zeros((steps,), dtype=np.float32)

    # Choose a constant injection norm target across the sweep.
    # We use the minimum unscaled Σ(src^2) so normalisation never amplifies the source (stability-first).
    E_src_target = float("nan")
    try:
        norms: list[float] = []
        for s in sigmas_f:
            src.fill(0.0)
            _gaussian_pulse3d(n, cx, cy, cz, float(s), float(pulse_amp), src)
            norms.append(float(np.sum(src.astype(np.float64) ** 2)))
        finite_norms = [v for v in norms if np.isfinite(v) and v > 0.0]
        if len(finite_norms) < 1:
            raise ValueError("ringdown: cannot compute finite source norms")
        E_src_target = float(min(finite_norms))
    except Exception as e:
        raise ValueError(f"ringdown: failed to compute E_src_target: {e}")

    _emit(f"[ringdown] E_src_target={E_src_target:.9e} (min Σ(src^2) across sweep; no amplification)")

    out_csv = resolve_out_path(__file__, out_csv)
    ensure_parent_dir(out_csv)

    _emit(f"[ringdown] out_csv={out_csv}")
    _emit(
        "[ringdown] n=%d steps=%d dt=%.6g c2=%.6g gamma=%.6g decay=%.6g pulse_amp=%.6g sigmas=%d probe_window=%d probe2=(%d,%d,%d) arrive_frac=%.3g" % (
            int(n), int(steps), float(traffic.dt), float(traffic.c2), float(traffic.gamma), float(traffic.decay),
            float(pulse_amp), int(len(sigmas_f)), int(probe_window), int(px), int(py), int(pz), float(arrive_frac)
        )
    )

    with open(out_csv, "w", newline="") as f:
        if provenance_header is not None and str(provenance_header).strip() != "":
            ph = str(provenance_header)
            f.write(ph)
            if not ph.endswith("\n"):
                f.write("\n")

        f.write(",".join([
            "sigma",
            "E_src_target",
            "src_scale",
            "src_peak",
            "land_tick",
            "E0",
            "E_peak",
            "t_E_peak",
            "E_land",
            "E_final",
            "Efinal_over_Eland",
            "Efinal_over_Esrc_target",
            "probe_peak_abs",
            "tail_rms",
            "tail_var",
            "freq_peak",
            "freq_bin",
            "amp_peak",
            "probe2_dx",
            "probe2_peak_abs",
            "tail2_rms",
            "tail2_var",
            "freq2_peak",
            "freq2_bin",
            "amp2_peak",
            "arrive_frac",
            "t_arrive",
        ]) + "\n")

        for i, sigma in enumerate(sigmas_f):
            phi.fill(0.0)
            vel.fill(0.0)
            src.fill(0.0)

            _gaussian_pulse3d(n, cx, cy, cz, float(sigma), float(pulse_amp), src)

            # Normalise the injected packet to constant Σ(src^2)=E_src_target (stability-first: no amplification).
            E_src_unscaled = float(np.sum(src.astype(np.float64) ** 2))
            if not np.isfinite(E_src_unscaled) or E_src_unscaled <= 0.0:
                raise ValueError("ringdown: non-finite/unusable source norm")
            if (not np.isfinite(E_src_target)) or E_src_target <= 0.0:
                raise ValueError("ringdown: non-finite E_src_target")
            src_scale = math.sqrt(float(E_src_target) / float(E_src_unscaled))
            src *= float(src_scale)
            src_peak = float(np.max(np.abs(src.astype(np.float64))))

            # Landed energy baseline: choose a sigma-aware settle time so wide packets aren't under-measured.
            # We still measure a single snapshot (E_land) to keep CSV stable, but land_tick must scale with sigma.
            # Heuristic: >=128 ticks, and ~8*sigma ticks, with a weak dependence on run length.
            land_tick = int(max(128, int(8.0 * float(sigma)), steps // 50))
            if land_tick >= steps:
                land_tick = max(1, steps - 1)

            energy_stride = int(max(1, steps // 512))

            E_land = float("nan")
            probe_peak_abs = 0.0
            probe2_peak_abs = 0.0
            t_arrive = int(-1)
            E0 = float("nan")
            E_peak = float("nan")
            t_E_peak = int(-1)

            for t in range(steps):
                if t > 0:
                    src.fill(0.0)
                phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, traffic, 1)

                E_t = float("nan")
                if t == 0:
                    E_t = float(_total_energy(phi, vel))
                    E0 = float(E_t)
                    E_peak = float(E_t)
                    t_E_peak = int(0)
                elif (t == (land_tick - 1)) or (t == (steps - 1)) or ((t % energy_stride) == 0):
                    E_t = float(_total_energy(phi, vel))
                    if np.isfinite(E_t) and (not np.isfinite(E_peak) or (E_t > float(E_peak))):
                        E_peak = float(E_t)
                        t_E_peak = int(t)

                if t == (land_tick - 1):
                    if not np.isfinite(E_t):
                        E_t = float(_total_energy(phi, vel))
                    E_land = float(E_t)

                v_probe = float(phi[cx, cy, cz])
                probe[t] = v_probe
                if abs(v_probe) > probe_peak_abs:
                    probe_peak_abs = abs(v_probe)
                v_probe2 = float(phi[px, py, pz])
                probe2[t] = v_probe2
                if abs(v_probe2) > probe2_peak_abs:
                    probe2_peak_abs = abs(v_probe2)
                if progress_cb is not None:
                    progress_cb(1)

            E_final = float(_total_energy(phi, vel))

            tail = probe[int(steps - probe_window):int(steps)].copy()
            tail64 = tail.astype(np.float64)
            tail64 = tail64 - float(np.mean(tail64))
            tail_var = float(np.mean(tail64 * tail64))
            tail_rms = float(math.sqrt(tail_var)) if np.isfinite(tail_var) and tail_var >= 0.0 else float("nan")
            freq_peak, amp_peak, freq_bin = _dominant_freq(tail, float(traffic.dt))

            tail2 = probe2[int(steps - probe_window):int(steps)].copy()
            tail2_64 = tail2.astype(np.float64)
            tail2_64 = tail2_64 - float(np.mean(tail2_64))
            tail2_var = float(np.mean(tail2_64 * tail2_64))
            tail2_rms = float(math.sqrt(tail2_var)) if np.isfinite(tail2_var) and tail2_var >= 0.0 else float("nan")
            freq2_peak, amp2_peak, freq2_bin = _dominant_freq(tail2, float(traffic.dt))

            if probe2_peak_abs > 0.0 and np.isfinite(probe2_peak_abs):
                thresh = float(arrive_frac) * float(probe2_peak_abs)
                for tt in range(int(steps)):
                    if abs(float(probe2[tt])) >= thresh:
                        t_arrive = int(tt)
                        break

            Efinal_over_Eland = float(E_final / E_land) if (np.isfinite(E_land) and E_land > 0.0) else float("nan")
            Efinal_over_Esrc_target = float(E_final / float(E_src_target)) if (np.isfinite(E_src_target) and E_src_target > 0.0) else float("nan")

            f.write(
                f"{float(sigma):.9g},"
                f"{float(E_src_target):.9e},"
                f"{float(src_scale):.9e},"
                f"{float(src_peak):.9e},"
                f"{int(land_tick)},"
                f"{float(E0):.9e},"
                f"{float(E_peak):.9e},"
                f"{int(t_E_peak)},"
                f"{float(E_land):.9e},"
                f"{float(E_final):.9e},"
                f"{float(Efinal_over_Eland):.9e},"
                f"{float(Efinal_over_Esrc_target):.9e},"
                f"{float(probe_peak_abs):.9e},"
                f"{float(tail_rms):.9e},"
                f"{float(tail_var):.9e},"
                f"{float(freq_peak):.9e},"
                f"{int(freq_bin)},"
                f"{float(amp_peak):.9e},"
                f"{int(probe2_dx)},"
                f"{float(probe2_peak_abs):.9e},"
                f"{float(tail2_rms):.9e},"
                f"{float(tail2_var):.9e},"
                f"{float(freq2_peak):.9e},"
                f"{int(freq2_bin)},"
                f"{float(amp2_peak):.9e},"
                f"{float(arrive_frac):.9g},"
                f"{int(t_arrive)}\n",
            )

            if bool(verbose):
                _emit("[ringdown] sigma=%g (%d/%d) Efinal/Eland=%.3e src_scale=%.3e freq_peak=%.3e" % (
                    float(sigma), int(i + 1), int(len(sigmas_f)), float(Efinal_over_Eland), float(src_scale), float(freq_peak)
                ))

    _emit(f"[ringdown] wrote {out_csv}")
    return out_csv