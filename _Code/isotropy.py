# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""isotropy.py — directional wave-speed calibration.

Purpose
- Measure effective group speed along axis vs diagonals for the current telegraph solver.
- This is a calibration step before making any SR/time-dilation claims.

Method
- Inject a single, band-limited pulse (3D Gaussian) at the center for one tick.
- Evolve in telegraph mode with the existing solver.
- Record intensity I(t)=phi^2 at several detector points.
- Define arrival time as the tick of peak intensity at each detector.

Outputs
- Writes a CSV with detector geometry, arrival tick, inferred c_eff, intensity diagnostics, and ratios vs axis.
- Also supports a sigma sweep mode which writes a single CSV containing all sigma points (one row per detector per sigma).

Notes
- This module is intentionally solver-agnostic: it uses `traffic.evolve_telegraph_traffic_steps`.
- No plotting here; keep it headless and scriptable.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

from params import TrafficParams
from traffic import evolve_telegraph_traffic_steps

from utils import ensure_parent_dir

VALID_SNR_THRESHOLD = 10.0


@dataclass(frozen=True)
class _Detector:
    name: str
    dx: int
    dy: int
    dz: int


def _gaussian_pulse(n: int, cx: int, cy: int, cz: int, sigma: float, amp: float) -> np.ndarray:
    """Return a centerd 3D Gaussian pulse as float32."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    x = np.arange(n, dtype=np.float32) - np.float32(cx)
    y = np.arange(n, dtype=np.float32) - np.float32(cy)
    z = np.arange(n, dtype=np.float32) - np.float32(cz)

    xx = x[:, None, None]
    yy = y[None, :, None]
    zz = z[None, None, :]

    r2 = (xx * xx) + (yy * yy) + (zz * zz)
    inv2s2 = np.float32(1.0 / (2.0 * sigma * sigma))

    g = np.exp(-r2 * inv2s2, dtype=np.float32)

    # Normalise so peak equals `amp`.
    # For a Gaussian, peak is at the center and equals 1.0.
    g *= np.float32(amp)

    return g.astype(np.float32, copy=False)

def _to_float(v: object, default: float = 0.0) -> float:
    if isinstance(v, (float, int, np.floating, np.integer)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return float(default)
    return float(default)

def _write_isotropy_summary_footer(f, rows: list[dict[str, object]]) -> None:
    by_sigma: dict[float, list[dict[str, object]]] = {}
    for r in rows:
        s = _to_float(r.get("sigma", 0.0), 0.0)
        by_sigma.setdefault(s, []).append(r)

    for s in sorted(by_sigma.keys()):
        rr = by_sigma[s]
        use: list[dict[str, object]] = []
        for r in rr:
            if bool(r.get("valid_peak", False)) and str(r.get("kind", "")) != "axis":
                use.append(r)

        if not use:
            f.write(f"# summary_sigma={float(s):.12g} count_valid=0\n")
            continue

        dcf: list[float] = []
        dif: list[float] = []
        for r in use:
            dcf.append(_to_float(r.get("delta_c_frac", 0.0), 0.0))
            dif.append(_to_float(r.get("delta_Ir2_frac", 0.0), 0.0))

        if dcf:
            max_abs_dcf = float(np.max(np.abs(np.asarray(dcf, dtype=np.float64))))
            rms_dcf = float(np.sqrt(np.mean(np.square(np.asarray(dcf, dtype=np.float64)))))
        else:
            max_abs_dcf = 0.0
            rms_dcf = 0.0

        if dif:
            max_abs_dif = float(np.max(np.abs(np.asarray(dif, dtype=np.float64))))
            rms_dif = float(np.sqrt(np.mean(np.square(np.asarray(dif, dtype=np.float64)))))
        else:
            max_abs_dif = 0.0
            rms_dif = 0.0

        f.write(
            "# summary_sigma={sigma:.12g} count_valid={n} "
            "max_abs_delta_c_frac={a:.12g} rms_delta_c_frac={r:.12g} "
            "max_abs_delta_Ir2_frac={ai:.12g} rms_delta_Ir2_frac={ri:.12g}\n".format(
                sigma=float(s),
                n=int(len(use)),
                a=float(max_abs_dcf),
                r=float(rms_dcf),
                ai=float(max_abs_dif),
                ri=float(rms_dif),
            )
        )
        
def _fill_gaussian_pulse_inplace(dst: np.ndarray, cx: int, cy: int, cz: int, sigma: float, amp: float) -> None:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if dst.dtype != np.float32 or dst.ndim != 3 or dst.shape[0] != dst.shape[1] or dst.shape[0] != dst.shape[2]:
        raise ValueError("dst must be float32 cube")

    n = int(dst.shape[0])
    x = np.arange(n, dtype=np.float32) - np.float32(cx)
    y = np.arange(n, dtype=np.float32) - np.float32(cy)
    z = np.arange(n, dtype=np.float32) - np.float32(cz)

    x2 = x * x
    y2 = y * y
    z2 = z * z

    inv2s2 = np.float32(1.0 / (2.0 * sigma * sigma))

    # r2 = x2[:,None,None] + y2[None,:,None] + z2[None,None,:]
    dst[:] = (x2[:, None, None] + y2[None, :, None] + z2[None, None, :]).astype(np.float32, copy=False)
    dst *= -inv2s2
    np.exp(dst, out=dst)
    dst *= np.float32(amp)


def _default_detectors(R: int) -> list[_Detector]:
    if R <= 0:
        raise ValueError("R must be > 0")

    d2 = int(round(R / math.sqrt(2.0)))
    d3 = int(round(R / math.sqrt(3.0)))

    dets: list[_Detector] = []

    # Axes (both directions)
    dets.append(_Detector("axis_x", +R, 0, 0))
    dets.append(_Detector("axis_y", 0, +R, 0))
    dets.append(_Detector("axis_z", 0, 0, +R))
    dets.append(_Detector("axis_nx", -R, 0, 0))
    dets.append(_Detector("axis_ny", 0, -R, 0))
    dets.append(_Detector("axis_nz", 0, 0, -R))

    # Face diagonals (xy, xz, yz) with sign variants
    for sx in (-1, 1):
        for sy in (-1, 1):
            dets.append(_Detector(f"diag_xy_{'p' if sx > 0 else 'm'}{'p' if sy > 0 else 'm'}", sx * d2, sy * d2, 0))
    for sx in (-1, 1):
        for sz in (-1, 1):
            dets.append(_Detector(f"diag_xz_{'p' if sx > 0 else 'm'}{'p' if sz > 0 else 'm'}", sx * d2, 0, sz * d2))
    for sy in (-1, 1):
        for sz in (-1, 1):
            dets.append(_Detector(f"diag_yz_{'p' if sy > 0 else 'm'}{'p' if sz > 0 else 'm'}", 0, sy * d2, sz * d2))

    # Body diagonals (xyz) with sign variants
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                dets.append(
                    _Detector(
                        f"diag_xyz_{'p' if sx > 0 else 'm'}{'p' if sy > 0 else 'm'}{'p' if sz > 0 else 'm'}",
                        sx * d3,
                        sy * d3,
                        sz * d3,
                    )
                )

    return dets


def _check_in_bounds(n: int, cx: int, cy: int, cz: int, dets: Iterable[_Detector]) -> None:
    for d in dets:
        x = cx + d.dx
        y = cy + d.dy
        z = cz + d.dz
        if not (0 <= x < n and 0 <= y < n and 0 <= z < n):
            raise ValueError(
                f"detector {d.name} out of bounds: ({x},{y},{z}) for n={n} (center=({cx},{cy},{cz}))"
            )


def _parabolic_peak(tr: list[float], idx: int) -> tuple[float, float]:
    """Return (delta, y_peak) where delta is sub-sample offset in [-1,1].

    Uses a 3-point parabola through (idx-1, idx, idx+1). If idx is at an edge or
    curvature is degenerate, returns (0.0, tr[idx]).
    """

    if idx <= 0 or idx >= (len(tr) - 1):
        y0 = float(tr[idx])
        return 0.0, y0

    y_m1 = float(tr[idx - 1])
    y0 = float(tr[idx])
    y_p1 = float(tr[idx + 1])

    denom = (y_m1 - 2.0 * y0 + y_p1)
    if denom == 0.0:
        return 0.0, y0

    delta = 0.5 * (y_m1 - y_p1) / denom
    if not np.isfinite(delta):
        return 0.0, y0
    if delta < -1.0:
        delta = -1.0
    elif delta > 1.0:
        delta = 1.0

    # Evaluate parabola at vertex (relative to idx)
    y_peak = y0 - 0.25 * (y_m1 - y_p1) * delta
    return float(delta), float(y_peak)


# Helper to classify detector group/kind for research diagnostics.
def _detector_kind(name: str) -> str:
    if name.startswith("axis_"):
        return "axis"
    if name.startswith("diag_xy_"):
        return "diag_xy"
    if name.startswith("diag_xz_"):
        return "diag_xz"
    if name.startswith("diag_yz_"):
        return "diag_yz"
    if name.startswith("diag_xyz_"):
        return "diag_xyz"
    return "other"


def _compute_isotropy_rows(
    *,
    n: int,
    R: int,
    steps: int,
    sigma: float,
    amp: float,
    traffic: TrafficParams,
    progress_cb: Optional[Callable[[int], None]] = None,
    phi_buf: Optional[np.ndarray] = None,
    vel_buf: Optional[np.ndarray] = None,
    src_buf: Optional[np.ndarray] = None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Run a single isotropy point and return (rows, meta).

    meta includes: c_theory, c_axis_mean, Ir2_axis_mean.
    rows do NOT include provenance, and will include sigma in each row.
    """

    if n < 8:
        raise ValueError("n too small")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    cx = n // 2
    cy = n // 2
    cz = n // 2

    dets = _default_detectors(R)
    _check_in_bounds(n, cx, cy, cz, dets)

    # Arrays (optionally reused across sweep points)
    if phi_buf is None:
        phi = np.zeros((n, n, n), dtype=np.float32)
    else:
        phi = phi_buf
        if phi.shape != (n, n, n) or phi.dtype != np.float32:
            raise ValueError("phi_buf must be float32 with shape (n,n,n)")
        phi.fill(np.float32(0.0))

    if vel_buf is None:
        vel = np.zeros((n, n, n), dtype=np.float32)
    else:
        vel = vel_buf
        if vel.shape != (n, n, n) or vel.dtype != np.float32:
            raise ValueError("vel_buf must be float32 with shape (n,n,n)")
        vel.fill(np.float32(0.0))

    if src_buf is None:
        src = _gaussian_pulse(n, cx, cy, cz, sigma=sigma, amp=amp)
    else:
        src = src_buf
        if src.shape != (n, n, n) or src.dtype != np.float32:
            raise ValueError("src_buf must be float32 with shape (n,n,n)")
        # Fill in-place efficiently, no large temp
        _fill_gaussian_pulse_inplace(src, cx, cy, cz, sigma=sigma, amp=amp)

    # Record intensity at detectors for each tick (after injection tick)
    traces: dict[str, list[float]] = {d.name: [] for d in dets}

    # Inject once (tick 0)
    phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, traffic, 1)

    # Then run with zero source
    src.fill(0.0)

    for _t in range(1, steps + 1):
        phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, traffic, 1)
        if progress_cb is not None:
            progress_cb(1)

        for d in dets:
            x = cx + d.dx
            y = cy + d.dy
            z = cz + d.dz
            v = float(phi[x, y, z])
            traces[d.name].append(v * v)

    # Compute arrival ticks by peak intensity (robust against threshold noise)
    arrivals: dict[str, int] = {}
    arrivals_f: dict[str, float] = {}
    peaks: dict[str, float] = {}
    peaks_f: dict[str, float] = {}

    for name, tr in traces.items():
        if not tr:
            arrivals[name] = -1
            arrivals_f[name] = -1.0
            peaks[name] = 0.0
            peaks_f[name] = 0.0
            continue

        arr = int(np.argmax(np.asarray(tr, dtype=np.float64)))
        t_int = arr + 1  # trace starts at tick=1
        delta, y_peak = _parabolic_peak(tr, arr)
        t_float = float(t_int) + float(delta)

        arrivals[name] = int(t_int)
        arrivals_f[name] = float(t_float)
        peaks[name] = float(tr[arr])
        peaks_f[name] = float(y_peak)

    # Baseline noise estimate from the tail of each trace (late-time floor)
    tail_window = 20
    tail_mean: dict[str, float] = {}
    tail_std: dict[str, float] = {}

    for name, tr in traces.items():
        if not tr:
            tail_mean[name] = 0.0
            tail_std[name] = 0.0
            continue
        w = int(min(tail_window, len(tr)))
        if w <= 0:
            tail_mean[name] = 0.0
            tail_std[name] = 0.0
            continue
        arr_tail = np.asarray(tr[-w:], dtype=np.float64)
        tail_mean[name] = float(np.mean(arr_tail))
        tail_std[name] = float(np.std(arr_tail))

    # Axis reference (mean over axes, both directions)
    axis_names = ("axis_x", "axis_y", "axis_z", "axis_nx", "axis_ny", "axis_nz")
    c_axis_list: list[float] = []
    Ir2_axis_list: list[float] = []

    for nm in axis_names:
        t0 = float(arrivals_f.get(nm, -1.0))
        if t0 > 0.0:
            c_axis_list.append(float(R) / float(t0))
            Ir2_axis_list.append(float(peaks_f.get(nm, 0.0)) * float(R * R))

    if not c_axis_list:
        raise ValueError("axis detectors did not register a peak")

    c_axis = float(np.mean(np.asarray(c_axis_list, dtype=np.float64)))
    Ir2_axis = float(np.mean(np.asarray(Ir2_axis_list, dtype=np.float64))) if Ir2_axis_list else 0.0

    rows: list[dict[str, object]] = []

    for d in dets:
        x = cx + d.dx
        y = cy + d.dy
        z = cz + d.dz

        dist = math.sqrt(float(d.dx * d.dx + d.dy * d.dy + d.dz * d.dz))
        dist_over_R = (float(dist) / float(R)) if R > 0 else 0.0
        t_peak = int(arrivals.get(d.name, -1))
        t_peak_f = float(arrivals_f.get(d.name, -1.0))

        if t_peak_f > 0.0:
            c_eff_f = dist / float(t_peak_f)
            ratio_f = c_eff_f / c_axis if c_axis > 0.0 else 0.0
        else:
            c_eff_f = 0.0
            ratio_f = 0.0

        # Additional diagnostics for research use
        if t_peak > 0:
            c_eff_int = dist / float(t_peak)
            ratio_int = c_eff_int / c_axis if c_axis > 0.0 else 0.0
        else:
            c_eff_int = 0.0
            ratio_int = 0.0

        if t_peak_f > 0.0 and steps > 0:
            arrival_frac = float(t_peak_f) / float(steps)
            tail_len = float(steps) - float(t_peak_f)
        else:
            arrival_frac = 0.0
            tail_len = 0.0

        if dist > 0.0:
            ux = float(d.dx) / float(dist)
            uy = float(d.dy) / float(dist)
            uz = float(d.dz) / float(dist)
        else:
            ux = 0.0
            uy = 0.0
            uz = 0.0

        kind = _detector_kind(d.name)

        peak_I = float(peaks.get(d.name, 0.0))
        peak_I_f = float(peaks_f.get(d.name, peak_I))
        peak_phi_f = math.sqrt(max(0.0, peak_I_f))

        Ir2 = peak_I_f * (dist * dist)
        Ir2_ratio = (Ir2 / Ir2_axis) if Ir2_axis > 0.0 else 0.0

        delta_c = float(c_eff_f) - float(c_axis)
        delta_c_frac = float(ratio_f) - 1.0
        delta_Ir2_frac = float(Ir2_ratio) - 1.0

        floor_mu = float(tail_mean.get(d.name, 0.0))
        floor_sd = float(tail_std.get(d.name, 0.0))
        eps = 1e-30
        snr_mu = float(peak_I_f / (floor_mu + eps))
        snr_sd = float(peak_I_f / (floor_sd + eps)) if floor_sd > 0.0 else 0.0

        valid_peak = bool((t_peak_f > 0.0) and (snr_mu >= float(VALID_SNR_THRESHOLD)))

        rows.append(
            {
                "sigma": float(sigma),
                "detector": d.name,
                "dx": int(d.dx),
                "dy": int(d.dy),
                "dz": int(d.dz),
                "x": int(x),
                "y": int(y),
                "z": int(z),
                "dist": float(dist),
                "dist_over_R": float(dist_over_R),
                "kind": str(kind),
                "ux": float(ux),
                "uy": float(uy),
                "uz": float(uz),
                "t_peak": int(t_peak),
                "t_peak_f": float(t_peak_f),
                "valid_peak": bool(valid_peak),
                "arrival_frac": float(arrival_frac),
                "tail_len": float(tail_len),
                "c_eff": float(c_eff_f),
                "c_eff_int": float(c_eff_int),
                "ratio_vs_axis_mean": float(ratio_f),
                "delta_c": float(delta_c),
                "delta_c_frac": float(delta_c_frac),
                "ratio_int_vs_axis_mean": float(ratio_int),
                "peak_I": float(peak_I_f),
                "peak_phi": float(peak_phi_f),
                "peak_I_r2": float(Ir2),
                "peak_I_r2_ratio": float(Ir2_ratio),
                "delta_Ir2_frac": float(delta_Ir2_frac),
                "tail_I_mean": float(floor_mu),
                "tail_I_std": float(floor_sd),
                "snr_peak_over_tail_mean": float(snr_mu),
                "snr_peak_over_tail_std": float(snr_sd),
            }
        )

    meta = {
        "c_theory": float(math.sqrt(float(traffic.c2))),
        "c_axis_mean": float(c_axis),
        "Ir2_axis_mean": float(Ir2_axis),
        "tail_window": float(20.0),
        "detectors": float(len(dets)),
    }

    return rows, meta


def run_isotropy_calibration(
    *,
    n: int,
    R: int,
    steps: int,
    sigma: float,
    amp: float,
    traffic: TrafficParams,
    out_csv: str,
    provenance_header: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """Run isotropy calibration and write CSV.

    Parameters
    - n: grid size (n,n,n)
    - R: detector radius (axis distance in cells)
    - steps: number of evolve ticks after the single injection tick
    - sigma: Gaussian pulse width (cells)
    - amp: pulse peak amplitude
    - traffic: TrafficParams (must be telegraph-compatible)
    - out_csv: output CSV path
    """

    if n < 8:
        raise ValueError("n too small")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    out_csv = str(out_csv)
    ensure_parent_dir(out_csv)

    rows, meta = _compute_isotropy_rows(
        n=n,
        R=R,
        steps=steps,
        sigma=sigma,
        amp=amp,
        traffic=traffic,
        progress_cb=progress_cb,
    )

    c_axis = float(meta["c_axis_mean"])
    dets_count = int(meta["detectors"])
    tail_window = int(meta["tail_window"])

    if progress_cb is None:
        print(f"[isotropy] n={n} R={R} steps={steps} sigma={sigma:g} amp={amp:g}")
        print(f"[isotropy] axis_mean: c_eff={c_axis:.6f} (axes=6)")

    with open(out_csv, "w", newline="") as f:
        if provenance_header is not None and str(provenance_header).strip() != "":
            h = str(provenance_header)
            if not h.endswith("\n"):
                h += "\n"
            f.write(h)
        f.write(f"# isotropy_R={int(R)}\n")
        f.write(f"# isotropy_steps={int(steps)}\n")
        f.write(f"# isotropy_sigma={float(sigma):.12g}\n")
        f.write(f"# isotropy_amp={float(amp):.12g}\n")
        f.write(f"# isotropy_detectors={int(dets_count)}\n")
        f.write(f"# isotropy_tail_window={int(tail_window)}\n")
        f.write(f"# traffic_mode={str(traffic.mode)}\n")
        f.write(f"# traffic_c2={float(traffic.c2):.12g}\n")
        f.write(f"# traffic_gamma={float(traffic.gamma):.12g}\n")
        f.write(f"# traffic_dt={float(traffic.dt):.12g}\n")
        f.write(f"# traffic_decay={float(traffic.decay):.12g}\n")
        f.write(f"# traffic_inject={float(getattr(traffic, 'inject', 0.0)):.12g}\n")
        b = getattr(traffic, "boundary", None)
        if b is not None and str(b) != "":
            f.write(f"# traffic_boundary={str(b)}\n")
            f.write(f"# traffic_sponge_width={int(getattr(traffic, 'sponge_width', 0))}\n")
            f.write(f"# traffic_sponge_strength={float(getattr(traffic, 'sponge_strength', 0.0)):.12g}\n")
        f.write(f"# c_theory={math.sqrt(float(traffic.c2)):.12g}\n")
        f.write(f"# c_axis_mean={float(c_axis):.12g}\n")
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sigma",
                "detector",
                "dx",
                "dy",
                "dz",
                "x",
                "y",
                "z",
                "dist",
                "dist_over_R",
                "kind", "ux", "uy", "uz",
                "t_peak",
                "t_peak_f",
                "valid_peak",
                "arrival_frac", "tail_len",
                "c_eff",
                "c_eff_int",
                "ratio_vs_axis_mean",
                "delta_c", "delta_c_frac",
                "ratio_int_vs_axis_mean",
                "peak_I",
                "peak_phi",
                "peak_I_r2",
                "peak_I_r2_ratio",
                "delta_Ir2_frac",
                "tail_I_mean",
                "tail_I_std",
                "snr_peak_over_tail_mean",
                "snr_peak_over_tail_std",
            ],
        )
        w.writeheader()
        w.writerows(rows)
        f.write("\n")
        _write_isotropy_summary_footer(f, rows)

    if progress_cb is None:
        print(f"[isotropy] out_csv={out_csv}")


# --- Public wrapper for core.py naming stability ---
def run_isotropy_test(
    *,
    n: int,
    R: int,
    steps: int,
    sigma: float,
    amp: float,
    traffic: TrafficParams,
    out_csv: str,
    provenance_header: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """Public entry-point used by core.py.

    This is a thin wrapper kept for naming stability while we iterate.
    """

    run_isotropy_calibration(
        n=n,
        R=R,
        steps=steps,
        sigma=sigma,
        amp=amp,
        traffic=traffic,
        out_csv=out_csv,
        provenance_header=provenance_header,
        progress_cb=progress_cb,
    )


def run_isotropy_sigma_sweep(
    *,
    n: int,
    R: int,
    steps: int,
    sigmas: list[float],
    amp: float,
    traffic: TrafficParams,
    out_csv: str,
    provenance_header: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """Run isotropy calibration over many sigmas and write a single CSV.

    The output contains one row per detector per sigma.
    """

    if not sigmas:
        raise ValueError("sigmas must be non-empty")

    out_csv = str(out_csv)
    ensure_parent_dir(out_csv)

    all_rows: list[dict[str, object]] = []
    meta0: Optional[dict[str, float]] = None

    # Reuse large buffers across sigma points to avoid repeated allocations.
    phi_buf = np.zeros((n, n, n), dtype=np.float32)
    vel_buf = np.zeros((n, n, n), dtype=np.float32)
    src_buf = np.zeros((n, n, n), dtype=np.float32)

    for s in sigmas:
        rows, meta = _compute_isotropy_rows(
            n=n,
            R=R,
            steps=steps,
            sigma=float(s),
            amp=amp,
            traffic=traffic,
            progress_cb=progress_cb,
            phi_buf=phi_buf,
            vel_buf=vel_buf,
            src_buf=src_buf,
        )
        if meta0 is None:
            meta0 = meta
        all_rows.extend(rows)

    if meta0 is None:
        raise RuntimeError("sigma sweep produced no results")

    c_axis = float(meta0["c_axis_mean"])
    tail_window = int(meta0["tail_window"])
    dets_count = int(meta0["detectors"])

    with open(out_csv, "w", newline="") as f:
        if provenance_header is not None and str(provenance_header).strip() != "":
            h = str(provenance_header)
            if not h.endswith("\n"):
                h += "\n"
            f.write(h)

        f.write(f"# isotropy_R={int(R)}\n")
        f.write(f"# isotropy_steps={int(steps)}\n")
        f.write(f"# isotropy_amp={float(amp):.12g}\n")
        f.write(f"# isotropy_sigmas={','.join([str(float(x)) for x in sigmas])}\n")
        f.write(f"# isotropy_detectors={int(dets_count)}\n")
        f.write(f"# isotropy_tail_window={int(tail_window)}\n")
        f.write(f"# traffic_mode={str(traffic.mode)}\n")
        f.write(f"# traffic_c2={float(traffic.c2):.12g}\n")
        f.write(f"# traffic_gamma={float(traffic.gamma):.12g}\n")
        f.write(f"# traffic_dt={float(traffic.dt):.12g}\n")
        f.write(f"# traffic_decay={float(traffic.decay):.12g}\n")
        f.write(f"# traffic_inject={float(getattr(traffic, 'inject', 0.0)):.12g}\n")
        b = getattr(traffic, "boundary", None)
        if b is not None and str(b) != "":
            f.write(f"# traffic_boundary={str(b)}\n")
            f.write(f"# traffic_sponge_width={int(getattr(traffic, 'sponge_width', 0))}\n")
            f.write(f"# traffic_sponge_strength={float(getattr(traffic, 'sponge_strength', 0.0)):.12g}\n")
        f.write(f"# c_theory={math.sqrt(float(traffic.c2)):.12g}\n")
        f.write(f"# c_axis_mean={float(c_axis):.12g}\n")

        w = csv.DictWriter(
            f,
            fieldnames=[
                "sigma",
                "detector",
                "dx",
                "dy",
                "dz",
                "x",
                "y",
                "z",
                "dist",
                "dist_over_R",
                "kind", "ux", "uy", "uz",
                "t_peak",
                "t_peak_f",
                "valid_peak",
                "arrival_frac", "tail_len",
                "c_eff",
                "c_eff_int",
                "ratio_vs_axis_mean",
                "delta_c", "delta_c_frac",
                "ratio_int_vs_axis_mean",
                "peak_I",
                "peak_phi",
                "peak_I_r2",
                "peak_I_r2_ratio",
                "delta_Ir2_frac",
                "tail_I_mean",
                "tail_I_std",
                "snr_peak_over_tail_mean",
                "snr_peak_over_tail_std",
            ],
        )
        w.writeheader()
        w.writerows(all_rows)
        f.write("\n")
        _write_isotropy_summary_footer(f, all_rows)

    if progress_cb is None:
        print(f"[isotropy] out_csv={out_csv}")


def _main() -> None:
    # Minimal manual entry-point for quick local sanity checks.
    tp = TrafficParams(
        mode="telegraph",
        c2=0.25,
        gamma=0.0,
        dt=1.0,
        inject=1.0,
        decay=0.0,
    )

    run_isotropy_test(
        n=128,
        R=40,
        steps=220,
        sigma=2.0,
        amp=50.0,
        traffic=tp,
        out_csv="_Output/isotropy_n128_R40.csv",
    )


if __name__ == "__main__":
    _main()