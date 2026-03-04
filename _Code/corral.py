# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""corral.py — Quantization via cavity resonances ("Quantum Corral").

This module builds a hard mask (Dirichlet boundary) that confines the field and then
sweeps a sinusoidal point-source frequency to measure stored energy.

Expected behaviour:
- Energy vs ω shows sharp peaks (standing-wave eigenmodes of the masked geometry).

Notes:
- Uses the masked telegraph solver in traffic.py (Jacobi-style, order independent).
- No plotting here. Write CSV only. Use visualiser.py for figures.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np

from params import TrafficParams
from traffic import evolve_telegraph_traffic_steps
from utils import Timer, ensure_parent_dir


@dataclass(frozen=True)
class CorralSweepConfig:
    geom: str = "sphere"  # sphere | cylinder
    radius: int = 32
    omega_start: float = 0.10
    omega_stop: float = 0.60
    omega_steps: int = 50
    steps: int = 1200
    burn_frac: float = 0.90
    amp: float = 1.0
    center_offset: Tuple[int, int, int] = (0, 0, 0)


def _build_mask(n: int, geom: str, radius: int) -> np.ndarray:
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if geom not in ("sphere", "cylinder"):
        raise ValueError("geom must be one of: sphere, cylinder")

    mask = np.zeros((n, n, n), dtype=np.int8)
    cx = n // 2
    cy = n // 2
    cz = n // 2

    r2 = float(radius * radius)

    if geom == "sphere":
        xs = (np.arange(n, dtype=np.float32) - float(cx)) ** 2
        ys = (np.arange(n, dtype=np.float32) - float(cy)) ** 2
        zs = (np.arange(n, dtype=np.float32) - float(cz)) ** 2
        d2 = xs[:, None, None] + ys[None, :, None] + zs[None, None, :]
        mask[d2 >= r2] = 1
        return mask

    xs = (np.arange(n, dtype=np.float32) - float(cx)) ** 2
    ys = (np.arange(n, dtype=np.float32) - float(cy)) ** 2
    d2_xy = xs[:, None] + ys[None, :]
    mask[d2_xy[:, :, None] >= r2] = 1
    return mask


def _energy_metrics(phi: np.ndarray, vel: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float, float]:
    if phi.shape != vel.shape or phi.shape != mask.shape:
        raise ValueError("phi/vel/mask shape mismatch")
    inside = (mask == 0)
    p = phi[inside].astype(np.float64, copy=False)
    v = vel[inside].astype(np.float64, copy=False)
    e_phi = float(np.sum(p * p))
    e_vel = float(np.sum(v * v))
    e_tot = e_phi + e_vel
    max_phi = float(np.max(np.abs(p))) if p.size else 0.0
    max_vel = float(np.max(np.abs(v))) if v.size else 0.0
    return e_phi, e_vel, e_tot, max_phi, max_vel


def run_quantum_corral_sweep(
    *,
    n: int,
    radius: int,
    omega_start: float,
    omega_stop: float,
    omega_steps: int,
    burn_in: int,
    warm_steps: int,
    out_csv: str,
    traffic: TrafficParams,
    geom: str = "sphere",
    burn_frac: float = 0.90,
    amp: float = 1.0,
    center_offset: Tuple[int, int, int] = (0, 0, 0),
    provenance_header: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Dict[str, object]:
    """Run a frequency sweep and write a spectrum CSV.

    This is wired from core.py. All outputs are written to a single CSV file.
    """

    n = int(n)
    if n <= 8:
        raise ValueError("n too small")

    radius = int(radius)
    if radius <= 0:
        raise ValueError("radius must be > 0")

    omega_steps = int(omega_steps)
    if omega_steps <= 1:
        raise ValueError("omega_steps must be > 1")

    burn_in = int(burn_in)
    if burn_in <= 10:
        raise ValueError("burn_in must be > 10")

    warm_steps = int(warm_steps)
    if warm_steps < 0:
        raise ValueError("warm_steps must be >= 0")

    burn_frac = float(burn_frac)
    if not (0.0 < burn_frac < 1.0):
        raise ValueError("burn_frac must be in (0, 1)")

    tp = traffic
    if str(getattr(tp, "mode", "")) != "telegraph":
        raise ValueError("corral requires telegraph mode")

    out_csv = str(out_csv)
    ensure_parent_dir(out_csv)

    timer = Timer().start()

    mask = _build_mask(n=n, geom=str(geom), radius=radius)
    inside_count = int(np.sum(mask == 0))

    cx = n // 2 + int(center_offset[0])
    cy = n // 2 + int(center_offset[1])
    cz = n // 2 + int(center_offset[2])
    if not (0 <= cx < n and 0 <= cy < n and 0 <= cz < n):
        raise ValueError("center_offset places source out of bounds")
    if int(mask[cx, cy, cz]) != 0:
        raise ValueError("source lies inside masked wall")

    meas_start = int(burn_in * burn_frac)
    meas_len = int(burn_in - meas_start)
    if meas_len <= 0:
        raise ValueError("measurement window is empty")

    if progress_cb is None:
        print(
            "[corral] n=%d geom=%s radius=%d inside=%d omegas=%d burn=%d warm=%d burn_frac=%.3f amp=%.4g out_csv=%s"
            % (
                int(n),
                str(geom),
                int(radius),
                int(inside_count),
                int(omega_steps),
                int(burn_in),
                int(warm_steps),
                float(burn_frac),
                float(amp),
                out_csv,
            )
        )

    omegas = np.linspace(float(omega_start), float(omega_stop), int(omega_steps), dtype=np.float64)

    phi = np.zeros((n, n, n), dtype=np.float32)
    vel = np.zeros((n, n, n), dtype=np.float32)
    src = np.zeros((n, n, n), dtype=np.float32)

    rows: List[Dict[str, object]] = []

    for i, omega in enumerate(omegas.tolist()):
        phi.fill(0.0)
        vel.fill(0.0)

        # Optional warm-up to settle transients before measurement run.
        if warm_steps:
            for t in range(warm_steps):
                src[cx, cy, cz] = float(amp) * float(math.sin(float(omega) * float(t)))
                phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, tp, 1, mask=mask)

        acc_phi = 0.0
        acc_vel = 0.0
        acc_tot = 0.0
        acc_tot_sq = 0.0
        max_e_tot = 0.0
        peak_phi = 0.0
        peak_vel = 0.0

        for t in range(burn_in):
            # Continue phase from warm-up so we don't introduce an artificial discontinuity.
            tt = t + warm_steps
            src[cx, cy, cz] = float(amp) * float(math.sin(float(omega) * float(tt)))

            phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, tp, 1, mask=mask)

            if progress_cb is not None and (t % 200) == 0:
                # Heartbeat: keep the shared progress bar clock ticking during heavy inner loops.
                progress_cb(0)

            if t >= meas_start:
                e_phi, e_vel, e_tot, max_phi, max_vel = _energy_metrics(phi, vel, mask)
                acc_phi += e_phi
                acc_vel += e_vel
                acc_tot += e_tot
                acc_tot_sq += float(e_tot) * float(e_tot)
                if e_tot > max_e_tot:
                    max_e_tot = float(e_tot)
                if max_phi > peak_phi:
                    peak_phi = max_phi
                if max_vel > peak_vel:
                    peak_vel = max_vel

        m = float(meas_len)
        mean_phi = float(acc_phi / m)
        mean_vel = float(acc_vel / m)
        mean_tot = float(acc_tot / m)
        var_tot = float((acc_tot_sq / m) - (mean_tot * mean_tot))
        if var_tot < 0.0:
            var_tot = 0.0
        std_tot = float(math.sqrt(var_tot))
        e_ratio = float(mean_vel / (mean_phi + 1.0e-12))

        rows.append(
            {
                "i_omega": int(i),
                "omega": float(omega),
                "e_phi": float(mean_phi),
                "e_vel": float(mean_vel),
                "e_tot": float(mean_tot),
                "e_tot_std": float(std_tot),
                "e_tot_max": float(max_e_tot),
                "e_ratio": float(e_ratio),
                "peak_phi": float(peak_phi),
                "peak_vel": float(peak_vel),
            }
        )

        if progress_cb is not None:
            progress_cb(1)
        elif (i % 5) == 0 or (i + 1) == int(omega_steps):
            print(
                "[corral] omega=%.6f (%d/%d) e_tot=%.6e peak_phi=%.6e"
                % (float(omega), i + 1, int(omega_steps), float(acc_tot / m), float(peak_phi))
            )

    with open(out_csv, "w", newline="") as f:
        if provenance_header is not None and str(provenance_header).strip() != "":
            h = str(provenance_header)
            if not h.endswith("\n"):
                h += "\n"
            f.write(h)
        # Telegraph transport parameters (for auditability / reproducibility).
        # Use getattr to avoid tight coupling to params schema.
        f.write(f"# traffic_mode={str(getattr(tp, 'mode', ''))}\n")
        f.write(f"# traffic_dt={float(getattr(tp, 'dt', 1.0)):.12g}\n")
        f.write(f"# traffic_c2={float(getattr(tp, 'c2', 0.0)):.12g}\n")
        f.write(f"# traffic_gamma={float(getattr(tp, 'gamma', 0.0)):.12g}\n")
        f.write(f"# traffic_decay={float(getattr(tp, 'decay', 0.0)):.12g}\n")
        f.write(f"# traffic_inject={float(getattr(tp, 'inject', 0.0)):.12g}\n")
        boundary = str(getattr(tp, "boundary", ""))
        if boundary != "":
            f.write(f"# traffic_boundary={boundary}\n")
        sponge_w = getattr(tp, "sponge_width", None)
        if sponge_w is None:
            sponge_w = getattr(tp, "sponge_w", None)
        if sponge_w is not None:
            f.write(f"# traffic_sponge_width={int(sponge_w)}\n")
        sponge_s = getattr(tp, "sponge_strength", None)
        if sponge_s is None:
            sponge_s = getattr(tp, "sponge_s", None)
        if sponge_s is not None:
            f.write(f"# traffic_sponge_strength={float(sponge_s):.12g}\n")
        f.write("# corral_mask=dirichlet\n")
        f.write(f"# corral_geom={str(geom)}\n")
        f.write(f"# corral_radius={int(radius)}\n")
        f.write(f"# corral_inside_count={int(inside_count)}\n")
        f.write(f"# corral_omega_start={float(omega_start):.12g}\n")
        f.write(f"# corral_omega_stop={float(omega_stop):.12g}\n")
        f.write(f"# corral_omega_steps={int(omega_steps)}\n")
        f.write(f"# corral_burn_in={int(burn_in)}\n")
        f.write(f"# corral_meas_start={int(meas_start)}\n")
        f.write(f"# corral_meas_len={int(meas_len)}\n")
        f.write(f"# corral_warm_steps={int(warm_steps)}\n")
        f.write(f"# corral_burn_frac={float(burn_frac):.12g}\n")
        f.write(f"# corral_amp={float(amp):.12g}\n")
        f.write(f"# corral_center_offset={int(center_offset[0])},{int(center_offset[1])},{int(center_offset[2])}\n")
        w = csv.DictWriter(
            f,
            fieldnames=[
                "i_omega",
                "omega",
                "e_phi",
                "e_vel",
                "e_tot",
                "e_tot_std",
                "e_tot_max",
                "e_ratio",
                "peak_phi",
                "peak_vel",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    elapsed = timer.stop_s()
    if progress_cb is None:
        print("[corral] wrote %s (%.3fs)" % (out_csv, float(elapsed)))

    if rows:
        idx = max(range(len(rows)), key=lambda j: float(cast(float, rows[j]["e_tot"])))
        best = rows[idx]
    else:
        best = {"omega": 0.0, "e_tot": 0.0}

    return {
        "out_csv": out_csv,
        "elapsed_s": float(elapsed),
        "best_omega": float(cast(float, best.get("omega", 0.0))),
        "best_e_tot": float(cast(float, best.get("e_tot", 0.0))),
        "inside_count": int(inside_count),
    }