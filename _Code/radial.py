# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""radial.py — radial diagnostics + regression fits

Role
----
Provides compact diagnostics for the steady 3D scalar field `phi` by collapsing it
into integer-radius shells around the lattice center:

  phi(x,y,z) -> mean(phi | floor(r)=k)

This is used to confirm the expected far-field behaviour (~1/r) under hard
boundary sinks.

Fits provided
------------
- `fit_powerlaw_inverse`      : log-log fit of `phi ≈ a * r^p` over a chosen window.
- `fit_linear_inv_r`          : linear fit of `phi ≈ a*(1/r) + b` over a chosen window.
- `fit_powerlaw_after_inv_r`  : fit `a*(1/r)+b` then log-log fit of `(phi-b) ≈ a0 * r^p`.
- `slope_scan_powerlaw`       : multi-window slope scan for robust slope summaries.

CSV helpers
-----------
- `dump_radial_csv`      : shell profile dump (r, mean_phi, count)
- `dump_radial_fit_csv`  : tabular fit rows used by the pipeline.

Contracts
---------
- Inputs are finite numpy arrays; fitting requires at least ~8 usable points.
- Raises `ValueError` on insufficient points or invalid windows (fail-fast).

Flat-module layout
------------------
Lives alongside `core.py` and is imported as:
  `from radial import radial_profile, fit_powerlaw_inverse, ...`
"""

from __future__ import annotations

import csv
import math
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from numba import njit

from params import RadialFit
from utils import ensure_parent_dir


@njit(cache=True)
def _nb_radial_sums_counts(phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n0, n1, n2 = phi.shape
    c = (float(n0) - 1.0) * 0.5

    dz2 = np.empty(n2, dtype=np.float64)
    for k in range(n2):
        dz = float(k) - c
        dz2[k] = dz * dz

    r_max_ub = int(math.floor(math.sqrt(3.0) * c)) + 2
    sums = np.zeros(r_max_ub, dtype=np.float64)
    sums_sq = np.zeros(r_max_ub, dtype=np.float64)
    counts = np.zeros(r_max_ub, dtype=np.int64)
    max_r = 0

    for i in range(n0):
        dx = float(i) - c
        dx2 = dx * dx
        for j in range(n1):
            dy = float(j) - c
            dy2 = dy * dy
            for k in range(n2):
                ri = int(math.floor(math.sqrt(dx2 + dy2 + dz2[k])))
                if ri > max_r:
                    max_r = ri
                v = float(phi[i, j, k])
                sums[ri] += v
                sums_sq[ri] += v * v
                counts[ri] += 1

    return sums, sums_sq, counts, int(max_r)


def radial_profile(phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (r_centers, mean_phi_in_shell, count_in_shell, std_phi_in_shell) for integer shells around center."""
    sums, sums_sq, counts, max_r = _nb_radial_sums_counts(phi.astype(np.float32, copy=False))
    sums = sums[: max_r + 1]
    sums_sq = sums_sq[: max_r + 1]
    counts = counts[: max_r + 1]

    denom = np.maximum(counts, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        means64 = sums / denom
        ex2 = sums_sq / denom
        var = ex2 - (means64 * means64)
        var = np.maximum(var, 0.0)
        std64 = np.sqrt(var)

    rs = np.arange(max_r + 1, dtype=np.float32)
    return rs, means64.astype(np.float32), counts.astype(np.int64), std64.astype(np.float32)


def fit_powerlaw_inverse(phi_r: np.ndarray, r: np.ndarray, r_min: float, r_max: float) -> RadialFit:
    """Fit phi ≈ a * r^p over a radial window using log-log regression."""
    mask = (r >= r_min) & (r <= r_max) & (phi_r > 0)
    rr = r[mask]
    yy = phi_r[mask]
    if rr.size < 8:
        raise ValueError("insufficient points for radial fit")
    x = np.log(rr.astype(np.float64))
    y = np.log(yy.astype(np.float64))
    X = np.vstack([x, np.ones_like(x)]).T
    p, b = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = p * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return RadialFit(slope=float(p), intercept=float(b), r2=float(r2))


def fit_linear_inv_r(phi_r: np.ndarray, r: np.ndarray, r_min: float, r_max: float) -> RadialFit:
    """Fit phi(r) ≈ a*(1/r) + b over a radial window using linear regression."""
    mask = (r >= r_min) & (r <= r_max) & (phi_r > 0) & (r > 0)
    rr = r[mask].astype(np.float64)
    yy = phi_r[mask].astype(np.float64)
    if rr.size < 8:
        raise ValueError("insufficient points for 1/r linear fit")
    x = (1.0 / rr)
    X = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(X, yy, rcond=None)[0]
    pred = a * x + b
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return RadialFit(slope=float(a), intercept=float(b), r2=float(r2))


def fit_powerlaw_after_inv_r(phi_r: np.ndarray, r: np.ndarray, r_min: float, r_max: float) -> Dict[str, float]:
    """Fit `phi(r) ≈ A/r + B` and also estimate the exponent on `phi(r)-B`.

    Why:
        A non-zero background `B` makes a raw log-log fit on `phi` look artificially shallow.
        For far-field checks we want to validate the `1/r` part, so we first fit `A/r + B`,
        then fit `(phi-B) ≈ a0 * r^p` on the same window (log-log), using only positive points.

    Returns a compact dict suitable for CSV row assembly.
    """
    inv = fit_linear_inv_r(phi_r, r, r_min=r_min, r_max=r_max)
    B = float(inv.intercept)

    mask = (r >= float(r_min)) & (r <= float(r_max)) & (r > 0)
    rr = r[mask].astype(np.float64)
    yy = (phi_r[mask].astype(np.float64) - B)

    # Require positive residuals for the log-log fit.
    pos = yy > 0
    rr = rr[pos]
    yy = yy[pos]
    if rr.size < 8:
        raise ValueError("insufficient points for de-offset powerlaw fit")

    x = np.log(rr)
    y = np.log(yy)
    X = np.vstack([x, np.ones_like(x)]).T
    p, b = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = p * x + b

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "A": float(inv.slope),
        "B": float(B),
        "r2_inv_r": float(inv.r2),
        "p": float(p),
        "log_a0": float(b),
        "r2_loglog": float(r2),
        "n_points": float(rr.size),
    }


def slope_scan_powerlaw(phi_r: np.ndarray, r: np.ndarray, r_min: float, r_max: float, windows: int = 7, width_frac: float = 0.6) -> Dict[str, float]:
    """Scan multiple sub-windows and report robust slope summary for phi ≈ a*r^p."""
    r_min = float(r_min)
    r_max = float(r_max)
    if r_max <= r_min:
        raise ValueError("invalid slope scan window")
    w = (r_max - r_min) * float(width_frac)
    if w <= 0:
        raise ValueError("invalid slope scan width")
    if windows < 1:
        raise ValueError("windows must be >= 1")
    starts = np.linspace(r_min, max(r_min, r_max - w), int(windows)).astype(np.float64)
    slopes: List[float] = []
    used = 0
    for s in starts:
        a = float(s)
        b = float(s + w)
        try:
            fit = fit_powerlaw_inverse(phi_r, r, r_min=a, r_max=b)
        except ValueError:
            continue
        if math.isfinite(fit.slope):
            slopes.append(float(fit.slope))
            used += 1
    if used < 3:
        return {"used": float(used), "median": float("nan"), "p16": float("nan"), "p84": float("nan")}
    arr = np.array(slopes, dtype=np.float64)
    med = float(np.median(arr))
    p16 = float(np.percentile(arr, 16.0))
    p84 = float(np.percentile(arr, 84.0))
    return {"used": float(used), "median": med, "p16": p16, "p84": p84}


def dump_radial_csv(path: str, r: np.ndarray, mean_phi: np.ndarray, count: np.ndarray, std_phi: np.ndarray, *, provenance: Optional[str] = None) -> None:
    ensure_parent_dir(path)
    with open(path, "w", newline="") as f:
        if provenance is not None and str(provenance) != "":
            prov = str(provenance)
            if not prov.endswith("\n"):
                prov += "\n"
            f.write(prov)
        w = csv.writer(f)
        w.writerow(["r", "mean_phi", "count", "std_phi"])
        for rr, mm, cc, ss in zip(r.tolist(), mean_phi.tolist(), count.tolist(), std_phi.tolist()):
            w.writerow([float(rr), float(mm), int(cc), float(ss)])


def dump_radial_fit_csv(path: str, rows: List[Dict[str, object]], *, provenance: Optional[str] = None) -> None:
    base_cols = ["fit_kind", "r_min", "r_max", "n_points", "slope", "intercept", "r2"]
    extra_cols = [
        "A",
        "B",
        "r2_inv_r",
        "p",
        "log_a0",
        "r2_loglog",
        "scan_used",
        "scan_median",
        "scan_p16",
        "scan_p84",
        # Monitoring / selection metadata (used by pipeline observers)
        "age_iters",
        "err_abs_p_plus_1",
    ]

    # Include any additional keys present in rows (fail-soft, stable ordering).
    extra_set: Set[str] = set(extra_cols)
    auto_cols: List[str] = []
    for row in rows:
        for k in row.keys():
            kk = str(k)
            if kk in base_cols:
                continue
            if kk in extra_set:
                continue
            extra_set.add(kk)
            auto_cols.append(kk)
    auto_cols = sorted(set(auto_cols))

    cols = base_cols + extra_cols + auto_cols

    ensure_parent_dir(path)
    with open(path, "w", newline="") as f:
        if provenance is not None and str(provenance) != "":
            prov = str(provenance)
            if not prov.endswith("\n"):
                prov += "\n"
            f.write(prov)
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([row.get(c, "") for c in cols])