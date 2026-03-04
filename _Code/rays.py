# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""rays.py — 2D ray tracing over an index map + Shapiro-style regression

Role
----
Implements a simple geometric-optics ray tracer over a 2D slice of the index map
`n_map(x,y)` produced by the pipeline:

  phi(x,y,z) -> n_map(x,y) = 1 + k * phi_slice

Given a set of impact parameters `b`, we integrate:
  D(b) = ∫ (n - 1) ds
using a midpoint update, and use an asinh regression as a finite-domain proxy for
log-like Shapiro behaviour:
  D ≈ C + K * asinh(X0 / b)

Important details
-----------------
- Sampling uses bilinear interpolation with central-difference gradients.
- Outside the map region we treat space as vacuum: n=1 and ∇n=0.
- This is a diagnostic ray tracer (not a full eikonal solver).

Contracts
---------
- `n_map` is 2D, finite, and expected to be >= 1 everywhere.
- Returns `D` as float64 for regression stability.
- Fail-fast on non-finite gradient samples.

Flat-module layout
------------------
Lives alongside `core.py` and is imported as:
  `from rays import ray_trace_delay, fit_asinh_delay`
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numba import njit, prange

from params import RayParams


def _bilinear(grid: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample from grid in index coordinates."""
    h, w = grid.shape
    x = float(np.clip(x, 0.0, w - 1.001))
    y = float(np.clip(y, 0.0, h - 1.001))

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - x0
    fy = y - y0

    v00 = float(grid[y0, x0])
    v10 = float(grid[y0, x1])
    v01 = float(grid[y1, x0])
    v11 = float(grid[y1, x1])

    v0 = v00 * (1 - fx) + v10 * fx
    v1 = v01 * (1 - fx) + v11 * fx
    return v0 * (1 - fy) + v1 * fy


def _grad_bilinear(grid: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Central-difference gradient (dn/dx, dn/dy) in index coordinates."""
    n0 = _bilinear(grid, x, y)
    nxp = _bilinear(grid, x + 1.0, y)
    nxm = _bilinear(grid, x - 1.0, y)
    nyp = _bilinear(grid, x, y + 1.0)
    nym = _bilinear(grid, x, y - 1.0)

    dn_dx = 0.5 * (nxp - nxm)
    dn_dy = 0.5 * (nyp - nym)

    if not (math.isfinite(n0) and math.isfinite(dn_dx) and math.isfinite(dn_dy)):
        raise ValueError("non-finite gradient sample")

    return dn_dx, dn_dy


@njit(cache=True)
def _nb_bilinear(grid: np.ndarray, x: float, y: float) -> float:
    h, w = grid.shape
    if x < 0.0:
        x = 0.0
    if y < 0.0:
        y = 0.0
    xmax = float(w) - 1.001
    ymax = float(h) - 1.001
    if x > xmax:
        x = xmax
    if y > ymax:
        y = ymax

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - float(x0)
    fy = y - float(y0)

    v00 = float(grid[y0, x0])
    v10 = float(grid[y0, x1])
    v01 = float(grid[y1, x0])
    v11 = float(grid[y1, x1])

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return v0 * (1.0 - fy) + v1 * fy


@njit(cache=True)
def _nb_bilinear_and_grad(grid: np.ndarray, x: float, y: float) -> Tuple[float, float, float]:
    """Bilinear sample plus analytic gradient (dn/dx, dn/dy) in index coordinates."""
    h, w = grid.shape
    if x < 0.0:
        x = 0.0
    if y < 0.0:
        y = 0.0
    xmax = float(w) - 1.001
    ymax = float(h) - 1.001
    if x > xmax:
        x = xmax
    if y > ymax:
        y = ymax

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - float(x0)
    fy = y - float(y0)

    v00 = float(grid[y0, x0])
    v10 = float(grid[y0, x1])
    v01 = float(grid[y1, x0])
    v11 = float(grid[y1, x1])

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    n = v0 * (1.0 - fy) + v1 * fy

    dn_dx = (1.0 - fy) * (v10 - v00) + fy * (v11 - v01)
    dn_dy = (1.0 - fx) * (v01 - v00) + fx * (v11 - v10)

    if not (math.isfinite(n) and math.isfinite(dn_dx) and math.isfinite(dn_dy)):
        raise ValueError("non-finite gradient sample")

    return float(n), float(dn_dx), float(dn_dy)


@njit(cache=True)
def _nb_bilinear_interior(grid: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample assuming (x,y) is strictly inside: 0 < x < w-1 and 0 < y < h-1."""
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - float(x0)
    fy = y - float(y0)

    v00 = float(grid[y0, x0])
    v10 = float(grid[y0, x1])
    v01 = float(grid[y1, x0])
    v11 = float(grid[y1, x1])

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    return v0 * (1.0 - fy) + v1 * fy


@njit(cache=True)
def _nb_bilinear_and_grad_interior(grid: np.ndarray, x: float, y: float) -> Tuple[float, float, float]:
    """Bilinear sample + analytic gradient assuming strict interior (no clamping)."""
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    fx = x - float(x0)
    fy = y - float(y0)

    v00 = float(grid[y0, x0])
    v10 = float(grid[y0, x1])
    v01 = float(grid[y1, x0])
    v11 = float(grid[y1, x1])

    v0 = v00 * (1.0 - fx) + v10 * fx
    v1 = v01 * (1.0 - fx) + v11 * fx
    n = v0 * (1.0 - fy) + v1 * fy

    dn_dx = (1.0 - fy) * (v10 - v00) + fy * (v11 - v01)
    dn_dy = (1.0 - fx) * (v01 - v00) + fx * (v11 - v10)

    if not (math.isfinite(n) and math.isfinite(dn_dx) and math.isfinite(dn_dy)):
        raise ValueError("non-finite gradient sample")

    return float(n), float(dn_dx), float(dn_dy)


@njit(cache=True)
def _nb_grad_bilinear(grid: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    n0 = _nb_bilinear(grid, x, y)
    nxp = _nb_bilinear(grid, x + 1.0, y)
    nxm = _nb_bilinear(grid, x - 1.0, y)
    nyp = _nb_bilinear(grid, x, y + 1.0)
    nym = _nb_bilinear(grid, x, y - 1.0)

    dn_dx = 0.5 * (nxp - nxm)
    dn_dy = 0.5 * (nyp - nym)

    if not (math.isfinite(n0) and math.isfinite(dn_dx) and math.isfinite(dn_dy)):
        raise ValueError("non-finite gradient sample")

    return dn_dx, dn_dy


@njit(parallel=True, cache=True)
def _nb_ray_trace_delay(n_map: np.ndarray, b_list: np.ndarray, X0: float, ds: float) -> np.ndarray:
    H, W = n_map.shape
    Wm1 = float(W - 1)
    Hm1 = float(H - 1)
    Wm2 = float(W - 2)
    Hm2 = float(H - 2)
    cx = float(W) / 2.0
    cy = float(H) / 2.0
    x0_map = -cx
    y0_map = -cy

    Ds = np.zeros(len(b_list), dtype=np.float64)
    max_steps = int(2.0 * X0 / ds * 2.0)

    for i in prange(len(b_list)):
        b = float(b_list[i])
        x = -X0
        y = b
        tx = 1.0
        ty = 0.0
        D = 0.0

        for _ in range(max_steps):
            if x >= X0:
                break

            xi = x - x0_map
            yi = y - y0_map

            if 0.0 <= xi < Wm1 and 0.0 <= yi < Hm1:
                if 1.0 <= xi < Wm2 and 1.0 <= yi < Hm2:
                    n, dn_dx, dn_dy = _nb_bilinear_and_grad_interior(n_map, xi, yi)
                else:
                    n = _nb_bilinear(n_map, xi, yi)
                    dn_dx, dn_dy = 0.0, 0.0
            else:
                n = 1.0
                dn_dx, dn_dy = 0.0, 0.0

            dn_ds = dn_dx * tx + dn_dy * ty
            dtx = (dn_dx - dn_ds * tx) / n
            dty = (dn_dy - dn_ds * ty) / n

            tmx = tx + 0.5 * ds * dtx
            tmy = ty + 0.5 * ds * dty
            inv_norm = 1.0 / (math.hypot(tmx, tmy) + 1e-12)
            tmx *= inv_norm
            tmy *= inv_norm

            x_mid = x + 0.5 * ds * tmx
            y_mid = y + 0.5 * ds * tmy

            xi_m = x_mid - x0_map
            yi_m = y_mid - y0_map

            if 0.0 <= xi_m < Wm1 and 0.0 <= yi_m < Hm1:
                if 1.0 <= xi_m < Wm2 and 1.0 <= yi_m < Hm2:
                    n_mid, dn_dx_mid, dn_dy_mid = _nb_bilinear_and_grad_interior(n_map, xi_m, yi_m)
                else:
                    n_mid = _nb_bilinear(n_map, xi_m, yi_m)
                    dn_dx_mid, dn_dy_mid = 0.0, 0.0
            else:
                n_mid = 1.0
                dn_dx_mid, dn_dy_mid = 0.0, 0.0

            dn_ds_mid = dn_dx_mid * tmx + dn_dy_mid * tmy
            dtx_mid = (dn_dx_mid - dn_ds_mid * tmx) / n_mid
            dty_mid = (dn_dy_mid - dn_ds_mid * tmy) / n_mid

            tx += ds * dtx_mid
            ty += ds * dty_mid
            inv_norm2 = 1.0 / (math.hypot(tx, ty) + 1e-12)
            tx *= inv_norm2
            ty *= inv_norm2

            x += ds * tmx
            y += ds * tmy

            D += (n_mid - 1.0) * ds

        Ds[i] = D

    return Ds


def ray_trace_delay(n_map: np.ndarray, b_list: np.ndarray, rp: RayParams) -> np.ndarray:
    """Trace rays across a 2D index map and return D(b)=∫(n-1)ds for each impact parameter."""
    n0 = np.ascontiguousarray(n_map)
    b0 = np.ascontiguousarray(b_list.astype(np.float64))
    Ds = _nb_ray_trace_delay(n0, b0, float(rp.X0), float(rp.ds))
    return Ds


def fit_asinh_delay(D: np.ndarray, b: np.ndarray, X0: float) -> Tuple[float, float, float]:
    """Fit D = C + K asinh(X0/b). Return (K, C, R^2)."""
    u = np.arcsinh(X0 / b)
    X = np.vstack([np.ones_like(u), u]).T
    C, K = np.linalg.lstsq(X, D, rcond=None)[0]
    pred = C + K * u

    ss_res = float(np.sum((D - pred) ** 2))
    ss_tot = float(np.sum((D - D.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return float(K), float(C), float(r2)