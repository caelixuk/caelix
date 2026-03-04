# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""lattice.py — ternary microstate initialisation + anneal

Role
----
Provides a tiny microstate generator used by the pipeline when *not* running in
`--delta-load` mode.

- `lattice_init()` seeds a mostly-zero 3D lattice with sparse ±1 excitations and
  forces a +1 at the center.
- `lattice_anneal()` performs a simple, fail-fast Metropolis-like relaxation that
  only accepts energy non-increasing moves (no temperature schedule).
- `lattice_init_multiscale()` builds a correlated ternary microstate via multiscale noise + a few full-grid smoothing passes (fast “malloc/inflation proxy”, no slow per-voxel anneal).

Energy surrogate
----------------
The local energy is deliberately minimal:
  - mismatch cost: |s_i - s_j| over valid 6-neighbour sites (hard boundaries)
  - occupancy cost: penalty for s_i != 0

This is a defect/edge counter, not a physical Hamiltonian.

Contracts
---------
- Microstate `s` is int8 with values expected in {-1,0,+1}.
- Shape is (n,n,n) with hard (non-periodic) boundaries.

Flat-module layout
------------------
Lives alongside `core.py` and is imported as:
  `from lattice import lattice_init, lattice_anneal`
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from numba import njit

from params import LatticeParams


def lattice_init(params: LatticeParams, rng: np.random.Generator) -> np.ndarray:
    """Initialise ternary lattice with sparse ±1 excitations in a mostly-zero vacuum."""
    n = params.n
    s = np.zeros((n, n, n), dtype=np.int8)

    mask = rng.random((n, n, n)) < params.p_seed
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n, n, n))
    s[mask] = signs[mask]

    c = n // 2
    s[c, c, c] = 1
    return s


def lattice_init_multiscale(params: LatticeParams, rng: np.random.Generator) -> np.ndarray:
    """Fast correlated ternary init via multiscale noise + light smoothing.

    This is a pragmatic “malloc/inflation proxy”: we sample a correlated field in one shot
    (no slow single-site Metropolis anneal), then threshold into {-1,0,+1} with roughly
    `p_seed` non-zero occupancy.

    Notes
    -----
    - Hard boundaries (no wrap).
    - Correlation structure is generic (multiscale smoothing), not a baked-in cosmology model.
    """
    n = int(params.n)
    if n <= 0:
        raise ValueError("n must be > 0")

    p = float(params.p_seed)
    if p <= 0.0:
        s = np.zeros((n, n, n), dtype=np.int8)
        c = n // 2
        s[c, c, c] = 1
        return s
    if p >= 1.0:
        p = 1.0

    g = np.zeros((n, n, n), dtype=np.float32)

    # Multiscale octaves: start coarse, upsample, and add finer detail with reduced amplitude.
    # Divisors are chosen to divide common experiment sizes (e.g. 128/256/384/512).
    divisors = [16, 8, 4, 2, 1]
    amp = 1.0
    for d in divisors:
        if d <= 0:
            continue
        if n % d != 0:
            continue
        cn = n // d
        if cn < 4:
            continue
        noise = rng.standard_normal((cn, cn, cn), dtype=np.float32)
        if d != 1:
            noise = noise.repeat(d, axis=0).repeat(d, axis=1).repeat(d, axis=2)
        g += (amp * noise)
        # Light smoothing after each octave to promote spatial coherence.
        _smooth_inplace(g, passes=2)
        amp *= 0.5

    # Final gentle smoothing: keeps the field coherent without erasing all texture.
    _smooth_inplace(g, passes=4)

    # Threshold to match requested occupancy fraction (approximately).
    a = np.abs(g).reshape(-1)
    q = 1.0 - float(p)
    if q <= 0.0:
        thr = 0.0
    else:
        thr = float(np.quantile(a.astype(np.float64, copy=False), q))

    s = np.zeros((n, n, n), dtype=np.int8)
    m = np.abs(g) >= thr
    s[m] = np.int8(1)
    s[g < 0.0] *= np.int8(-1)

    c = n // 2
    s[c, c, c] = 1
    return s
# ==================== Numba kernels ====================

def _smooth_inplace(g: np.ndarray, passes: int) -> None:
    if passes <= 0:
        return
    if g.ndim != 3:
        raise ValueError("expected 3D array")
    p = int(passes)
    if p <= 0:
        return
    orig = g
    work = g
    tmp = np.empty_like(g)
    for _ in range(p):
        _nb_smooth_pass(work, tmp)
        work, tmp = tmp, work
    if work is not orig:
        orig[:] = work


def _local_energy(s: np.ndarray, x: int, y: int, z: int, j_mismatch: float, j_nonzero: float) -> float:
    """Local energy contribution for a single site (6-neighbour, periodic off, hard boundaries).

    Energy surrogate:
      - mismatch: sum |s_i - s_j| over valid neighbours
      - occupancy: penalty for s_i != 0

    This is deliberately simple: it's a defect/edge counter, not a full field theory.
    """
    n = s.shape[0]
    si = int(s[x, y, z])
    e = 0.0

    if si != 0:
        e += j_nonzero

    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        xx = x + dx
        yy = y + dy
        zz = z + dz
        if 0 <= xx < n and 0 <= yy < n and 0 <= zz < n:
            sj = int(s[xx, yy, zz])
            e += j_mismatch * abs(si - sj)

    return e


# ==================== Numba kernels ====================

@njit(cache=True)
def _nb_local_energy(s: np.ndarray, x: int, y: int, z: int, j_mismatch: float, j_nonzero: float) -> float:
    n = s.shape[0]
    si = int(s[x, y, z])
    e = 0.0
    if si != 0:
        e += float(j_nonzero)

    if x + 1 < n:
        sj = int(s[x + 1, y, z])
        d = si - sj
        if d < 0:
            d = -d
        e += float(j_mismatch) * float(d)
    if x - 1 >= 0:
        sj = int(s[x - 1, y, z])
        d = si - sj
        if d < 0:
            d = -d
        e += float(j_mismatch) * float(d)
    if y + 1 < n:
        sj = int(s[x, y + 1, z])
        d = si - sj
        if d < 0:
            d = -d
        e += float(j_mismatch) * float(d)
    if y - 1 >= 0:
        sj = int(s[x, y - 1, z])
        d = si - sj
        if d < 0:
            d = -d
        e += float(j_mismatch) * float(d)
    if z + 1 < n:
        sj = int(s[x, y, z + 1])
        d = si - sj
        if d < 0:
            d = -d
        e += float(j_mismatch) * float(d)
    if z - 1 >= 0:
        sj = int(s[x, y, z - 1])
        d = si - sj
        if d < 0:
            d = -d
        e += float(j_mismatch) * float(d)

    return float(e)


@njit(cache=True)
def _nb_smooth_pass(src: np.ndarray, dst: np.ndarray) -> None:
    n0, n1, n2 = src.shape
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                v = float(src[i, j, k])
                c = 1.0
                if i + 1 < n0:
                    v += float(src[i + 1, j, k])
                    c += 1.0
                if i - 1 >= 0:
                    v += float(src[i - 1, j, k])
                    c += 1.0
                if j + 1 < n1:
                    v += float(src[i, j + 1, k])
                    c += 1.0
                if j - 1 >= 0:
                    v += float(src[i, j - 1, k])
                    c += 1.0
                if k + 1 < n2:
                    v += float(src[i, j, k + 1])
                    c += 1.0
                if k - 1 >= 0:
                    v += float(src[i, j, k - 1])
                    c += 1.0
                dst[i, j, k] = np.float32(v / c)


@njit(cache=True)
def _nb_pick_prop(cur: int, bit01: int) -> int:
    """Pick a proposal value uniformly from the two choices != cur.

    This matches the rejection-sampling logic in the Python version but avoids a loop.
    bit01 must be 0 or 1.
    """
    if cur == -1:
        return 0 if bit01 == 0 else 1
    if cur == 0:
        return -1 if bit01 == 0 else 1
    return -1 if bit01 == 0 else 0


@njit(cache=True)
def _nb_anneal_apply(
    s: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    bits: np.ndarray,
    j_mismatch: float,
    j_nonzero: float,
) -> None:
    steps = xs.shape[0]
    for i in range(steps):
        x = int(xs[i])
        y = int(ys[i])
        z = int(zs[i])

        cur = int(s[x, y, z])
        prop = _nb_pick_prop(cur, int(bits[i]))

        e0 = _nb_local_energy(s, x, y, z, j_mismatch, j_nonzero)
        s[x, y, z] = np.int8(prop)
        e1 = _nb_local_energy(s, x, y, z, j_mismatch, j_nonzero)

        if e1 > e0:
            s[x, y, z] = np.int8(cur)


def lattice_anneal(
    s: np.ndarray,
    params: LatticeParams,
    rng: np.random.Generator,
    progress_cb: Callable[[int], None] | None = None
) -> np.ndarray:
    """Metropolis-like, but strictly energy-non-increasing (fail-loud: no temperature)."""
    n = params.n
    j_mismatch = params.j_mismatch
    j_nonzero = params.j_nonzero

    steps = int(params.steps)
    xs = rng.integers(0, n, size=steps, dtype=np.int32)
    ys = rng.integers(0, n, size=steps, dtype=np.int32)
    zs = rng.integers(0, n, size=steps, dtype=np.int32)
    bits = rng.integers(0, 2, size=steps, dtype=np.int8)

    if progress_cb is None:
        _nb_anneal_apply(s, xs, ys, zs, bits, float(j_mismatch), float(j_nonzero))
        return s

    # Progress-aware chunking: keeps the hot loop in numba while providing coarse updates.
    j_m = float(j_mismatch)
    j_nz = float(j_nonzero)
    chunk = 4096
    i0 = 0
    while i0 < steps:
        i1 = i0 + chunk
        if i1 > steps:
            i1 = steps
        _nb_anneal_apply(s, xs[i0:i1], ys[i0:i1], zs[i0:i1], bits[i0:i1], j_m, j_nz)
        progress_cb(i1 - i0)
        i0 = i1

    return s