# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""traffic.py — load → steady scalar field (diffusion + telegraph kernels)

Role
----
Evolves a scalar field `phi` on a hard-bounded 3D lattice, driven by an injected
source (`load` / `src`).

This is the main hot-path candidate for Numba optimisation. Keep the API tight
and avoid dragging unrelated utilities into this module.

Evolution modes
---------------
- Diffusion (`mode="diffuse"`):
    A first-order relaxer that mixes towards the 6-neighbour average each tick.
    With sustained injection and Dirichlet boundaries (phi=0 on faces), the
    steady-state around a compact source approaches the familiar ~1/r behaviour.

- Telegraph (`mode="telegraph"`):
    A damped second-order surrogate with a velocity field `vel`.

    Important: parameters here are *per-tick* coefficients in a discrete-time
    update, not continuous-time PDE constants. In particular, `gamma`, `inject`
    and `c2` act directly on each tick; `dt` is applied only when integrating
    velocity into position (`phi`). If you change `dt` and want comparable
    dynamics, you must generally re-tune `gamma`/`inject`/`c2` accordingly.

    Intended for experiments that want wave-like transients without claiming a
    textbook hyperbolic PDE discretisation.


- Nonlinear (`mode="nonlinear"`):
    Telegraph stepping with a local restoring potential and stiffening cubic term.
    Acceleration adds: `-k*phi - lambda*phi^3` (lambda>=0; k may be negative for double-well/symmetry-breaking scans).

- Sine-Gordon (mode="sine_gordon"):
    Telegraph stepping with bounded restoring force: acceleration adds -k*sin(phi).

Obstacles (masked telegraph)
----------------------------
Telegraph stepping supports hard obstacles via an optional `mask` array:
- `mask[i,j,k] != 0` is a solid wall (Dirichlet clamp: `phi=0`, `vel=0`).
- Masked neighbours contribute as zero to the Laplacian (no “leaking” through
  walls).

This is used by the double-slit experiment (hard wall with carved apertures).

Boundary semantics
------------------
Hard (non-periodic) boundaries throughout:
- Neighbour sums use explicit hard checks (no wrap).
- Boundary handling supports either a hard clamp (`boundary_mode="zero"`) or an
  absorbing sponge layer (`boundary_mode="sponge"`). Hard clamp sets faces to
  zero each tick. Sponge applies a smooth damping ramp near faces (no hard
  clamp), reducing reflections while keeping the interior solution clean.

Numerics and determinism
------------------------
- Internal dtype is float32 for `phi/vel/src`.
- Telegraph stepping is Jacobi-style: reads from the previous state and writes
  into a separate buffer each tick (order-independent within a step).
- Stability checks in this module (e.g. the `sqrt(c2)*dt` bound) should be
  treated as pragmatic guards for this discrete-time scheme, not as a strict
  CFL condition derived from a continuous PDE.
- Fail-fast parameter checks; unknown modes raise.

Public API
----------
Batch solvers (allocate and run for `tp.iters`):
- `evolve_diffusion_traffic(load, tp) -> phi`
- `evolve_telegraph_traffic(load, tp) -> phi`
- `evolve_traffic(load, tp) -> phi` (dispatch)

Steppers (advance an existing state by `steps`):
- `evolve_diffusion_traffic_steps(phi, src, tp, steps) -> phi`
- `evolve_telegraph_traffic_steps(phi, vel, src, tp, steps, mask=None, chirality=None, chiral_select=0) -> (phi, vel)`

The stepper wrappers use `np.ascontiguousarray(..., dtype=...)` to avoid per-tick
full-grid copies when arrays are already contiguous and correctly typed.

Threading and environment
-------------------------
This module imports Numba at import-time. For consistent CPU utilisation on
Apple Silicon (P/E cores) and to avoid BLAS/vecLib oversubscription, `core.py`
calls:

  `apply_metal_thread_defaults()`

*before* importing runner modules that depend on Numba (including this module).
If you import `traffic.py` directly (outside `core.py`), you are responsible for
setting thread environment variables first.

Flat-module layout
------------------
Lives alongside `core.py` and is imported as:
  `from traffic import evolve_traffic`
"""

from __future__ import annotations

import math
from typing import Callable, Tuple, Optional

import numpy as np

from numba import njit, prange

from params import TrafficParams
from utils import _assert_finite


def _sum6_hard(arr: np.ndarray) -> np.ndarray:
    """Sum of 6-neighbour values with hard boundaries (no wrap)."""
    out = np.zeros_like(arr, dtype=np.float32)

    out[1:, :, :] += arr[:-1, :, :]
    out[:-1, :, :] += arr[1:, :, :]

    out[:, 1:, :] += arr[:, :-1, :]
    out[:, :-1, :] += arr[:, 1:, :]

    out[:, :, 1:] += arr[:, :, :-1]
    out[:, :, :-1] += arr[:, :, 1:]

    return out


def _lap6_hard(arr: np.ndarray) -> np.ndarray:
    """6-neighbour Laplacian with hard boundaries: sum(neigh) - 6*center."""
    return _sum6_hard(arr) - (6.0 * arr.astype(np.float32))


# --- Numba kernels for diffusion (private helpers) ---

@njit(parallel=True, cache=True)
def _nb_sum6_hard_into(arr: np.ndarray, out: np.ndarray) -> None:
    n0, n1, n2 = arr.shape
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                s = 0.0
                if i > 0:
                    s += float(arr[i - 1, j, k])
                if i + 1 < n0:
                    s += float(arr[i + 1, j, k])
                if j > 0:
                    s += float(arr[i, j - 1, k])
                if j + 1 < n1:
                    s += float(arr[i, j + 1, k])
                if k > 0:
                    s += float(arr[i, j, k - 1])
                if k + 1 < n2:
                    s += float(arr[i, j, k + 1])
                out[i, j, k] = s


@njit(parallel=True, cache=True)
def _nb_inject(phi: np.ndarray, src: np.ndarray, inject: float, phi_inj: np.ndarray) -> None:
    n0, n1, n2 = phi.shape
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                phi_inj[i, j, k] = float(phi[i, j, k]) + inject * float(src[i, j, k])


@njit(parallel=True, cache=True)
def _nb_diffuse_update(
    phi_inj: np.ndarray,
    neigh: np.ndarray,
    rate_rise: float,
    rate_fall: float,
    decay: float,
    phi_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi_inj.shape
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                p = float(phi_inj[i, j, k])
                target = float(neigh[i, j, k]) / 6.0
                delta = target - p
                rate = rate_rise if delta > 0.0 else rate_fall
                v = p + rate * delta
                if decay != 0.0:
                    v *= mul
                phi_next[i, j, k] = v


@njit(parallel=True, cache=True)
def _nb_diffuse_update_fused(
    phi_inj: np.ndarray,
    rate_rise: float,
    rate_fall: float,
    decay: float,
    phi_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi_inj.shape
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                s = 0.0
                if i > 0:
                    s += float(phi_inj[i - 1, j, k])
                if i + 1 < n0:
                    s += float(phi_inj[i + 1, j, k])
                if j > 0:
                    s += float(phi_inj[i, j - 1, k])
                if j + 1 < n1:
                    s += float(phi_inj[i, j + 1, k])
                if k > 0:
                    s += float(phi_inj[i, j, k - 1])
                if k + 1 < n2:
                    s += float(phi_inj[i, j, k + 1])

                p = float(phi_inj[i, j, k])
                target = s / 6.0
                delta = target - p
                rate = rate_rise if delta > 0.0 else rate_fall
                v = p + rate * delta
                if decay != 0.0:
                    v *= mul
                phi_next[i, j, k] = v


@njit(parallel=True, cache=True)
def _nb_clamp_zero_faces(arr: np.ndarray) -> None:
    n0, n1, n2 = arr.shape
    for j in prange(n1):
        for k in range(n2):
            arr[0, j, k] = 0.0
            arr[n0 - 1, j, k] = 0.0
    for i in prange(n0):
        for k in range(n2):
            arr[i, 0, k] = 0.0
            arr[i, n1 - 1, k] = 0.0
    for i in prange(n0):
        for j in range(n1):
            arr[i, j, 0] = 0.0
            arr[i, j, n2 - 1] = 0.0

# --- Neumann (zero-normal-gradient) boundary clamp ---
@njit(parallel=True, cache=True)
def _nb_clamp_neumann_faces(arr: np.ndarray) -> None:
    n0, n1, n2 = arr.shape
    # X faces
    for j in prange(n1):
        for k in range(n2):
            arr[0, j, k] = float(arr[1, j, k])
            arr[n0 - 1, j, k] = float(arr[n0 - 2, j, k])
    # Y faces
    for i in prange(n0):
        for k in range(n2):
            arr[i, 0, k] = float(arr[i, 1, k])
            arr[i, n1 - 1, k] = float(arr[i, n1 - 2, k])
    # Z faces
    for i in prange(n0):
        for j in range(n1):
            arr[i, j, 0] = float(arr[i, j, 1])
            arr[i, j, n2 - 1] = float(arr[i, j, n2 - 2])

@njit(parallel=True, cache=True)
def _nb_apply_sponge(arr: np.ndarray, width: int, strength: float, ax_x: int, ax_y: int, ax_z: int) -> None:
    """Apply a smooth absorbing sponge ramp near selected faces.

    ax_x/ax_y/ax_z are 0/1 enables for damping near the corresponding axis faces.
    This reduces boundary reflections without imposing a hard clamp.
    """
    if width <= 0 or strength <= 0.0:
        return
    if ax_x == 0 and ax_y == 0 and ax_z == 0:
        return
    n0, n1, n2 = arr.shape
    w = int(width)
    smax = float(strength)
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                d = 1 << 30
                if ax_x != 0:
                    dx0 = i
                    dx1 = n0 - 1 - i
                    if dx1 < dx0:
                        dx0 = dx1
                    if dx0 < d:
                        d = dx0
                if ax_y != 0:
                    dy0 = j
                    dy1 = n1 - 1 - j
                    if dy1 < dy0:
                        dy0 = dy1
                    if dy0 < d:
                        d = dy0
                if ax_z != 0:
                    dz0 = k
                    dz1 = n2 - 1 - k
                    if dz1 < dz0:
                        dz0 = dz1
                    if dz0 < d:
                        d = dz0
                if d >= w:
                    continue
                u = float(w - d) / float(w)
                damp = smax * (u * u)
                if damp >= 1.0:
                    arr[i, j, k] = 0.0
                elif damp > 0.0:
                    arr[i, j, k] = float(arr[i, j, k]) * (1.0 - damp)

@njit(cache=True)
def _nb_diffuse_steps(
    phi0: np.ndarray,
    src: np.ndarray,
    inject: float,
    rate_rise: float,
    rate_fall: float,
    decay: float,
    clamp_zero: int,
    clamp_neumann: int,
    sponge_width: int,
    sponge_strength: float,
    sponge_ax_x: int,
    sponge_ax_y: int,
    sponge_ax_z: int,
    steps: int,
    phi_inj: np.ndarray,
    phi_next: np.ndarray,
) -> np.ndarray:
    phi = phi0
    nxt = phi_next
    for _ in range(int(steps)):
        _nb_inject(phi, src, inject, phi_inj)
        _nb_diffuse_update_fused(phi_inj, rate_rise, rate_fall, decay, nxt)
        _nb_apply_sponge(nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        if clamp_zero != 0:
            _nb_clamp_zero_faces(nxt)
        elif clamp_neumann != 0:
            _nb_clamp_neumann_faces(nxt)
        tmp = phi
        phi = nxt
        nxt = tmp
    return phi


# --- Chiral helpers (Parity/Chirality Patch 1/3) ---

@njit(cache=True)
def _nb_chiral_weight(ch: int, select: int) -> float:
    # ch: -1 left, +1 right, 0 neutral. Neutral always couples.
    # select: 0 off/both, -1 left-only, +1 right-only
    if select == 0:
        return 1.0
    if ch == 0:
        return 1.0
    return 1.0 if ch == select else 0.0

# --- Numba kernels for telegraph (private helpers) ---

@njit(parallel=True, cache=True)
def _nb_lap6_hard_into(arr: np.ndarray, out: np.ndarray) -> None:
    n0, n1, n2 = arr.shape
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                s = 0.0
                if i > 0:
                    s += float(arr[i - 1, j, k])
                if i + 1 < n0:
                    s += float(arr[i + 1, j, k])
                if j > 0:
                    s += float(arr[i, j - 1, k])
                if j + 1 < n1:
                    s += float(arr[i, j + 1, k])
                if k > 0:
                    s += float(arr[i, j, k - 1])
                if k + 1 < n2:
                    s += float(arr[i, j, k + 1])
                out[i, j, k] = s - 6.0 * float(arr[i, j, k])


@njit(parallel=True, cache=True)
def _nb_telegraph_update(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    lap: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi.shape
    damp = 1.0 - gamma
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                v = float(vel[i, j, k]) * damp
                v += inject * float(src[i, j, k])
                v += c2 * float(lap[i, j, k])
                p = float(phi[i, j, k]) + dt * v
                if decay != 0.0:
                    p *= mul
                vel_next[i, j, k] = v
                phi_next[i, j, k] = p


@njit(parallel=True, cache=True)
def _nb_telegraph_update_fused(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi.shape
    damp = 1.0 - gamma
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                s = 0.0
                if i > 0:
                    s += float(phi[i - 1, j, k])
                if i + 1 < n0:
                    s += float(phi[i + 1, j, k])
                if j > 0:
                    s += float(phi[i, j - 1, k])
                if j + 1 < n1:
                    s += float(phi[i, j + 1, k])
                if k > 0:
                    s += float(phi[i, j, k - 1])
                if k + 1 < n2:
                    s += float(phi[i, j, k + 1])

                lap = s - 6.0 * float(phi[i, j, k])

                v = float(vel[i, j, k]) * damp
                v += inject * float(src[i, j, k])
                v += c2 * lap
                p = float(phi[i, j, k]) + dt * v
                if decay != 0.0:
                    p *= mul
                vel_next[i, j, k] = v
                phi_next[i, j, k] = p


# --- Nonlinear update kernels ---

@njit(parallel=True, cache=True)
def _nb_nonlinear_update_fused(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    k_lin: float,
    lam_cubic: float,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi.shape
    damp = 1.0 - gamma
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                s = 0.0
                if i > 0:
                    s += float(phi[i - 1, j, k])
                if i + 1 < n0:
                    s += float(phi[i + 1, j, k])
                if j > 0:
                    s += float(phi[i, j - 1, k])
                if j + 1 < n1:
                    s += float(phi[i, j + 1, k])
                if k > 0:
                    s += float(phi[i, j, k - 1])
                if k + 1 < n2:
                    s += float(phi[i, j, k + 1])

                p_c = float(phi[i, j, k])
                lap = s - 6.0 * p_c

                v = float(vel[i, j, k]) * damp

                # Force/acc terms (telegraph-style): accumulate into v, then position uses dt.
                acc = inject * float(src[i, j, k])
                acc += c2 * lap
                if k_lin != 0.0:
                    acc += -k_lin * p_c
                if lam_cubic != 0.0:
                    acc += -lam_cubic * (p_c * p_c * p_c)

                v += acc

                p = p_c + dt * v
                if decay != 0.0:
                    p *= mul

                vel_next[i, j, k] = v
                phi_next[i, j, k] = p


@njit(cache=True)
def _nb_nonlinear_steps(
    phi0: np.ndarray,
    vel0: np.ndarray,
    src: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    k_lin: float,
    lam_cubic: float,
    clamp_zero: int,
    clamp_neumann: int,
    sponge_width: int,
    sponge_strength: float,
    sponge_ax_x: int,
    sponge_ax_y: int,
    sponge_ax_z: int,
    steps: int,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = phi0
    vel = vel0
    p_nxt = phi_next
    v_nxt = vel_next
    for _ in range(int(steps)):
        _nb_nonlinear_update_fused(phi, vel, src, gamma, inject, c2, dt, decay, k_lin, lam_cubic, p_nxt, v_nxt)
        _nb_apply_sponge(p_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        _nb_apply_sponge(v_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        if clamp_zero != 0:
            _nb_clamp_zero_faces(p_nxt)
            _nb_clamp_zero_faces(v_nxt)
        elif clamp_neumann != 0:
            _nb_clamp_neumann_faces(p_nxt)
            _nb_clamp_neumann_faces(v_nxt)
        tmp = phi
        phi = p_nxt
        p_nxt = tmp
        tmpv = vel
        vel = v_nxt
        v_nxt = tmpv
    return phi, vel


# --- Sine-Gordon update kernels ---

@njit(parallel=True, cache=True)
def _nb_sine_gordon_update_fused(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    k_sg: float,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi.shape
    damp = 1.0 - gamma
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                s = 0.0
                if i > 0:
                    s += float(phi[i - 1, j, k])
                if i + 1 < n0:
                    s += float(phi[i + 1, j, k])
                if j > 0:
                    s += float(phi[i, j - 1, k])
                if j + 1 < n1:
                    s += float(phi[i, j + 1, k])
                if k > 0:
                    s += float(phi[i, j, k - 1])
                if k + 1 < n2:
                    s += float(phi[i, j, k + 1])

                p_c = float(phi[i, j, k])
                lap = s - 6.0 * p_c

                v = float(vel[i, j, k]) * damp

                # Force/acc terms (telegraph-style): accumulate into v, then position uses dt.
                acc = inject * float(src[i, j, k])
                acc += c2 * lap
                if k_sg != 0.0:
                    # Bounded restoring force: -k * sin(phi)
                    acc += -k_sg * math.sin(p_c)

                v += acc

                p = p_c + dt * v
                if decay != 0.0:
                    p *= mul

                vel_next[i, j, k] = v
                phi_next[i, j, k] = p

# --- Sine-Gordon update with spatial k-grid ---

@njit(parallel=True, cache=True)
def _nb_sine_gordon_update_fused_kgrid(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    k_grid: np.ndarray,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi.shape
    damp = 1.0 - gamma
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                s = 0.0
                if i > 0:
                    s += float(phi[i - 1, j, k])
                if i + 1 < n0:
                    s += float(phi[i + 1, j, k])
                if j > 0:
                    s += float(phi[i, j - 1, k])
                if j + 1 < n1:
                    s += float(phi[i, j + 1, k])
                if k > 0:
                    s += float(phi[i, j, k - 1])
                if k + 1 < n2:
                    s += float(phi[i, j, k + 1])

                p_c = float(phi[i, j, k])
                lap = s - 6.0 * p_c

                v = float(vel[i, j, k]) * damp

                # Force/acc terms (telegraph-style): accumulate into v, then position uses dt.
                acc = inject * float(src[i, j, k])
                acc += c2 * lap

                k_loc = float(k_grid[i, j, k])
                if k_loc != 0.0:
                    acc += -k_loc * math.sin(p_c)

                v += acc

                p = p_c + dt * v
                if decay != 0.0:
                    p *= mul

                vel_next[i, j, k] = v
                phi_next[i, j, k] = p

# --- Sine-Gordon update with domain mask (inactive cells are "outside the junction") ---
@njit(parallel=True, cache=True)
def _nb_sine_gordon_update_masked_fused(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    domain_mask: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    k_sg: float,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi.shape
    damp = 1.0 - gamma
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                if domain_mask[i, j, k] != 0:
                    vel_next[i, j, k] = 0.0
                    phi_next[i, j, k] = 0.0
                    continue

                p_c = float(phi[i, j, k])

                # Laplacian with hard boundaries, but "outside" neighbours reflect p_c (Neumann on domain edge)
                s = 0.0
                if i > 0:
                    if domain_mask[i - 1, j, k] == 0:
                        s += float(phi[i - 1, j, k])
                    else:
                        s += p_c
                if i + 1 < n0:
                    if domain_mask[i + 1, j, k] == 0:
                        s += float(phi[i + 1, j, k])
                    else:
                        s += p_c
                if j > 0:
                    if domain_mask[i, j - 1, k] == 0:
                        s += float(phi[i, j - 1, k])
                    else:
                        s += p_c
                if j + 1 < n1:
                    if domain_mask[i, j + 1, k] == 0:
                        s += float(phi[i, j + 1, k])
                    else:
                        s += p_c
                if k > 0:
                    if domain_mask[i, j, k - 1] == 0:
                        s += float(phi[i, j, k - 1])
                    else:
                        s += p_c
                if k + 1 < n2:
                    if domain_mask[i, j, k + 1] == 0:
                        s += float(phi[i, j, k + 1])
                    else:
                        s += p_c

                lap = s - 6.0 * p_c

                v = float(vel[i, j, k]) * damp

                # Force/acc terms (telegraph-style): accumulate into v, then position uses dt.
                acc = inject * float(src[i, j, k])
                acc += c2 * lap
                if k_sg != 0.0:
                    acc += -k_sg * math.sin(p_c)

                v += acc

                p = p_c + dt * v
                if decay != 0.0:
                    p *= mul

                vel_next[i, j, k] = v
                phi_next[i, j, k] = p


# --- Masked Sine-Gordon update with spatial k-grid ---
@njit(parallel=True, cache=True)
def _nb_sine_gordon_update_masked_fused_kgrid(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    domain_mask: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    k_grid: np.ndarray,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi.shape
    damp = 1.0 - gamma
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                if domain_mask[i, j, k] != 0:
                    vel_next[i, j, k] = 0.0
                    phi_next[i, j, k] = 0.0
                    continue

                p_c = float(phi[i, j, k])

                # Laplacian with domain-edge Neumann reflection.
                s = 0.0
                if i > 0:
                    if domain_mask[i - 1, j, k] == 0:
                        s += float(phi[i - 1, j, k])
                    else:
                        s += p_c
                if i + 1 < n0:
                    if domain_mask[i + 1, j, k] == 0:
                        s += float(phi[i + 1, j, k])
                    else:
                        s += p_c
                if j > 0:
                    if domain_mask[i, j - 1, k] == 0:
                        s += float(phi[i, j - 1, k])
                    else:
                        s += p_c
                if j + 1 < n1:
                    if domain_mask[i, j + 1, k] == 0:
                        s += float(phi[i, j + 1, k])
                    else:
                        s += p_c
                if k > 0:
                    if domain_mask[i, j, k - 1] == 0:
                        s += float(phi[i, j, k - 1])
                    else:
                        s += p_c
                if k + 1 < n2:
                    if domain_mask[i, j, k + 1] == 0:
                        s += float(phi[i, j, k + 1])
                    else:
                        s += p_c

                lap = s - 6.0 * p_c

                v = float(vel[i, j, k]) * damp

                acc = inject * float(src[i, j, k])
                acc += c2 * lap

                k_loc = float(k_grid[i, j, k])
                if k_loc != 0.0:
                    acc += -k_loc * math.sin(p_c)

                v += acc

                p = p_c + dt * v
                if decay != 0.0:
                    p *= mul

                vel_next[i, j, k] = v
                phi_next[i, j, k] = p


@njit(cache=True)
def _nb_sine_gordon_steps(
    phi0: np.ndarray,
    vel0: np.ndarray,
    src: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    k_sg: float,
    clamp_zero: int,
    clamp_neumann: int,
    sponge_width: int,
    sponge_strength: float,
    sponge_ax_x: int,
    sponge_ax_y: int,
    sponge_ax_z: int,
    steps: int,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = phi0
    vel = vel0
    p_nxt = phi_next
    v_nxt = vel_next
    for _ in range(int(steps)):
        _nb_sine_gordon_update_fused(phi, vel, src, gamma, inject, c2, dt, decay, k_sg, p_nxt, v_nxt)
        _nb_apply_sponge(p_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        _nb_apply_sponge(v_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        if clamp_zero != 0:
            _nb_clamp_zero_faces(p_nxt)
            _nb_clamp_zero_faces(v_nxt)
        elif clamp_neumann != 0:
            _nb_clamp_neumann_faces(p_nxt)
            _nb_clamp_neumann_faces(v_nxt)
        tmp = phi
        phi = p_nxt
        p_nxt = tmp
        tmpv = vel
        vel = v_nxt
        v_nxt = tmpv
    return phi, vel

# --- Sine-Gordon step-loop with spatial k-grid ---

@njit(cache=True)
def _nb_sine_gordon_steps_kgrid(
    phi0: np.ndarray,
    vel0: np.ndarray,
    src: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    k_grid: np.ndarray,
    clamp_zero: int,
    clamp_neumann: int,
    sponge_width: int,
    sponge_strength: float,
    sponge_ax_x: int,
    sponge_ax_y: int,
    sponge_ax_z: int,
    steps: int,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = phi0
    vel = vel0
    p_nxt = phi_next
    v_nxt = vel_next
    for _ in range(int(steps)):
        _nb_sine_gordon_update_fused_kgrid(phi, vel, src, gamma, inject, c2, dt, decay, k_grid, p_nxt, v_nxt)
        _nb_apply_sponge(p_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        _nb_apply_sponge(v_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        if clamp_zero != 0:
            _nb_clamp_zero_faces(p_nxt)
            _nb_clamp_zero_faces(v_nxt)
        elif clamp_neumann != 0:
            _nb_clamp_neumann_faces(p_nxt)
            _nb_clamp_neumann_faces(v_nxt)
        tmp = phi
        phi = p_nxt
        p_nxt = tmp
        tmpv = vel
        vel = v_nxt
        v_nxt = tmpv
    return phi, vel

# --- Masked sine-gordon step-loop (domain mask) ---
@njit(cache=True)
def _nb_sine_gordon_steps_masked(
    phi0: np.ndarray,
    vel0: np.ndarray,
    src: np.ndarray,
    domain_mask: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    k_sg: float,
    clamp_zero: int,
    clamp_neumann: int,
    sponge_width: int,
    sponge_strength: float,
    sponge_ax_x: int,
    sponge_ax_y: int,
    sponge_ax_z: int,
    steps: int,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = phi0
    vel = vel0
    p_nxt = phi_next
    v_nxt = vel_next
    for _ in range(int(steps)):
        _nb_sine_gordon_update_masked_fused(
            phi,
            vel,
            src,
            domain_mask,
            gamma,
            inject,
            c2,
            dt,
            decay,
            k_sg,
            p_nxt,
            v_nxt,
        )
        _nb_apply_sponge(p_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        _nb_apply_sponge(v_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        if clamp_zero != 0:
            _nb_clamp_zero_faces(p_nxt)
            _nb_clamp_zero_faces(v_nxt)
        elif clamp_neumann != 0:
            _nb_clamp_neumann_faces(p_nxt)
            _nb_clamp_neumann_faces(v_nxt)
        tmp = phi
        phi = p_nxt
        p_nxt = tmp
        tmpv = vel
        vel = v_nxt
        v_nxt = tmpv
    return phi, vel


# --- Masked sine-gordon step-loop (domain mask + k-grid) ---
@njit(cache=True)
def _nb_sine_gordon_steps_masked_kgrid(
    phi0: np.ndarray,
    vel0: np.ndarray,
    src: np.ndarray,
    domain_mask: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    k_grid: np.ndarray,
    clamp_zero: int,
    clamp_neumann: int,
    sponge_width: int,
    sponge_strength: float,
    sponge_ax_x: int,
    sponge_ax_y: int,
    sponge_ax_z: int,
    steps: int,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = phi0
    vel = vel0
    p_nxt = phi_next
    v_nxt = vel_next
    for _ in range(int(steps)):
        _nb_sine_gordon_update_masked_fused_kgrid(
            phi,
            vel,
            src,
            domain_mask,
            gamma,
            inject,
            c2,
            dt,
            decay,
            k_grid,
            p_nxt,
            v_nxt,
        )
        _nb_apply_sponge(p_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        _nb_apply_sponge(v_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        if clamp_zero != 0:
            _nb_clamp_zero_faces(p_nxt)
            _nb_clamp_zero_faces(v_nxt)
        elif clamp_neumann != 0:
            _nb_clamp_neumann_faces(p_nxt)
            _nb_clamp_neumann_faces(v_nxt)
        tmp = phi
        phi = p_nxt
        p_nxt = tmp
        tmpv = vel
        vel = v_nxt
        v_nxt = tmpv
    return phi, vel


# --- Fused chiral telegraph update (unmasked) ---
@njit(parallel=True, cache=True)
def _nb_telegraph_update_fused_chiral(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    chirality: np.ndarray,
    chiral_select: int,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi.shape
    damp = 1.0 - gamma
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                s = 0.0
                if i > 0:
                    s += float(phi[i - 1, j, k])
                if i + 1 < n0:
                    s += float(phi[i + 1, j, k])
                if j > 0:
                    s += float(phi[i, j - 1, k])
                if j + 1 < n1:
                    s += float(phi[i, j + 1, k])
                if k > 0:
                    s += float(phi[i, j, k - 1])
                if k + 1 < n2:
                    s += float(phi[i, j, k + 1])

                lap = s - 6.0 * float(phi[i, j, k])

                w = _nb_chiral_weight(int(chirality[i, j, k]), int(chiral_select))

                v = float(vel[i, j, k]) * damp
                v += (inject * w) * float(src[i, j, k])
                v += c2 * lap

                p = float(phi[i, j, k]) + dt * v
                if decay != 0.0:
                    p *= mul

                vel_next[i, j, k] = v
                phi_next[i, j, k] = p


# Masked version: Dirichlet walls (phi=0, vel=0 where mask!=0)
@njit(parallel=True, cache=True)
def _nb_telegraph_update_masked_fused(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    mask: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi.shape
    damp = 1.0 - gamma
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                if mask[i, j, k] != 0:
                    vel_next[i, j, k] = 0.0
                    phi_next[i, j, k] = 0.0
                    continue

                s = 0.0
                if i > 0 and mask[i - 1, j, k] == 0:
                    s += float(phi[i - 1, j, k])
                if i + 1 < n0 and mask[i + 1, j, k] == 0:
                    s += float(phi[i + 1, j, k])
                if j > 0 and mask[i, j - 1, k] == 0:
                    s += float(phi[i, j - 1, k])
                if j + 1 < n1 and mask[i, j + 1, k] == 0:
                    s += float(phi[i, j + 1, k])
                if k > 0 and mask[i, j, k - 1] == 0:
                    s += float(phi[i, j, k - 1])
                if k + 1 < n2 and mask[i, j, k + 1] == 0:
                    s += float(phi[i, j, k + 1])

                lap = s - 6.0 * float(phi[i, j, k])

                v = float(vel[i, j, k]) * damp
                v += inject * float(src[i, j, k])
                v += c2 * lap
                p = float(phi[i, j, k]) + dt * v
                if decay != 0.0:
                    p *= mul
                vel_next[i, j, k] = v
                phi_next[i, j, k] = p


# --- Fused chiral telegraph update (masked) ---
@njit(parallel=True, cache=True)
def _nb_telegraph_update_masked_fused_chiral(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    mask: np.ndarray,
    chirality: np.ndarray,
    chiral_select: int,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> None:
    n0, n1, n2 = phi.shape
    damp = 1.0 - gamma
    mul = 1.0 - decay
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                if mask[i, j, k] != 0:
                    vel_next[i, j, k] = 0.0
                    phi_next[i, j, k] = 0.0
                    continue

                s = 0.0
                if i > 0 and mask[i - 1, j, k] == 0:
                    s += float(phi[i - 1, j, k])
                if i + 1 < n0 and mask[i + 1, j, k] == 0:
                    s += float(phi[i + 1, j, k])
                if j > 0 and mask[i, j - 1, k] == 0:
                    s += float(phi[i, j - 1, k])
                if j + 1 < n1 and mask[i, j + 1, k] == 0:
                    s += float(phi[i, j + 1, k])
                if k > 0 and mask[i, j, k - 1] == 0:
                    s += float(phi[i, j, k - 1])
                if k + 1 < n2 and mask[i, j, k + 1] == 0:
                    s += float(phi[i, j, k + 1])

                lap = s - 6.0 * float(phi[i, j, k])

                w = _nb_chiral_weight(int(chirality[i, j, k]), int(chiral_select))

                v = float(vel[i, j, k]) * damp
                v += (inject * w) * float(src[i, j, k])
                v += c2 * lap

                p = float(phi[i, j, k]) + dt * v
                if decay != 0.0:
                    p *= mul

                vel_next[i, j, k] = v
                phi_next[i, j, k] = p


@njit(cache=True)
def _nb_telegraph_steps(
    phi0: np.ndarray,
    vel0: np.ndarray,
    src: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    clamp_zero: int,
    clamp_neumann: int,
    sponge_width: int,
    sponge_strength: float,
    sponge_ax_x: int,
    sponge_ax_y: int,
    sponge_ax_z: int,
    steps: int,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = phi0
    vel = vel0
    p_nxt = phi_next
    v_nxt = vel_next
    for _ in range(int(steps)):
        _nb_telegraph_update_fused(phi, vel, src, gamma, inject, c2, dt, decay, p_nxt, v_nxt)
        _nb_apply_sponge(p_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        _nb_apply_sponge(v_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        if clamp_zero != 0:
            _nb_clamp_zero_faces(p_nxt)
            _nb_clamp_zero_faces(v_nxt)
        elif clamp_neumann != 0:
            _nb_clamp_neumann_faces(p_nxt)
            _nb_clamp_neumann_faces(v_nxt)
        tmp = phi
        phi = p_nxt
        p_nxt = tmp
        tmpv = vel
        vel = v_nxt
        v_nxt = tmpv
    return phi, vel


# --- Parity/Chirality Patch 2/3: Chiral step-loop (unmasked) ---
@njit(cache=True)
def _nb_telegraph_steps_chiral(
    phi0: np.ndarray,
    vel0: np.ndarray,
    src: np.ndarray,
    chirality: np.ndarray,
    chiral_select: int,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    clamp_zero: int,
    clamp_neumann: int,
    sponge_width: int,
    sponge_strength: float,
    sponge_ax_x: int,
    sponge_ax_y: int,
    sponge_ax_z: int,
    steps: int,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = phi0
    vel = vel0
    p_nxt = phi_next
    v_nxt = vel_next
    for _ in range(int(steps)):
        _nb_telegraph_update_fused_chiral(
            phi,
            vel,
            src,
            chirality,
            int(chiral_select),
            gamma,
            inject,
            c2,
            dt,
            decay,
            p_nxt,
            v_nxt,
        )
        _nb_apply_sponge(p_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        _nb_apply_sponge(v_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        if clamp_zero != 0:
            _nb_clamp_zero_faces(p_nxt)
            _nb_clamp_zero_faces(v_nxt)
        elif clamp_neumann != 0:
            _nb_clamp_neumann_faces(p_nxt)
            _nb_clamp_neumann_faces(v_nxt)
        tmp = phi
        phi = p_nxt
        p_nxt = tmp
        tmpv = vel
        vel = v_nxt
        v_nxt = tmpv
    return phi, vel


# Masked version
@njit(cache=True)
def _nb_telegraph_steps_masked(
    phi0: np.ndarray,
    vel0: np.ndarray,
    src: np.ndarray,
    mask: np.ndarray,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    clamp_zero: int,
    clamp_neumann: int,
    sponge_width: int,
    sponge_strength: float,
    sponge_ax_x: int,
    sponge_ax_y: int,
    sponge_ax_z: int,
    steps: int,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = phi0
    vel = vel0
    p_nxt = phi_next
    v_nxt = vel_next
    for _ in range(int(steps)):
        _nb_telegraph_update_masked_fused(phi, vel, src, mask, gamma, inject, c2, dt, decay, p_nxt, v_nxt)
        _nb_apply_sponge(p_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        _nb_apply_sponge(v_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        if clamp_zero != 0:
            _nb_clamp_zero_faces(p_nxt)
            _nb_clamp_zero_faces(v_nxt)
        elif clamp_neumann != 0:
            _nb_clamp_neumann_faces(p_nxt)
            _nb_clamp_neumann_faces(v_nxt)
        tmp = phi
        phi = p_nxt
        p_nxt = tmp
        tmpv = vel
        vel = v_nxt
        v_nxt = tmpv
    return phi, vel


# --- Parity/Chirality: Chiral step-loop (masked) ---
@njit(cache=True)
def _nb_telegraph_steps_masked_chiral(
    phi0: np.ndarray,
    vel0: np.ndarray,
    src: np.ndarray,
    mask: np.ndarray,
    chirality: np.ndarray,
    chiral_select: int,
    gamma: float,
    inject: float,
    c2: float,
    dt: float,
    decay: float,
    clamp_zero: int,
    clamp_neumann: int,
    sponge_width: int,
    sponge_strength: float,
    sponge_ax_x: int,
    sponge_ax_y: int,
    sponge_ax_z: int,
    steps: int,
    phi_next: np.ndarray,
    vel_next: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    phi = phi0
    vel = vel0
    p_nxt = phi_next
    v_nxt = vel_next
    for _ in range(int(steps)):
        _nb_telegraph_update_masked_fused_chiral(
            phi,
            vel,
            src,
            mask,
            chirality,
            int(chiral_select),
            gamma,
            inject,
            c2,
            dt,
            decay,
            p_nxt,
            v_nxt,
        )
        _nb_apply_sponge(p_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        _nb_apply_sponge(v_nxt, int(sponge_width), float(sponge_strength), int(sponge_ax_x), int(sponge_ax_y), int(sponge_ax_z))
        if clamp_zero != 0:
            _nb_clamp_zero_faces(p_nxt)
            _nb_clamp_zero_faces(v_nxt)
        elif clamp_neumann != 0:
            _nb_clamp_neumann_faces(p_nxt)
            _nb_clamp_neumann_faces(v_nxt)
        tmp = phi
        phi = p_nxt
        p_nxt = tmp
        tmpv = vel
        vel = v_nxt
        v_nxt = tmpv
    return phi, vel


def evolve_diffusion_traffic(
    load: np.ndarray,
    tp: TrafficParams,
    progress_cb: Callable[[int], None] | None = None,
    state_cb: Optional[Callable[[np.ndarray, int], None]] = None,
) -> np.ndarray:
    """Evolve a congestion/traffic field driven by the load source.

    Mechanism (single-path):
      - Each tick injects `inject * load` into the field.
      - The field diffuses by mixing towards the 6-neighbour average.
      - Optional decay bleeds traffic out over time.
      - Boundary handling is either a hard clamp (faces=0) or a sponge layer.

    With hard boundaries and sustained injection, the steady state reproduces
    the familiar ~1/r behaviour around compact sources in 3D without solving
    Poisson explicitly.
    """
    if load.ndim != 3:
        raise ValueError("load must be 3D")

    if not (0.0 <= float(tp.rate_rise) <= 1.0):
        raise ValueError("TrafficParams.rate_rise must be in [0,1]")

    if not (0.0 <= float(tp.rate_fall) <= 1.0):
        raise ValueError("TrafficParams.rate_fall must be in [0,1]")

    if not (0.0 <= float(tp.decay) < 1.0):
        raise ValueError("TrafficParams.decay must be in [0,1)")

    bm = str(tp.boundary_mode).strip().lower()
    if bm not in ("zero", "sponge", "neumann", "open"):
        raise ValueError("TrafficParams.boundary_mode must be one of: zero, sponge, neumann, open")
    sponge_w = int(tp.sponge_width)
    sponge_s = float(tp.sponge_strength)
    sponge_axes = str(getattr(tp, "sponge_axes", "xyz")).strip().lower()
    if bm == "sponge":
        if sponge_w <= 0:
            raise ValueError("TrafficParams.sponge_width must be > 0 when boundary_mode=sponge")
        if not (0.0 < sponge_s <= 1.0):
            raise ValueError("TrafficParams.sponge_strength must be in (0,1] when boundary_mode=sponge")
        if sponge_axes == "":
            raise ValueError("TrafficParams.sponge_axes must be non-empty when boundary_mode=sponge")
        for ch in sponge_axes:
            if ch not in ("x", "y", "z"):
                raise ValueError("TrafficParams.sponge_axes must contain only x/y/z")
        ax_x = 1 if ("x" in sponge_axes) else 0
        ax_y = 1 if ("y" in sponge_axes) else 0
        ax_z = 1 if ("z" in sponge_axes) else 0
        clamp_zero = 0
        clamp_neumann = 0
    else:
        if sponge_w != 0 or sponge_s != 0.0 or sponge_axes not in ("", "xyz"):
            raise ValueError("sponge params must be zero/default when boundary_mode!=sponge")
        ax_x = 0
        ax_y = 0
        ax_z = 0
        if bm == "zero":
            clamp_zero = 1
            clamp_neumann = 0
        elif bm == "neumann":
            clamp_zero = 0
            clamp_neumann = 1
        else:
            # open
            clamp_zero = 0
            clamp_neumann = 0

    phi0 = np.zeros_like(load, dtype=np.float32)
    src = load.astype(np.float32)

    phi_inj = np.empty_like(phi0)
    phi_next = np.empty_like(phi0)

    if progress_cb is None:
        phi = _nb_diffuse_steps(
            np.ascontiguousarray(phi0),
            np.ascontiguousarray(src),
            float(tp.inject),
            float(tp.rate_rise),
            float(tp.rate_fall),
            float(tp.decay),
            int(clamp_zero),
            int(clamp_neumann),
            int(sponge_w),
            float(sponge_s),
            int(ax_x), int(ax_y), int(ax_z),
            int(tp.iters),
            phi_inj,
            phi_next,
        )
        if state_cb is not None:
            try:
                state_cb(phi, int(tp.iters))
            except Exception:
                pass
        _assert_finite(phi, "phi")
        return phi

    phi = np.ascontiguousarray(phi0)
    src0 = np.ascontiguousarray(src)
    total = int(tp.iters)
    # Smaller chunks when streaming state (live viewer) so updates are frequent.
    chunk = 16 if state_cb is not None else 512
    done = 0
    while done < total:
        steps = chunk
        if done + steps > total:
            steps = total - done
        phi = _nb_diffuse_steps(
            phi,
            src0,
            float(tp.inject),
            float(tp.rate_rise),
            float(tp.rate_fall),
            float(tp.decay),
            int(clamp_zero),
            int(clamp_neumann),
            int(sponge_w),
            float(sponge_s),
            int(ax_x), int(ax_y), int(ax_z),
            int(steps),
            phi_inj,
            phi_next,
        )
        done += int(steps)
        progress_cb(int(steps))
        if state_cb is not None:
            state_cb(phi, int(done))

    _assert_finite(phi, "phi")
    return phi


def evolve_telegraph_traffic(
    load: np.ndarray,
    tp: TrafficParams,
    progress_cb: Callable[[int], None] | None = None,
    state_cb: Optional[Callable[[np.ndarray, int], None]] = None,
) -> np.ndarray:
    """Damped second-order field evolution (telegraph / wave surrogate)."""
    if load.ndim != 3:
        raise ValueError("load must be 3D")

    if not (0.0 <= float(tp.gamma) < 1.0):
        raise ValueError("TrafficParams.gamma must be in [0,1)")

    if not (0.0 <= float(tp.decay) < 1.0):
        raise ValueError("TrafficParams.decay must be in [0,1)")

    if not math.isfinite(float(tp.c2)) or float(tp.c2) < 0.0:
        raise ValueError("TrafficParams.c2 must be finite and >= 0")

    if not math.isfinite(float(tp.dt)) or float(tp.dt) <= 0.0:
        raise ValueError("TrafficParams.dt must be finite and > 0")
    if float(tp.c2) > 0.0:
        cfl = math.sqrt(float(tp.c2)) * float(tp.dt)
        # Empirical stability guard for this discrete-time scheme (not a strict PDE CFL condition).
        limit = (1.0 / math.sqrt(3.0)) * 0.98
        if not math.isfinite(cfl) or cfl >= limit:
            raise ValueError(
                "unstable telegraph params: sqrt(c2)*dt=%.6g must be < %.6g (reduce c2 or dt)" % (cfl, limit)
            )

    bm = str(tp.boundary_mode).strip().lower()
    if bm not in ("zero", "sponge", "neumann", "open"):
        raise ValueError("TrafficParams.boundary_mode must be one of: zero, sponge, neumann, open")
    sponge_w = int(tp.sponge_width)
    sponge_s = float(tp.sponge_strength)
    sponge_axes = str(getattr(tp, "sponge_axes", "xyz")).strip().lower()
    if bm == "sponge":
        if sponge_w <= 0:
            raise ValueError("TrafficParams.sponge_width must be > 0 when boundary_mode=sponge")
        if not (0.0 < sponge_s <= 1.0):
            raise ValueError("TrafficParams.sponge_strength must be in (0,1] when boundary_mode=sponge")
        if sponge_axes == "":
            raise ValueError("TrafficParams.sponge_axes must be non-empty when boundary_mode=sponge")
        for ch in sponge_axes:
            if ch not in ("x", "y", "z"):
                raise ValueError("TrafficParams.sponge_axes must contain only x/y/z")
        ax_x = 1 if ("x" in sponge_axes) else 0
        ax_y = 1 if ("y" in sponge_axes) else 0
        ax_z = 1 if ("z" in sponge_axes) else 0
        clamp_zero = 0
        clamp_neumann = 0
    else:
        if sponge_w != 0 or sponge_s != 0.0 or sponge_axes not in ("", "xyz"):
            raise ValueError("sponge params must be zero/default when boundary_mode!=sponge")
        ax_x = 0
        ax_y = 0
        ax_z = 0
        if bm == "zero":
            clamp_zero = 1
            clamp_neumann = 0
        elif bm == "neumann":
            clamp_zero = 0
            clamp_neumann = 1
        else:
            # open
            clamp_zero = 0
            clamp_neumann = 0

    phi0 = np.zeros_like(load, dtype=np.float32)
    vel0 = np.zeros_like(load, dtype=np.float32)
    src = load.astype(np.float32)

    phi_next = np.empty_like(phi0)
    vel_next = np.empty_like(phi0)

    if progress_cb is None:
        phi, vel = _nb_telegraph_steps(
            np.ascontiguousarray(phi0),
            np.ascontiguousarray(vel0),
            np.ascontiguousarray(src),
            float(tp.gamma),
            float(tp.inject),
            float(tp.c2),
            float(tp.dt),
            float(tp.decay),
            int(clamp_zero),
            int(clamp_neumann),
            int(sponge_w),
            float(sponge_s),
            int(ax_x), int(ax_y), int(ax_z),
            int(tp.iters),
            phi_next,
            vel_next,
        )
        if state_cb is not None:
            try:
                state_cb(phi, int(tp.iters))
            except Exception:
                pass
        _assert_finite(phi, "phi")
        _assert_finite(vel, "vel")
        return phi

    phi = np.ascontiguousarray(phi0)
    vel = np.ascontiguousarray(vel0)
    src0 = np.ascontiguousarray(src)
    total = int(tp.iters)
    # Smaller chunks when streaming state (live viewer) so updates are frequent.
    chunk = 8 if state_cb is not None else 256
    done = 0
    while done < total:
        steps = chunk
        if done + steps > total:
            steps = total - done
        phi, vel = _nb_telegraph_steps(
            phi,
            vel,
            src0,
            float(tp.gamma),
            float(tp.inject),
            float(tp.c2),
            float(tp.dt),
            float(tp.decay),
            int(clamp_zero),
            int(clamp_neumann),
            int(sponge_w),
            float(sponge_s),
            int(ax_x), int(ax_y), int(ax_z),
            int(steps),
            phi_next,
            vel_next,
        )
        done += int(steps)
        progress_cb(int(steps))
        if state_cb is not None:
            state_cb(phi, int(done))

    _assert_finite(phi, "phi")
    _assert_finite(vel, "vel")
    return phi


def evolve_nonlinear_traffic(
    load: np.ndarray,
    tp: TrafficParams,
    progress_cb: Callable[[int], None] | None = None,
    state_cb: Optional[Callable[[np.ndarray, int], None]] = None,
) -> np.ndarray:
    """Nonlinear telegraph evolution (Klein–Gordon + stiffening cubic term)."""
    if load.ndim != 3:
        raise ValueError("load must be 3D")

    if not (0.0 <= float(tp.gamma) < 1.0):
        raise ValueError("TrafficParams.gamma must be in [0,1)")

    if not (0.0 <= float(tp.decay) < 1.0):
        raise ValueError("TrafficParams.decay must be in [0,1)")

    if not math.isfinite(float(tp.c2)) or float(tp.c2) < 0.0:
        raise ValueError("TrafficParams.c2 must be finite and >= 0")

    if not math.isfinite(float(tp.dt)) or float(tp.dt) <= 0.0:
        raise ValueError("TrafficParams.dt must be finite and > 0")
    if float(tp.c2) > 0.0:
        cfl = math.sqrt(float(tp.c2)) * float(tp.dt)
        # Empirical stability guard for this discrete-time scheme (not a strict PDE CFL condition).
        limit = (1.0 / math.sqrt(3.0)) * 0.98
        if not math.isfinite(cfl) or cfl >= limit:
            raise ValueError(
                "unstable telegraph params: sqrt(c2)*dt=%.6g must be < %.6g (reduce c2 or dt)" % (cfl, limit)
            )

    k_lin = float(tp.traffic_k)
    lam_cubic = float(tp.traffic_lambda)
    if not math.isfinite(k_lin):
        raise ValueError("TrafficParams.traffic_k must be finite")
    if not math.isfinite(lam_cubic) or lam_cubic < 0.0:
        raise ValueError("TrafficParams.traffic_lambda must be finite and >= 0")

    bm = str(tp.boundary_mode).strip().lower()
    if bm not in ("zero", "sponge", "neumann", "open"):
        raise ValueError("TrafficParams.boundary_mode must be one of: zero, sponge, neumann, open")
    sponge_w = int(tp.sponge_width)
    sponge_s = float(tp.sponge_strength)
    sponge_axes = str(getattr(tp, "sponge_axes", "xyz")).strip().lower()
    if bm == "sponge":
        if sponge_w <= 0:
            raise ValueError("TrafficParams.sponge_width must be > 0 when boundary_mode=sponge")
        if not (0.0 < sponge_s <= 1.0):
            raise ValueError("TrafficParams.sponge_strength must be in (0,1] when boundary_mode=sponge")
        if sponge_axes == "":
            raise ValueError("TrafficParams.sponge_axes must be non-empty when boundary_mode=sponge")
        for ch in sponge_axes:
            if ch not in ("x", "y", "z"):
                raise ValueError("TrafficParams.sponge_axes must contain only x/y/z")
        ax_x = 1 if ("x" in sponge_axes) else 0
        ax_y = 1 if ("y" in sponge_axes) else 0
        ax_z = 1 if ("z" in sponge_axes) else 0
        clamp_zero = 0
        clamp_neumann = 0
    else:
        if sponge_w != 0 or sponge_s != 0.0 or sponge_axes not in ("", "xyz"):
            raise ValueError("sponge params must be zero/default when boundary_mode!=sponge")
        ax_x = 0
        ax_y = 0
        ax_z = 0
        if bm == "zero":
            clamp_zero = 1
            clamp_neumann = 0
        elif bm == "neumann":
            clamp_zero = 0
            clamp_neumann = 1
        else:
            # open
            clamp_zero = 0
            clamp_neumann = 0

    phi0 = np.zeros_like(load, dtype=np.float32)
    vel0 = np.zeros_like(load, dtype=np.float32)
    src = load.astype(np.float32)

    phi_next = np.empty_like(phi0)
    vel_next = np.empty_like(phi0)

    if progress_cb is None:
        phi, vel = _nb_nonlinear_steps(
            np.ascontiguousarray(phi0),
            np.ascontiguousarray(vel0),
            np.ascontiguousarray(src),
            float(tp.gamma),
            float(tp.inject),
            float(tp.c2),
            float(tp.dt),
            float(tp.decay),
            float(k_lin),
            float(lam_cubic),
            int(clamp_zero),
            int(clamp_neumann),
            int(sponge_w),
            float(sponge_s),
            int(ax_x), int(ax_y), int(ax_z),
            int(tp.iters),
            phi_next,
            vel_next,
        )
        if state_cb is not None:
            try:
                state_cb(phi, int(tp.iters))
            except Exception:
                pass
        _assert_finite(phi, "phi")
        _assert_finite(vel, "vel")
        return phi

    phi = np.ascontiguousarray(phi0)
    vel = np.ascontiguousarray(vel0)
    src0 = np.ascontiguousarray(src)
    total = int(tp.iters)
    # Smaller chunks when streaming state (live viewer) so updates are frequent.
    chunk = 8 if state_cb is not None else 256
    done = 0
    while done < total:
        steps = chunk
        if done + steps > total:
            steps = total - done
        phi, vel = _nb_nonlinear_steps(
            phi,
            vel,
            src0,
            float(tp.gamma),
            float(tp.inject),
            float(tp.c2),
            float(tp.dt),
            float(tp.decay),
            float(k_lin),
            float(lam_cubic),
            int(clamp_zero),
            int(clamp_neumann),
            int(sponge_w),
            float(sponge_s),
            int(ax_x), int(ax_y), int(ax_z),
            int(steps),
            phi_next,
            vel_next,
        )
        done += int(steps)
        progress_cb(int(steps))
        if state_cb is not None:
            state_cb(phi, int(done))

    _assert_finite(phi, "phi")
    _assert_finite(vel, "vel")
    return phi


def evolve_nonlinear_traffic_steps(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    tp: TrafficParams,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance (phi, vel) by `steps` ticks using the nonlinear telegraph update."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if phi.shape != src.shape or vel.shape != src.shape:
        raise ValueError("phi/vel/src shape mismatch")

    m = str(tp.mode).strip().lower()
    if m != "nonlinear":
        raise ValueError(f"evolve_nonlinear_traffic_steps requires TrafficParams.mode=nonlinear (got {m!r})")

    if not (0.0 <= float(tp.gamma) < 1.0):
        raise ValueError("TrafficParams.gamma must be in [0,1)")

    if not (0.0 <= float(tp.decay) < 1.0):
        raise ValueError("TrafficParams.decay must be in [0,1)")

    if not math.isfinite(float(tp.c2)) or float(tp.c2) < 0.0:
        raise ValueError("TrafficParams.c2 must be finite and >= 0")

    if not math.isfinite(float(tp.dt)) or float(tp.dt) <= 0.0:
        raise ValueError("TrafficParams.dt must be finite and > 0")
    if float(tp.c2) > 0.0:
        cfl = math.sqrt(float(tp.c2)) * float(tp.dt)
        # Empirical stability guard for this discrete-time scheme (not a strict PDE CFL condition).
        limit = (1.0 / math.sqrt(3.0)) * 0.98
        if not math.isfinite(cfl) or cfl >= limit:
            raise ValueError(
                "unstable telegraph params: sqrt(c2)*dt=%.6g must be < %.6g (reduce c2 or dt)" % (cfl, limit)
            )

    k_lin = float(tp.traffic_k)
    lam_cubic = float(tp.traffic_lambda)
    if not math.isfinite(k_lin):
        raise ValueError("TrafficParams.traffic_k must be finite")
    if not math.isfinite(lam_cubic) or lam_cubic < 0.0:
        raise ValueError("TrafficParams.traffic_lambda must be finite and >= 0")

    bm = str(tp.boundary_mode).strip().lower()
    if bm not in ("zero", "sponge", "neumann", "open"):
        raise ValueError("TrafficParams.boundary_mode must be one of: zero, sponge, neumann, open")
    sponge_w = int(tp.sponge_width)
    sponge_s = float(tp.sponge_strength)
    sponge_axes = str(getattr(tp, "sponge_axes", "xyz")).strip().lower()
    if bm == "sponge":
        if sponge_w <= 0:
            raise ValueError("TrafficParams.sponge_width must be > 0 when boundary_mode=sponge")
        if not (0.0 < sponge_s <= 1.0):
            raise ValueError("TrafficParams.sponge_strength must be in (0,1] when boundary_mode=sponge")
        if sponge_axes == "":
            raise ValueError("TrafficParams.sponge_axes must be non-empty when boundary_mode=sponge")
        for ch in sponge_axes:
            if ch not in ("x", "y", "z"):
                raise ValueError("TrafficParams.sponge_axes must contain only x/y/z")
        ax_x = 1 if ("x" in sponge_axes) else 0
        ax_y = 1 if ("y" in sponge_axes) else 0
        ax_z = 1 if ("z" in sponge_axes) else 0
        clamp_zero = 0
        clamp_neumann = 0
    else:
        if sponge_w != 0 or sponge_s != 0.0 or sponge_axes not in ("", "xyz"):
            raise ValueError("sponge params must be zero/default when boundary_mode!=sponge")
        ax_x = 0
        ax_y = 0
        ax_z = 0
        if bm == "zero":
            clamp_zero = 1
            clamp_neumann = 0
        elif bm == "neumann":
            clamp_zero = 0
            clamp_neumann = 1
        else:
            # open
            clamp_zero = 0
            clamp_neumann = 0

    phi0 = np.ascontiguousarray(phi, dtype=np.float32)
    vel0 = np.ascontiguousarray(vel, dtype=np.float32)
    src0 = np.ascontiguousarray(src, dtype=np.float32)

    phi_next = np.empty_like(phi0)
    vel_next = np.empty_like(phi0)

    phi_out, vel_out = _nb_nonlinear_steps(
        phi0,
        vel0,
        src0,
        float(tp.gamma),
        float(tp.inject),
        float(tp.c2),
        float(tp.dt),
        float(tp.decay),
        float(k_lin),
        float(lam_cubic),
        int(clamp_zero),
        int(clamp_neumann),
        int(sponge_w),
        float(sponge_s),
        int(ax_x), int(ax_y), int(ax_z),
        int(steps),
        phi_next,
        vel_next,
    )

    _assert_finite(phi_out, "phi")
    _assert_finite(vel_out, "vel")
    return phi_out, vel_out


# --- Sine-Gordon wrappers ---

def evolve_sine_gordon_traffic(
    load: np.ndarray,
    tp: TrafficParams,
    progress_cb: Callable[[int], None] | None = None,
    state_cb: Optional[Callable[[np.ndarray, int], None]] = None,
) -> np.ndarray:
    """Sine-Gordon telegraph evolution (bounded restoring force: -k*sin(phi))."""
    if load.ndim != 3:
        raise ValueError("load must be 3D")

    if not (0.0 <= float(tp.gamma) < 1.0):
        raise ValueError("TrafficParams.gamma must be in [0,1)")

    if not (0.0 <= float(tp.decay) < 1.0):
        raise ValueError("TrafficParams.decay must be in [0,1)")

    if not math.isfinite(float(tp.c2)) or float(tp.c2) < 0.0:
        raise ValueError("TrafficParams.c2 must be finite and >= 0")

    if not math.isfinite(float(tp.dt)) or float(tp.dt) <= 0.0:
        raise ValueError("TrafficParams.dt must be finite and > 0")
    if float(tp.c2) > 0.0:
        cfl = math.sqrt(float(tp.c2)) * float(tp.dt)
        # Empirical stability guard for this discrete-time scheme (not a strict PDE CFL condition).
        limit = (1.0 / math.sqrt(3.0)) * 0.98
        if not math.isfinite(cfl) or cfl >= limit:
            raise ValueError(
                "unstable telegraph params: sqrt(c2)*dt=%.6g must be < %.6g (reduce c2 or dt)" % (cfl, limit)
            )

    k_sg = float(tp.traffic_k)
    if not math.isfinite(k_sg) or k_sg < 0.0:
        raise ValueError("TrafficParams.traffic_k must be finite and >= 0 for sine_gordon")

    bm = str(tp.boundary_mode).strip().lower()
    if bm not in ("zero", "sponge", "neumann", "open"):
        raise ValueError("TrafficParams.boundary_mode must be one of: zero, sponge, neumann, open")
    sponge_w = int(tp.sponge_width)
    sponge_s = float(tp.sponge_strength)
    sponge_axes = str(getattr(tp, "sponge_axes", "xyz")).strip().lower()
    if bm == "sponge":
        if sponge_w <= 0:
            raise ValueError("TrafficParams.sponge_width must be > 0 when boundary_mode=sponge")
        if not (0.0 < sponge_s <= 1.0):
            raise ValueError("TrafficParams.sponge_strength must be in (0,1] when boundary_mode=sponge")
        if sponge_axes == "":
            raise ValueError("TrafficParams.sponge_axes must be non-empty when boundary_mode=sponge")
        for ch in sponge_axes:
            if ch not in ("x", "y", "z"):
                raise ValueError("TrafficParams.sponge_axes must contain only x/y/z")
        ax_x = 1 if ("x" in sponge_axes) else 0
        ax_y = 1 if ("y" in sponge_axes) else 0
        ax_z = 1 if ("z" in sponge_axes) else 0
        clamp_zero = 0
        clamp_neumann = 0
    else:
        if sponge_w != 0 or sponge_s != 0.0 or sponge_axes not in ("", "xyz"):
            raise ValueError("sponge params must be zero/default when boundary_mode!=sponge")
        ax_x = 0
        ax_y = 0
        ax_z = 0
        if bm == "zero":
            clamp_zero = 1
            clamp_neumann = 0
        elif bm == "neumann":
            clamp_zero = 0
            clamp_neumann = 1
        else:
            # open
            clamp_zero = 0
            clamp_neumann = 0

    phi0 = np.zeros_like(load, dtype=np.float32)
    vel0 = np.zeros_like(load, dtype=np.float32)
    src = load.astype(np.float32)

    phi_next = np.empty_like(phi0)
    vel_next = np.empty_like(phi0)

    if progress_cb is None:
        phi, vel = _nb_sine_gordon_steps(
            np.ascontiguousarray(phi0),
            np.ascontiguousarray(vel0),
            np.ascontiguousarray(src),
            float(tp.gamma),
            float(tp.inject),
            float(tp.c2),
            float(tp.dt),
            float(tp.decay),
            float(k_sg),
            int(clamp_zero),
            int(clamp_neumann),
            int(sponge_w),
            float(sponge_s),
            int(ax_x), int(ax_y), int(ax_z),
            int(tp.iters),
            phi_next,
            vel_next,
        )
        if state_cb is not None:
            try:
                state_cb(phi, int(tp.iters))
            except Exception:
                pass
        _assert_finite(phi, "phi")
        _assert_finite(vel, "vel")
        return phi

    phi = np.ascontiguousarray(phi0)
    vel = np.ascontiguousarray(vel0)
    src0 = np.ascontiguousarray(src)
    total = int(tp.iters)
    # Smaller chunks when streaming state (live viewer) so updates are frequent.
    chunk = 8 if state_cb is not None else 256
    done = 0
    while done < total:
        steps = chunk
        if done + steps > total:
            steps = total - done
        phi, vel = _nb_sine_gordon_steps(
            phi,
            vel,
            src0,
            float(tp.gamma),
            float(tp.inject),
            float(tp.c2),
            float(tp.dt),
            float(tp.decay),
            float(k_sg),
            int(clamp_zero),
            int(clamp_neumann),
            int(sponge_w),
            float(sponge_s),
            int(ax_x), int(ax_y), int(ax_z),
            int(steps),
            phi_next,
            vel_next,
        )
        done += int(steps)
        progress_cb(int(steps))
        if state_cb is not None:
            state_cb(phi, int(done))

    _assert_finite(phi, "phi")
    _assert_finite(vel, "vel")
    return phi


def evolve_sine_gordon_traffic_steps(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    tp: TrafficParams,
    steps: int,
    k_grid: np.ndarray | None = None,
    domain_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance (phi, vel) by `steps` ticks using the sine-gordon update."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if phi.shape != src.shape or vel.shape != src.shape:
        raise ValueError("phi/vel/src shape mismatch")

    m = str(tp.mode).strip().lower()
    if m != "sine_gordon":
        raise ValueError(f"evolve_sine_gordon_traffic_steps requires TrafficParams.mode=sine_gordon (got {m!r})")

    if not (0.0 <= float(tp.gamma) < 1.0):
        raise ValueError("TrafficParams.gamma must be in [0,1)")

    if not (0.0 <= float(tp.decay) < 1.0):
        raise ValueError("TrafficParams.decay must be in [0,1)")

    if not math.isfinite(float(tp.c2)) or float(tp.c2) < 0.0:
        raise ValueError("TrafficParams.c2 must be finite and >= 0")

    if not math.isfinite(float(tp.dt)) or float(tp.dt) <= 0.0:
        raise ValueError("TrafficParams.dt must be finite and > 0")
    if float(tp.c2) > 0.0:
        cfl = math.sqrt(float(tp.c2)) * float(tp.dt)
        # Empirical stability guard for this discrete-time scheme (not a strict PDE CFL condition).
        limit = (1.0 / math.sqrt(3.0)) * 0.98
        if not math.isfinite(cfl) or cfl >= limit:
            raise ValueError(
                "unstable telegraph params: sqrt(c2)*dt=%.6g must be < %.6g (reduce c2 or dt)" % (cfl, limit)
            )

    k_sg = float(tp.traffic_k)
    if not math.isfinite(k_sg) or k_sg < 0.0:
        raise ValueError("TrafficParams.traffic_k must be finite and >= 0 for sine_gordon")

    k0 = None
    if k_grid is not None:
        if k_grid.shape != src.shape:
            raise ValueError("k_grid/src shape mismatch")
        k0 = np.ascontiguousarray(k_grid, dtype=np.float32)

    dm0 = None
    if domain_mask is not None:
        if domain_mask.shape != src.shape:
            raise ValueError("domain_mask/src shape mismatch")
        dm0 = np.ascontiguousarray(domain_mask, dtype=np.int8)

    bm = str(tp.boundary_mode).strip().lower()
    if bm not in ("zero", "sponge", "neumann", "open"):
        raise ValueError("TrafficParams.boundary_mode must be one of: zero, sponge, neumann, open")
    sponge_w = int(tp.sponge_width)
    sponge_s = float(tp.sponge_strength)
    sponge_axes = str(getattr(tp, "sponge_axes", "xyz")).strip().lower()
    if bm == "sponge":
        if sponge_w <= 0:
            raise ValueError("TrafficParams.sponge_width must be > 0 when boundary_mode=sponge")
        if not (0.0 < sponge_s <= 1.0):
            raise ValueError("TrafficParams.sponge_strength must be in (0,1] when boundary_mode=sponge")
        if sponge_axes == "":
            raise ValueError("TrafficParams.sponge_axes must be non-empty when boundary_mode=sponge")
        for ch in sponge_axes:
            if ch not in ("x", "y", "z"):
                raise ValueError("TrafficParams.sponge_axes must contain only x/y/z")
        ax_x = 1 if ("x" in sponge_axes) else 0
        ax_y = 1 if ("y" in sponge_axes) else 0
        ax_z = 1 if ("z" in sponge_axes) else 0
        clamp_zero = 0
        clamp_neumann = 0
    else:
        if sponge_w != 0 or sponge_s != 0.0 or sponge_axes not in ("", "xyz"):
            raise ValueError("sponge params must be zero/default when boundary_mode!=sponge")
        ax_x = 0
        ax_y = 0
        ax_z = 0
        if bm == "zero":
            clamp_zero = 1
            clamp_neumann = 0
        elif bm == "neumann":
            clamp_zero = 0
            clamp_neumann = 1
        else:
            # open
            clamp_zero = 0
            clamp_neumann = 0

    phi0 = np.ascontiguousarray(phi, dtype=np.float32)
    vel0 = np.ascontiguousarray(vel, dtype=np.float32)
    src0 = np.ascontiguousarray(src, dtype=np.float32)

    phi_next = np.empty_like(phi0)
    vel_next = np.empty_like(phi0)

    if dm0 is None:
        if k0 is None:
            phi_out, vel_out = _nb_sine_gordon_steps(
                phi0,
                vel0,
                src0,
                float(tp.gamma),
                float(tp.inject),
                float(tp.c2),
                float(tp.dt),
                float(tp.decay),
                float(k_sg),
                int(clamp_zero),
                int(clamp_neumann),
                int(sponge_w),
                float(sponge_s),
                int(ax_x), int(ax_y), int(ax_z),
                int(steps),
                phi_next,
                vel_next,
            )
        else:
            phi_out, vel_out = _nb_sine_gordon_steps_kgrid(
                phi0,
                vel0,
                src0,
                float(tp.gamma),
                float(tp.inject),
                float(tp.c2),
                float(tp.dt),
                float(tp.decay),
                k0,
                int(clamp_zero),
                int(clamp_neumann),
                int(sponge_w),
                float(sponge_s),
                int(ax_x), int(ax_y), int(ax_z),
                int(steps),
                phi_next,
                vel_next,
            )
    else:
        if k0 is None:
            phi_out, vel_out = _nb_sine_gordon_steps_masked(
                phi0,
                vel0,
                src0,
                dm0,
                float(tp.gamma),
                float(tp.inject),
                float(tp.c2),
                float(tp.dt),
                float(tp.decay),
                float(k_sg),
                int(clamp_zero),
                int(clamp_neumann),
                int(sponge_w),
                float(sponge_s),
                int(ax_x), int(ax_y), int(ax_z),
                int(steps),
                phi_next,
                vel_next,
            )
        else:
            phi_out, vel_out = _nb_sine_gordon_steps_masked_kgrid(
                phi0,
                vel0,
                src0,
                dm0,
                float(tp.gamma),
                float(tp.inject),
                float(tp.c2),
                float(tp.dt),
                float(tp.decay),
                k0,
                int(clamp_zero),
                int(clamp_neumann),
                int(sponge_w),
                float(sponge_s),
                int(ax_x), int(ax_y), int(ax_z),
                int(steps),
                phi_next,
                vel_next,
            )

    _assert_finite(phi_out, "phi")
    _assert_finite(vel_out, "vel")
    return phi_out, vel_out


def evolve_telegraph_traffic_steps(
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    tp: TrafficParams,
    steps: int,
    mask: np.ndarray | None = None,
    chirality: np.ndarray | None = None,
    chiral_select: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance (phi, vel) by `steps` ticks using the telegraph update."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if phi.shape != src.shape or vel.shape != src.shape:
        raise ValueError("phi/vel/src shape mismatch")

    if mask is not None and mask.shape != src.shape:
        raise ValueError("mask/src shape mismatch")

    sel = int(chiral_select)
    if sel not in (-1, 0, 1):
        raise ValueError("chiral_select must be one of: -1, 0, 1")

    if chirality is not None and chirality.shape != src.shape:
        raise ValueError("chirality/src shape mismatch")

    if sel != 0 and chirality is None:
        raise ValueError("chiral_select requires chirality array")

    if not (0.0 <= float(tp.gamma) < 1.0):
        raise ValueError("TrafficParams.gamma must be in [0,1)")

    if not (0.0 <= float(tp.decay) < 1.0):
        raise ValueError("TrafficParams.decay must be in [0,1)")

    if not math.isfinite(float(tp.c2)) or float(tp.c2) < 0.0:
        raise ValueError("TrafficParams.c2 must be finite and >= 0")

    if not math.isfinite(float(tp.dt)) or float(tp.dt) <= 0.0:
        raise ValueError("TrafficParams.dt must be finite and > 0")
    if float(tp.c2) > 0.0:
        cfl = math.sqrt(float(tp.c2)) * float(tp.dt)
        # Empirical stability guard for this discrete-time scheme (not a strict PDE CFL condition).
        limit = (1.0 / math.sqrt(3.0)) * 0.98
        if not math.isfinite(cfl) or cfl >= limit:
            raise ValueError(
                "unstable telegraph params: sqrt(c2)*dt=%.6g must be < %.6g (reduce c2 or dt)" % (cfl, limit)
            )

    bm = str(tp.boundary_mode).strip().lower()
    if bm not in ("zero", "sponge", "neumann", "open"):
        raise ValueError("TrafficParams.boundary_mode must be one of: zero, sponge, neumann, open")
    sponge_w = int(tp.sponge_width)
    sponge_s = float(tp.sponge_strength)
    sponge_axes = str(getattr(tp, "sponge_axes", "xyz")).strip().lower()
    if bm == "sponge":
        if sponge_w <= 0:
            raise ValueError("TrafficParams.sponge_width must be > 0 when boundary_mode=sponge")
        if not (0.0 < sponge_s <= 1.0):
            raise ValueError("TrafficParams.sponge_strength must be in (0,1] when boundary_mode=sponge")
        if sponge_axes == "":
            raise ValueError("TrafficParams.sponge_axes must be non-empty when boundary_mode=sponge")
        for ch in sponge_axes:
            if ch not in ("x", "y", "z"):
                raise ValueError("TrafficParams.sponge_axes must contain only x/y/z")
        ax_x = 1 if ("x" in sponge_axes) else 0
        ax_y = 1 if ("y" in sponge_axes) else 0
        ax_z = 1 if ("z" in sponge_axes) else 0
        clamp_zero = 0
        clamp_neumann = 0
    else:
        if sponge_w != 0 or sponge_s != 0.0 or sponge_axes not in ("", "xyz"):
            raise ValueError("sponge params must be zero/default when boundary_mode!=sponge")
        ax_x = 0
        ax_y = 0
        ax_z = 0
        if bm == "zero":
            clamp_zero = 1
            clamp_neumann = 0
        elif bm == "neumann":
            clamp_zero = 0
            clamp_neumann = 1
        else:
            # open
            clamp_zero = 0
            clamp_neumann = 0

    phi0 = np.ascontiguousarray(phi, dtype=np.float32)
    vel0 = np.ascontiguousarray(vel, dtype=np.float32)
    src0 = np.ascontiguousarray(src, dtype=np.float32)
    m0 = None
    if mask is not None:
        m0 = np.ascontiguousarray(mask, dtype=np.int8)

    ch0 = None
    if chirality is not None:
        ch0 = np.ascontiguousarray(chirality, dtype=np.int8)

    phi_next = np.empty_like(phi0)
    vel_next = np.empty_like(phi0)

    if m0 is None:
        if ch0 is None or sel == 0:
            phi_out, vel_out = _nb_telegraph_steps(
                phi0,
                vel0,
                src0,
                float(tp.gamma),
                float(tp.inject),
                float(tp.c2),
                float(tp.dt),
                float(tp.decay),
                int(clamp_zero),
                int(clamp_neumann),
                int(sponge_w),
                float(sponge_s),
                int(ax_x), int(ax_y), int(ax_z),
                int(steps),
                phi_next,
                vel_next,
            )
        else:
            phi_out, vel_out = _nb_telegraph_steps_chiral(
                phi0,
                vel0,
                src0,
                ch0,
                sel,
                float(tp.gamma),
                float(tp.inject),
                float(tp.c2),
                float(tp.dt),
                float(tp.decay),
                int(clamp_zero),
                int(clamp_neumann),
                int(sponge_w),
                float(sponge_s),
                int(ax_x), int(ax_y), int(ax_z),
                int(steps),
                phi_next,
                vel_next,
            )
    else:
        if ch0 is None or sel == 0:
            phi_out, vel_out = _nb_telegraph_steps_masked(
                phi0,
                vel0,
                src0,
                m0,
                float(tp.gamma),
                float(tp.inject),
                float(tp.c2),
                float(tp.dt),
                float(tp.decay),
                int(clamp_zero),
                int(clamp_neumann),
                int(sponge_w),
                float(sponge_s),
                int(ax_x), int(ax_y), int(ax_z),
                int(steps),
                phi_next,
                vel_next,
            )
        else:
            phi_out, vel_out = _nb_telegraph_steps_masked_chiral(
                phi0,
                vel0,
                src0,
                m0,
                ch0,
                sel,
                float(tp.gamma),
                float(tp.inject),
                float(tp.c2),
                float(tp.dt),
                float(tp.decay),
                int(clamp_zero),
                int(clamp_neumann),
                int(sponge_w),
                float(sponge_s),
                int(ax_x), int(ax_y), int(ax_z),
                int(steps),
                phi_next,
                vel_next,
            )

    phi = phi_out
    vel = vel_out

    _assert_finite(phi, "phi")
    _assert_finite(vel, "vel")
    return phi, vel


def evolve_diffusion_traffic_steps(phi: np.ndarray, src: np.ndarray, tp: TrafficParams, steps: int) -> np.ndarray:
    """Advance the traffic field by `steps` ticks using the diffusion update."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if phi.shape != src.shape:
        raise ValueError("phi/src shape mismatch")

    bm = str(tp.boundary_mode).strip().lower()
    if bm not in ("zero", "sponge", "neumann", "open"):
        raise ValueError("TrafficParams.boundary_mode must be one of: zero, sponge, neumann, open")
    sponge_w = int(tp.sponge_width)
    sponge_s = float(tp.sponge_strength)
    sponge_axes = str(getattr(tp, "sponge_axes", "xyz")).strip().lower()
    if bm == "sponge":
        if sponge_w <= 0:
            raise ValueError("TrafficParams.sponge_width must be > 0 when boundary_mode=sponge")
        if not (0.0 < sponge_s <= 1.0):
            raise ValueError("TrafficParams.sponge_strength must be in (0,1] when boundary_mode=sponge")
        if sponge_axes == "":
            raise ValueError("TrafficParams.sponge_axes must be non-empty when boundary_mode=sponge")
        for ch in sponge_axes:
            if ch not in ("x", "y", "z"):
                raise ValueError("TrafficParams.sponge_axes must contain only x/y/z")
        ax_x = 1 if ("x" in sponge_axes) else 0
        ax_y = 1 if ("y" in sponge_axes) else 0
        ax_z = 1 if ("z" in sponge_axes) else 0
        clamp_zero = 0
        clamp_neumann = 0
    else:
        if sponge_w != 0 or sponge_s != 0.0 or sponge_axes not in ("", "xyz"):
            raise ValueError("sponge params must be zero/default when boundary_mode!=sponge")
        ax_x = 0
        ax_y = 0
        ax_z = 0
        if bm == "zero":
            clamp_zero = 1
            clamp_neumann = 0
        elif bm == "neumann":
            clamp_zero = 0
            clamp_neumann = 1
        else:
            # open
            clamp_zero = 0
            clamp_neumann = 0

    phi0 = np.ascontiguousarray(phi, dtype=np.float32)
    src0 = np.ascontiguousarray(src, dtype=np.float32)

    phi_inj = np.empty_like(phi0)
    phi_next = np.empty_like(phi0)

    out = _nb_diffuse_steps(
        phi0,
        src0,
        float(tp.inject),
        float(tp.rate_rise),
        float(tp.rate_fall),
        float(tp.decay),
        int(clamp_zero),
        int(clamp_neumann),
        int(sponge_w),
        float(sponge_s),
        int(ax_x), int(ax_y), int(ax_z),
        int(steps),
        phi_inj,
        phi_next,
    )

    _assert_finite(out, "phi")
    return out


def evolve_traffic(
    load: np.ndarray,
    tp: TrafficParams,
    progress_cb: Callable[[int], None] | None = None,
    state_cb: Optional[Callable[[np.ndarray, int], None]] = None,
) -> np.ndarray:
    """Dispatch traffic evolution by mode (fail-fast on unknown modes)."""
    m = str(tp.mode).strip().lower()
    if m == "diffuse":
        return evolve_diffusion_traffic(load, tp, progress_cb=progress_cb, state_cb=state_cb)
    if m == "telegraph":
        return evolve_telegraph_traffic(load, tp, progress_cb=progress_cb, state_cb=state_cb)
    if m == "nonlinear":
        return evolve_nonlinear_traffic(load, tp, progress_cb=progress_cb, state_cb=state_cb)
    if m == "sine_gordon":
        return evolve_sine_gordon_traffic(load, tp, progress_cb=progress_cb, state_cb=state_cb)
    raise ValueError("TrafficParams.mode must be one of: diffuse, telegraph, nonlinear, sine_gordon")