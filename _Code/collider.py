# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""collider.py — Two-walker collision experiment

Intent
------
We test whether collision behaviour depends on *spin phase* when two moving sources
(“walkers”) interact on the same field.

This module deliberately keeps the mechanics simple and verifiable:
- Two moving delta sources are injected into the same 3D grid.
- Each source follows a helical (axis + transverse orbit) trajectory to imprint a
  handedness (“spin”) into the wake.
- The field is evolved with the telegraph (wave) solver so momentum/spin is
  preserved during interaction (diffusion would simply smear the wakes).

Collider always uses moving delta injections internally; it does not depend on the global --delta-load pipeline switch.

Key point
---------
The goal here is not to assert a physical interpretation (Pauli, etc.).
It's to produce a *repeatable diagnostic* that can be compared across:
  - spin_a = +1 (CCW) vs spin_b = +1 (CCW)
  - spin_a = +1 (CCW) vs spin_b = -1 (CW)

Flat-module layout
------------------
Lives alongside `core.py` and imports local modules directly:
  from traffic import evolve_telegraph_traffic_steps

No plotting is performed here; graphical output belongs in `visualiser.py`.
"""

from __future__ import annotations

import math
import os
import sys
import csv
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from numba import njit

from params import PipelineParams
from traffic import evolve_telegraph_traffic_steps
from utils import Timer, ensure_parent_dir, wallclock_iso, write_csv_provenance_header


@dataclass(frozen=True)
class WalkerSpec:
    """
    Defines a moving point source with a helical transverse orbit for the collider experiment.

    All coordinates are in *grid index space* (0..n-1), but may be non-integer (continuous).

    Parameters
    ----------
    x0 : float
        Initial x position of the walker (head-on direction, grid units).
    vx : float
        Initial velocity in the x direction (head-on axis, grid units per step).
    y0 : float
        Initial y position (center of helix).
    z0 : float
        Initial z position (center of helix).
    radius : float
        Radius of the helical orbit in the (y, z) plane.
    omega : float
        Angular velocity (radians per step) for the helical orbit.
    phase0 : float
        Initial phase (radians) of the helical orbit.
    spin : int
        +1 => CCW in the (y, z) plane as x increases.
        -1 => CW  in the (y, z) plane as x increases.
    q : float
        Source strength (charge) injected at each step.
    """

    x0: float
    vx: float
    y0: float
    z0: float
    radius: float
    omega: float
    phase0: float
    spin: int
    q: float


def _clamp_int(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _nearest_cell(pos: Tuple[float, float, float], n: int) -> Tuple[int, int, int]:
    x, y, z = pos
    xi = _clamp_int(int(round(x)), 0, n - 1)
    yi = _clamp_int(int(round(y)), 0, n - 1)
    zi = _clamp_int(int(round(z)), 0, n - 1)
    return xi, yi, zi


def _in_bounds(pos: Tuple[float, float, float], n: int) -> int:
    x, y, z = pos
    return int((0.0 <= x < float(n)) and (0.0 <= y < float(n)) and (0.0 <= z < float(n)))


def _walker_pos(ws: WalkerSpec, step: int) -> Tuple[float, float, float]:
    t = float(step)
    x = ws.x0 + ws.vx * t
    ang = ws.phase0 + (float(ws.spin) * ws.omega * t)
    y = ws.y0 + ws.radius * math.cos(ang)
    z = ws.z0 + ws.radius * math.sin(ang)
    return float(x), float(y), float(z)


def _inject_delta(src: np.ndarray, pos: Tuple[float, float, float], q: float) -> None:
    n = int(src.shape[0])
    xi, yi, zi = _nearest_cell(pos, n)
    src[xi, yi, zi] += np.float32(q)


 

def _energy_slab_x(arr: np.ndarray, x_center: int, half_width: int) -> float:
    n = int(arr.shape[0])
    x0 = _clamp_int(int(x_center) - int(half_width), 0, n - 1)
    x1 = _clamp_int(int(x_center) + int(half_width), 0, n - 1)
    a = arr[x0 : x1 + 1].astype(np.float64, copy=False)
    return float(np.sum(a * a))


# --- Numba kernels ---
@njit(fastmath=True)
def _nb_energy_and_maxabs(arr: np.ndarray) -> Tuple[float, float]:
    n0 = int(arr.shape[0])
    acc = 0.0
    m = 0.0
    for x in range(n0):
        for y in range(n0):
            for z in range(n0):
                v = float(arr[x, y, z])
                av = v if v >= 0.0 else -v
                if av > m:
                    m = av
                acc += v * v
    return float(acc), float(m)



@njit(fastmath=True)
def _nb_energy_slab_x(arr: np.ndarray, x_center: int, half_width: int) -> float:
    n0 = int(arr.shape[0])
    x0 = x_center - int(half_width)
    x1 = x_center + int(half_width)
    if x0 < 0:
        x0 = 0
    if x1 >= n0:
        x1 = n0 - 1
    acc = 0.0
    for x in range(x0, x1 + 1):
        for y in range(n0):
            for z in range(n0):
                v = float(arr[x, y, z])
                acc += v * v
    return float(acc)


# --- Gradient kernel for back-reaction (Stage-3) ---

@njit(fastmath=True)
def _nb_grad_phi_at(phi: np.ndarray, x: float, y: float, z: float) -> Tuple[float, float, float]:
    n = int(phi.shape[0])
    xi = int(round(x))
    yi = int(round(y))
    zi = int(round(z))
    if xi < 1 or xi > n - 2 or yi < 1 or yi > n - 2 or zi < 1 or zi > n - 2:
        return 0.0, 0.0, 0.0
    gx = 0.5 * (float(phi[xi + 1, yi, zi]) - float(phi[xi - 1, yi, zi]))
    gy = 0.5 * (float(phi[xi, yi + 1, zi]) - float(phi[xi, yi - 1, zi]))
    gz = 0.5 * (float(phi[xi, yi, zi + 1]) - float(phi[xi, yi, zi - 1]))
    return float(gx), float(gy), float(gz)


# --- Nucleus injection kernels ---

@njit(fastmath=True)
def _nb_inject_point(src: np.ndarray, xi: int, yi: int, zi: int, q: float) -> None:
    n = int(src.shape[0])
    if xi < 0:
        xi = 0
    elif xi >= n:
        xi = n - 1
    if yi < 0:
        yi = 0
    elif yi >= n:
        yi = n - 1
    if zi < 0:
        zi = 0
    elif zi >= n:
        zi = n - 1
    src[xi, yi, zi] += np.float32(q)


@njit(fastmath=True)
def _nb_inject_ball_uniform(src: np.ndarray, x: float, y: float, z: float, q: float, r: int) -> None:
    """Inject q uniformly into a small ball around (x,y,z).

    This is used for the nucleus because a stationary per-step delta source can
    excite numeric blow-ups in long runs.
    """
    n = int(src.shape[0])
    cx = int(round(x))
    cy = int(round(y))
    cz = int(round(z))
    rr = int(r) * int(r)
    if r <= 0:
        _nb_inject_point(src, cx, cy, cz, q)
        return

    cnt = 0
    for dx in range(-int(r), int(r) + 1):
        xi = cx + dx
        if xi < 0 or xi >= n:
            continue
        for dy in range(-int(r), int(r) + 1):
            yi = cy + dy
            if yi < 0 or yi >= n:
                continue
            for dz in range(-int(r), int(r) + 1):
                zi = cz + dz
                if zi < 0 or zi >= n:
                    continue
                if (dx * dx + dy * dy + dz * dz) <= rr:
                    cnt += 1

    if cnt <= 0:
        _nb_inject_point(src, cx, cy, cz, q)
        return

    q_each = float(q) / float(cnt)
    for dx in range(-int(r), int(r) + 1):
        xi = cx + dx
        if xi < 0 or xi >= n:
            continue
        for dy in range(-int(r), int(r) + 1):
            yi = cy + dy
            if yi < 0 or yi >= n:
                continue
            for dz in range(-int(r), int(r) + 1):
                zi = cz + dz
                if zi < 0 or zi >= n:
                    continue
                if (dx * dx + dy * dy + dz * dz) <= rr:
                    src[xi, yi, zi] += np.float32(q_each)

# --- Halo damping kernel ---
@njit(fastmath=True)
def _nb_apply_halo_damping(vel: np.ndarray, cx: int, cy: int, cz: int, r: int, strength: float, profile: int) -> int:
    """Apply a radial damping halo to `vel` around (cx,cy,cz).

    Returns the number of voxels actually damped (non-zero value with factor < 1).

    profile:
      0 = linear:   factor = 1 - strength*(1-d)
      1 = quadratic:factor = 1 - strength*(1-d)^2
      2 = exp:      factor = exp(-strength*(1-d))
    """
    n = int(vel.shape[0])
    rr = int(r) * int(r)
    touched = 0
    if r <= 0 or strength <= 0.0:
        return 0
    r_f = float(r)
    x0 = cx - int(r)
    x1 = cx + int(r)
    y0 = cy - int(r)
    y1 = cy + int(r)
    z0 = cz - int(r)
    z1 = cz + int(r)
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if z0 < 0:
        z0 = 0
    if x1 >= n:
        x1 = n - 1
    if y1 >= n:
        y1 = n - 1
    if z1 >= n:
        z1 = n - 1

    for x in range(x0, x1 + 1):
        dx = x - cx
        dx2 = dx * dx
        for y in range(y0, y1 + 1):
            dy = y - cy
            dxy2 = dx2 + dy * dy
            for z in range(z0, z1 + 1):
                dz = z - cz
                d2 = dxy2 + dz * dz
                if d2 > rr:
                    continue
                d = math.sqrt(float(d2)) / r_f  # 0..1
                one_minus = 1.0 - d
                if profile == 0:
                    factor = 1.0 - float(strength) * one_minus
                elif profile == 1:
                    factor = 1.0 - float(strength) * (one_minus * one_minus)
                else:
                    factor = math.exp(-float(strength) * one_minus)
                if factor < 0.0:
                    factor = 0.0
                v0 = float(vel[x, y, z])
                if (factor < 0.999999999999) and (v0 != 0.0):
                    touched += 1
                vel[x, y, z] = np.float32(v0 * factor)
    return int(touched)



@njit(fastmath=True)
def _nb_energy_local_ball(arr: np.ndarray, cx: int, cy: int, cz: int, r: int) -> float:
    n = int(arr.shape[0])
    rr = int(r) * int(r)
    acc = 0.0
    for dx in range(-int(r), int(r) + 1):
        x = cx + dx
        if x < 0 or x >= n:
            continue
        for dy in range(-int(r), int(r) + 1):
            y = cy + dy
            if y < 0 or y >= n:
                continue
            for dz in range(-int(r), int(r) + 1):
                z = cz + dz
                if z < 0 or z >= n:
                    continue
                if (dx * dx + dy * dy + dz * dz) <= rr:
                    v = float(arr[x, y, z])
                    acc += v * v
    return float(acc)


def _energy_local_ball(arr: np.ndarray, pos: Tuple[float, float, float], r: int) -> float:
    n = int(arr.shape[0])
    cx, cy, cz = _nearest_cell(pos, n)
    return float(_nb_energy_local_ball(arr, int(cx), int(cy), int(cz), int(r)))


def _precompute_shell_offsets(r_inner: int, r_outer: int, stride: int) -> np.ndarray:
    rin = int(r_inner)
    rout = int(r_outer)
    s = int(stride)
    if rin < 0:
        rin = 0
    if rout <= 0:
        return np.zeros((0, 3), dtype=np.int16)
    if rout < rin:
        rout = rin
    if s <= 0:
        s = 1
    rin2 = rin * rin
    rout2 = rout * rout
    offs: List[Tuple[int, int, int]] = []
    for dx in range(-rout, rout + 1, s):
        for dy in range(-rout, rout + 1, s):
            for dz in range(-rout, rout + 1, s):
                d2 = dx * dx + dy * dy + dz * dz
                if rin2 <= d2 <= rout2:
                    offs.append((dx, dy, dz))
    if not offs:
        return np.zeros((0, 3), dtype=np.int16)
    return np.asarray(offs, dtype=np.int16)


# Helper: count how many shell-offset voxels are in-bounds at given cell
def _count_shell_voxels_at_cell(n: int, cx: int, cy: int, cz: int, offsets: np.ndarray) -> int:
    if offsets.size == 0:
        return 0
    nn = int(n)
    cxi = int(cx)
    cyi = int(cy)
    czi = int(cz)
    cnt = 0
    for i in range(int(offsets.shape[0])):
        x = cxi + int(offsets[i, 0])
        y = cyi + int(offsets[i, 1])
        z = czi + int(offsets[i, 2])
        if 0 <= x < nn and 0 <= y < nn and 0 <= z < nn:
            cnt += 1
    return int(cnt)




@njit(fastmath=True)
def _nb_energy_shell(arr: np.ndarray, cx: int, cy: int, cz: int, offsets: np.ndarray) -> Tuple[float, int]:
    n = int(arr.shape[0])
    acc = 0.0
    cnt = 0
    for i in range(int(offsets.shape[0])):
        dx = int(offsets[i, 0])
        dy = int(offsets[i, 1])
        dz = int(offsets[i, 2])
        x = cx + dx
        y = cy + dy
        z = cz + dz
        if x < 0 or x >= n or y < 0 or y >= n or z < 0 or z >= n:
            continue
        v = float(arr[x, y, z])
        acc += v * v
        cnt += 1
    return float(acc), int(cnt)


# --- Octant energy binned shell kernel ---
@njit(fastmath=True)
def _nb_energy_shell_octants(arr: np.ndarray, cx: int, cy: int, cz: int, offsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (E_oct[8], N_oct[8]) for shell offsets around (cx,cy,cz).

    Octant index uses sign bits of (dx,dy,dz) where dx>=0 is bit 0, dy>=0 is bit 1, dz>=0 is bit 2.
    Index = sx + 2*sy + 4*sz in [0..7].
    """
    n = int(arr.shape[0])
    E = np.zeros(8, dtype=np.float64)
    N = np.zeros(8, dtype=np.int32)
    for i in range(int(offsets.shape[0])):
        dx = int(offsets[i, 0])
        dy = int(offsets[i, 1])
        dz = int(offsets[i, 2])
        x = cx + dx
        y = cy + dy
        z = cz + dz
        if x < 0 or x >= n or y < 0 or y >= n or z < 0 or z >= n:
            continue
        sx = 1 if dx >= 0 else 0
        sy = 1 if dy >= 0 else 0
        sz = 1 if dz >= 0 else 0
        j = sx + 2 * sy + 4 * sz
        v = float(arr[x, y, z])
        E[j] += v * v
        N[j] += 1
    return E, N


def _energy_shell_at_cell(arr: np.ndarray, cx: int, cy: int, cz: int, offsets: np.ndarray) -> Tuple[float, int]:
    if offsets.size == 0:
        return 0.0, 0
    e, cnt = _nb_energy_shell(arr, int(cx), int(cy), int(cz), offsets)
    return float(e), int(cnt)


def _energy_shell(arr: np.ndarray, center: Tuple[float, float, float], offsets: np.ndarray) -> Tuple[float, int]:
    if offsets.size == 0:
        return 0.0, 0
    n = int(arr.shape[0])
    cx, cy, cz = _nearest_cell(center, n)
    e, cnt = _nb_energy_shell(arr, int(cx), int(cy), int(cz), offsets)
    return float(e), int(cnt)


def _default_specs(params: PipelineParams, spin_b: int) -> Tuple[WalkerSpec, WalkerSpec, int]:
    n = int(params.lattice.n)
    c = n // 2

    # Default run length (used only if caller does not pass `steps`).
    # Keep both walkers in-bounds for the head-on run.
    # With xB0≈c+0.5*sep and vx>0, require: xB0 - vx*(steps-1) >= 0 (with margin).

    vx = float(params.collider_vx)

    # Choose a separation that yields an *integer* collision step.
    # The near-miss in 07A is expected when t_c is fractional because the helix phase
    # and the x-positions cannot line up perfectly on an integer lattice step.
    # We snap sep so that: t_c = sep / (2*vx) is an integer.
    sep_raw = max(6.0, float(n) * 0.25)
    t_c_raw = sep_raw / (2.0 * float(vx))
    t_c_i = max(1, int(round(t_c_raw)))
    sep = 2.0 * float(vx) * float(t_c_i)

    xA0 = float(c) - 0.5 * sep
    xB0 = float(c) + 0.5 * sep
    radius = float(params.collider_orbit_radius)
    omega = float(params.collider_orbit_omega)

    if vx <= 0.0:
        raise ValueError("collider_vx must be > 0")
    if radius < 0.0:
        raise ValueError("collider_orbit_radius must be >= 0")
    if omega < 0.0:
        raise ValueError("collider_orbit_omega must be >= 0")

    # Compute a safe default run length (only used when `run_collider` receives no explicit `steps`).
    margin = 8.0
    steps = max(1, int((0.625 * float(n) - margin) / float(vx)))

    # Choose phase for walker B so the trajectories have identical transverse position at the collision time.
    # This isolates "spin" from impact-parameter effects.
    # Condition at collision: angB(t_c) == angA(t_c) (mod 2π).
    # A: angA = 0 + (+1)*omega*t
    # B: angB = phase0B + (spin_b)*omega*t
    # Solve for spin_b=-1 => phase0B = 2*omega*t_c (mod 2π).
    t_c = float(xB0 - xA0) / (2.0 * float(vx))
    # Sanity: collision time should be (very nearly) an integer step.
    # (Floating error tolerated.)
    if abs(t_c - round(t_c)) > 1e-4:
        raise ValueError("collider sep snap failed: t_c not integer")
    phase0B = 0.0
    if int(spin_b) == -1:
        phase0B = (2.0 * float(omega) * t_c) % (2.0 * math.pi)

    # Stage-4: impact parameter (transverse offsets) for scattering runs.
    # Defaults to 0.0 so baseline runs are unchanged.
    impact_b = float(getattr(params, "collider_impact_b", 0.0))
    impact_bz = float(getattr(params, "collider_impact_bz", 0.0))


    q = 1.0

    A = WalkerSpec(
        x0=xA0,
        vx=+vx,
        y0=float(c),
        z0=float(c),
        radius=radius,
        omega=omega,
        phase0=0.0,
        spin=+1,
        q=q,
    )

    B = WalkerSpec(
        x0=xB0,
        vx=-vx,
        y0=float(c) + float(impact_b),
        z0=float(c) + float(impact_bz),
        radius=radius,
        omega=omega,
        phase0=phase0B,
        spin=int(spin_b),
        q=q,
    )

    return A, B, steps




def run_collider(
    params: PipelineParams,
    *,
    spin_b: int = -1,
    steps: Optional[int] = None,
    exp_code: str = "07A",
    variant: str = "collider",
    out_dir: Optional[str] = None,
    out_csv: Optional[str] = None,
    provenance_header: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    log_path: Optional[str] = None,
) -> None:
    """Run a two-walker collision experiment.

    Uses the telegraph (wave) solver so wake momentum/spin is preserved during
    interaction.

    Output
    ------
    If `out_csv` is provided, writes exactly to that path.
    Otherwise writes into the CAELIX bundle layout:
      <out_dir>/_csv/<exp_code>_collider_<variant>_n<n>.csv
    where `out_dir` is typically the timestamped experiment bundle created by `core.py`.
    """

    A, B, steps0 = _default_specs(params, int(spin_b))
    if steps is None or int(steps) <= 0:
        steps = int(steps0)
    else:
        steps = int(steps)

    n = int(params.lattice.n)

    sep0 = float(B.x0 - A.x0)
    t_c = sep0 / (2.0 * float(A.vx))
    t_c_int = int(round(t_c))
    center_tc = (
        0.5 * (_walker_pos(A, t_c_int)[0] + _walker_pos(B, t_c_int)[0]),
        0.5 * (_walker_pos(A, t_c_int)[1] + _walker_pos(B, t_c_int)[1]),
        0.5 * (_walker_pos(A, t_c_int)[2] + _walker_pos(B, t_c_int)[2]),
    )
    center_tc_cell = _nearest_cell(center_tc, int(n))

    # Stage-2: post-collision hold / decay (injection shutter).
    # Default OFF so 07A/07B remain unchanged.
    collider_hold = bool(getattr(params, "collider_hold", False))
    hold_grace_steps = int(getattr(params, "collider_hold_grace", 0))
    hold_steps = int(getattr(params, "collider_hold_steps", 0))
    center_ball_r = int(getattr(params, "collider_center_ball_r", 6))
    if hold_grace_steps < 0:
        hold_grace_steps = 0
    if hold_steps < 0:
        hold_steps = 0
    if center_ball_r < 1:
        center_ball_r = 1


    # Stage-0: optionally disable walker B (single-body runs, e.g. Hydrogen).
    # Default ON so prior experiments are unchanged.
    collider_enable_b = bool(getattr(params, "collider_enable_b", True))

    # Stage-3: back-reaction (field-responsive motion).
    # Default OFF so 07A–07F remain unchanged.
    collider_backreact = bool(getattr(params, "collider_backreact", False))
    backreact_k = float(getattr(params, "collider_backreact_k", 0.0))
    backreact_mode = str(getattr(params, "collider_backreact_mode", "repel")).strip().lower()
    backreact_vmax = float(getattr(params, "collider_backreact_vmax", float(abs(A.vx))))
    if backreact_k < 0.0:
        backreact_k = 0.0
    if backreact_vmax <= 0.0:
        backreact_vmax = float(abs(A.vx))
    if backreact_mode not in ("repel", "attract"):
        raise ValueError("collider_backreact_mode must be 'repel' or 'attract'")
    backreact_axes = str(getattr(params, "collider_backreact_axes", "x")).strip().lower()
    if backreact_axes not in ("x", "xyz"):
        raise ValueError("collider_backreact_axes must be 'x' or 'xyz'")

    # Stage-6: central nucleus (third-body momentum sink / anchor).
    # Default OFF so prior experiments are unchanged.
    collider_nucleus = bool(getattr(params, "collider_nucleus", False))
    nucleus_q = float(getattr(params, "collider_nucleus_q", 1.0))
    nucleus_mode = str(getattr(params, "collider_nucleus_mode", "sin")).strip().lower()
    nucleus_omega = float(getattr(params, "collider_nucleus_omega", float(A.omega)))
    nucleus_phase = float(getattr(params, "collider_nucleus_phase", 0.0))
    if nucleus_q < 0.0:
        nucleus_q = 0.0
    if nucleus_omega < 0.0:
        nucleus_omega = 0.0
    if nucleus_mode not in ("dc", "sin"):
        raise ValueError("collider_nucleus_mode must be 'dc' or 'sin'")

    # Use a tiny ball injection for the nucleus to avoid stationary per-step delta blow-ups.
    nucleus_r = 1
    # DC nucleus "soft pin" (stability): avoid hard Dirichlet clamps which can pump energy.
    # alpha pins phi toward nucleus_q; beta damps vel at the nucleus voxel.
    nucleus_dc_alpha = 0.05
    nucleus_dc_beta = 0.25

    # Stage-7: local damping halo (phenomenological radiative cooling).
    # Default OFF so previous experiments remain unchanged.
    collider_halo = bool(getattr(params, "collider_halo", False))
    halo_center = str(getattr(params, "collider_halo_center", "nucleus")).strip().lower()
    halo_r = int(getattr(params, "collider_halo_r", 0))
    halo_strength = float(getattr(params, "collider_halo_strength", 0.0))
    halo_profile = str(getattr(params, "collider_halo_profile", "linear")).strip().lower()

    if halo_center not in ("nucleus", "collision"):
        raise ValueError("collider_halo_center must be 'nucleus' or 'collision'")
    if halo_profile not in ("linear", "quadratic", "exp"):
        raise ValueError("collider_halo_profile must be 'linear', 'quadratic', or 'exp'")
    if collider_halo:
        if halo_r <= 0:
            raise ValueError("collider_halo_r must be > 0 when collider_halo is enabled")
        if halo_strength <= 0.0:
            raise ValueError("collider_halo_strength must be > 0 when collider_halo is enabled")

    halo_profile_i = 0
    if halo_profile == "quadratic":
        halo_profile_i = 1
    elif halo_profile == "exp":
        halo_profile_i = 2

    # Optional calorimetry detectors (shell energies around the collision center).
    # These are intentionally *off by default* so 07A/07B remain lightweight.
    collider_detectors = bool(getattr(params, "collider_detectors", False))
    collider_octants = bool(getattr(params, "collider_octants", False))

    # Defaults mirror CLI (fractions of n) and are overridden if CLI wiring exists.
    shell_stride = int(getattr(params, "collider_shell_stride", 2))
    shell1_inner_frac = float(getattr(params, "collider_shell1_inner_frac", 0.18))
    shell1_outer_frac = float(getattr(params, "collider_shell1_outer_frac", 0.22))
    shell2_inner_frac = float(getattr(params, "collider_shell2_inner_frac", 0.22))
    shell2_outer_frac = float(getattr(params, "collider_shell2_outer_frac", 0.26))

    # Compute cut step and adjust steps if hold mode is enabled
    cut_step = int(t_c_int) + int(hold_grace_steps)
    if collider_hold and hold_steps > 0:
        min_steps = int(cut_step) + int(hold_steps)
        if int(steps) < int(min_steps):
            steps = int(min_steps)

    if not collider_detectors:
        shell_stride = 0
        shell1_r_inner = 0
        shell1_r_outer = 0
        shell2_r_inner = 0
        shell2_r_outer = 0
        shell1_offsets = np.zeros((0, 3), dtype=np.int16)
        shell2_offsets = np.zeros((0, 3), dtype=np.int16)
    else:
        if shell_stride <= 0:
            shell_stride = 1
        shell1_r_inner = max(0, int(round(float(shell1_inner_frac) * float(n))))
        shell1_r_outer = max(shell1_r_inner, int(round(float(shell1_outer_frac) * float(n))))
        shell2_r_inner = max(0, int(round(float(shell2_inner_frac) * float(n))))
        shell2_r_outer = max(shell2_r_inner, int(round(float(shell2_outer_frac) * float(n))))
        shell1_offsets = _precompute_shell_offsets(shell1_r_inner, shell1_r_outer, shell_stride)
        shell2_offsets = _precompute_shell_offsets(shell2_r_inner, shell2_r_outer, shell_stride)
        # Fail-loud: shells must actually cover voxels at the collision center.
        s1_cnt = _count_shell_voxels_at_cell(int(n), int(center_tc_cell[0]), int(center_tc_cell[1]), int(center_tc_cell[2]), shell1_offsets)
        s2_cnt = _count_shell_voxels_at_cell(int(n), int(center_tc_cell[0]), int(center_tc_cell[1]), int(center_tc_cell[2]), shell2_offsets)
        if s1_cnt <= 0 or s2_cnt <= 0:
            raise ValueError(
                "collider detectors shell voxel_count==0 at center; check center/origin/radii/stride (shell1_cnt=%d shell2_cnt=%d center=%s shell1=[%d,%d] shell2=[%d,%d] stride=%d n=%d)" % (
                    int(s1_cnt),
                    int(s2_cnt),
                    str(center_tc_cell),
                    int(shell1_r_inner),
                    int(shell1_r_outer),
                    int(shell2_r_inner),
                    int(shell2_r_outer),
                    int(shell_stride),
                    int(n),
                )
            )

    if out_csv is not None and str(out_csv).strip() != "":
        csv_path = str(out_csv)
        ensure_parent_dir(csv_path)
    else:
        if out_dir is None:
            out_dir = os.path.join("_Output", "07_collider", str(exp_code))
        csv_dir = os.path.join(str(out_dir), "_csv")
        os.makedirs(csv_dir, exist_ok=True)
        spin_tag = "same_spin" if int(spin_b) == +1 else "opp_spin"
        ts = os.path.basename(str(out_dir).rstrip(os.sep))
        # Match CAELIX convention: include n and the bundle timestamp in the filename.
        csv_name = f"{str(exp_code)}_{str(variant)}_{spin_tag}_n{int(n)}_{ts}.csv"
        csv_path = os.path.join(csv_dir, csv_name)
        ensure_parent_dir(csv_path)

    phi = np.zeros((n, n, n), dtype=np.float32)
    vel = np.zeros_like(phi)
    src = np.zeros((n, n, n), dtype=np.float32)

    tp = params.traffic

    t_run = Timer().start()
    warned_oob = False
    last_progress = 0

    def _log(msg: str) -> None:
        s = str(msg)
        if log_path is not None and str(log_path).strip() != "":
            with open(str(log_path), "a", encoding="utf-8") as lf:
                lf.write(s.rstrip("\n") + "\n")
        else:
            print(s)

    with open(csv_path, "w", encoding="utf-8") as f:
        # Track halo diagnostics even if halo is disabled.
        halo_touched_vel_max = 0
        halo_touched_phi_max = 0

        if provenance_header is not None and str(provenance_header) != "":
            f.write(str(provenance_header))
        else:
            f.write(
                write_csv_provenance_header(
                    producer="CAELIX",
                    command=" ".join(sys.argv),
                    cwd=os.getcwd(),
                    python_exe=sys.executable,
                    when_iso=wallclock_iso(),
                    experiment=f"{exp_code}_{variant}",
                    extra={
                        "n": int(n),
                        "steps": int(steps),
                        "work_units": int(steps),
                        "spin_b": int(spin_b),
                        "collider_enable_b": int(1 if collider_enable_b else 0),
                        "csv": str(csv_path),
                        "t_c_int": int(t_c_int),
                        "collider_hold": int(1 if collider_hold else 0),
                        "hold_grace_steps": int(hold_grace_steps),
                        "hold_steps": int(hold_steps),
                        "cut_step": int(cut_step),
                        "center_ball_r": int(center_ball_r),
                        "collider_detectors": int(1 if collider_detectors else 0),
                        "shell_stride": int(shell_stride),
                        "shell1_r_inner": int(shell1_r_inner),
                        "shell1_r_outer": int(shell1_r_outer),
                        "shell1_points": int(shell1_offsets.shape[0]),
                        "shell1_voxels_at_c": int(_count_shell_voxels_at_cell(int(n), int(center_tc_cell[0]), int(center_tc_cell[1]), int(center_tc_cell[2]), shell1_offsets)),
                        "shell2_r_inner": int(shell2_r_inner),
                        "shell2_r_outer": int(shell2_r_outer),
                        "shell2_points": int(shell2_offsets.shape[0]),
                        "shell2_voxels_at_c": int(_count_shell_voxels_at_cell(int(n), int(center_tc_cell[0]), int(center_tc_cell[1]), int(center_tc_cell[2]), shell2_offsets)),
                        "collider_backreact": int(1 if collider_backreact else 0),
                        "backreact_k": float(backreact_k),
                        "backreact_mode": str(backreact_mode),
                        "backreact_vmax": float(backreact_vmax),
                        "impact_b": float(getattr(params, "collider_impact_b", 0.0)),
                        "impact_bz": float(getattr(params, "collider_impact_bz", 0.0)),
                        "backreact_axes": str(backreact_axes),
                        "collider_nucleus": int(1 if collider_nucleus else 0),
                        "nucleus_q": float(nucleus_q),
                        "nucleus_mode": str(nucleus_mode),
                        "nucleus_omega": float(nucleus_omega),
                        "nucleus_phase": float(nucleus_phase),
                        "collider_halo": int(1 if collider_halo else 0),
                        "halo_center": str(halo_center),
                        "halo_r": int(halo_r),
                        "halo_strength": float(halo_strength),
                        "halo_profile": str(halo_profile),
                    },
                )
            )
        # CSV metadata (comment lines) + column header
        f.write("# collider\n")
        f.write("# n=%d steps=%d spin_b=%d\n" % (int(n), int(steps), int(spin_b)))
        f.write("# vx=%.9g orbit_r=%.9g orbit_omega=%.9g\n" % (float(A.vx), float(A.radius), float(A.omega)))
        f.write("# sep0=%.9g t_c=%.9g phase0A=%.9g phase0B=%.9g\n" % (float(sep0), float(t_c), float(A.phase0), float(B.phase0)))
        f.write("# detectors=%d enable_b=%d shells: stride=%d shell1=[%d,%d] pts=%d shell2=[%d,%d] pts=%d center_tc=(%.3f,%.3f,%.3f) t_c_int=%d hold=%d cut_step=%d hold_grace=%d hold_steps=%d center_ball_r=%d backreact=%d backreact_k=%.9g backreact_mode=%s backreact_axes=%s backreact_vmax=%.9g nucleus=%d nucleus_q=%.9g nucleus_mode=%s nucleus_omega=%.9g nucleus_phase=%.9g halo=%d halo_center=%s halo_r=%d halo_strength=%.9g halo_profile=%s\n" % (
            int(1 if collider_detectors else 0),
            int(1 if collider_enable_b else 0),
            int(shell_stride),
            int(shell1_r_inner),
            int(shell1_r_outer),
            int(shell1_offsets.shape[0]),
            int(shell2_r_inner),
            int(shell2_r_outer),
            int(shell2_offsets.shape[0]),
            float(center_tc[0]),
            float(center_tc[1]),
            float(center_tc[2]),
            int(t_c_int),
            int(1 if collider_hold else 0),
            int(cut_step),
            int(hold_grace_steps),
            int(hold_steps),
            int(center_ball_r),
            int(1 if collider_backreact else 0),
            float(backreact_k),
            str(backreact_mode),
            str(backreact_axes),
            float(backreact_vmax),
            int(1 if collider_nucleus else 0),
            float(nucleus_q),
            str(nucleus_mode),
            float(nucleus_omega),
            float(nucleus_phase),
            int(1 if collider_halo else 0),
            str(halo_center),
            int(halo_r),
            float(halo_strength),
            str(halo_profile),
        ))
        f.write(
            "step,t_c_int,t_rel,inj_scale,in_hold,backreact_on,vxA,vxB,xA,yA,zA,xB,yB,zB,sep,inA,inB,phi_max,vel_max,phi_energy,vel_energy,E_tot,E_tot_dE,hold_E_tot_dE,phi_midE,vel_midE,phi_A_E,phi_B_E,vel_A_E,vel_B_E,sep_x,sep_yz,mid_frac_phi,mid_frac_vel,mid_frac_tot,localA_frac_phi,localB_frac_phi,localA_frac_vel,localB_frac_vel,localA_frac_tot,localB_frac_tot,center_phi_E,center_vel_E,center_tot_E,center_frac_tot,center_dE_tot,shell1E_phi,shell1E_vel,shell1N,shell1_frac_phi,shell1_frac_vel,shell1_frac_tot,shell2E_phi,shell2E_vel,shell2N,shell2_frac_phi,shell2_frac_vel,shell2_frac_tot,shell2_minus_shell1_tot,shell2_pow_tot,gxA,gxB,dvA,dvB,axA,axB,PA,PB,sep_min,sep_min_t,cut_step,hold_grace_steps,hold_steps,vyA,vzA,vyB,vzB,gyA,gzA,gyB,gzB,dvyA,dvzA,dvyB,dvzB,thetaA,thetaB,theta_degA,theta_degB,octE0,octE1,octE2,octE3,octE4,octE5,octE6,octE7,octN0,octN1,octN2,octN3,octN4,octN5,octN6,octN7,angA,angB,dphi,phi_c,vel_c,src_max,src_E,src_qA,src_qB,src_qN,cellAx,cellAy,cellAz,cellBx,cellBy,cellBz,cellNx,cellNy,cellNz,halo_touched_vel,halo_touched_phi,halo_touched_vel_max,halo_touched_phi_max\n"
        )
        w = csv.writer(f)

        # Stage-0 derived observables tracked across the whole run.
        # In single-body mode (B disabled), separation is undefined; keep NaN rather than +inf.
        if collider_enable_b:
            sep_min = float("inf")
        else:
            sep_min = float("nan")
        sep_min_t = -1

        # Shell transport proxy (finite difference on shell2 total energy).
        prev_shell2_tot = None  # type: Optional[float]

        # center energy for finite difference (hold stage)
        prev_center_tot = None  # type: Optional[float]
        prev_E_tot = None  # type: Optional[float]

        # Dynamic state (Stage-3 back-reaction updates vx; x integrates vx).
        # Stage-4 (axes=xyz) adds induced drift in y/z while preserving the helical drive.
        xA = float(A.x0)
        xB = float(B.x0)
        vxA = float(A.vx)
        vxB = float(B.vx)
        vyA = 0.0
        vzA = 0.0
        vyB = 0.0
        vzB = 0.0
        yA_drift = 0.0
        zA_drift = 0.0
        yB_drift = 0.0
        zB_drift = 0.0
        backreact_on = int(1 if (collider_backreact and backreact_k > 0.0) else 0)
        prev_vxA = float(vxA)
        prev_vxB = float(vxB)
        prev_vyA = float(vyA)
        prev_vzA = float(vzA)
        prev_vyB = float(vyB)
        prev_vzB = float(vzB)
        gxA = 0.0
        gxB = 0.0
        gyA = 0.0
        gzA = 0.0
        gyB = 0.0
        gzB = 0.0
        dvA = 0.0
        dvB = 0.0
        dvyA = 0.0
        dvzA = 0.0
        dvyB = 0.0
        dvzB = 0.0
        axA = 0.0
        axB = 0.0
        PA = 0.0
        PB = 0.0
        halo_touched_vel = 0
        halo_touched_phi = 0

        for step_i in range(int(steps)):
            src[:, :, :] = 0.0

            t = float(step_i)
            angA = float(A.phase0) + (float(A.spin) * float(A.omega) * t)
            angB = float(B.phase0) + (float(B.spin) * float(B.omega) * t)

            # Wrapped phase-difference scalar
            dphi = float(angB - angA)
            two_pi = 2.0 * math.pi
            dphi = (dphi + math.pi) % two_pi - math.pi

            pA = (float(xA), float(A.y0 + yA_drift + A.radius * math.cos(angA)), float(A.z0 + zA_drift + A.radius * math.sin(angA)))
            if collider_enable_b:
                pB = (float(xB), float(B.y0 + yB_drift + B.radius * math.cos(angB)), float(B.z0 + zB_drift + B.radius * math.sin(angB)))
            else:
                # Sentinel: keep B out-of-bounds so any downstream metrics are fail-obvious.
                pB = (-1.0, -1.0, -1.0)

            inA = _in_bounds(pA, n)
            inB = _in_bounds(pB, n) if collider_enable_b else 0
            warn_oob = (inA == 0) or (bool(collider_enable_b) and (inB == 0))
            if (not warned_oob) and warn_oob:
                warned_oob = True
                _log(
                    "[collider] WARNING: walker out-of-bounds at step=%d (inA=%d inB=%d enable_b=%d). Injection will be clamped; interpret results with care." % (
                        step_i,
                        inA,
                        inB,
                        int(1 if collider_enable_b else 0),
                    )
                )

            # Injection bookkeeping and cell indices
            in_hold = int(1 if (collider_hold and int(step_i) >= int(cut_step)) else 0)
            inj_scale = 0.0 if in_hold == 1 else 1.0

            src_qA = float(A.q) * float(inj_scale)
            src_qB = float(B.q) * float(inj_scale) if collider_enable_b else 0.0
            src_qN = 0.0

            cellAx, cellAy, cellAz = _nearest_cell(pA, n)
            if collider_enable_b:
                cellBx, cellBy, cellBz = _nearest_cell(pB, n)
            else:
                cellBx, cellBy, cellBz = -1, -1, -1
            cellNx, cellNy, cellNz = int(n // 2), int(n // 2), int(n // 2)

            _inject_delta(src, pA, float(src_qA))
            if collider_enable_b:
                _inject_delta(src, pB, float(src_qB))
            if collider_nucleus:
                cx = 0.5 * float(n)
                cy = 0.5 * float(n)
                cz = 0.5 * float(n)
                # Nucleus is a permanent anchor: it does not follow the walkers' injection shutter.
                # IMPORTANT: DC mode is treated as a static anchor, not a per-step forcing term.
                # Per-step DC forcing can accumulate energy and trigger non-finite blow-ups on long runs.
                if nucleus_mode == "sin":
                    src_qN = float(nucleus_q) * math.sin(float(nucleus_omega) * float(t) + float(nucleus_phase))
                    _nb_inject_ball_uniform(src, float(cx), float(cy), float(cz), float(src_qN), int(nucleus_r))
                else:
                    # DC nucleus is applied as a post-evolve soft pin (see below).
                    src_qN = 0.0

            # Source diagnostics prior to evolve
            src_E, src_max = _nb_energy_and_maxabs(src)

            try:
                phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, tp, 1)
            except ValueError as e:
                # Fail-loud with useful diagnostics for non-finite blow-ups.
                src_e, src_max2 = _nb_energy_and_maxabs(src)
                phi_e, phi_max2 = _nb_energy_and_maxabs(phi)
                vel_e, vel_max2 = _nb_energy_and_maxabs(vel)
                _log(
                    "[collider] ERROR evolve failed at step=%d t=%.3f: %s" % (int(step_i), float(t), str(e))
                )
                _log(
                    "[collider] diag src_max=%.6g src_E=%.6g phi_max=%.6g phi_E=%.6g vel_max=%.6g vel_E=%.6g nucleus=%d nucleus_mode=%s nucleus_q=%.6g halo=%d halo_r=%d halo_strength=%.6g backreact=%d k=%.6g axes=%s vmax=%.6g" % (
                        float(src_max2),
                        float(src_e),
                        float(phi_max2),
                        float(phi_e),
                        float(vel_max2),
                        float(vel_e),
                        int(1 if collider_nucleus else 0),
                        str(nucleus_mode),
                        float(nucleus_q),
                        int(1 if collider_halo else 0),
                        int(halo_r),
                        float(halo_strength),
                        int(backreact_on),
                        float(backreact_k),
                        str(backreact_axes),
                        float(backreact_vmax),
                    )
                )
                raise

            # DC nucleus as a *soft* anchor (stability): relax toward nucleus_q rather than hard-clamping.
            if collider_nucleus and nucleus_mode == "dc":
                nx = int(n // 2)
                ny = int(n // 2)
                nz = int(n // 2)
                a = float(nucleus_dc_alpha)
                if a > 0.0:
                    phi0 = float(phi[nx, ny, nz])
                    phi[nx, ny, nz] = np.float32((1.0 - a) * phi0 + a * float(nucleus_q))
                b = float(nucleus_dc_beta)
                if b > 0.0:
                    vel[nx, ny, nz] = np.float32(float(vel[nx, ny, nz]) * (1.0 - b))

            if collider_halo:
                if halo_center == "nucleus":
                    hc = int(n // 2)
                    hy = int(n // 2)
                    hz = int(n // 2)
                else:
                    hc = int(center_tc_cell[0])
                    hy = int(center_tc_cell[1])
                    hz = int(center_tc_cell[2])
                halo_touched_vel = int(_nb_apply_halo_damping(vel, int(hc), int(hy), int(hz), int(halo_r), float(halo_strength), int(halo_profile_i)))
                halo_touched_phi = int(_nb_apply_halo_damping(phi, int(hc), int(hy), int(hz), int(halo_r), float(halo_strength), int(halo_profile_i)))
                if halo_touched_vel > halo_touched_vel_max:
                    halo_touched_vel_max = int(halo_touched_vel)
                if halo_touched_phi > halo_touched_phi_max:
                    halo_touched_phi_max = int(halo_touched_phi)
            else:
                halo_touched_vel = 0
                halo_touched_phi = 0

            # Sample centre-point values after post-processing (DC nucleus + halo).
            c0 = int(n // 2)
            phi_c = float(phi[c0, c0, c0])
            vel_c = float(vel[c0, c0, c0])

            if backreact_on == 1:
                prev_vxA = float(vxA)
                prev_vxB = float(vxB)
                prev_vyA = float(vyA)
                prev_vzA = float(vzA)
                prev_vyB = float(vyB)
                prev_vzB = float(vzB)
                gxA, gyA, gzA = _nb_grad_phi_at(phi, float(pA[0]), float(pA[1]), float(pA[2]))
                if collider_enable_b:
                    gxB, gyB, gzB = _nb_grad_phi_at(phi, float(pB[0]), float(pB[1]), float(pB[2]))
                else:
                    gxB, gyB, gzB = 0.0, 0.0, 0.0
                # Mode controls the force sign. Axes controls whether we induce transverse drift.
                sgn = -1.0 if str(backreact_mode) == "repel" else 1.0
                dvA = float(sgn) * float(backreact_k) * float(gxA)
                if collider_enable_b:
                    dvB = float(sgn) * float(backreact_k) * float(gxB)
                    vxB = float(vxB + dvB)
                else:
                    dvB = 0.0
                vxA = float(vxA + dvA)

                if backreact_axes == "xyz":
                    dvyA = float(sgn) * float(backreact_k) * float(gyA)
                    dvzA = float(sgn) * float(backreact_k) * float(gzA)
                    if collider_enable_b:
                        dvyB = float(sgn) * float(backreact_k) * float(gyB)
                        dvzB = float(sgn) * float(backreact_k) * float(gzB)
                        vyB = float(vyB + dvyB)
                        vzB = float(vzB + dvzB)
                    else:
                        dvyB = 0.0
                        dvzB = 0.0
                    vyA = float(vyA + dvyA)
                    vzA = float(vzA + dvzA)
                else:
                    dvyA = 0.0
                    dvzA = 0.0
                    dvyB = 0.0
                    dvzB = 0.0

                vmax = float(backreact_vmax)
                if vxA > vmax:
                    vxA = vmax
                elif vxA < -vmax:
                    vxA = -vmax
                if collider_enable_b:
                    if vxB > vmax:
                        vxB = vmax
                    elif vxB < -vmax:
                        vxB = -vmax
                if backreact_axes == "xyz":
                    if vyA > vmax:
                        vyA = vmax
                    elif vyA < -vmax:
                        vyA = -vmax
                    if vzA > vmax:
                        vzA = vmax
                    elif vzA < -vmax:
                        vzA = -vmax
                    if collider_enable_b:
                        if vyB > vmax:
                            vyB = vmax
                        elif vyB < -vmax:
                            vyB = -vmax
                        if vzB > vmax:
                            vzB = vmax
                        elif vzB < -vmax:
                            vzB = -vmax

                axA = float(vxA - prev_vxA)
                axB = float(vxB - prev_vxB)
                PA = float(vxA) * float(dvA)
                PB = float(vxB) * float(dvB)
            else:
                gxA = 0.0
                gxB = 0.0
                gyA = 0.0
                gzA = 0.0
                gyB = 0.0
                gzB = 0.0
                dvA = 0.0
                dvB = 0.0
                dvyA = 0.0
                dvzA = 0.0
                dvyB = 0.0
                dvzB = 0.0
                axA = 0.0
                axB = 0.0
                PA = 0.0
                PB = 0.0

            # Integrate x for next step (y/z remain prescribed helix vs step index).
            xA = float(xA + vxA)
            xB = float(xB + vxB)
            # Integrate transverse drift only when enabled.
            if backreact_on == 1 and backreact_axes == "xyz":
                yA_drift = float(yA_drift + vyA)
                zA_drift = float(zA_drift + vzA)
                if collider_enable_b:
                    yB_drift = float(yB_drift + vyB)
                    zB_drift = float(zB_drift + vzB)

            if collider_enable_b:
                dx = float(pB[0] - pA[0])
                dy = float(pB[1] - pA[1])
                dz = float(pB[2] - pA[2])
                sep_x = abs(dx)
                sep_yz = math.sqrt(dy * dy + dz * dz)
                sep = math.sqrt(dx * dx + dy * dy + dz * dz)
                if sep < sep_min:
                    sep_min = float(sep)
                    sep_min_t = int(step_i)
            else:
                # Single-body mode: separation is undefined.
                dx = float("nan")
                dy = float("nan")
                dz = float("nan")
                sep_x = float("nan")
                sep_yz = float("nan")
                sep = float("nan")

            # Stage-4 scattering: instantaneous deflection angle from the collision axis.
            # theta := arctan(v_perp / |v_x|). Use induced drift (vy/vz) when enabled.
            vperpA = math.sqrt(float(vyA) * float(vyA) + float(vzA) * float(vzA))
            vperpB = math.sqrt(float(vyB) * float(vyB) + float(vzB) * float(vzB))
            ax_absA = abs(float(vxA))
            ax_absB = abs(float(vxB))
            thetaA = math.atan2(vperpA, ax_absA + 1e-12)
            thetaB = math.atan2(vperpB, ax_absB + 1e-12)
            theta_degA = float(thetaA) * (180.0 / math.pi)
            theta_degB = float(thetaB) * (180.0 / math.pi)
            octE = [0.0] * 8
            octN = [0] * 8

            e_phi, phi_max = _nb_energy_and_maxabs(phi)
            e_vel, vel_max = _nb_energy_and_maxabs(vel)

            e_tot = float(e_phi + e_vel)
            eps = 1e-12
            E_tot_dE = 0.0
            hold_E_tot_dE = 0.0
            if prev_E_tot is not None:
                E_tot_dE = float(e_tot - float(prev_E_tot))
                if in_hold == 1:
                    hold_E_tot_dE = float(E_tot_dE)
            prev_E_tot = float(e_tot)

            if collider_enable_b:
                x_mid = int(round(0.5 * (pA[0] + pB[0])))
            else:
                x_mid = int(round(pA[0]))
            phi_midE = _nb_energy_slab_x(phi, x_mid, 3)
            vel_midE = _nb_energy_slab_x(vel, x_mid, 3)

            phi_A_E = _energy_local_ball(phi, pA, 3)
            vel_A_E = _energy_local_ball(vel, pA, 3)
            if collider_enable_b:
                phi_B_E = _energy_local_ball(phi, pB, 3)
                vel_B_E = _energy_local_ball(vel, pB, 3)
            else:
                phi_B_E = float("nan")
                vel_B_E = float("nan")
            center_phi_E = _energy_local_ball(phi, center_tc, int(center_ball_r))
            center_vel_E = _energy_local_ball(vel, center_tc, int(center_ball_r))
            center_tot_E = float(center_phi_E + center_vel_E)
            center_frac_tot = float(center_tot_E) / float(e_tot + eps)
            center_dE_tot = 0.0
            if prev_center_tot is not None and in_hold == 1:
                center_dE_tot = float(center_tot_E - float(prev_center_tot))
            prev_center_tot = float(center_tot_E)

            mid_frac_phi = float(phi_midE) / float(e_phi + eps)
            mid_frac_vel = float(vel_midE) / float(e_vel + eps)
            localA_frac_phi = float(phi_A_E) / float(e_phi + eps)
            localA_frac_vel = float(vel_A_E) / float(e_vel + eps)
            if collider_enable_b:
                localB_frac_phi = float(phi_B_E) / float(e_phi + eps)
                localB_frac_vel = float(vel_B_E) / float(e_vel + eps)
            else:
                localB_frac_phi = float("nan")
                localB_frac_vel = float("nan")

            mid_frac_tot = float(phi_midE + vel_midE) / float(e_tot + eps)
            localA_frac_tot = float(phi_A_E + vel_A_E) / float(e_tot + eps)
            if collider_enable_b:
                localB_frac_tot = float(phi_B_E + vel_B_E) / float(e_tot + eps)
            else:
                localB_frac_tot = float("nan")

            if collider_detectors:
                shell1E_phi, shell1N_phi = _energy_shell_at_cell(phi, center_tc_cell[0], center_tc_cell[1], center_tc_cell[2], shell1_offsets)
                shell1E_vel, shell1N_vel = _energy_shell_at_cell(vel, center_tc_cell[0], center_tc_cell[1], center_tc_cell[2], shell1_offsets)
                shell1N = int(min(shell1N_phi, shell1N_vel))
                shell1_frac_phi = float(shell1E_phi) / float(e_phi + eps)
                shell1_frac_vel = float(shell1E_vel) / float(e_vel + eps)
                shell1_frac_tot = float(shell1E_phi + shell1E_vel) / float(e_tot + eps)

                shell2E_phi, shell2N_phi = _energy_shell_at_cell(phi, center_tc_cell[0], center_tc_cell[1], center_tc_cell[2], shell2_offsets)
                shell2E_vel, shell2N_vel = _energy_shell_at_cell(vel, center_tc_cell[0], center_tc_cell[1], center_tc_cell[2], shell2_offsets)
                shell2N = int(min(shell2N_phi, shell2N_vel))
                shell2_frac_phi = float(shell2E_phi) / float(e_phi + eps)
                shell2_frac_vel = float(shell2E_vel) / float(e_vel + eps)
                shell2_tot = float(shell2E_phi + shell2E_vel)
                shell2_frac_tot = float(shell2_tot) / float(e_tot + eps)

                shell2_minus_shell1_tot = float(shell2_tot - (shell1E_phi + shell1E_vel))

                shell2_pow_tot = 0.0
                if prev_shell2_tot is not None:
                    shell2_pow_tot = float(shell2_tot - float(prev_shell2_tot))
                prev_shell2_tot = float(shell2_tot)

                if collider_octants:
                    oct_phi, octN_phi = _nb_energy_shell_octants(phi, int(center_tc_cell[0]), int(center_tc_cell[1]), int(center_tc_cell[2]), shell2_offsets)
                    oct_vel, octN_vel = _nb_energy_shell_octants(vel, int(center_tc_cell[0]), int(center_tc_cell[1]), int(center_tc_cell[2]), shell2_offsets)
                    # Use the min counts (phi/vel) to keep bookkeeping consistent.
                    for j in range(8):
                        octE[j] = float(oct_phi[j] + oct_vel[j])
                        octN[j] = int(octN_phi[j]) if int(octN_phi[j]) < int(octN_vel[j]) else int(octN_vel[j])
            else:
                shell1E_phi = 0.0
                shell1E_vel = 0.0
                shell1N = 0
                shell1_frac_phi = 0.0
                shell1_frac_vel = 0.0
                shell1_frac_tot = 0.0

                shell2E_phi = 0.0
                shell2E_vel = 0.0
                shell2N = 0
                shell2_frac_phi = 0.0
                shell2_frac_vel = 0.0
                shell2_tot = 0.0
                shell2_frac_tot = 0.0

                shell2_minus_shell1_tot = 0.0
                shell2_pow_tot = 0.0

                center_phi_E = 0.0
                center_vel_E = 0.0
                center_tot_E = 0.0
                center_frac_tot = 0.0
                center_dE_tot = 0.0

            t_rel = int(step_i) - int(t_c_int)

            w.writerow([
                int(step_i),
                int(t_c_int),
                int(t_rel),
                float(inj_scale),
                int(in_hold),
                int(backreact_on),
                float(vxA),
                float(vxB),
                float(pA[0]), float(pA[1]), float(pA[2]),
                float(pB[0]), float(pB[1]), float(pB[2]),
                float(sep),
                int(inA),
                int(inB),
                float(phi_max),
                float(vel_max),
                float(e_phi),
                float(e_vel),
                float(e_tot),
                float(E_tot_dE),
                float(hold_E_tot_dE),
                float(phi_midE),
                float(vel_midE),
                float(phi_A_E),
                float(phi_B_E),
                float(vel_A_E),
                float(vel_B_E),
                float(sep_x),
                float(sep_yz),
                float(mid_frac_phi),
                float(mid_frac_vel),
                float(mid_frac_tot),
                float(localA_frac_phi),
                float(localB_frac_phi),
                float(localA_frac_vel),
                float(localB_frac_vel),
                float(localA_frac_tot),
                float(localB_frac_tot),
                float(center_phi_E),
                float(center_vel_E),
                float(center_tot_E),
                float(center_frac_tot),
                float(center_dE_tot),
                float(shell1E_phi),
                float(shell1E_vel),
                int(shell1N),
                float(shell1_frac_phi),
                float(shell1_frac_vel),
                float(shell1_frac_tot),
                float(shell2E_phi),
                float(shell2E_vel),
                int(shell2N),
                float(shell2_frac_phi),
                float(shell2_frac_vel),
                float(shell2_frac_tot),
                float(shell2_minus_shell1_tot),
                float(shell2_pow_tot),
                float(gxA),
                float(gxB),
                float(dvA),
                float(dvB),
                float(axA),
                float(axB),
                float(PA),
                float(PB),
                float(sep_min),
                int(sep_min_t),
                int(cut_step),
                int(hold_grace_steps),
                int(hold_steps),
                float(vyA),
                float(vzA),
                float(vyB),
                float(vzB),
                float(gyA),
                float(gzA),
                float(gyB),
                float(gzB),
                float(dvyA),
                float(dvzA),
                float(dvyB),
                float(dvzB),
                float(thetaA),
                float(thetaB),
                float(theta_degA),
                float(theta_degB),
                float(octE[0]), float(octE[1]), float(octE[2]), float(octE[3]), float(octE[4]), float(octE[5]), float(octE[6]), float(octE[7]),
                int(octN[0]), int(octN[1]), int(octN[2]), int(octN[3]), int(octN[4]), int(octN[5]), int(octN[6]), int(octN[7]),
                float(angA),
                float(angB),
                float(dphi),
                float(phi_c),
                float(vel_c),
                float(src_max),
                float(src_E),
                float(src_qA),
                float(src_qB),
                float(src_qN),
                int(cellAx), int(cellAy), int(cellAz),
                int(cellBx), int(cellBy), int(cellBz),
                int(cellNx), int(cellNy), int(cellNz),
                int(halo_touched_vel),
                int(halo_touched_phi),
                int(halo_touched_vel_max),
                int(halo_touched_phi_max),
            ])

            if progress_cb is not None:
                if (step_i == 0) or ((step_i + 1) % 200 == 0) or (step_i + 1 == int(steps)):
                    done = int(step_i + 1)
                    delta = int(done - last_progress)
                    if delta > 0:
                        progress_cb(delta)
                        last_progress = done

        f.write("# summary\n")
        f.write("# sep_min=%.6f sep_min_t=%d t_c_int=%d\n" % (float(sep_min), int(sep_min_t), int(t_c_int)))
        f.write("# hold=%d cut_step=%d hold_grace_steps=%d hold_steps=%d center_ball_r=%d\n" % (
            int(1 if collider_hold else 0),
            int(cut_step),
            int(hold_grace_steps),
            int(hold_steps),
            int(center_ball_r),
        ))
        f.write("# halo_touched_vel_max=%d halo_touched_phi_max=%d\n" % (int(halo_touched_vel_max), int(halo_touched_phi_max)))

    wall_s = float(t_run.stop_s())
    _log(
        "[collider] done steps=%d spin_b=%d csv=%s wall=%.3fs" % (
            int(steps),
            int(spin_b),
            csv_path,
            wall_s,
        )
    )
    # Terminal summary is safe only if we are not mid-progress-bar updates.
    if progress_cb is None:
        print(
            "[collider] done steps=%d spin_b=%d csv=%s wall=%.3fs" % (
                int(steps),
                int(spin_b),
                csv_path,
                wall_s,
            )
        )