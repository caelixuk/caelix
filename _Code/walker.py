# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""walker.py — dynamic Heavy Walker experiment (moving delta source)

Role
----
The Heavy Walker is a diagnostic experiment for lag / wake effects in the traffic
solver. We move a unit point source through the lattice and measure asymmetry in
the resulting field.

This runner is intentionally self-contained: it does not build the global index
map; it directly steps the traffic solver on a 3D lattice with a moving source.

Modes
-----
- Linear path: axis-aligned unit steps (±x, ±y, ±z) with optional hold phase.
- Circle path: 4-connected approximation to a circle in the x-y plane.
- Mach-1 convenience: telegraph mode with fixed tick iters (2/2).

Measured outputs
----------------
Per step we record (among other fields):
- `phi_max`
- front/back sums in a local cube split along the motion axis
- `asym = (front - back)/(front + back)`
- `hubble = (front - back)/total_field`
- optional probe dipole metrics at four cardinal points around the center

Sweep helper
------------
`run_heavy_walker_sweep()` runs a tick-iter list and writes a summary CSV.

Contracts
---------
- Walker uses moving delta injections internally; it does not depend on the global --delta-load pipeline switch.
- Hard boundaries; walker positions must stay inside a safe diagnostic margin.
- Uses `traffic.evolve_*_traffic_steps` for deterministic stepping.

Flat-module layout
------------------
Lives alongside `core.py` and is imported as:
  `from walker import run_heavy_walker, run_heavy_walker_sweep`
"""

from __future__ import annotations

import math
import os
import csv
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np

from params import PipelineParams
from traffic import evolve_diffusion_traffic_steps, evolve_telegraph_traffic_steps
from utils import _as_float, ensure_parent_dir


def _axis_from_step(dx: int, dy: int, dz: int) -> Tuple[int, int]:
    """Return (axis, sign) for an axis-aligned unit step.

    Special case: (0,0,0) is treated as a 'hold' (no motion). We still need a
    stable axis for front/back diagnostics, so we define this as +x.
    """
    if (dx, dy, dz) == (0, 0, 0):
        return 0, 1
    if (dx, dy, dz) == (1, 0, 0):
        return 0, 1
    if (dx, dy, dz) == (-1, 0, 0):
        return 0, -1
    if (dx, dy, dz) == (0, 1, 0):
        return 1, 1
    if (dx, dy, dz) == (0, -1, 0):
        return 1, -1
    if (dx, dy, dz) == (0, 0, 1):
        return 2, 1
    if (dx, dy, dz) == (0, 0, -1):
        return 2, -1
    raise ValueError("walker step must be axis-aligned unit (±x, ±y, ±z), or (0,0,0) for hold")


def _axis_from_vec(dx: int, dy: int, dz: int, fallback: Tuple[int, int]) -> Tuple[int, int]:
    """Pick (axis, sign) from a general motion vector.

    Uses the largest-magnitude component; falls back when dx=dy=dz=0.
    """
    if (dx, dy, dz) == (0, 0, 0):
        return fallback
    adx = abs(int(dx))
    ady = abs(int(dy))
    adz = abs(int(dz))
    if adx >= ady and adx >= adz:
        return 0, (1 if int(dx) > 0 else -1)
    if ady >= adx and ady >= adz:
        return 1, (1 if int(dy) > 0 else -1)
    return 2, (1 if int(dz) > 0 else -1)


def _circle_pos(cxyz: Tuple[int, int, int], radius: int, theta: float) -> Tuple[int, int, int]:
    cx, cy, cz = int(cxyz[0]), int(cxyz[1]), int(cxyz[2])
    r = int(radius)
    if r < 1:
        raise ValueError("--walker-circle-radius must be >= 1")
    x = int(round(float(cx) + float(r) * math.cos(float(theta))))
    y = int(round(float(cy) + float(r) * math.sin(float(theta))))
    z = int(cz)
    return x, y, z


def _circle_step_next(
    pos: Tuple[int, int, int],
    cxyz: Tuple[int, int, int],
    radius: int,
    theta: float,
    theta_step: float,
) -> Tuple[Tuple[int, int, int], float, Tuple[int, int, int]]:
    """Advance one lattice step along an approximate circle using only 4-connected moves.

    We track a continuous angle (theta) to define a moving target point on the ideal circle,
    but we move towards the rounded target by ONE axis step per walker step.

    Guarantees:
      - Returned step delta is (±1,0,0) or (0,±1,0) or (0,0,0) only.
      - Avoids (1,1,0) diagonal jumps caused by rounding.
    """
    cx, cy, cz = int(cxyz[0]), int(cxyz[1]), int(cxyz[2])
    x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
    r = int(radius)
    if r < 1:
        raise ValueError("--walker-circle-radius must be >= 1")

    tries = 0
    tx, ty, tz = x, y, z
    th = float(theta)
    while tries < 32:
        th = th + float(theta_step)
        tx = int(round(float(cx) + float(r) * math.cos(th)))
        ty = int(round(float(cy) + float(r) * math.sin(th)))
        tz = int(cz)
        if (tx, ty, tz) != (x, y, z):
            break
        tries += 1

    dx = int(tx) - int(x)
    dy = int(ty) - int(y)
    dz = int(tz) - int(z)

    step_dx = 0
    step_dy = 0
    step_dz = 0

    adx = abs(dx)
    ady = abs(dy)
    adz = abs(dz)

    if adx >= ady and adx >= adz and dx != 0:
        step_dx = 1 if dx > 0 else -1
    elif ady >= adx and ady >= adz and dy != 0:
        step_dy = 1 if dy > 0 else -1
    elif dz != 0:
        step_dz = 1 if dz > 0 else -1

    nx = int(x + step_dx)
    ny = int(y + step_dy)
    nz = int(z + step_dz)

    return (nx, ny, nz), float(th), (int(step_dx), int(step_dy), int(step_dz))


def _front_back_sums(phi: np.ndarray, pos: Tuple[int, int, int], axis: int, sign: int, r_local: int) -> Tuple[float, float]:
    """Sum phi in a local cube split into front/back halves along (axis, sign)."""
    n = int(phi.shape[0])
    x0, y0, z0 = (int(pos[0]), int(pos[1]), int(pos[2]))

    r = int(r_local)
    if r < 1:
        raise ValueError("--walker-r-local must be >= 1")

    x1 = max(0, x0 - r)
    x2 = min(n, x0 + r + 1)
    y1 = max(0, y0 - r)
    y2 = min(n, y0 + r + 1)
    z1 = max(0, z0 - r)
    z2 = min(n, z0 + r + 1)

    cube = phi[x1:x2, y1:y2, z1:z2].astype(np.float64)

    if axis == 0:
        idx0 = x0 - x1
    elif axis == 1:
        idx0 = y0 - y1
    else:
        idx0 = z0 - z1

    if sign > 0:
        sl_front = [slice(None), slice(None), slice(None)]
        sl_back = [slice(None), slice(None), slice(None)]
        sl_front[axis] = slice(idx0 + 1, None)
        sl_back[axis] = slice(None, idx0)
    else:
        sl_front = [slice(None), slice(None), slice(None)]
        sl_back = [slice(None), slice(None), slice(None)]
        sl_front[axis] = slice(None, idx0)
        sl_back[axis] = slice(idx0 + 1, None)

    front = float(np.sum(cube[tuple(sl_front)]))
    back = float(np.sum(cube[tuple(sl_back)]))
    return front, back


def walker_work_units(params: PipelineParams, tick_iters_move: int, tick_iters_hold: int) -> int:
    """Estimate 'work units' for a single walker run.

    Purpose: provide a proportional cost model for progress bars / sweeps.
    We count traffic evolution iterations (hot cost), not Python loop iterations.
    """
    warm = int(params.traffic.iters)
    move_steps = int(params.walker_steps)
    hold_steps = int(params.walker_hold_steps)
    if bool(getattr(params, "walker_mach1", False)):
        tick_iters_move = 2
        tick_iters_hold = 2
    if warm < 0:
        warm = 0
    if move_steps < 0:
        move_steps = 0
    if hold_steps < 0:
        hold_steps = 0
    if int(tick_iters_move) < 1:
        raise ValueError("walker_work_units: tick_iters_move must be >= 1")
    if int(tick_iters_hold) < 1:
        raise ValueError("walker_work_units: tick_iters_hold must be >= 1")
    return int(warm + (move_steps * int(tick_iters_move)) + (hold_steps * int(tick_iters_hold)))


def walker_sweep_work_units(params: PipelineParams, tick_list: List[int]) -> int:
    """Total work units for a sweep over tick iters."""
    total = 0
    for t in tick_list:
        total += int(walker_work_units(params, int(t), int(t)))
    return int(total)


def _write_provenance_header(f, provenance_header: Optional[str], artefact: str) -> None:
    """Write provenance/comment header lines to an already-open text file."""
    if provenance_header is not None:
        h = str(provenance_header)
        if h != "":
            if not h.endswith("\n"):
                h = h + "\n"
            f.write(h)
    if str(artefact).strip() != "":
        f.write(f"# artefact={str(artefact).strip()}\n")


def run_heavy_walker(
    params: PipelineParams,
    provenance_header: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    log_line: Optional[Callable[[str], None]] = None,
    verbose: bool = False,
    tick_iters_move_override: Optional[int] = None,
    tick_iters_hold_override: Optional[int] = None,
    dump_walker_path: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Dynamic experiment: move a point source and watch the traffic field lag."""

    path = str(params.walker_path).strip().lower()
    if path not in ("linear", "circle"):
        raise ValueError("--walker-path must be one of: linear, circle")

    dx = int(params.walker_dx)
    dy = int(params.walker_dy)
    dz = int(params.walker_dz)

    n = int(params.lattice.n)
    c0 = n // 2
    margin = int(params.delta_margin)
    if margin < 1:
        raise ValueError("--delta-margin must be >= 1")

    move_steps = int(params.walker_steps)
    if move_steps < 0:
        raise ValueError("--walker-steps must be >= 0")

    move_tick_iters = int(params.walker_tick_iters)
    if tick_iters_move_override is not None:
        move_tick_iters = int(tick_iters_move_override)
    if move_tick_iters < 1:
        raise ValueError("--walker-tick-iters must be >= 1")

    hold_steps = int(params.walker_hold_steps)
    if hold_steps < 0:
        raise ValueError("--walker-hold-steps must be >= 0")

    hold_tick_iters = int(params.walker_hold_tick_iters)
    if tick_iters_hold_override is not None:
        hold_tick_iters = int(tick_iters_hold_override)
    if hold_tick_iters < 1:
        raise ValueError("--walker-hold-tick-iters must be >= 1")
    if bool(params.walker_mach1):
        tp_mode = str(params.traffic.mode).strip().lower()
        if tp_mode != "telegraph":
            raise ValueError("--walker-mach1 requires --traffic-mode telegraph")
        move_tick_iters = 2
        hold_tick_iters = 2

    r_local = int(params.walker_r_local)
    if r_local < 1:
        raise ValueError("--walker-r-local must be >= 1")

    safe_margin = int(margin + r_local + 1)
    if safe_margin < 1:
        raise ValueError("internal: safe_margin must be >= 1")

    if (c0 - safe_margin) < 0 or (c0 + safe_margin) >= n:
        raise ValueError("walker has no legal start position with this --n/--delta-margin/--walker-r-local")

    cx = int(params.walker_center_x) if int(params.walker_center_x) >= 0 else int(c0)
    cy = int(params.walker_center_y) if int(params.walker_center_y) >= 0 else int(c0)
    cz = int(params.walker_center_z) if int(params.walker_center_z) >= 0 else int(c0)

    if not (safe_margin <= cx <= (n - 1 - safe_margin)):
        raise ValueError("--walker-center-x places the orbit too close to boundary")
    if not (safe_margin <= cy <= (n - 1 - safe_margin)):
        raise ValueError("--walker-center-y places the orbit too close to boundary")
    if not (safe_margin <= cz <= (n - 1 - safe_margin)):
        raise ValueError("--walker-center-z places the orbit too close to boundary")

    probe_r = int(params.walker_probe_r)
    if probe_r < 0:
        raise ValueError("--walker-probe-r must be >= 0")

    if probe_r > 0:
        lim_x = min(int(cx - safe_margin), int((n - 1 - safe_margin) - cx))
        lim_y = min(int(cy - safe_margin), int((n - 1 - safe_margin) - cy))
        lim = int(min(lim_x, lim_y))
        if probe_r > lim:
            raise ValueError("--walker-probe-r is too large for this --n/--delta-margin/--walker-r-local and center")

        probes = (
            (int(cx + probe_r), int(cy), int(cz)),
            (int(cx), int(cy + probe_r), int(cz)),
            (int(cx - probe_r), int(cy), int(cz)),
            (int(cx), int(cy - probe_r), int(cz)),
        )
    else:
        probes = ()

    if path == "linear":
        if move_steps > 0 and (dx, dy, dz) == (0, 0, 0):
            raise ValueError("move phase requires a non-zero unit step: choose one of ±x, ±y, ±z")
        if move_steps > 0:
            step_len = abs(dx) + abs(dy) + abs(dz)
            if step_len != 1:
                raise ValueError("walker step must be axis-aligned unit (±x, ±y, ±z)")

        axis, sign = _axis_from_step(dx, dy, dz)

        max_disp = int(move_steps)
        max_pos = int((n - safe_margin - 1) - cx)
        max_neg = int(cx - safe_margin)
        max_allowed = int(max_pos if sign > 0 else max_neg)
        if max_allowed < 0:
            raise ValueError("walker has no legal start position with this --n/--delta-margin/--walker-r-local")
        if max_disp > max_allowed:
            raise ValueError(
                "walker would hit diagnostic boundary after %d steps (requested %d); reduce --walker-steps or increase --n/--delta-margin, or reduce --walker-r-local" % (
                    max_allowed, max_disp,
                )
            )

        pos = [cx, cy, cz]
        circle_radius = 0
        circle_period = 0
    else:
        circle_radius = int(params.walker_circle_radius)
        if circle_radius < 1:
            raise ValueError("--walker-circle-radius must be >= 1")
        circle_period = int(params.walker_circle_period)
        if circle_period < 0:
            raise ValueError("--walker-circle-period must be >= 0")
        if circle_period == 0:
            circle_period = max(1, int(move_steps) if int(move_steps) > 0 else 1)

        lim_x = min(int(cx - safe_margin), int((n - 1 - safe_margin) - cx))
        lim_y = min(int(cy - safe_margin), int((n - 1 - safe_margin) - cy))
        lim_z = min(int(cz - safe_margin), int((n - 1 - safe_margin) - cz))
        lim = int(min(lim_x, lim_y, lim_z))
        if circle_radius > lim:
            raise ValueError("--walker-circle-radius is too large for this --n/--delta-margin/--walker-r-local and center")

        x0, y0, z0 = _circle_pos((cx, cy, cz), circle_radius, 0.0)
        pos = [int(x0), int(y0), int(z0)]

        theta = 0.0
        theta_step = (2.0 * math.pi) / float(circle_period)

        axis, sign = 0, 1

    phi = np.zeros((n, n, n), dtype=np.float32)
    vel = np.zeros((n, n, n), dtype=np.float32)
    src = np.zeros((n, n, n), dtype=np.float32)

    src[:, :, :] = 0.0
    src[pos[0], pos[1], pos[2]] = 1.0
    if str(params.traffic.mode).strip().lower() == "telegraph":
        phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, params.traffic, int(params.traffic.iters))
    else:
        phi = evolve_diffusion_traffic_steps(phi, src, params.traffic, int(params.traffic.iters))

    if progress_cb is not None:
        progress_cb(int(max(0, int(params.traffic.iters))))

    rows: List[Dict[str, object]] = []

    def _probe_metrics(phi_in: np.ndarray, vel_in: np.ndarray) -> Dict[str, float]:
        if probe_r <= 0:
            return {}

        p_phi = []
        p_vel = []
        for (px, py, pz) in probes:
            p_phi.append(float(phi_in[int(px), int(py), int(pz)]))
            p_vel.append(float(vel_in[int(px), int(py), int(pz)]))

        dx_phi = float(p_phi[0] - p_phi[2])
        dy_phi = float(p_phi[1] - p_phi[3])
        dx_vel = float(p_vel[0] - p_vel[2])
        dy_vel = float(p_vel[1] - p_vel[3])

        mag_phi = float(math.hypot(dx_phi, dy_phi))
        ang_phi = float(math.atan2(dy_phi, dx_phi))
        mag_vel = float(math.hypot(dx_vel, dy_vel))
        ang_vel = float(math.atan2(dy_vel, dx_vel))

        return {
            "probe_r": float(probe_r),
            "phi_p0": float(p_phi[0]),
            "phi_p1": float(p_phi[1]),
            "phi_p2": float(p_phi[2]),
            "phi_p3": float(p_phi[3]),
            "vel_p0": float(p_vel[0]),
            "vel_p1": float(p_vel[1]),
            "vel_p2": float(p_vel[2]),
            "vel_p3": float(p_vel[3]),
            "dipole_mag_phi": float(mag_phi),
            "dipole_ang_phi": float(ang_phi),
            "dipole_mag_vel": float(mag_vel),
            "dipole_ang_vel": float(ang_vel),
        }

    tp_mode = str(params.traffic.mode).strip().lower()
    if tp_mode == "telegraph":
        c_est = float(math.sqrt(float(params.traffic.c2)) * float(params.traffic.dt))
    else:
        c_est = float("nan")

    def _v_est(step_dx: int, step_dy: int, step_dz: int, tick_iters: int) -> float:
        step_len = abs(int(step_dx)) + abs(int(step_dy)) + abs(int(step_dz))
        if step_len == 0:
            return 0.0
        return float(step_len) / float(tick_iters)

    t_global = 0

    last_axis_sign = (axis, sign)
    last_step = (0, 0, 0)

    for _ in range(1, move_steps + 1):
        t_global += 1

        prev = (int(pos[0]), int(pos[1]), int(pos[2]))

        if path == "linear":
            pos[0] += dx
            pos[1] += dy
            pos[2] += dz
        else:
            next_pos, theta, step_vec = _circle_step_next(
                (int(pos[0]), int(pos[1]), int(pos[2])),
                (int(cx), int(cy), int(cz)),
                int(circle_radius),
                float(theta),
                float(theta_step),
            )
            pos[0] = int(next_pos[0])
            pos[1] = int(next_pos[1])
            pos[2] = int(next_pos[2])
            step_dx, step_dy, step_dz = (int(step_vec[0]), int(step_vec[1]), int(step_vec[2]))

        if path == "linear":
            step_dx = int(pos[0]) - int(prev[0])
            step_dy = int(pos[1]) - int(prev[1])
            step_dz = int(pos[2]) - int(prev[2])
        last_step = (step_dx, step_dy, step_dz)

        axis, sign = _axis_from_vec(step_dx, step_dy, step_dz, fallback=last_axis_sign)
        last_axis_sign = (axis, sign)

        src[:, :, :] = 0.0
        src[pos[0], pos[1], pos[2]] = 1.0

        if str(params.traffic.mode).strip().lower() == "telegraph":
            phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, params.traffic, move_tick_iters)
        else:
            phi = evolve_diffusion_traffic_steps(phi, src, params.traffic, move_tick_iters)

        if progress_cb is not None:
            progress_cb(int(move_tick_iters))

        front, back = _front_back_sums(phi, (pos[0], pos[1], pos[2]), axis=axis, sign=sign, r_local=r_local)
        denom = front + back + 1e-12
        asym = (front - back) / denom
        total_field = float(np.sum(phi.astype(np.float64)))
        hubble = (front - back) / (total_field + 1e-12)

        pm = float(np.max(phi))

        v = float(_v_est(step_dx, step_dy, step_dz, move_tick_iters))
        mach = (v / c_est) if (math.isfinite(c_est) and c_est > 0.0) else float("nan")

        row = {
            "phase": "move",
            "path": str(path),
            "cx": float(cx),
            "cy": float(cy),
            "cz": float(cz),
            "circle_radius": float(circle_radius),
            "circle_period": float(circle_period),
            "step_dx": float(step_dx),
            "step_dy": float(step_dy),
            "step_dz": float(step_dz),
            "axis": float(axis),
            "sign": float(sign),
            "t": float(t_global),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "phi_max": float(pm),
            "front": float(front),
            "back": float(back),
            "total_field": float(total_field),
            "hubble": float(hubble),
            "asym": float(asym),
            "tick_iters": float(move_tick_iters),
            "v_est": float(v),
            "c_est": float(c_est),
            "mach": float(mach),
            "stop_event": 0.0,
        }
        row.update(_probe_metrics(phi, vel))
        rows.append(row)

        line = "[walker] phase=move t=%d pos=(%d,%d,%d) phi_max=%.6g front=%.6g back=%.6g total=%.6g hubble=%.6g asym=%.6g mach=%.6g" % (
            t_global, pos[0], pos[1], pos[2], pm, front, back, total_field, hubble, asym, mach,
        )
        if bool(verbose):
            if log_line is not None:
                log_line(str(line))
            else:
                print(str(line))
        else:
            if log_line is not None:
                log_line(str(line))

    for _ in range(int(hold_steps)):
        t_global += 1

        src[:, :, :] = 0.0
        src[pos[0], pos[1], pos[2]] = float(params.walker_hold_inject)

        if str(params.traffic.mode).strip().lower() == "telegraph":
            phi, vel = evolve_telegraph_traffic_steps(phi, vel, src, params.traffic, hold_tick_iters)
        else:
            phi = evolve_diffusion_traffic_steps(phi, src, params.traffic, hold_tick_iters)

        if progress_cb is not None:
            progress_cb(int(hold_tick_iters))

        axis, sign = last_axis_sign
        front, back = _front_back_sums(phi, (pos[0], pos[1], pos[2]), axis=axis, sign=sign, r_local=r_local)
        denom = front + back + 1e-12
        asym = (front - back) / denom
        total_field = float(np.sum(phi.astype(np.float64)))
        hubble = (front - back) / (total_field + 1e-12)

        pm = float(np.max(phi))

        v = float(_v_est(0, 0, 0, hold_tick_iters))
        mach = (v / c_est) if (math.isfinite(c_est) and c_est > 0.0) else float("nan")
        stop_event = 1.0 if t_global == float(move_steps) + 1.0 else 0.0

        row = {
            "phase": "hold",
            "path": str(path),
            "cx": float(cx),
            "cy": float(cy),
            "cz": float(cz),
            "circle_radius": float(circle_radius),
            "circle_period": float(circle_period),
            "step_dx": 0.0,
            "step_dy": 0.0,
            "step_dz": 0.0,
            "axis": float(axis),
            "sign": float(sign),
            "t": float(t_global),
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "phi_max": float(pm),
            "front": float(front),
            "back": float(back),
            "total_field": float(total_field),
            "hubble": float(hubble),
            "asym": float(asym),
            "tick_iters": float(hold_tick_iters),
            "v_est": float(v),
            "c_est": float(c_est),
            "mach": float(mach),
            "stop_event": float(stop_event),
        }
        row.update(_probe_metrics(phi, vel))
        rows.append(row)

        line = "[walker] phase=hold t=%d pos=(%d,%d,%d) phi_max=%.6g front=%.6g back=%.6g total=%.6g hubble=%.6g asym=%.6g mach=%.6g stop=%.0f" % (
            t_global, pos[0], pos[1], pos[2], pm, front, back, total_field, hubble, asym, mach, stop_event,
        )
        if bool(verbose):
            if log_line is not None:
                log_line(str(line))
            else:
                print(str(line))
        else:
            if log_line is not None:
                log_line(str(line))

    dump_path = str(dump_walker_path if dump_walker_path is not None else params.dump_walker).strip()
    if dump_path != "":
        ensure_parent_dir(dump_path)
        cols = [
            "phase", "path", "t", "x", "y", "z",
            "cx", "cy", "cz", "circle_radius", "circle_period",
            "step_dx", "step_dy", "step_dz", "axis", "sign",
            "phi_max", "front", "back", "total_field", "hubble", "asym",
            "tick_iters", "v_est", "c_est", "mach", "stop_event",
            "probe_r",
            "phi_p0", "phi_p1", "phi_p2", "phi_p3",
            "vel_p0", "vel_p1", "vel_p2", "vel_p3",
            "dipole_mag_phi", "dipole_ang_phi",
            "dipole_mag_vel", "dipole_ang_vel",
        ]
        with open(dump_path, "w", newline="") as f:
            _write_provenance_header(f, provenance_header, artefact="walker_trace")
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in cols])
        line = f"[walker] wrote {dump_path}"
        if bool(verbose):
            if log_line is not None:
                log_line(str(line))
            else:
                print(str(line))
        else:
            if log_line is not None:
                log_line(str(line))

    return rows


def _parse_int_list(s: str, name: str) -> List[int]:
    raw = str(s).strip()
    if raw == "":
        raise ValueError(f"{name} must be a non-empty comma-separated list")
    parts = [p.strip() for p in raw.split(",")]
    out: List[int] = []
    for p in parts:
        if p == "":
            raise ValueError(f"{name} contains an empty entry")
        try:
            v = int(p)
        except ValueError as e:
            raise ValueError(f"{name} contains a non-integer entry: {p!r}") from e
        if v < 1:
            raise ValueError(f"{name} entries must be >= 1")
        out.append(v)
    if len(out) < 1:
        raise ValueError(f"{name} must contain at least one entry")
    return out


def run_heavy_walker_sweep(params: PipelineParams, tick_list: List[int], out_csv: str, provenance_header: Optional[str] = None, progress_cb: Optional[Callable[[int], None]] = None, log_line: Optional[Callable[[str], None]] = None, verbose: bool = False) -> None:
    out_csv = str(out_csv).strip()
    if out_csv == "":
        out_csv = str(getattr(params, "walker_sweep_out", "")).strip()

    if out_csv == "":
        # Default: place sweep output alongside other run artefacts.
        # Prefer a per-run directory if available; fall back to the configured output root.
        run_dir = str(getattr(params, "run_dir", "")).strip()
        if run_dir == "":
            run_dir = str(getattr(params, "out_dir", "")).strip()
        if run_dir == "":
            run_dir = str(getattr(params, "out", "")).strip()
        if run_dir == "":
            raise ValueError("walker sweep: no output directory available (set --walker-sweep-out)")
        out_csv = os.path.join(run_dir, "_csv", "walker_sweep.csv")

    ensure_parent_dir(out_csv)

    cols = [
        "move_tick_iters",
        "hold_tick_iters",
        "mach",
        "move_asym_min",
        "move_asym_last",
        "hold_asym_max",
        "hold_asym_final",
        "overtake_step",
    ]

    summary_rows: List[Dict[str, object]] = []

    for tick in tick_list:
        rows = run_heavy_walker(
            params,
            provenance_header=provenance_header,
            progress_cb=progress_cb,
            log_line=log_line,
            verbose=bool(verbose),
            tick_iters_move_override=int(tick),
            tick_iters_hold_override=int(tick),
            dump_walker_path="",
        )

        move_rows = [r for r in rows if str(r.get("phase")) == "move"]
        hold_rows = [r for r in rows if str(r.get("phase")) == "hold"]
        if len(move_rows) < 1:
            raise ValueError("walker sweep: no move rows produced")
        if int(params.walker_hold_steps) > 0 and len(hold_rows) < 1:
            raise ValueError("walker sweep: hold requested but no hold rows produced")

        move_asym = np.array([
            float(_as_float(r.get("asym"), "walker.row.asym"))
            for r in move_rows
        ], dtype=np.float64)
        move_asym_min = float(np.min(move_asym))
        move_asym_last = float(move_asym[-1])

        if len(hold_rows) > 0:
            hold_asym = np.array([
                float(_as_float(r.get("asym"), "walker.row.asym"))
                for r in hold_rows
            ], dtype=np.float64)
            hold_asym_max = float(np.max(hold_asym))
            hold_asym_final = float(hold_asym[-1])

            overtake_step = ""
            for r in hold_rows:
                if float(_as_float(r.get("asym"), "walker.row.asym")) > 0.0:
                    overtake_step = int(float(_as_float(r.get("t"), "walker.row.t")))
                    break
        else:
            hold_asym_max = float("nan")
            hold_asym_final = float("nan")
            overtake_step = ""

        mach = float(_as_float(move_rows[0].get("mach"), "walker.move_row.mach"))

        rec: Dict[str, object] = {
            "move_tick_iters": int(tick),
            "hold_tick_iters": int(tick),
            "mach": float(mach),
            "move_asym_min": float(move_asym_min),
            "move_asym_last": float(move_asym_last),
            "hold_asym_max": float(hold_asym_max),
            "hold_asym_final": float(hold_asym_final),
            "overtake_step": overtake_step,
        }
        summary_rows.append(rec)

        line = "[walker-sweep] tick=%d mach=%.6g move_min=%.6g hold_max=%.6g overtake=%s" % (
            int(tick), float(mach), float(move_asym_min), float(hold_asym_max), str(overtake_step),
        )
        if bool(verbose):
            if log_line is not None:
                log_line(str(line))
            else:
                print(str(line))
        else:
            if log_line is not None:
                log_line(str(line))

    with open(out_csv, "w", newline="") as f:
        _write_provenance_header(f, provenance_header, artefact="walker_tick_sweep")
        w = csv.writer(f)
        w.writerow(cols)
        for r in summary_rows:
            w.writerow([r.get(c, "") for c in cols])

    line = f"[walker-sweep] wrote {out_csv}"
    if bool(verbose):
        if log_line is not None:
            log_line(str(line))
        else:
            print(str(line))
    else:
        if log_line is not None:
            log_line(str(line))