# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""collidersg.py — Experiment 09 (Sine-Gordon multi-sprite interaction)

Purpose

A minimal, fail-loud collider harness for the Sine-Gordon traffic mode.

Design goals
- Initialise N localised “sprites” at arbitrary (x,y,z) with per-sprite velocity vectors.
- Use **finite-shift** velocity initialisation (reduces lattice gradient noise):
    vel0 ≈ (phi0(x - v*dt) - phi0(x)) / dt
- Track each sprite using **local-window** maxima around a predicted position (fast, robust).
- Emit an append-safe CSV with a provenance header.

Scenarios (experiment wiring lives in core.py / experiments)

  A: Head-on breather collider (sprite assets; phase offset; merger / rebound studies).
  B: Off-axis / orbit / near-miss scattering (sprite assets; impact parameter b via Y/Z offsets).
  C: Static initialisation / horizon mapping (sprite assets; v=0; distance sweep).
  D: Kink-wall collision sanity (procedural planar kink walls; Neumann boundary; tracker uses |∂x phi|).
  E: Binary “wire” transistor prototype (kink walls trimmed into a soft-masked bullet inside a k-grid wire).

Notes

- “D/E” use `kind="kink_wall"` (no sprite asset) and rely on gradient tracking.
- The wire is enabled by setting `sg_k_outside>0` plus wire bounds; bevel softens impedance.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import sys
import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import ensure_parent_dir, resolve_out_path, now_s, wallclock_iso, write_csv_provenance_header
from traffic import evolve_sine_gordon_traffic_steps
from params import TrafficParams


Vec3 = Tuple[float, float, float]
I3 = Tuple[int, int, int]


@dataclass(frozen=True)
class SpriteSpec:
    """Immutable sprite descriptor."""

    sid: int
    pos: I3
    vel: Vec3
    amp: float
    sigma: float
    kind: str = "gaussian"
    phase: float = 0.0
    track_only: bool = False
    fixed_probe: bool = False  # if true, do not move during tracking; sample local peak only
    bullet_sigma: float = 0.0
    bullet_axis: str = "y"  # ribbon trim axis: 'y' or 'z'


@dataclass
class ActiveSprite:
    """Runtime tracker for a sprite."""

    sid: int
    pos: np.ndarray  # float xyz
    vel: np.ndarray  # float xyz
    kind: str = "gaussian"
    track_only: bool = False
    fixed_probe: bool = False
    status: str = "ALIVE"  # ALIVE|LOST|MERGED|ANNIHILATED


# --- SpriteAsset helpers ---

@dataclass(frozen=True)
class SpriteAsset:
    """A compact extracted sprite patch asset."""

    L: int
    phi: np.ndarray
    vel: np.ndarray
    src: np.ndarray
    load: np.ndarray
    meta: dict[str, Any]


def read_sprite_asset_h5(path: str) -> SpriteAsset:
    """Read a sprite asset HDF5 containing datasets: phi, vel, src, load."""

    p = str(path or "").strip()
    if not p:
        raise ValueError("sprite_asset_h5 path is empty")
    if not os.path.exists(p):
        raise FileNotFoundError(p)

    import h5py
    import numpy as np

    def _attr_to_py(v: Any) -> Any:
        if v is None:
            return None
        # h5py may return numpy scalars / bytes.
        try:
            if isinstance(v, (np.generic,)):
                v = v.item()
        except Exception:
            pass
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8")
            except Exception:
                return str(v)
        if isinstance(v, (list, tuple)):
            return [_attr_to_py(x) for x in v]
        return v

    with h5py.File(p, "r") as f:
        for k in ("phi", "vel", "src", "load"):
            if k not in f:
                raise ValueError(f"sprite asset missing dataset '{k}' in {p}")
        phi = np.asarray(f["phi"], dtype=np.float32)
        vel = np.asarray(f["vel"], dtype=np.float32)
        src = np.asarray(f["src"], dtype=np.float32)
        load = np.asarray(f["load"], dtype=np.float32)

        meta: dict[str, Any] = {}
        try:
            for k in f.attrs.keys():
                meta[str(k)] = _attr_to_py(f.attrs[k])
        except Exception:
            meta = {}

    if phi.ndim != 3:
        raise ValueError(f"sprite phi must be 3D, got {phi.shape}")
    if phi.shape != vel.shape or phi.shape != src.shape or phi.shape != load.shape:
        raise ValueError(f"sprite datasets must share shape, got phi={phi.shape} vel={vel.shape} src={src.shape} load={load.shape}")

    L = int(phi.shape[0])
    if L != int(phi.shape[1]) or L != int(phi.shape[2]) or L < 3:
        raise ValueError(f"sprite patch must be cubic and >=3, got {phi.shape}")

    return SpriteAsset(L=L, phi=phi, vel=vel, src=src, load=load, meta=meta)


# --- Asset finite-shift helpers for translational velocity kick ---


def _infer_bulk_v_from_patch(phi: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Infer an effective bulk translation velocity v from a patch.

    We fit: vel ≈ -(v · ∇phi) in a weighted least-squares sense over the patch.

    This estimates the *directional drift* component embedded in the asset so
    we can remove it before applying an explicit collider kick.
    """

    if phi.ndim != 3 or vel.ndim != 3:
        raise ValueError(f"infer_bulk_v expects 3D arrays, got phi={phi.shape} vel={vel.shape}")
    if phi.shape != vel.shape:
        raise ValueError(f"infer_bulk_v requires matching shapes, got phi={phi.shape} vel={vel.shape}")

    # Gradients (array order is [x,y,z]).
    gx, gy, gz = np.gradient(phi.astype(np.float32), edge_order=1)

    amax = float(np.max(np.abs(phi)))
    if not math.isfinite(amax) or amax <= 0.0:
        return np.zeros((3,), dtype=np.float32)

    # Core-focused weights: emphasise regions with significant |phi|.
    w = (np.abs(phi).astype(np.float32) / float(amax))
    w = (w * w).astype(np.float32)

    # Mask tiny-amplitude regions to avoid fitting noise.
    m = w > np.float32(0.01)
    if not bool(np.any(m)):
        return np.zeros((3,), dtype=np.float32)

    # Weighted least squares: solve A v ≈ b, where b = -vel and A = [gx, gy, gz].
    ww = np.sqrt(w[m]).astype(np.float32)
    A = np.stack([gx[m], gy[m], gz[m]], axis=1).astype(np.float32)
    b = (-vel[m]).astype(np.float32)

    Aw = (A * ww[:, None]).astype(np.float32)
    bw = (b * ww).astype(np.float32)

    try:
        v_fit, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    except Exception:
        return np.zeros((3,), dtype=np.float32)

    v_fit = np.asarray(v_fit, dtype=np.float32).reshape(3)
    if not (np.isfinite(v_fit).all()):
        return np.zeros((3,), dtype=np.float32)

    return v_fit


def _remove_bulk_translation_from_asset(asset: SpriteAsset) -> tuple[SpriteAsset, np.ndarray]:
    """Remove the inferred bulk translation component from an asset's velocity field.

    Returns (asset_clean, v_fit), where v_fit is the inferred drift velocity.
    """

    phi0 = asset.phi.astype(np.float32)
    vel0 = asset.vel.astype(np.float32)

    v_fit = _infer_bulk_v_from_patch(phi0, vel0)
    if float(np.max(np.abs(v_fit))) < 1.0e-9:
        return asset, v_fit

    gx, gy, gz = np.gradient(phi0, edge_order=1)
    vel_trans = -(float(v_fit[0]) * gx + float(v_fit[1]) * gy + float(v_fit[2]) * gz).astype(np.float32)
    vel_clean = (vel0 - vel_trans).astype(np.float32)

    return (
        SpriteAsset(
            L=asset.L,
            phi=asset.phi,
            vel=vel_clean,
            src=asset.src,
            load=asset.load,
            meta=asset.meta,
        ),
        v_fit,
    )


def _shift_patch_nn(a: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Nearest-neighbour shift of a cubic patch by (dx,dy,dz) in voxel units.

    Samples `a` at (i - dx, j - dy, k - dz). Out-of-bounds samples are zero.

    This is intentionally simple (no SciPy dependency). It's used only to form
    a finite-shift translational kick for sprite assets.
    """

    if a.ndim != 3:
        raise ValueError(f"patch must be 3D, got {a.shape}")

    L = int(a.shape[0])
    if L != int(a.shape[1]) or L != int(a.shape[2]):
        raise ValueError(f"patch must be cubic, got {a.shape}")

    # Build integer sample coordinates.
    ii, jj, kk = np.indices((L, L, L), dtype=np.int32)
    si = np.rint(ii.astype(np.float32) - float(dx)).astype(np.int32)
    sj = np.rint(jj.astype(np.float32) - float(dy)).astype(np.int32)
    sk = np.rint(kk.astype(np.float32) - float(dz)).astype(np.int32)

    m = (si >= 0) & (si < L) & (sj >= 0) & (sj < L) & (sk >= 0) & (sk < L)
    out = np.zeros((L, L, L), dtype=np.float32)
    if not bool(np.any(m)):
        return out

    out[m] = a[si[m], sj[m], sk[m]].astype(np.float32)
    return out


# --- Tri-linear shift helper for sub-voxel finite-shift ---
def _shift_patch_trilinear(a: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Tri-linear shift of a cubic patch by (dx,dy,dz) in voxel units.

    Samples `a` at (i - dx, j - dy, k - dz) using tri-linear interpolation.
    Out-of-bounds samples are zero.

    Array order is [x, y, z].

    This avoids the sub-voxel quantisation of nearest-neighbour shifting.
    """

    if a.ndim != 3:
        raise ValueError(f"patch must be 3D, got {a.shape}")

    L = int(a.shape[0])
    if L != int(a.shape[1]) or L != int(a.shape[2]):
        raise ValueError(f"patch must be cubic, got {a.shape}")

    ii, jj, kk = np.indices((L, L, L), dtype=np.float32)
    xf = ii - float(dx)
    yf = jj - float(dy)
    zf = kk - float(dz)

    x0 = np.floor(xf).astype(np.int32)
    y0 = np.floor(yf).astype(np.int32)
    z0 = np.floor(zf).astype(np.int32)

    wx = (xf - x0.astype(np.float32)).astype(np.float32)
    wy = (yf - y0.astype(np.float32)).astype(np.float32)
    wz = (zf - z0.astype(np.float32)).astype(np.float32)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    out = np.zeros((L, L, L), dtype=np.float32)

    def _acc(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, w: np.ndarray) -> None:
        m = (ix >= 0) & (ix < L) & (iy >= 0) & (iy < L) & (iz >= 0) & (iz < L)
        if not bool(np.any(m)):
            return
        out[m] += w[m] * a[ix[m], iy[m], iz[m]].astype(np.float32)

    _acc(x0, y0, z0, (1.0 - wx) * (1.0 - wy) * (1.0 - wz))
    _acc(x1, y0, z0, (wx) * (1.0 - wy) * (1.0 - wz))
    _acc(x0, y1, z0, (1.0 - wx) * (wy) * (1.0 - wz))
    _acc(x1, y1, z0, (wx) * (wy) * (1.0 - wz))

    _acc(x0, y0, z1, (1.0 - wx) * (1.0 - wy) * (wz))
    _acc(x1, y0, z1, (wx) * (1.0 - wy) * (wz))
    _acc(x0, y1, z1, (1.0 - wx) * (wy) * (wz))
    _acc(x1, y1, z1, (wx) * (wy) * (wz))

    return out


def _boosted_asset_patch(asset: SpriteAsset, v: Vec3, dt: float) -> SpriteAsset:
    """Return a patch asset with an added translational kick using sub-voxel finite-shift transport.

    The extracted sprite asset already carries internal `vel` (the breather heartbeat).
    For collider scenarios we also want a bulk translation specified by `v`.

    We approximate the translational component as:
        vel_trans ≈ (phi(x - v*dt) - phi(x)) / dt

    Using tri-linear interpolation avoids sub-voxel quantisation (|v*dt| << 1 voxel)
    that can otherwise zero-out the kick when using nearest-neighbour shifting.
    """

    if float(dt) <= 0.0:
        raise ValueError(f"dt must be >0, got {dt}")

    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    if abs(vx) < 1.0e-12 and abs(vy) < 1.0e-12 and abs(vz) < 1.0e-12:
        return asset

    dx = vx * float(dt)
    dy = vy * float(dt)
    dz = vz * float(dt)

    phi0 = asset.phi.astype(np.float32)
    phi_shift = _shift_patch_trilinear(phi0, dx, dy, dz)
    vel_trans = (phi0 - phi_shift) / float(dt)

    # Calibrate the kick so the inferred bulk drift matches the requested velocity.
    # We fit: vel ≈ -(v · ∇phi) using the same LSQ estimator; apply a scalar gain.
    v_req = np.array([vx, vy, vz], dtype=np.float32)
    v_req_abs = float(np.linalg.norm(v_req))
    gain = 1.0
    if v_req_abs > 1.0e-12:
        u = (v_req / np.float32(v_req_abs)).astype(np.float32)
        v_imp = _infer_bulk_v_from_patch(phi0, vel_trans.astype(np.float32))
        a = float(np.dot(v_imp.astype(np.float32), u))
        b = float(v_req_abs)
        if math.isfinite(a) and abs(a) > 1.0e-8:
            gain = b / a
            # Safety clamp: avoid shock injection from pathological frames / flat gradients.
            if not math.isfinite(gain):
                gain = 1.0
            gain = float(max(-200.0, min(200.0, gain)))
        else:
            # If the implied translation along the requested direction is ~0, do not amplify.
            gain = 1.0

    vel_boost = (asset.vel.astype(np.float32) + float(gain) * vel_trans.astype(np.float32)).astype(np.float32)

    # Keep src/load as-is; they are not part of SG evolution but travel with the sprite.
    return SpriteAsset(
        L=asset.L,
        phi=asset.phi,
        vel=vel_boost,
        src=asset.src,
        load=asset.load,
        meta=asset.meta,
    )


def _stamp_sprite_patch(
    *,
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    load: np.ndarray,
    asset: SpriteAsset,
    centre_xyz: I3,
    phase: float,
) -> None:
    """Stamp a sprite patch (phi,vel,src,load) into the full grids (in-place).

    `centre_xyz` is the target centre voxel in the full grid.
    `phase` is applied as a real-valued sign control via cos(phase) to both phi and vel.
    """

    n = int(phi.shape[0])
    L = int(asset.L)
    r = L // 2
    cx, cy, cz = int(centre_xyz[0]), int(centre_xyz[1]), int(centre_xyz[2])

    x0, x1 = cx - r, cx - r + L
    y0, y1 = cy - r, cy - r + L
    z0, z1 = cz - r, cz - r + L

    if x0 < 0 or y0 < 0 or z0 < 0 or x1 > n or y1 > n or z1 > n:
        raise ValueError(f"sprite stamp out of bounds: centre={centre_xyz} L={L} n={n}")

    s = float(math.cos(float(phase)))

    phi[x0:x1, y0:y1, z0:z1] += s * asset.phi
    vel[x0:x1, y0:y1, z0:z1] += s * asset.vel
    src[x0:x1, y0:y1, z0:z1] += s * asset.src
    load[x0:x1, y0:y1, z0:z1] += s * asset.load


# --- Kick leverage helpers for sprite asset phase selection ---

def _core_vabs_max(vel: np.ndarray) -> float:
    """Max |vel| in a small central cube (core) of the patch."""

    if vel.ndim != 3:
        return float(np.max(np.abs(vel)))
    L = int(vel.shape[0])
    r = max(1, L // 6)
    c = L // 2
    x0, x1 = max(0, c - r), min(L, c + r + 1)
    block = vel[x0:x1, x0:x1, x0:x1]
    return float(np.max(np.abs(block)))


def _kick_leverage_along_dir(phi: np.ndarray, u: np.ndarray, dt: float) -> float:
    """Return implied translation leverage 'a' along direction u for a unit kick.

    We compute vel_trans for a unit requested velocity along u and infer the implied
    bulk drift via the LSQ estimator. The returned scalar is a = v_imp · u.
    """

    ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
    dx = ux * float(dt)
    dy = uy * float(dt)
    dz = uz * float(dt)

    phi0 = phi.astype(np.float32)
    phi_shift = _shift_patch_trilinear(phi0, dx, dy, dz)
    vel_trans = (phi_shift - phi0) / float(dt)

    v_imp = _infer_bulk_v_from_patch(phi0, vel_trans.astype(np.float32))
    a = float(np.dot(v_imp.astype(np.float32), u.astype(np.float32)))
    if not math.isfinite(a):
        return 0.0
    return a


def _phase_select_for_kick(asset: SpriteAsset, v_req: Vec3, tp: TrafficParams, *, steps: int = 256) -> tuple[SpriteAsset, dict[str, float]]:
    """Evolve the patch locally and choose a phase that maximises peak |phi| (tall/low-kinetic breather phase).

    This avoids harvesting a frame where the finite-shift kick has ~zero effect
    along the requested direction.

    Returns (asset_best, info) where info includes:
      - phase_steps: steps searched
      - best_step: selected local step
      - best_peak_amp: max |phi| at selected step
      - best_a: kick leverage scalar along requested direction at selected step (diagnostic)
      - best_core_vabs: core |vel| at selected step
    """

    vx, vy, vz = float(v_req[0]), float(v_req[1]), float(v_req[2])
    v_abs = float(math.sqrt(vx * vx + vy * vy + vz * vz))
    if v_abs <= 1.0e-12:
        return asset, {
            "phase_steps": 0.0,
            "best_step": 0.0,
            "best_peak_amp": float(np.max(np.abs(asset.phi.astype(np.float32)))),
            "best_a": 0.0,
            "best_core_vabs": _core_vabs_max(asset.vel),
        }

    u = np.array([vx / v_abs, vy / v_abs, vz / v_abs], dtype=np.float32)

    # Local copies.
    phi = asset.phi.astype(np.float32)
    vel = asset.vel.astype(np.float32)
    src = asset.src.astype(np.float32)

    best_step = 0
    best_core = _core_vabs_max(vel)
    best_phi = phi
    best_vel = vel
    best_peak = float(np.max(np.abs(phi)))

    # Search forward.
    iters = int(max(0, steps))
    for t in range(1, iters + 1):
        phi, vel = evolve_sine_gordon_traffic_steps(phi, vel, src, tp, 1)
        peak = float(np.max(np.abs(phi)))

        # Primary objective: maximise peak amplitude (tall phase). Tie-break: smaller core |vel|.
        if peak > best_peak + 1.0e-9:
            best_step = t
            best_peak = float(peak)
            best_core = _core_vabs_max(vel)
            best_phi = phi
            best_vel = vel
        elif abs(peak - best_peak) <= 1.0e-9:
            c = _core_vabs_max(vel)
            if c < best_core:
                best_step = t
                best_peak = float(peak)
                best_core = float(c)
                best_phi = phi
                best_vel = vel

    try:
        best_a = float(_kick_leverage_along_dir(best_phi, u, float(tp.dt)))
    except Exception:
        best_a = 0.0

    asset_best = SpriteAsset(
        L=asset.L,
        phi=best_phi.astype(np.float32),
        vel=best_vel.astype(np.float32),
        src=asset.src,
        load=asset.load,
        meta=asset.meta,
    )

    return asset_best, {
        "phase_steps": float(iters),
        "best_step": float(best_step),
        "best_peak_amp": float(best_peak),
        "best_a": float(best_a),
        "best_core_vabs": float(best_core),
    }


def initialise_from_sprite_asset(n: int, sprites: Sequence[SpriteSpec], asset_h5: str, dt: float, sg_k: float, c2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Initialise (phi,vel,src,load) by stamping a sprite asset at each sprite position, with optional translational kick."""

    asset = read_sprite_asset_h5(asset_h5)

    phi0 = np.zeros((n, n, n), dtype=np.float32)
    vel0 = np.zeros((n, n, n), dtype=np.float32)
    src0 = np.zeros((n, n, n), dtype=np.float32)
    load0 = np.zeros((n, n, n), dtype=np.float32)

    pll_info: list[dict[str, Any]] = []

    for sp in sprites:
        x, y, z = sp.pos
        if not (0 <= x < n and 0 <= y < n and 0 <= z < n):
            raise ValueError(f"sprite id={sp.sid} pos out of bounds: {sp.pos}")

        if bool(getattr(sp, "track_only", False)):
            # Passive detector: do not stamp the sprite asset into the field.
            # It will still be tracked via the normal tracking loop in run_collidersg.
            pll_info.append({
                "sid": int(sp.sid),
                "track_only": 1,
                "v_req": [float(sp.vel[0]), float(sp.vel[1]), float(sp.vel[2])],
            })
            continue

        # Phase-select a local patch frame that has non-zero kick leverage along the requested direction.
        # This prevents stamping a phase where the finite-shift kick is swallowed (a≈0).
        tp_local = TrafficParams(
            mode="sine_gordon",
            dt=float(dt),
            c2=float(c2),
            gamma=0.0,
            decay=0.0,
            boundary_mode="zero",
            boundary_zero=True,
            sponge_width=0,
            sponge_strength=0.0,
            traffic_k=float(sg_k),
            inject=0.0,
        )
        asset_sel, _sel_info = _phase_select_for_kick(asset, sp.vel, tp_local, steps=256)
        _pll: dict[str, Any] = {
            "sid": int(sp.sid),
            "phase": float(sp.phase),
            "v_req": [float(sp.vel[0]), float(sp.vel[1]), float(sp.vel[2])],
            "sel": dict(_sel_info or {}),
        }

        # NOTE: Do NOT attempt LSQ drift removal for SG sprite assets (breathers are standing waves).
        # The advection fit can hallucinate bulk motion and sabotage the kick leverage.
        asset_q = asset_sel
        _v_fit = np.zeros((3,), dtype=np.float32)
        _pll["v_fit"] = [0.0, 0.0, 0.0]

        # Build the kick component on the *unphased* fields, then apply phase only to the
        # internal fields. This prevents phase=pi from cancelling or flipping the kick.
        vx, vy, vz = float(sp.vel[0]), float(sp.vel[1]), float(sp.vel[2])
        dx = vx * float(dt)
        dy = vy * float(dt)
        dz = vz * float(dt)

        phi0p = asset_q.phi.astype(np.float32)
        phi_shift = _shift_patch_trilinear(phi0p, dx, dy, dz)
        vel_trans = (phi0p - phi_shift) / float(dt)
        _pll["vel_trans_abs_max"] = float(np.max(np.abs(vel_trans)))

        # Calibrate gain along requested direction using the LSQ drift estimator.
        v_req = np.array([vx, vy, vz], dtype=np.float32)
        v_req_abs = float(np.linalg.norm(v_req))
        gain = 1.0
        if v_req_abs > 1.0e-12:
            u = (v_req / np.float32(v_req_abs)).astype(np.float32)
            v_imp = _infer_bulk_v_from_patch(phi0p, vel_trans.astype(np.float32))
            a = float(np.dot(v_imp.astype(np.float32), u))
            _pll["a"] = float(a)
            b = float(v_req_abs)
            if math.isfinite(a) and abs(a) > 1.0e-8:
                gain = b / a
                if not math.isfinite(gain):
                    gain = 1.0
                # Tighter clamp than the generic helper; prevents runaway on noisy a.
                gain = float(max(-50.0, min(50.0, gain)))
                _pll["gain"] = float(gain)
            else:
                gain = 1.0
                _pll["gain"] = 1.0
        else:
            gain = 1.0
            _pll["gain"] = 1.0

        s = float(math.cos(float(sp.phase)))
        phi_ph = (s * asset_q.phi).astype(np.float32)
        src_ph = (s * asset_q.src).astype(np.float32)
        load_ph = (s * asset_q.load).astype(np.float32)

        # Phase flips the sprite polarity (hill vs valley). The translational kick must match that polarity
        # so a phase=pi (s=-1) sprite moves in the requested direction rather than interpreting the kick as filling-in.
        vel_internal_ph = (s * asset_q.vel.astype(np.float32)).astype(np.float32)
        vel_boost = (vel_internal_ph + float(s) * float(gain) * vel_trans.astype(np.float32)).astype(np.float32)
        _pll["vel_internal_abs_max"] = float(np.max(np.abs(vel_internal_ph)))
        _pll["vel_boost_abs_max"] = float(np.max(np.abs(vel_boost)))
        pll_info.append(_pll)

        asset2 = SpriteAsset(
            L=asset_q.L,
            phi=phi_ph,
            vel=vel_boost,
            src=src_ph,
            load=load_ph,
            meta=asset_q.meta,
        )

        _stamp_sprite_patch(
            phi=phi0,
            vel=vel0,
            src=src0,
            load=load0,
            asset=asset2,
            centre_xyz=sp.pos,
            phase=0.0,
        )

    return phi0, vel0, src0, load0, pll_info


def _as_i3(v: Sequence[Any]) -> I3:
    if len(v) != 3:
        raise ValueError(f"pos must be 3-vector, got {v!r}")
    return (int(v[0]), int(v[1]), int(v[2]))


def _as_v3(v: Sequence[Any]) -> Vec3:
    if len(v) != 3:
        raise ValueError(f"vel must be 3-vector, got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


def parse_sprites_json(payload: str) -> List[SpriteSpec]:
    """Parse sprites from a JSON string or JSON file path."""

    s = payload.strip()
    if not s:
        raise ValueError("sprites JSON is empty")
    if os.path.exists(s) and s.lower().endswith(".json"):
        with open(s, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(s)

    if not isinstance(data, list):
        raise ValueError("sprites JSON must be a list")

    out: List[SpriteSpec] = []
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise ValueError(f"sprite[{i}] must be an object")

        # Accept both `sid` (preferred) and legacy `id`.
        if "sid" in obj:
            sid = int(obj["sid"])
        elif "id" in obj:
            sid = int(obj["id"])
        else:
            sid = i + 1

        pos = _as_i3(obj["pos"])
        vel = _as_v3(obj.get("vel", (0.0, 0.0, 0.0)))
        amp = float(obj.get("amp", 0.0))
        sigma = float(obj.get("sigma", 1.0))
        kind = str(obj.get("kind", "gaussian")).strip() or "gaussian"
        phase = float(obj.get("phase", 0.0))
        track_only = bool(obj.get("track_only", False))
        fixed_probe = bool(obj.get("fixed_probe", False))
        bullet_sigma = float(obj.get("bullet_sigma", 0.0) or 0.0)
        bullet_axis = str(obj.get("bullet_axis", "y") or "y").strip().lower()
        if bullet_axis not in ("y", "z"):
            raise ValueError(f"bullet_axis must be 'y' or 'z', got {bullet_axis!r}")
        out.append(SpriteSpec(sid=sid, pos=pos, vel=vel, amp=amp, sigma=sigma, kind=kind, phase=phase, track_only=track_only, fixed_probe=fixed_probe, bullet_sigma=bullet_sigma, bullet_axis=bullet_axis))

    # Fail-loud: no duplicate ids.
    ids = [s.sid for s in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate sprite ids in JSON: {ids}")

    return out
def _kink_profile_1d(x: np.ndarray, x0: float, ell: float, charge: float) -> np.ndarray:
    """1D sine-Gordon kink/antikink profile along x.

    Returns a monotonic step of magnitude 2π*charge (charge=+1 kink, charge=-1 antikink).
    We keep the field unwrapped; sin(phi) is 2π-periodic.
    """

    if ell <= 0.0:
        raise ValueError(f"ell must be >0, got {ell}")
    # Overflow-safe base kink: 0 -> 2π.
    s = ((x - float(x0)) / float(ell)).astype(np.float32)
    # Avoid exp overflow: for large |s|, arctan(exp(s)) saturates to {0, pi/2}.
    s_clip = np.clip(s, -60.0, 60.0)
    e = np.exp(s_clip).astype(np.float32)
    base = (4.0 * np.arctan(e)).astype(np.float32)
    # Hard saturate tails to the analytic limits for extra numerical stability.
    base = np.where(s > 60.0, np.float32(2.0 * math.pi), base)
    base = np.where(s < -60.0, np.float32(0.0), base)
    if float(charge) >= 0.0:
        return base.astype(np.float32)
    # Antikink: 2π -> 0.
    return (2.0 * math.pi - base).astype(np.float32)


def initialise_kink_walls(n: int, sprites: Sequence[SpriteSpec], dt: float, c2: float, sg_k: float) -> tuple[np.ndarray, np.ndarray]:
    """Initialise planar kink/antikink walls with arbitrary normal direction.

    This is a first-pass 'hard-ball' species for collider experiments.

    Constraints (fail-loud):
      - All sprites must have kind == 'kink_wall'
      - |v| must be < c (where c = sqrt(c2))
      - The wall normal is taken as n = v/|v| (for |v|>0). If |v|==0, we default to +X normal.

    Field model (single wall i):
      phi(r) = kink( n · (r - r0) )

    Velocity initialisation uses a finite-shift along the motion direction by shifting each wall centre by v*dt:
      vel0 = (phi_shift - phi0) / dt

    Notes:
      - Phase selects kink vs antikink via cos(phase) sign (phase≈π => antikink).
      - This implementation is intentionally simple and prioritises correct geometry over speed.
    """

    if dt <= 0.0:
        raise ValueError(f"dt must be >0, got {dt}")

    if float(sg_k) <= 0.0:
        raise ValueError(f"sg_k must be >0 for kink walls, got {sg_k}")

    c = float(math.sqrt(float(c2)))
    if not math.isfinite(c) or c <= 0.0:
        raise ValueError(f"invalid c from c2: c2={c2} c={c}")

    ell0 = float(math.sqrt(float(c2) / float(sg_k)))
    if not math.isfinite(ell0) or ell0 <= 0.0:
        raise ValueError(f"invalid kink ell from c2/sg_k: c2={c2} sg_k={sg_k} ell={ell0}")

    def _charge_from_phase(ph: float) -> float:
        s = float(math.cos(float(ph)))
        return -1.0 if s < 0.0 else 1.0

    def _unit_from_vel(vx: float, vy: float, vz: float) -> tuple[np.ndarray, float]:
        v2 = float(vx * vx + vy * vy + vz * vz)
        if v2 <= 1.0e-24:
            u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            return u, 0.0
        vabs = float(math.sqrt(v2))
        if vabs >= c:
            raise ValueError(f"kink_wall |v| must be < c, got v={vabs} c={c}")
        u = np.array([vx / vabs, vy / vabs, vz / vabs], dtype=np.float32)
        return u, vabs

    # Precompute coordinate grids once (array order is [x,y,z]).
    X, Y, Z = np.indices((n, n, n), dtype=np.float32)

    def _kink_profile_signed(s: np.ndarray, ell: float, charge: float) -> np.ndarray:
        if ell <= 0.0:
            raise ValueError(f"ell must be >0, got {ell}")
        q = (s / float(ell)).astype(np.float32)
        q_clip = np.clip(q, -60.0, 60.0)
        e = np.exp(q_clip).astype(np.float32)
        base = (4.0 * np.arctan(e)).astype(np.float32)
        base = np.where(q > 60.0, np.float32(2.0 * math.pi), base)
        base = np.where(q < -60.0, np.float32(0.0), base)
        if float(charge) >= 0.0:
            return base.astype(np.float32)
        return (np.float32(2.0 * math.pi) - base).astype(np.float32)

    phi0 = np.zeros((n, n, n), dtype=np.float32)

    # Build phi0 as a sum of planar kinks along each wall normal.
    for sp in sprites:
        if str(sp.kind) != "kink_wall":
            raise ValueError(f"initialise_kink_walls requires kind='kink_wall' for all sprites, got sid={sp.sid} kind={sp.kind!r}")

        vx, vy, vz = float(sp.vel[0]), float(sp.vel[1]), float(sp.vel[2])
        u, vabs = _unit_from_vel(vx, vy, vz)
        gamma = 1.0 / float(math.sqrt(max(1.0e-12, 1.0 - (vabs * vabs) / (c * c))))
        ell = float(ell0 / gamma)

        x0, y0, z0 = float(sp.pos[0]), float(sp.pos[1]), float(sp.pos[2])
        s = (float(u[0]) * (X - x0) + float(u[1]) * (Y - y0) + float(u[2]) * (Z - z0)).astype(np.float32)
        q = _charge_from_phase(float(sp.phase))
        prof = _kink_profile_signed(s, ell, q)

        bs = float(getattr(sp, "bullet_sigma", 0.0) or 0.0)
        if bs > 0.0:
            inv2 = 1.0 / (2.0 * bs * bs)
            dx = (X - np.float32(x0)).astype(np.float32)
            dy = (Y - np.float32(y0)).astype(np.float32)
            dz = (Z - np.float32(z0)).astype(np.float32)
            # Ribbon window (v2): trim along a single transverse axis while leaving the other
            # transverse axis untrimmed so the wall can remain anchored to the pipe.
            # We still remove the component along u to get a transverse displacement, but then
            # we project onto a chosen world axis ('y' or 'z').
            du = (np.float32(float(u[0])) * dx + np.float32(float(u[1])) * dy + np.float32(float(u[2])) * dz).astype(np.float32)
            px = (dx - np.float32(float(u[0])) * du).astype(np.float32)
            py = (dy - np.float32(float(u[1])) * du).astype(np.float32)
            pz = (dz - np.float32(float(u[2])) * du).astype(np.float32)

            bax = str(getattr(sp, "bullet_axis", "y") or "y").strip().lower()
            if bax == "z":
                d1 = pz
            else:
                d1 = py

            m = np.exp(-(d1 * d1) * np.float32(inv2)).astype(np.float32)
            prof = (prof * m).astype(np.float32)

        phi0 += prof

    # Finite-shift velocity init by shifting along v*dt.
    phi_shift = np.zeros_like(phi0)
    for sp in sprites:
        vx, vy, vz = float(sp.vel[0]), float(sp.vel[1]), float(sp.vel[2])
        u, vabs = _unit_from_vel(vx, vy, vz)
        gamma = 1.0 / float(math.sqrt(max(1.0e-12, 1.0 - (vabs * vabs) / (c * c))))
        ell = float(ell0 / gamma)

        x0s = float(sp.pos[0]) + vx * float(dt)
        y0s = float(sp.pos[1]) + vy * float(dt)
        z0s = float(sp.pos[2]) + vz * float(dt)
        s = (float(u[0]) * (X - x0s) + float(u[1]) * (Y - y0s) + float(u[2]) * (Z - z0s)).astype(np.float32)
        q = _charge_from_phase(float(sp.phase))
        prof = _kink_profile_signed(s, ell, q)

        bs = float(getattr(sp, "bullet_sigma", 0.0) or 0.0)
        if bs > 0.0:
            inv2 = 1.0 / (2.0 * bs * bs)
            dx = (X - np.float32(x0s)).astype(np.float32)
            dy = (Y - np.float32(y0s)).astype(np.float32)
            dz = (Z - np.float32(z0s)).astype(np.float32)
            # Ribbon window (v2): trim along a single transverse axis while leaving the other
            # transverse axis untrimmed so the wall can remain anchored to the pipe.
            # We still remove the component along u to get a transverse displacement, but then
            # we project onto a chosen world axis ('y' or 'z').
            du = (np.float32(float(u[0])) * dx + np.float32(float(u[1])) * dy + np.float32(float(u[2])) * dz).astype(np.float32)
            px = (dx - np.float32(float(u[0])) * du).astype(np.float32)
            py = (dy - np.float32(float(u[1])) * du).astype(np.float32)
            pz = (dz - np.float32(float(u[2])) * du).astype(np.float32)

            bax = str(getattr(sp, "bullet_axis", "y") or "y").strip().lower()
            if bax == "z":
                d1 = pz
            else:
                d1 = py

            m = np.exp(-(d1 * d1) * np.float32(inv2)).astype(np.float32)
            prof = (prof * m).astype(np.float32)

        phi_shift += prof

    vel0 = (phi_shift - phi0) / float(dt)
    return phi0.astype(np.float32), vel0.astype(np.float32)


def sprites_digest(sprites: Sequence[SpriteSpec]) -> str:
    """Stable short digest for a sprite list.

    We canonicalise to a minimal JSON with sorted ids so we can refer to a configuration
    without embedding the full payload in the CSV.
    """

    rows = []
    for sp in sorted(sprites, key=lambda s: s.sid):
        rows.append({
            "id": int(sp.sid),
            "pos": [int(sp.pos[0]), int(sp.pos[1]), int(sp.pos[2])],
            "vel": [float(sp.vel[0]), float(sp.vel[1]), float(sp.vel[2])],
            "amp": float(sp.amp),
            "sigma": float(sp.sigma),
            "phase": float(sp.phase),
        })
    payload = json.dumps(rows, separators=(",", ":"), sort_keys=True)
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return h[:12]


def _nb_gaussian_stamp_f(phi: np.ndarray, cx: float, cy: float, cz: float, amp: float, sigma: float, phase: float) -> None:
    """Stamp a compact 3D Gaussian into phi (in-place), with a float centre.

    This supports fractional voxel shifts (used by finite-shift velocity initialisation).

    Note: This is *not* an analytic SG breather profile. It is a localised initial condition.

    `phase` is a simple sign/offset control via cos(phase), keeping the initial field real.
    """

    if sigma <= 0:
        raise ValueError(f"sigma must be >0, got {sigma}")

    n = phi.shape[0]
    r = max(3, int(math.ceil(3.5 * sigma)))

    icx, icy, icz = int(round(cx)), int(round(cy)), int(round(cz))
    x0 = max(0, icx - r)
    x1 = min(n, icx + r + 1)
    y0 = max(0, icy - r)
    y1 = min(n, icy + r + 1)
    z0 = max(0, icz - r)
    z1 = min(n, icz + r + 1)

    sx = np.arange(x0, x1, dtype=np.float32) - float(cx)
    sy = np.arange(y0, y1, dtype=np.float32) - float(cy)
    sz = np.arange(z0, z1, dtype=np.float32) - float(cz)

    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    gx = np.exp(-inv2s2 * sx * sx)
    gy = np.exp(-inv2s2 * sy * sy)
    gz = np.exp(-inv2s2 * sz * sz)

    scale = float(amp) * float(math.cos(phase))
    phi[x0:x1, y0:y1, z0:z1] += scale * gx[:, None, None] * gy[None, :, None] * gz[None, None, :]


def _nb_gaussian_stamp(phi: np.ndarray, cx: int, cy: int, cz: int, amp: float, sigma: float, phase: float) -> None:
    """Integer-centred convenience wrapper."""

    _nb_gaussian_stamp_f(phi, float(cx), float(cy), float(cz), amp, sigma, phase)


def initialise_phi(n: int, sprites: Sequence[SpriteSpec]) -> np.ndarray:
    phi0 = np.zeros((n, n, n), dtype=np.float32)
    for sp in sprites:
        x, y, z = sp.pos
        if not (0 <= x < n and 0 <= y < n and 0 <= z < n):
            raise ValueError(f"sprite id={sp.sid} pos out of bounds: {sp.pos}")
        _nb_gaussian_stamp(phi0, x, y, z, sp.amp, sp.sigma, sp.phase)

    return phi0


def initialise_vel_finite_shift(phi0: np.ndarray, sprites: Sequence[SpriteSpec], dt: float) -> np.ndarray:
    """Finite-shift velocity initialisation.

    We form a shifted field by shifting each sprite stamp by Δ = v*dt (in voxels), using fractional centres.
    Then vel0 = (phi_shift - phi0) / dt.

    This avoids directly differentiating phi on the lattice.
    """

    if dt <= 0:
        raise ValueError(f"dt must be >0, got {dt}")

    n = phi0.shape[0]
    phi_shift = np.zeros_like(phi0)

    for sp in sprites:
        vx, vy, vz = sp.vel
        x, y, z = sp.pos
        xs = float(x) - float(vx) * float(dt)
        ys = float(y) - float(vy) * float(dt)
        zs = float(z) - float(vz) * float(dt)
        if not (0.0 <= xs < float(n) and 0.0 <= ys < float(n) and 0.0 <= zs < float(n)):
            raise ValueError(f"finite-shift moved sprite id={sp.sid} out of bounds: shifted_pos={(xs,ys,zs)}")
        _nb_gaussian_stamp_f(phi_shift, xs, ys, zs, sp.amp, sp.sigma, sp.phase)

    vel0 = (phi_shift - phi0) / float(dt)
    return vel0.astype(np.float32)


def _max_in_window(phi: np.ndarray, centre: np.ndarray, r: int) -> Tuple[np.ndarray, float]:
    """Return argmax position and max |phi| in a cube window."""

    n = phi.shape[0]
    cx, cy, cz = int(round(float(centre[0]))), int(round(float(centre[1]))), int(round(float(centre[2])))
    x0 = max(0, cx - r)
    x1 = min(n, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(n, cy + r + 1)
    z0 = max(0, cz - r)
    z1 = min(n, cz + r + 1)

    block = phi[x0:x1, y0:y1, z0:z1]
    if block.size == 0:
        return centre.copy(), 0.0

    a = np.abs(block)
    k = int(np.argmax(a))
    ix, iy, iz = np.unravel_index(k, a.shape)

    peak = float(a[ix, iy, iz])

    # Sub-voxel refinement: weighted centroid of |phi| in a tiny neighbourhood around the argmax.
    # This avoids predictor snapping to integer voxels when the peak moves sub-voxel per tick.
    rr = 2
    x0c = max(0, int(ix) - rr)
    x1c = min(int(a.shape[0]), int(ix) + rr + 1)
    y0c = max(0, int(iy) - rr)
    y1c = min(int(a.shape[1]), int(iy) + rr + 1)
    z0c = max(0, int(iz) - rr)
    z1c = min(int(a.shape[2]), int(iz) + rr + 1)

    sub = a[x0c:x1c, y0c:y1c, z0c:z1c]
    wsum = float(np.sum(sub))
    if wsum > 0.0:
        sii, sjj, skk = np.indices(sub.shape, dtype=np.float32)
        cx_sub = float(np.sum(sub * (sii + float(x0c)))) / wsum
        cy_sub = float(np.sum(sub * (sjj + float(y0c)))) / wsum
        cz_sub = float(np.sum(sub * (skk + float(z0c)))) / wsum
        pos = np.array([float(x0) + cx_sub, float(y0) + cy_sub, float(z0) + cz_sub], dtype=np.float32)
    else:
        pos = np.array([x0 + ix, y0 + iy, z0 + iz], dtype=np.float32)

    return pos, peak


def _max_in_window_masked(phi: np.ndarray, centre: np.ndarray, r: int, valid_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return argmax position and max |phi| in a cube window, restricted to valid_mask!=0.

    `valid_mask` must be a uint8/boolean grid with the same shape as `phi`.
    """

    if phi.shape != valid_mask.shape:
        raise ValueError(f"valid_mask shape must match phi: phi={phi.shape} mask={valid_mask.shape}")

    n = phi.shape[0]
    cx, cy, cz = int(round(float(centre[0]))), int(round(float(centre[1]))), int(round(float(centre[2])))
    x0 = max(0, cx - r)
    x1 = min(n, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(n, cy + r + 1)
    z0 = max(0, cz - r)
    z1 = min(n, cz + r + 1)

    block = phi[x0:x1, y0:y1, z0:z1]
    mblk = valid_mask[x0:x1, y0:y1, z0:z1]
    if block.size == 0:
        return centre.copy(), 0.0

    a = np.abs(block).astype(np.float32)
    a[mblk == 0] = np.float32(-1.0)

    maxv = float(np.max(a))
    if maxv <= 0.0:
        return centre.copy(), 0.0

    # If there is a plateau, prefer the voxel closest to the requested centre.
    idx = np.argwhere(a == np.float32(maxv))
    if idx.shape[0] == 0:
        return centre.copy(), 0.0

    cc = np.array([
        float(centre[0]) - float(x0),
        float(centre[1]) - float(y0),
        float(centre[2]) - float(z0),
    ], dtype=np.float32)
    d2 = np.sum((idx.astype(np.float32) - cc[None, :]) ** 2, axis=1)
    sel = idx[int(np.argmin(d2))]
    ix, iy, iz = int(sel[0]), int(sel[1]), int(sel[2])

    peak = float(maxv)

    # Sub-voxel refinement: weighted centroid of |phi| in a tiny neighbourhood.
    rr = 2
    x0c = max(0, int(ix) - rr)
    x1c = min(int(a.shape[0]), int(ix) + rr + 1)
    y0c = max(0, int(iy) - rr)
    y1c = min(int(a.shape[1]), int(iy) + rr + 1)
    z0c = max(0, int(iz) - rr)
    z1c = min(int(a.shape[2]), int(iz) + rr + 1)

    sub = a[x0c:x1c, y0c:y1c, z0c:z1c]
    msub = mblk[x0c:x1c, y0c:y1c, z0c:z1c]
    sub = (sub * (msub != 0).astype(np.float32)).astype(np.float32)

    wsum = float(np.sum(sub))
    if wsum > 0.0:
        sii, sjj, skk = np.indices(sub.shape, dtype=np.float32)
        cx_sub = float(np.sum(sub * (sii + float(x0c)))) / wsum
        cy_sub = float(np.sum(sub * (sjj + float(y0c)))) / wsum
        cz_sub = float(np.sum(sub * (skk + float(z0c)))) / wsum
        pos = np.array([float(x0) + cx_sub, float(y0) + cy_sub, float(z0) + cz_sub], dtype=np.float32)
    else:
        pos = np.array([x0 + ix, y0 + iy, z0 + iz], dtype=np.float32)

    return pos, peak

def _build_k_grid(
    n: int,
    *,
    k_inside: float,
    k_outside: float,
    wire_y0: int,
    wire_y1: int,
    wire_z0: int,
    wire_z1: int,
    bevel: int,
) -> Optional[np.ndarray]:
    """Optional per-voxel SG stiffness mask.

    Returns None when disabled.

    The wire is an axis-aligned rectangular tube spanning all X, with Y in [wire_y0, wire_y1)
    and Z in [wire_z0, wire_z1). Outside is k_outside.

    If bevel>0, we linearly ramp k across a bevel band on the wire edges to reduce impedance cliffs.
    """

    # Disabled unless the wire box is provided.
    if wire_y0 < 0 or wire_y1 < 0 or wire_z0 < 0 or wire_z1 < 0:
        return None

    if not (0 <= int(wire_y0) < int(wire_y1) <= int(n)):
        raise ValueError(f"wire_y0/wire_y1 out of bounds: y0={wire_y0} y1={wire_y1} n={n}")
    if not (0 <= int(wire_z0) < int(wire_z1) <= int(n)):
        raise ValueError(f"wire_z0/wire_z1 out of bounds: z0={wire_z0} z1={wire_z1} n={n}")

    k_in = float(k_inside)
    k_out = float(k_outside)
    if not math.isfinite(k_in) or k_in <= 0.0:
        raise ValueError(f"k_inside must be finite and >0, got {k_in}")
    if not math.isfinite(k_out) or k_out <= 0.0:
        raise ValueError(f"k_outside must be finite and >0, got {k_out}")

    k = np.full((n, n, n), k_out, dtype=np.float32)

    y0, y1 = int(wire_y0), int(wire_y1)
    z0, z1 = int(wire_z0), int(wire_z1)
    k[:, y0:y1, z0:z1] = np.float32(k_in)

    b = int(max(0, bevel))
    if b <= 0:
        return k

    # Bevel bands: ramp from k_in to k_out over b voxels.
    def _ramp(t: np.float32) -> np.float32:
        return (np.float32(1.0) - t) * np.float32(k_in) + t * np.float32(k_out)

    # Y-low face
    for i in range(1, b + 1):
        yy = y0 - i
        if yy < 0:
            break
        t = np.float32(i) / np.float32(b + 1)
        k[:, yy:yy + 1, z0:z1] = _ramp(t)

    # Y-high face
    for i in range(0, b):
        yy = y1 + i
        if yy >= n:
            break
        t = np.float32(i + 1) / np.float32(b + 1)
        k[:, yy:yy + 1, z0:z1] = _ramp(t)

    # Z-low face
    for i in range(1, b + 1):
        zz = z0 - i
        if zz < 0:
            break
        t = np.float32(i) / np.float32(b + 1)
        k[:, y0:y1, zz:zz + 1] = _ramp(t)

    # Z-high face
    for i in range(0, b):
        zz = z1 + i
        if zz >= n:
            break
        t = np.float32(i + 1) / np.float32(b + 1)
        k[:, y0:y1, zz:zz + 1] = _ramp(t)

    return k


# --- Soft wire mask for trimming planar kink walls ---
def _build_wire_mask(
    n: int,
    *,
    wire_y0: int,
    wire_y1: int,
    wire_z0: int,
    wire_z1: int,
    bevel: int,
) -> np.ndarray:
    """Build a soft wire mask (float32) to trim planar kink walls into a bullet.

    Mask is 1.0 inside the wire box [y0,y1)×[z0,z1) and ramps to 0.0 across an
    outer bevel band of width `bevel` voxels. Outside the bevel band the mask is 0.0.

    This is applied only when the per-voxel k-grid is enabled, to avoid injecting a
    full YZ wall into a stiff outside medium.
    """

    if not (0 <= int(wire_y0) < int(wire_y1) <= int(n)):
        raise ValueError(f"wire_y0/wire_y1 out of bounds: y0={wire_y0} y1={wire_y1} n={n}")
    if not (0 <= int(wire_z0) < int(wire_z1) <= int(n)):
        raise ValueError(f"wire_z0/wire_z1 out of bounds: z0={wire_z0} z1={wire_z1} n={n}")

    y0, y1 = int(wire_y0), int(wire_y1)
    z0, z1 = int(wire_z0), int(wire_z1)
    b = int(max(0, bevel))

    ys = np.arange(n, dtype=np.int32)
    zs = np.arange(n, dtype=np.int32)

    dy = np.zeros((n,), dtype=np.int32)
    dy[ys < y0] = (y0 - ys[ys < y0])
    dy[ys >= y1] = (ys[ys >= y1] - (y1 - 1))

    dz = np.zeros((n,), dtype=np.int32)
    dz[zs < z0] = (z0 - zs[zs < z0])
    dz[zs >= z1] = (zs[zs >= z1] - (z1 - 1))

    # Chebyshev distance to the wire rectangle in the YZ plane.
    d = np.maximum(dy[:, None], dz[None, :]).astype(np.int32)

    m2 = np.zeros((n, n), dtype=np.float32)
    if b <= 0:
        m2[(d == 0)] = np.float32(1.0)
        return m2

    # 1.0 in-core; linearly ramp down to 0.0 over the bevel band (d=1..b).
    m2[(d == 0)] = np.float32(1.0)
    in_band = (d > 0) & (d <= b)
    if bool(np.any(in_band)):
        m2[in_band] = (np.float32(1.0) - (d[in_band].astype(np.float32) / np.float32(b + 1))).astype(np.float32)

    return m2



# --- T-junction mask for wire experiments (v0: hard mask, no bevel) ---

def _build_t_junction_mask(
    n: int,
    *,
    wire_y0: int,
    wire_y1: int,
    wire_z0: int,
    wire_z1: int,
    junction_x: int,
    branch_len: int,
    branch_thick: int = 2,
) -> np.ndarray:
    """Build a hard 3D domain mask for a simple T-junction.

    Geometry (array order is [x,y,z]):
      - Trunk: spans x in [0, junction_x+1) with YZ cross-section [wire_y0, wire_y1)×[wire_z0, wire_z1).
      - Branch: spans x in [junction_x, junction_x+branch_thick) with the same Z bounds and
        Y in [wire_y0-branch_len, wire_y1+branch_len).

    Returns a float32 mask m3 with 1.0 inside the junction volume, 0.0 outside.

    Notes:
      - v0 is intentionally blocky (no bevel / smoothing).
      - Callers convert m3 to the `domain_mask` convention expected by traffic kernels.
    """

    if not (0 <= int(wire_y0) < int(wire_y1) <= int(n)):
        raise ValueError(f"wire_y0/wire_y1 out of bounds: y0={wire_y0} y1={wire_y1} n={n}")
    if not (0 <= int(wire_z0) < int(wire_z1) <= int(n)):
        raise ValueError(f"wire_z0/wire_z1 out of bounds: z0={wire_z0} z1={wire_z1} n={n}")

    jx = int(junction_x)
    if not (0 <= jx < int(n)):
        raise ValueError(f"junction_x out of bounds: junction_x={junction_x} n={n}")

    bl = int(branch_len)
    if bl <= 0:
        raise ValueError(f"branch_len must be >0, got {branch_len}")

    bt = int(max(1, branch_thick))

    y0, y1 = int(wire_y0), int(wire_y1)
    z0, z1 = int(wire_z0), int(wire_z1)

    # Branch runs along Y centred on the trunk, extending equally above and below.
    by0 = max(0, y0 - bl)
    by1 = min(int(n), y1 + bl)

    # Trunk spans the full X extent (so the junction is a true T, not a dead-end).
    tx0, tx1 = 0, int(n)

    # Branch is a short slab in X at the junction.
    bx0, bx1 = max(0, jx), min(int(n), jx + bt)

    m3 = np.zeros((n, n, n), dtype=np.float32)

    # Trunk volume.
    m3[tx0:tx1, y0:y1, z0:z1] = np.float32(1.0)

    # Branch volume.
    m3[bx0:bx1, by0:by1, z0:z1] = np.float32(1.0)

    return m3


# --- OR-junction mask for wire experiments (v0: hard mask, shared dump cavity) ---

def _build_or_junction_mask(
    n: int,
    *,
    wire_y0: int,
    wire_y1: int,
    wire_z0: int,
    wire_z1: int,
    junction_x: int,
    branch_len: int,
    branch_thick: int = 2,
    dump_len: int = 24,
    dump_throat: int = 2,
    dump_y_pad: int = 0,
) -> np.ndarray:
    """Build a hard 3D domain mask for an OR-style impact junction with a shared dump cavity.

    This is a geometry primitive intended for *kink-wall* logic where each input acts like a piston.
    Rather than attempting fan-in of travelling particles, each input can strike an internal wall and
    eject a pulse into the trunk; rebound/backwash energy is preferentially routed into a shared dump.

    Array order is [x,y,z].

    Geometry:
      - Trunk: spans full X extent with YZ cross-section [wire_y0, wire_y1)×[wire_z0, wire_z1).
      - Junction slab: x in [junction_x, junction_x+branch_thick).
      - Input branches: within the junction slab, extend Y to [wire_y0-branch_len, wire_y1+branch_len)
        (same Z bounds as the trunk).
      - Shared dump cavity: a widened pocket downstream of the junction slab, connected via a narrow
        throat centred on the trunk centreline. The dump is *inside* the active domain; callers may
        optionally apply local damping (gamma) in this region later.

    Parameters:
      - dump_len: X length of the dump cavity beyond the junction slab.
      - dump_throat: half-width (in voxels) of the throat in Y that connects trunk->dump.
      - dump_y_pad: extra padding (in voxels) to widen the dump cavity in Y beyond the trunk band.

    Returns float32 mask m3 with 1.0 inside the active domain, 0.0 outside.

    Notes:
      - v0 is intentionally blocky (no bevel / smoothing).
      - This does not implement dissipation by itself; it only provides geometry.
    """

    if not (0 <= int(wire_y0) < int(wire_y1) <= int(n)):
        raise ValueError(f"wire_y0/wire_y1 out of bounds: y0={wire_y0} y1={wire_y1} n={n}")
    if not (0 <= int(wire_z0) < int(wire_z1) <= int(n)):
        raise ValueError(f"wire_z0/wire_z1 out of bounds: z0={wire_z0} z1={wire_z1} n={n}")

    jx = int(junction_x)
    if not (0 <= jx < int(n)):
        raise ValueError(f"junction_x out of bounds: junction_x={junction_x} n={n}")

    bl = int(branch_len)
    if bl <= 0:
        raise ValueError(f"branch_len must be >0, got {branch_len}")

    bt = int(max(1, branch_thick))
    dl = int(max(0, dump_len))
    dtw = int(max(1, dump_throat))
    dpad = int(max(0, dump_y_pad))

    y0, y1 = int(wire_y0), int(wire_y1)
    z0, z1 = int(wire_z0), int(wire_z1)

    # Branch runs along Y centred on the trunk, extending equally above and below.
    by0 = max(0, y0 - bl)
    by1 = min(int(n), y1 + bl)

    # Trunk spans the full X extent.
    tx0, tx1 = 0, int(n)

    # Junction slab in X.
    bx0, bx1 = max(0, jx), min(int(n), jx + bt)

    # Dump cavity in X (downstream of the slab).
    dx0, dx1 = bx1, min(int(n), bx1 + dl)

    # Trunk centreline (voxel-centre convention).
    yc = 0.5 * (float(y0) + float(y1 - 1))
    cy = int(round(yc))

    # Throat band in Y (connects trunk to dump).
    ty0 = max(0, cy - dtw)
    ty1 = min(int(n), cy + dtw + 1)

    # Dump width in Y: trunk band expanded by dump_y_pad.
    dy0 = max(0, y0 - dpad)
    dy1 = min(int(n), y1 + dpad)

    m3 = np.zeros((n, n, n), dtype=np.float32)

    # Trunk volume.
    m3[tx0:tx1, y0:y1, z0:z1] = np.float32(1.0)

    # Input branch volume (slab only).
    m3[bx0:bx1, by0:by1, z0:z1] = np.float32(1.0)

    # Shared dump cavity: widen in Y, same Z as trunk, downstream of slab.
    if dx1 > dx0:
        m3[dx0:dx1, dy0:dy1, z0:z1] = np.float32(1.0)
        # Narrow throat to encourage backwash energy into the dump without opening a huge cavity.
        m3[dx0:dx1, ty0:ty1, z0:z1] = np.float32(1.0)

    return m3


# --- Y-junction mask for wire experiments (v0: hard mask, no bevel) ---

def _build_y_junction_mask(
    n: int,
    *,
    wire_y0: int,
    wire_y1: int,
    wire_z0: int,
    wire_z1: int,
    junction_x: int,
    arm_len: int,
    arm_thick: int = 2,
) -> np.ndarray:
    """Build a hard 3D domain mask for a simple symmetric Y-junction.

    Array order is [x,y,z].

    Trunk:
      - spans full X extent
      - YZ cross-section is [wire_y0, wire_y1)×[wire_z0, wire_z1)

    Arms (engineering corridors):
      - two diagonal corridors in the XY plane that feed into the trunk at x=junction_x
      - corridor centreline: y = yc ± (junction_x - x) for x in [junction_x-arm_len, junction_x]
      - a +X-moving kink_wall can traverse the corridor and enter the trunk

    Returns float32 mask m3 with 1.0 inside the active domain, 0.0 outside.

    Notes:
      - v0 is intentionally blocky (no bevel / smoothing).
      - This is a geometry primitive for OR/AND probing; it does not guarantee fan-in.
    """

    if not (0 <= int(wire_y0) < int(wire_y1) <= int(n)):
        raise ValueError(f"wire_y0/wire_y1 out of bounds: y0={wire_y0} y1={wire_y1} n={n}")
    if not (0 <= int(wire_z0) < int(wire_z1) <= int(n)):
        raise ValueError(f"wire_z0/wire_z1 out of bounds: z0={wire_z0} z1={wire_z1} n={n}")

    jx = int(junction_x)
    if not (0 <= jx < int(n)):
        raise ValueError(f"junction_x out of bounds: junction_x={junction_x} n={n}")

    al = int(arm_len)
    if al <= 0:
        raise ValueError(f"arm_len must be >0, got {arm_len}")

    at = int(max(1, arm_thick))

    y0, y1 = int(wire_y0), int(wire_y1)
    z0, z1 = int(wire_z0), int(wire_z1)

    # Trunk centreline (voxel-centre convention).
    yc = 0.5 * (float(y0) + float(y1 - 1))

    m3 = np.zeros((n, n, n), dtype=np.float32)

    # Trunk volume (downstream of junction only): avoid a backward-facing open waveguide
    # that can soak energy or create back-propagating junk.
    tx_start = max(0, jx - at)
    m3[tx_start:, y0:y1, z0:z1] = np.float32(1.0)

    # Merge slab at the junction: keep it tight to the trunk cross-section.
    # A large cavity invites corner pinning / slosh on a discrete lattice.
    mx0, mx1 = max(0, jx), min(int(n), jx + at)
    m3[mx0:mx1, y0:y1, z0:z1] = np.float32(1.0)

    # Arm half-width: reuse the trunk half-width so corridors can carry a full wall segment.
    hw = max(1, int((y1 - y0) // 2))

    # Chamfer band: in the last few voxels before the junction, taper the arm corridors into the trunk
    # to avoid a hard corner snag.
    chamfer = int(min(10, max(2, al // 2)))

    # Diagonal arms upstream of the junction.
    x_start = max(0, jx - al)
    for xx in range(x_start, jx + 1):
        d = int(jx - xx)  # 0..arm_len
        for sgn in (+1, -1):
            y_c = int(round(yc + float(sgn * d)))
            ay0 = max(0, y_c - hw)
            ay1 = min(int(n), y_c + hw + 1)

            # Near the junction, taper the corridor toward the trunk band.
            if d <= chamfer:
                t = float(d) / float(max(1, chamfer))  # 0 at junction, 1 upstream
                # Interpolate the allowed band toward the trunk cross-section.
                ty0 = int(round((1.0 - t) * float(y0) + t * float(ay0)))
                ty1 = int(round((1.0 - t) * float(y1) + t * float(ay1)))
                ty0 = max(0, min(int(n), ty0))
                ty1 = max(0, min(int(n), ty1))
                if ty1 > ty0:
                    m3[xx:xx + 1, ty0:ty1, z0:z1] = np.float32(1.0)
            else:
                m3[xx:xx + 1, ay0:ay1, z0:z1] = np.float32(1.0)

    return m3

def run_collidersg(
    *,
    n: int,
    steps: int,
    dt: float,
    c2: float,
    sg_k: float,
    # Optional spatial SG stiffness mask (“wire”): if enabled, k is per-voxel.
    # Inside the wire box we use sg_k. Outside we use sg_k_outside.
    sg_k_outside: float = 0.0,
    wire_y0: int = -1,
    wire_y1: int = -1,
    wire_z0: int = -1,
    wire_z1: int = -1,
    wire_bevel: int = 0,
    wire_geom: str = "straight",  # straight|t_junction|y_junction|or_junction
    junction_x: int = -1,
    branch_len: int = 0,
    branch_thick: int = 2,
    dump_len: int = 24,
    dump_throat: int = 2,
    dump_y_pad: int = 0,
    sprites: Sequence[SpriteSpec],
    sprite_asset_h5: str = "",
    out_csv: str,
    log_every: int = 10,
    track_r: int = 14,
    peak_thresh: float = 0.05,
    phi_abs_every: int = 0,
    boundary_mode: str = "sponge",
    sponge_width: int = 32,
    sponge_strength: float = 0.1,
    gamma: float = 0.0,
    decay: float = 0.0,
    scenario: str = "",
    progress: Optional[Callable[[int], None]] = None,
    provenance_header: Optional[str] = None,
) -> str:
    """Run an SG interaction experiment and write a trajectory CSV.

    Output CSV is long-format (one row per sprite per logging tick).

    Scenario “E” is the wire-constrained kink-bullet prototype: enable `sg_k_outside>0` and
    provide `wire_y0/wire_y1/wire_z0/wire_z1` (optionally `wire_bevel`) to build a per-voxel k-grid.
    For kink walls, when the wire is enabled, we also build a `domain_mask` so the SG field only evolves
    inside the wire+bevel domain.
    """

    if steps <= 0:
        raise ValueError(f"steps must be >0, got {steps}")
    if log_every <= 0:
        raise ValueError(f"log_every must be >0, got {log_every}")
    if track_r <= 1:
        raise ValueError(f"track_r must be >1, got {track_r}")
    if phi_abs_every < 0:
        raise ValueError(f"phi_abs_every must be >=0, got {phi_abs_every}")
    if boundary_mode not in ("open", "zero", "neumann", "sponge"):
        raise ValueError(
            "boundary_mode must be one of 'open', 'zero', 'neumann', 'sponge', got "
            f"{boundary_mode!r}"
        )
    if sponge_width < 0:
        raise ValueError(f"sponge_width must be >=0, got {sponge_width}")
    if sponge_strength < 0:
        raise ValueError(f"sponge_strength must be >=0, got {sponge_strength}")
    if gamma < 0:
        raise ValueError(f"gamma must be >=0, got {gamma}")
    if decay < 0:
        raise ValueError(f"decay must be >=0, got {decay}")
    if float(sg_k_outside) < 0.0:
        raise ValueError(f"sg_k_outside must be >=0, got {sg_k_outside}")
    if int(wire_bevel) < 0:
        raise ValueError(f"wire_bevel must be >=0, got {wire_bevel}")

    wg = str(wire_geom or "").strip() or "straight"
    if wg not in ("straight", "t_junction", "y_junction", "or_junction"):
        raise ValueError(f"wire_geom must be 'straight', 't_junction', 'y_junction', or 'or_junction', got {wire_geom!r}")

    if wg == "t_junction":
        if int(wire_bevel) != 0:
            raise ValueError(f"t_junction wire_geom requires wire_bevel=0 (v0 hard mask), got {wire_bevel}")
        if int(junction_x) < 0:
            raise ValueError(f"t_junction requires junction_x >=0, got {junction_x}")
        if int(branch_len) <= 0:
            raise ValueError(f"t_junction requires branch_len >0, got {branch_len}")
        if int(branch_thick) <= 0:
            raise ValueError(f"t_junction requires branch_thick >0, got {branch_thick}")

    if wg == "or_junction":
        if int(wire_bevel) != 0:
            raise ValueError(f"or_junction wire_geom requires wire_bevel=0 (v0 hard mask), got {wire_bevel}")
        if int(junction_x) < 0:
            raise ValueError(f"or_junction requires junction_x >=0, got {junction_x}")
        if int(branch_len) <= 0:
            raise ValueError(f"or_junction requires branch_len >0, got {branch_len}")
        if int(branch_thick) <= 0:
            raise ValueError(f"or_junction requires branch_thick >0, got {branch_thick}")
        if int(dump_len) < 0:
            raise ValueError(f"or_junction requires dump_len >=0, got {dump_len}")
        if int(dump_throat) <= 0:
            raise ValueError(f"or_junction requires dump_throat >0, got {dump_throat}")
        if int(dump_y_pad) < 0:
            raise ValueError(f"or_junction requires dump_y_pad >=0, got {dump_y_pad}")

    if wg == "y_junction":
        if int(wire_bevel) != 0:
            raise ValueError(f"y_junction wire_geom requires wire_bevel=0 (v0 hard mask), got {wire_bevel}")
        if int(junction_x) < 0:
            raise ValueError(f"y_junction requires junction_x >=0, got {junction_x}")
        if int(branch_len) <= 0:
            raise ValueError(f"y_junction requires branch_len >0 (arm length), got {branch_len}")
        if int(branch_thick) <= 0:
            raise ValueError(f"y_junction requires branch_thick >0 (merge slab thickness), got {branch_thick}")

    # Wire/k-grid is enabled only when sg_k_outside > 0.
    # When enabled, all wire bounds must be provided (fail-loud).
    if float(sg_k_outside) > 0.0:
        if wire_y0 < 0 or wire_y1 < 0 or wire_z0 < 0 or wire_z1 < 0:
            raise ValueError(
                "wire bounds must be set when sg_k_outside > 0; "
                f"got wire_y0={wire_y0} wire_y1={wire_y1} wire_z0={wire_z0} wire_z1={wire_z1}"
            )

    asset_path = str(sprite_asset_h5 or "").strip()

    asset_meta: dict[str, Any] = {}
    asset_sg_k = float("nan")
    if asset_path != "":
        # Read once here so we can validate/infer SG parameters.
        _asset_meta0 = read_sprite_asset_h5(asset_path)
        asset_meta = dict(_asset_meta0.meta or {})
        try:
            asset_sg_k = float(asset_meta.get("sg.k", float("nan")))
        except Exception:
            asset_sg_k = float("nan")

        # Auto-set: if sg_k is effectively unset, adopt the asset's k.
        if (not math.isfinite(float(sg_k))) or (abs(float(sg_k)) < 1.0e-12):
            if not math.isfinite(asset_sg_k):
                raise ValueError(f"sprite asset missing sg.k attribute: {asset_path}")
            sg_k = float(asset_sg_k)
        else:
            # Verify: fail-loud on mismatch.
            if math.isfinite(asset_sg_k):
                if abs(float(sg_k) - float(asset_sg_k)) > 1.0e-6:
                    raise ValueError(
                        f"collidersg sg_k mismatch: collider_k={float(sg_k)} asset_k={float(asset_sg_k)} asset={asset_path}"
                    )

    # Build initial state.
    # If a sprite asset is provided, we stamp the *living* (phi,vel,src,load) patch for each sprite.
    # Otherwise we fall back to procedural Gaussian stamps + finite-shift velocity initialisation.
    init_mode = "gaussian"
    if asset_path != "":
        init_mode = "sprite_asset"
        phi, vel, src, load, pll_info = initialise_from_sprite_asset(n, sprites, asset_path, dt, float(sg_k), float(c2))
    else:
        any_kink = any((str(getattr(sp, "kind", "gaussian")) == "kink_wall") for sp in sprites)
        if any_kink:
            # Kink-wall mode. Allow extra tracker-only sprites (track_only=True) so we can add detectors
            # without injecting additional walls. Passive detectors (track_only=True) may be non-kink species.
            inject_kinks: List[SpriteSpec] = []
            for sp in sprites:
                if bool(getattr(sp, "track_only", False)):
                    # Passive detectors are allowed to be non-kink species (e.g. gaussian) even in kink-wall runs.
                    continue
                if str(sp.kind) != "kink_wall":
                    raise ValueError(f"cannot mix kink_wall with other sprite kinds in one run: sid={sp.sid} kind={sp.kind!r}")
                inject_kinks.append(sp)

            if len(inject_kinks) == 0:
                raise ValueError("kink_wall mode requires at least one non-track_only kink_wall injector")

            init_mode = "kink_wall"
            phi, vel = initialise_kink_walls(n, inject_kinks, dt, float(c2), float(sg_k))
            src = np.zeros_like(phi)
            load = np.zeros_like(phi)
            pll_info = []

            # (Soft trim of phi/vel by m3 removed.)
        else:
            phi = initialise_phi(n, sprites)
            vel = initialise_vel_finite_shift(phi, sprites, dt)
            src = np.zeros_like(phi)
            load = np.zeros_like(phi)
            pll_info = []

    # --- Diagnostics: asset kick scale ---
    asset_phi_abs_max = 0.0
    asset_vel_abs_max = 0.0
    vel_trans_abs_max = 0.0
    kick_dx = 0.0
    kick_dy = 0.0
    kick_dz = 0.0

    if asset_path != "" and len(sprites) > 0:
        _asset0 = read_sprite_asset_h5(asset_path)
        asset_phi_abs_max = float(np.max(np.abs(_asset0.phi)))
        asset_vel_abs_max = float(np.max(np.abs(_asset0.vel)))

        _sp0 = sprites[0]
        _vx, _vy, _vz = float(_sp0.vel[0]), float(_sp0.vel[1]), float(_sp0.vel[2])
        kick_dx = _vx * float(dt)
        kick_dy = _vy * float(dt)
        kick_dz = _vz * float(dt)

        # Diagnostic: tri-linear finite-shift kick magnitude across all sprites (worst-case).
        _phi0 = _asset0.phi.astype(np.float32)
        vel_trans_abs_max = 0.0
        for _sp in sprites:
            _vx, _vy, _vz = float(_sp.vel[0]), float(_sp.vel[1]), float(_sp.vel[2])
            _dx = _vx * float(dt)
            _dy = _vy * float(dt)
            _dz = _vz * float(dt)
            if abs(_dx) < 1.0e-12 and abs(_dy) < 1.0e-12 and abs(_dz) < 1.0e-12:
                continue
            _phi_shift = _shift_patch_trilinear(_phi0, _dx, _dy, _dz)
            _vel_trans = (_phi_shift - _phi0) / float(dt)
            vel_trans_abs_max = max(float(vel_trans_abs_max), float(np.max(np.abs(_vel_trans))))

    vel0_abs_max = float(np.max(np.abs(vel)))

    # Fail-fast overlap sanity.
    # This guard is ONLY valid for *procedural Gaussian* stamps; kink walls and sprite assets can exceed pi.
    phi_abs_max = float(np.max(np.abs(phi)))
    if init_mode == "gaussian" and phi_abs_max > math.pi:
        raise RuntimeError(f"initial phi_abs_max={phi_abs_max:.3f} exceeds pi; sprites overlap/too strong")
    phi_abs_max_cached = float(phi_abs_max)

    # Track state.
    actives: List[ActiveSprite] = []
    for sp in sprites:
        actives.append(ActiveSprite(
            sid=sp.sid,
            pos=np.array(sp.pos, dtype=np.float32),
            vel=np.array(sp.vel, dtype=np.float32),
            kind=str(getattr(sp, "kind", "gaussian")),
            track_only=bool(getattr(sp, "track_only", False)),
            fixed_probe=bool(getattr(sp, "fixed_probe", False)),
        ))

    exit_pos_ct: Dict[int, int] = {int(a.sid): 0 for a in actives}
    exit_neg_ct: Dict[int, int] = {int(a.sid): 0 for a in actives}
    last_hit_pos: Dict[int, int] = {int(a.sid): 0 for a in actives}
    last_hit_neg: Dict[int, int] = {int(a.sid): 0 for a in actives}

    trunk_out_ct: Dict[int, int] = {int(a.sid): 0 for a in actives}
    last_hit_trunk: Dict[int, int] = {int(a.sid): 0 for a in actives}

    # Kink walls are plateau states (phi -> 2π in half-space). Any boundary that clamps phi->0
    # (Dirichlet 'zero') or damps toward 0 ('sponge') injects an effective anti-kink at the edges.
    # For kink-wall runs we therefore force a *non-clamping* boundary and disable sponge params.
    # Prefer Neumann (zero-gradient) because it supports a stable plateau at the edges.
    if init_mode == "kink_wall":
        if boundary_mode in ("zero", "sponge"):
            boundary_mode = "neumann"
        sponge_width = 0
        sponge_strength = 0.0


    # Canonicalise sponge parameters: only meaningful in sponge mode.
    if boundary_mode != "sponge":
        sponge_width = 0
        sponge_strength = 0.0
    else:
        if int(sponge_width) <= 0:
            raise ValueError(f"sponge_width must be >0 when boundary_mode='sponge', got {sponge_width}")
        if float(sponge_strength) <= 0.0:
            raise ValueError(f"sponge_strength must be >0 when boundary_mode='sponge', got {sponge_strength}")

    # Optional per-voxel SG stiffness mask ("wire"). Disabled unless all wire bounds are provided.
    k_grid: Optional[np.ndarray] = None
    domain_mask: Optional[np.ndarray] = None
    if float(sg_k_outside) > 0.0:
        if wg == "straight":
            k_grid = _build_k_grid(
                n,
                k_inside=float(sg_k),
                k_outside=float(sg_k_outside),
                wire_y0=int(wire_y0),
                wire_y1=int(wire_y1),
                wire_z0=int(wire_z0),
                wire_z1=int(wire_z1),
                bevel=int(wire_bevel),
            )
            m2 = _build_wire_mask(
                n,
                wire_y0=int(wire_y0),
                wire_y1=int(wire_y1),
                wire_z0=int(wire_z0),
                wire_z1=int(wire_z1),
                bevel=int(wire_bevel),
            )
            # domain_mask convention (traffic kernels): 0 = active, 1 = wall.
            # Our 2D mask m2 is >0 inside wire+bevel, 0 outside.
            wall2 = (m2 <= np.float32(0.0)).astype(np.uint8)
            domain_mask = np.tile(wall2[None, :, :], (n, 1, 1)).astype(np.uint8)
        else:
            # 3D hard-mask junction geometries.
            if wg == "t_junction":
                m3 = _build_t_junction_mask(
                    n,
                    wire_y0=int(wire_y0),
                    wire_y1=int(wire_y1),
                    wire_z0=int(wire_z0),
                    wire_z1=int(wire_z1),
                    junction_x=int(junction_x),
                    branch_len=int(branch_len),
                    branch_thick=int(branch_thick),
                )
            elif wg == "y_junction":
                m3 = _build_y_junction_mask(
                    n,
                    wire_y0=int(wire_y0),
                    wire_y1=int(wire_y1),
                    wire_z0=int(wire_z0),
                    wire_z1=int(wire_z1),
                    junction_x=int(junction_x),
                    arm_len=int(branch_len),
                    arm_thick=int(branch_thick),
                )
            elif wg == "or_junction":
                m3 = _build_or_junction_mask(
                    n,
                    wire_y0=int(wire_y0),
                    wire_y1=int(wire_y1),
                    wire_z0=int(wire_z0),
                    wire_z1=int(wire_z1),
                    junction_x=int(junction_x),
                    branch_len=int(branch_len),
                    branch_thick=int(branch_thick),
                    dump_len=int(dump_len),
                    dump_throat=int(dump_throat),
                    dump_y_pad=int(dump_y_pad),
                )
            else:
                raise ValueError(f"unsupported wire_geom for 3D mask build: {wg!r}")

            k_grid = np.full((n, n, n), float(sg_k_outside), dtype=np.float32)
            k_grid[m3 > np.float32(0.0)] = np.float32(float(sg_k))
            domain_mask = (m3 <= np.float32(0.0)).astype(np.uint8)

        # Enforce "junction exists only inside the domain": outside is non-participating.
        # This prevents stiff outside-k from snapping the initial condition.
        phi = phi.copy()
        vel = vel.copy()
        phi[domain_mask != 0] = np.float32(0.0)
        vel[domain_mask != 0] = np.float32(0.0)
        src = src.copy()
        load = load.copy()
        src[domain_mask != 0] = np.float32(0.0)
        load[domain_mask != 0] = np.float32(0.0)

    # Use the canonical TrafficParams so typing + validation remain consistent.
    # TrafficParams is immutable; construct it with the required kwargs.
    tp = TrafficParams(
        mode="sine_gordon",
        dt=float(dt),
        c2=float(c2),
        gamma=float(gamma),
        decay=float(decay),
        boundary_mode=str(boundary_mode),
        boundary_zero=(str(boundary_mode) == "zero"),
        sponge_width=int(sponge_width),
        sponge_strength=float(sponge_strength),
        traffic_k=float(sg_k),
        inject=0.0,
    )

    out_path = resolve_out_path(__file__, out_csv)
    ensure_parent_dir(out_path)

    if provenance_header is None:
        provenance_header = write_csv_provenance_header(
            producer="collidersg",
            command=" ".join(sys.argv),
            cwd=os.getcwd(),
            python_exe=sys.executable,
            when_iso=wallclock_iso(),
            experiment="09",
            extra={
                "artefact": "sg_collision_tracks",
                "n": n,
                "steps": steps,
                "dt": dt,
                "c2": c2,
                "sg_k": sg_k,
                "asset_sg_k": float(asset_sg_k),
                "asset_meta_has_k": int(1 if math.isfinite(asset_sg_k) else 0),
                "init_mode": init_mode,
                "asset_kick": "finite_shift_trilinear_patch" if asset_path != "" else "finite_shift_gaussian",
                "log_every": log_every,
                "phi_abs_every": phi_abs_every,
                "track_r": track_r,
                "peak_thresh": peak_thresh,
                "sprite_count": len(sprites),
                "boundary_mode": boundary_mode,
                "sponge_width": sponge_width,
                "sponge_strength": sponge_strength,
                "gamma": gamma,
                "decay": decay,
                "sprite_asset_h5": asset_path,
                "asset_phi_abs_max": float(asset_phi_abs_max),
                "asset_vel_abs_max": float(asset_vel_abs_max),
                "vel_trans_abs_max_max": float(vel_trans_abs_max),
                "kick_dx": float(kick_dx),
                "kick_dy": float(kick_dy),
                "kick_dz": float(kick_dz),
                "vel0_abs_max": float(vel0_abs_max),
                "pll_info_json": json.dumps(pll_info, separators=(",", ":")),
                "pll_info_count": int(len(pll_info)),
                "asset_drift_removed": int(0 if asset_path != "" else 0),
                "boundary_mode_eff": str(boundary_mode),
                "sponge_width_eff": int(sponge_width),
                "sponge_strength_eff": float(sponge_strength),
                "k_grid_enabled": int(1 if k_grid is not None else 0),
                "sg_k_outside": float(sg_k_outside),
                "wire_y0": int(wire_y0),
                "wire_y1": int(wire_y1),
                "wire_z0": int(wire_z0),
                "wire_z1": int(wire_z1),
                "wire_bevel": int(wire_bevel),
                "wire_geom": str(wg),
                "junction_x": int(junction_x),
                "branch_len": int(branch_len),
                "branch_thick": int(branch_thick),
                "dump_len": int(dump_len),
                "dump_throat": int(dump_throat),
                "dump_y_pad": int(dump_y_pad),
                "domain_mask_enabled": int(1 if 'domain_mask' in locals() and domain_mask is not None else 0),
            },
        )

    t0 = now_s()
    wall_s = 0.0

    with open(out_path, "w", newline="") as f:
        ph = provenance_header
        if ph and not ph.endswith("\n"):
            ph += "\n"
        if ph:
            f.write(ph)
        f.write("# artefact=sg_collision_tracks\n")
        sc = str(scenario).strip()
        if sc != "":
            f.write(f"# scenario={sc}\n")
        f.write(f"# sprites_digest={sprites_digest(sprites)}\n")
        f.write(f"# boundary_mode_eff={boundary_mode} sponge_width_eff={int(sponge_width)} sponge_strength_eff={float(sponge_strength):.6f}\n")
        f.write(f"# k_grid_enabled={1 if k_grid is not None else 0} sg_k_outside={float(sg_k_outside):.6f} wire_geom={wg} wire_y0={int(wire_y0)} wire_y1={int(wire_y1)} wire_z0={int(wire_z0)} wire_z1={int(wire_z1)} wire_bevel={int(wire_bevel)} junction_x={int(junction_x)} branch_len={int(branch_len)} branch_thick={int(branch_thick)} dump_len={int(dump_len)} dump_throat={int(dump_throat)} dump_y_pad={int(dump_y_pad)} domain_mask_enabled={1 if 'domain_mask' in locals() and domain_mask is not None else 0}\n")
        if asset_path != "":
            f.write(f"# asset_phi_abs_max={asset_phi_abs_max:.6f}\n")
            f.write(f"# asset_vel_abs_max={asset_vel_abs_max:.6f}\n")
            f.write(f"# vel_trans_abs_max_max={vel_trans_abs_max:.6f}\n")
            f.write(f"# kick_dx={kick_dx:.6f} kick_dy={kick_dy:.6f} kick_dz={kick_dz:.6f}\n")
            if len(pll_info) > 0:
                f.write(f"# pll_info_json={json.dumps(pll_info, separators=(',', ':'))}\n")
        f.write(f"# vel0_abs_max={vel0_abs_max:.6f}\n")

        trunk_yc = float("nan")
        if (wire_y0 >= 0) and (wire_y1 >= 0):
            trunk_yc = 0.5 * (float(wire_y0) + float(wire_y1 - 1))

        f.write("t,sid,x,y,z,vx,vy,vz,peak_abs,status,phi_abs_max,wall_s,in_trunk,in_branch_pos,in_branch_neg,in_branch,dy_from_trunk_c,hit_exit_pos,hit_exit_neg,exit_pos_ct,exit_neg_ct,hit_trunk_out,trunk_out_ct\n")

        for t in range(steps + 1):
            if t % log_every == 0:
                if phi_abs_every == 0 or (t % int(phi_abs_every) == 0):
                    phi_abs_max_cached = float(np.max(np.abs(phi)))
                phi_abs_max = float(phi_abs_max_cached)
                # For kink walls, tracking by |phi| fails (plateau). Track the wall by a directional
                # derivative along the wall normal: |∂_u phi| = |ux*∂x phi + uy*∂y phi + uz*∂z phi|.
                # Use signed forward differences (not per-axis absolute magnitudes) so diagonal walls remain trackable.
                gradx_abs = None
                grady_abs = None
                gradz_abs = None
                if any(a.kind == "kink_wall" and a.status == "ALIVE" for a in actives):
                    gx = np.zeros_like(phi, dtype=np.float32)
                    gy = np.zeros_like(phi, dtype=np.float32)
                    gz = np.zeros_like(phi, dtype=np.float32)
                    gx[:-1, :, :] = (phi[1:, :, :] - phi[:-1, :, :]).astype(np.float32)
                    gy[:, :-1, :] = (phi[:, 1:, :] - phi[:, :-1, :]).astype(np.float32)
                    gz[:, :, :-1] = (phi[:, :, 1:] - phi[:, :, :-1]).astype(np.float32)
                    gradx_abs = gx
                    grady_abs = gy
                    gradz_abs = gz
                wall_s = now_s() - t0
                for a in actives:
                    if a.status != "ALIVE":
                        f.write(f"{t},{a.sid},{a.pos[0]:.3f},{a.pos[1]:.3f},{a.pos[2]:.3f},{a.vel[0]:.5f},{a.vel[1]:.5f},{a.vel[2]:.5f},0.0,{a.status},{phi_abs_max:.6f},{wall_s:.3f},0,0,0,0,0.0,0,0,0,0,0,0\n")
                        continue

                    # Predictor step. Fixed probes do not move.
                    if bool(getattr(a, "fixed_probe", False)):
                        pred = a.pos.copy()
                    else:
                        pred = a.pos + a.vel * (float(log_every) * float(dt))
                    # For kink walls in confined geometries, full 3D argmax tracking is ill-posed:
                    # the wall is planar and corner gradients win. Track only along the dominant
                    # motion axis and lock the transverse axes to the predicted centreline.
                    if a.kind == "kink_wall" and (k_grid is not None) and (domain_mask is not None) and (wire_y0 >= 0) and (wire_y1 >= 0) and (wire_z0 >= 0) and (wire_z1 >= 0):
                        # Determine dominant motion axis from requested velocity.
                        vx0, vy0, vz0 = float(a.vel[0]), float(a.vel[1]), float(a.vel[2])
                        ax, ay, az = abs(vx0), abs(vy0), abs(vz0)
                        # Default to X if effectively stationary.
                        dom = 0
                        if ay >= ax and ay >= az and ay > 1.0e-9:
                            dom = 1
                        elif az >= ax and az >= ay and az > 1.0e-9:
                            dom = 2

                        # Centreline targets.
                        yc = np.float32(0.5 * (float(wire_y0) + float(wire_y1 - 1)))
                        zc = np.float32(0.5 * (float(wire_z0) + float(wire_z1 - 1)))
                        # For junction slabs, anchor X to the slab centreline so Y/Z pistons cannot drift
                        # toward corner gradients and get "stuck" in tracking.
                        xc = np.float32(float(a.pos[0]))
                        if wg in ("t_junction", "or_junction") and int(junction_x) >= 0 and int(branch_thick) > 0:
                            xc = np.float32(float(int(junction_x)) + 0.5 * float(int(branch_thick) - 1))

                        if dom == 0:
                            # Move along X; lock Y/Z.
                            pred[0] = a.pos[0] + a.vel[0] * (float(log_every) * float(dt))
                            pred[1] = yc
                            pred[2] = zc
                        elif dom == 1:
                            # Move along Y; lock X/Z.
                            pred[0] = xc
                            pred[1] = a.pos[1] + a.vel[1] * (float(log_every) * float(dt))
                            pred[2] = zc
                        else:
                            # Move along Z; lock X/Y.
                            pred[0] = xc
                            pred[1] = yc
                            pred[2] = a.pos[2] + a.vel[2] * (float(log_every) * float(dt))
                    if a.kind == "kink_wall":
                        # Track using directional gradient |∂_u phi| along the wall normal u.
                        # For moving walls we take u parallel to velocity; for v≈0 default to +X.
                        ux, uy, uz = 1.0, 0.0, 0.0
                        try:
                            vx0, vy0, vz0 = float(a.vel[0]), float(a.vel[1]), float(a.vel[2])
                            v2 = float(vx0 * vx0 + vy0 * vy0 + vz0 * vz0)
                            if v2 > 1.0e-18:
                                vabs = float(math.sqrt(v2))
                                ux, uy, uz = vx0 / vabs, vy0 / vabs, vz0 / vabs
                        except Exception:
                            ux, uy, uz = 1.0, 0.0, 0.0

                        if gradx_abs is None or grady_abs is None or gradz_abs is None:
                            field = phi
                        else:
                            field = np.abs(float(ux) * gradx_abs + float(uy) * grady_abs + float(uz) * gradz_abs).astype(np.float32)
                        if domain_mask is not None:
                            # valid_mask: 1 where active (domain_mask==0)
                            valid = (domain_mask == 0).astype(np.uint8)

                            # Kink-wall tracking in confined geometries: 3D argmax tends to snap to corners/endcaps.
                            # Track only along the dominant motion axis at the predicted transverse centre.
                            vx0, vy0, vz0 = float(a.vel[0]), float(a.vel[1]), float(a.vel[2])
                            ax0, ay0, az0 = abs(vx0), abs(vy0), abs(vz0)
                            dom = 0
                            if ay0 >= ax0 and ay0 >= az0 and ay0 > 1.0e-9:
                                dom = 1
                            elif az0 >= ax0 and az0 >= ay0 and az0 > 1.0e-9:
                                dom = 2

                            n0 = int(field.shape[0])
                            ix = int(round(float(pred[0])))
                            iy = int(round(float(pred[1])))
                            iz = int(round(float(pred[2])))
                            ix = max(0, min(n0 - 1, ix))
                            iy = max(0, min(n0 - 1, iy))
                            iz = max(0, min(n0 - 1, iz))

                            r1 = int(track_r)
                            if dom == 0:
                                a0 = max(0, ix - r1)
                                a1 = min(n0, ix + r1 + 1)
                                line = np.abs(field[a0:a1, iy:iy + 1, iz:iz + 1]).reshape(-1)
                                vline = valid[a0:a1, iy:iy + 1, iz:iz + 1].reshape(-1)
                                line[vline == 0] = np.float32(-1.0)
                                k = int(np.argmax(line))
                                peak = float(line[k]) if float(line[k]) > 0.0 else 0.0
                                pos = np.array([float(a0 + k), float(iy), float(iz)], dtype=np.float32)
                            elif dom == 1:
                                a0 = max(0, iy - r1)
                                a1 = min(n0, iy + r1 + 1)
                                line = np.abs(field[ix:ix + 1, a0:a1, iz:iz + 1]).reshape(-1)
                                vline = valid[ix:ix + 1, a0:a1, iz:iz + 1].reshape(-1)
                                line[vline == 0] = np.float32(-1.0)
                                k = int(np.argmax(line))
                                peak = float(line[k]) if float(line[k]) > 0.0 else 0.0
                                pos = np.array([float(ix), float(a0 + k), float(iz)], dtype=np.float32)
                            else:
                                a0 = max(0, iz - r1)
                                a1 = min(n0, iz + r1 + 1)
                                line = np.abs(field[ix:ix + 1, iy:iy + 1, a0:a1]).reshape(-1)
                                vline = valid[ix:ix + 1, iy:iy + 1, a0:a1].reshape(-1)
                                line[vline == 0] = np.float32(-1.0)
                                k = int(np.argmax(line))
                                peak = float(line[k]) if float(line[k]) > 0.0 else 0.0
                                pos = np.array([float(ix), float(iy), float(a0 + k)], dtype=np.float32)
                        else:
                            pos, peak = _max_in_window(field, pred, track_r)
                            if (k_grid is not None) and (domain_mask is not None) and (wire_y0 >= 0) and (wire_y1 >= 0) and (wire_z0 >= 0) and (wire_z1 >= 0) and (wg == "straight"):
                                pos = pos.copy()
                                pos[1] = pred[1]
                                pos[2] = pred[2]
                    else:
                        pos, peak = _max_in_window(phi, pred, track_r)
                    if peak < peak_thresh:
                        # Passive detectors are allowed to "hear nothing" until a wave arrives.
                        # Keep them alive so they can wake up later.
                        if not bool(getattr(a, "track_only", False)):
                            a.status = "LOST"
                    else:
                        if bool(getattr(a, "fixed_probe", False)):
                            # Fixed probe: do not move; do not update velocity.
                            pass
                        else:
                            old = a.pos.copy()
                            a.pos = pos
                            if t != 0:
                                v_meas = (a.pos - old) / (float(log_every) * float(dt))
                                # Do not feed kink-wall tracking back into a.vel.
                                # For kink walls, a.vel encodes the commanded wall normal/speed used by the
                                # tracker (u≈v/|v|). Updating it from noisy argmax motion can collapse the
                                # commanded velocity to ~0 and make pistons appear frozen.
                                if a.kind != "kink_wall":
                                    # Avoid snapping predictor velocity to ~0 due to integer-voxel argmax quantisation.
                                    if float(np.max(np.abs(v_meas))) > 1.0e-6:
                                        a.vel = v_meas

                    in_trunk = 0
                    in_branch_pos = 0
                    in_branch_neg = 0
                    in_branch = 0
                    dy_from_trunk_c = 0.0

                    hit_exit_pos = 0
                    hit_exit_neg = 0
                    hit_trunk_out = 0
                    try:
                        if math.isfinite(trunk_yc):
                            dy_from_trunk_c = float(a.pos[1]) - float(trunk_yc)
                    except Exception:
                        dy_from_trunk_c = 0.0

                    if (wg in ("t_junction", "y_junction", "or_junction", "straight")) and (k_grid is not None) and (domain_mask is not None) and (wire_y0 >= 0) and (wire_y1 >= 0) and (wire_z0 >= 0) and (wire_z1 >= 0):
                        x = float(a.pos[0])
                        y = float(a.pos[1])
                        z = float(a.pos[2])
                        y0 = float(wire_y0)
                        y1 = float(wire_y1)
                        z0 = float(wire_z0)
                        z1 = float(wire_z1)
                        jx = float(junction_x)
                        bt = float(max(1, int(branch_thick)))
                        bl = float(max(1, int(branch_len)))

                        # Trunk occupancy: inside the straight channel cross-section.
                        if (y >= y0) and (y < y1) and (z >= z0) and (z < z1):
                            in_trunk = 1

                        if wg == "t_junction":
                            # Branch occupancy split (v1 diagnostic): within the junction slab in X and same Z bounds.
                            # Positive branch is y >= y1; negative branch is y < y0.
                            by0 = max(0.0, y0 - bl)
                            by1 = min(float(n), y1 + bl)
                            if (x >= jx) and (x < jx + bt) and (z >= z0) and (z < z1) and (y >= by0) and (y < by1):
                                if y >= y1:
                                    in_branch_pos = 1
                                elif y < y0:
                                    in_branch_neg = 1
                            in_branch = 1 if (in_branch_pos != 0 or in_branch_neg != 0) else 0

                            # Exit hit flags: near the dead-end caps of the +Y and -Y branches.
                            # Treat the endcaps as a small band (in voxels) inside the branch extent.
                            exit_band = 2.0
                            if in_branch_pos != 0:
                                if y >= (by1 - 1.0 - exit_band):
                                    hit_exit_pos = 1
                            if in_branch_neg != 0:
                                if y <= (by0 + exit_band):
                                    hit_exit_neg = 1

                        # Trunk output hit: detect a tracker entering the trunk cross-section sufficiently far downstream.
                        # Default threshold is n-42 (x≈150 at n=192), but never earlier than junction_x+10.
                        x_out = float(max(int(junction_x) + 10, int(n) - 42))
                        if in_trunk != 0 and x >= x_out:
                            hit_trunk_out = 1

                    sid_key = int(a.sid)
                    if hit_exit_pos != 0 and last_hit_pos.get(sid_key, 0) == 0:
                        exit_pos_ct[sid_key] = int(exit_pos_ct.get(sid_key, 0)) + 1
                    if hit_exit_neg != 0 and last_hit_neg.get(sid_key, 0) == 0:
                        exit_neg_ct[sid_key] = int(exit_neg_ct.get(sid_key, 0)) + 1
                    last_hit_pos[sid_key] = int(1 if hit_exit_pos != 0 else 0)
                    last_hit_neg[sid_key] = int(1 if hit_exit_neg != 0 else 0)

                    if hit_trunk_out != 0 and last_hit_trunk.get(sid_key, 0) == 0:
                        trunk_out_ct[sid_key] = int(trunk_out_ct.get(sid_key, 0)) + 1
                    last_hit_trunk[sid_key] = int(1 if hit_trunk_out != 0 else 0)

                    f.write(f"{t},{a.sid},{a.pos[0]:.3f},{a.pos[1]:.3f},{a.pos[2]:.3f},{a.vel[0]:.5f},{a.vel[1]:.5f},{a.vel[2]:.5f},{peak:.6f},{a.status},{phi_abs_max:.6f},{wall_s:.3f},{in_trunk},{in_branch_pos},{in_branch_neg},{in_branch},{dy_from_trunk_c:.3f},{hit_exit_pos},{hit_exit_neg},{int(exit_pos_ct.get(int(a.sid), 0))},{int(exit_neg_ct.get(int(a.sid), 0))},{hit_trunk_out},{int(trunk_out_ct.get(int(a.sid), 0))}\n")

            if t == steps:
                break

            # Physics step: single tick. The traffic stepper returns new arrays (it is not in-place).
            phi, vel = evolve_sine_gordon_traffic_steps(phi, vel, src, tp, 1, k_grid=k_grid, domain_mask=domain_mask)
            if progress is not None:
                progress(1)

    wall_s = now_s() - t0
    with open(out_path, "a", newline="") as f:
        f.write(f"# wall_s={wall_s:.3f}\n")

    return out_path