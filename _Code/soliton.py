# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

from __future__ import annotations

import csv
import dataclasses
import time
import os
from typing import Any, Callable, cast

import numpy as np

from traffic import evolve_nonlinear_traffic_steps
# (keep as-is; we'll pass an evolve_fn into the scan; nonlinear remains the default)
from exporters import dump_pipeline_state_h5
from utils import ensure_parent_dir

from extractsprite import SpriteExtractSpec, extract_sprite_from_fields


def _total_energy(phi: np.ndarray, vel: np.ndarray) -> float:
    # Fast diagnostic magnitude (phi^2 + vel^2). Not the physical energy used in _energy_terms().
    # We log both: this one is cheap/robust; _energy_terms() provides KG/SG energy breakdown.
    return float(np.sum(phi * phi) + np.sum(vel * vel))


# Physical energy breakdown (cropped cube, for practical n=512)
def _energy_terms(phi: np.ndarray, vel: np.ndarray, tp: Any, *, crop_r: int | None) -> dict[str, float]:
    """Compute a simple Klein–Gordon-style energy breakdown.

    Notes:
    - We deliberately compute this over the same cropped cube used for R_g to keep n=512 fast.
    - Potential corresponds to force terms used in `traffic.py`:
        F_stiff = -k * phi        => V_stiff = 0.5 * k * phi^2
        F_nl    = -λ * phi^3      => V_nl    = 0.25 * λ * phi^4
    - Gradient term uses forward differences (discrete ∇phi) within the cropped cube.
    """
    n = int(phi.shape[0])
    cx = n // 2
    cy = n // 2
    cz = n // 2

    if crop_r is None:
        x0, x1 = 0, n
        y0, y1 = 0, n
        z0, z1 = 0, n
    else:
        r = int(max(1, crop_r))
        x0 = max(0, cx - r)
        x1 = min(n, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(n, cy + r + 1)
        z0 = max(0, cz - r)
        z1 = min(n, cz + r + 1)

    sub_phi = phi[x0:x1, y0:y1, z0:z1]
    sub_vel = vel[x0:x1, y0:y1, z0:z1]

    # Coefficients
    c2 = float(getattr(tp, "c2", 0.0))
    k = float(getattr(tp, "traffic_k", 0.0))
    lam = float(getattr(tp, "traffic_lambda", 0.0))

    # Kinetic
    E_kin = 0.5 * float(np.sum(sub_vel * sub_vel))

    # Gradient (forward diffs). Use the overlap region so diffs stay inside crop.
    # This slightly undercounts edge contributions, but is consistent across the sweep.
    if sub_phi.shape[0] >= 2 and sub_phi.shape[1] >= 2 and sub_phi.shape[2] >= 2:
        dx = sub_phi[1:, :, :] - sub_phi[:-1, :, :]
        dy = sub_phi[:, 1:, :] - sub_phi[:, :-1, :]
        dz = sub_phi[:, :, 1:] - sub_phi[:, :, :-1]
        E_grad = 0.5 * c2 * float(np.sum(dx * dx) + np.sum(dy * dy) + np.sum(dz * dz))
    else:
        E_grad = 0.0

    # Potential terms
    mode = str(getattr(tp, "mode", ""))

    # Default: Klein–Gordon + phi^4 scaffold (current nonlinear mode).
    E_stiff = 0.0
    E_phi4 = 0.0
    E_sine = 0.0

    if mode == "sine_gordon":
        # Sine–Gordon potential: V = k * (1 - cos(phi))  (so F = -dV/dphi = -k * sin(phi))
        # Note: this is naturally bounded and acts as a saturating non-linearity.
        if k != 0.0:
            E_sine = float(k) * float(np.sum(1.0 - np.cos(sub_phi)))
    else:
        # KG stiffness term: V = 0.5 * k * phi^2
        if k != 0.0:
            E_stiff = 0.5 * k * float(np.sum(sub_phi * sub_phi))
        # phi^4 term: V = 0.25 * lambda * phi^4
        if lam != 0.0:
            p2 = sub_phi * sub_phi
            E_phi4 = 0.25 * lam * float(np.sum(p2 * p2))

    E_total = float(E_kin + E_grad + E_stiff + E_phi4 + E_sine)

    mean_phi = float(np.mean(sub_phi)) if sub_phi.size != 0 else 0.0
    mean_abs_phi = float(np.mean(np.abs(sub_phi))) if sub_phi.size != 0 else 0.0

    # Shifted total energy: for k<0 and lambda>0, the true-vacuum minima have V_min = -k^2/(4*lambda).
    # We add back (-V_min) per cell so the vacuum baseline is ~0, which makes phase-transition runs easier to read.
    if lam > 0.0 and k < 0.0 and sub_phi.size != 0:
        V_min = - (k * k) / (4.0 * lam)
        E_total_shifted = float(E_total - (V_min * float(sub_phi.size)))
    else:
        E_total_shifted = float(E_total)

    # φ distribution summary (cropped cube) for symmetry-breaking diagnostics.
    # Keep this cheap: percentiles over the crop, not the full grid.
    if sub_phi.size != 0:
        flat = sub_phi.reshape(-1).astype(np.float32, copy=False)
        phi_std = float(np.std(flat))
        # Percentiles as a compact “hist summary”.
        p01, p05, p16, p50, p84, p95, p99 = np.quantile(
            flat,
            [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99],
            method="linear",
        ).tolist()
        phi_p01 = float(p01)
        phi_p05 = float(p05)
        phi_p16 = float(p16)
        phi_p50 = float(p50)
        phi_p84 = float(p84)
        phi_p95 = float(p95)
        phi_p99 = float(p99)
    else:
        phi_std = 0.0
        phi_p01 = 0.0
        phi_p05 = 0.0
        phi_p16 = 0.0
        phi_p50 = 0.0
        phi_p84 = 0.0
        phi_p95 = 0.0
        phi_p99 = 0.0

    return {
        "crop_r": float(crop_r) if crop_r is not None else -1.0,
        "mean_phi": float(mean_phi),
        "mean_abs_phi": float(mean_abs_phi),
        "phi_std": float(phi_std),
        "phi_p01": float(phi_p01),
        "phi_p05": float(phi_p05),
        "phi_p16": float(phi_p16),
        "phi_p50": float(phi_p50),
        "phi_p84": float(phi_p84),
        "phi_p95": float(phi_p95),
        "phi_p99": float(phi_p99),
        "E_total": float(E_total),
        "E_total_shifted": float(E_total_shifted),
        "E_kin": float(E_kin),
        "E_grad": float(E_grad),
        "E_stiff": float(E_stiff),
        "E_phi4": float(E_phi4),
        "E_sine": float(E_sine),
        "k": float(k),
        "lambda": float(lam),
    }


# --- Added helpers for cube bounds and phi summary ---
def _cube_bounds(n: int, *, crop_r: int | None, scale: float = 1.0) -> tuple[int, int, int, int, int, int]:
    cx = n // 2
    cy = n // 2
    cz = n // 2
    if crop_r is None:
        return 0, n, 0, n, 0, n
    r = int(max(1, int(float(crop_r) * float(scale))))
    x0 = max(0, cx - r)
    x1 = min(n, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(n, cy + r + 1)
    z0 = max(0, cz - r)
    z1 = min(n, cz + r + 1)
    return x0, x1, y0, y1, z0, z1

def _cube_bounds_about(n: int, *, cx: int, cy: int, cz: int, crop_r: int) -> tuple[int, int, int, int, int, int]:
    r = int(max(1, int(crop_r)))
    x0 = max(0, int(cx) - r)
    x1 = min(int(n), int(cx) + r + 1)
    y0 = max(0, int(cy) - r)
    y1 = min(int(n), int(cy) + r + 1)
    z0 = max(0, int(cz) - r)
    z1 = min(int(n), int(cz) + r + 1)
    return x0, x1, y0, y1, z0, z1



def _phi_summary(phi: np.ndarray, *, bounds: tuple[int, int, int, int, int, int]) -> dict[str, float]:
    x0, x1, y0, y1, z0, z1 = bounds
    sub = phi[x0:x1, y0:y1, z0:z1]
    if sub.size == 0:
        return {
            "mean_phi": 0.0,
            "mean_abs_phi": 0.0,
            "phi_std": 0.0,
            "phi_p01": 0.0,
            "phi_p05": 0.0,
            "phi_p16": 0.0,
            "phi_p50": 0.0,
            "phi_p84": 0.0,
            "phi_p95": 0.0,
            "phi_p99": 0.0,
        }
    mean_phi = float(np.mean(sub))
    mean_abs_phi = float(np.mean(np.abs(sub)))
    flat = sub.reshape(-1).astype(np.float32, copy=False)
    phi_std = float(np.std(flat))
    p01, p05, p16, p50, p84, p95, p99 = np.quantile(
        flat,
        [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99],
        method="linear",
    ).tolist()
    return {
        "mean_phi": float(mean_phi),
        "mean_abs_phi": float(mean_abs_phi),
        "phi_std": float(phi_std),
        "phi_p01": float(p01),
        "phi_p05": float(p05),
        "phi_p16": float(p16),
        "phi_p50": float(p50),
        "phi_p84": float(p84),
        "phi_p95": float(p95),
        "phi_p99": float(p99),
    }

# --- Helper for shell moments ---

def _phi_moments(phi: np.ndarray, *, bounds: tuple[int, int, int, int, int, int]) -> dict[str, float]:
    x0, x1, y0, y1, z0, z1 = bounds
    sub = phi[x0:x1, y0:y1, z0:z1]
    if sub.size == 0:
        return {"count": 0.0, "sum": 0.0, "sum_abs": 0.0, "sum2": 0.0}
    # Accumulate in float64 for stability.
    s = float(np.sum(sub, dtype=np.float64))
    sa = float(np.sum(np.abs(sub), dtype=np.float64))
    s2 = float(np.sum((sub * sub).astype(np.float64, copy=False)))
    return {"count": float(sub.size), "sum": s, "sum_abs": sa, "sum2": s2}


# --- Helper to find the global max |phi| location and radius from centre ---

def _peak_loc_abs(phi: np.ndarray) -> tuple[int, int, int, float]:
    # Return (x, y, z, r_from_centre) of the global max |phi|.
    # Used to detect drift / boundary hits and to select “in-phase” extraction windows.
    n = int(phi.shape[0])
    if n <= 0:
        return 0, 0, 0, 0.0
    flat_idx = int(np.argmax(np.abs(phi)))
    x, y, z = np.unravel_index(flat_idx, phi.shape)
    cx = n // 2
    cy = n // 2
    cz = n // 2
    dx = float(int(x) - int(cx))
    dy = float(int(y) - int(cy))
    dz = float(int(z) - int(cz))
    r = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    return int(x), int(y), int(z), float(r)


# --- Helper for centre-of-mass of rho = phi^2 within bounds ---

def _rho_centroid(phi: np.ndarray, *, bounds: tuple[int, int, int, int, int, int]) -> tuple[float, float, float, float]:
    # Centre-of-mass of rho = phi^2 within bounds, returned as (dx, dy, dz, r) relative to lattice centre.
    # Useful for drift/translation diagnostics that are less brittle than peak-picking.
    x0, x1, y0, y1, z0, z1 = bounds
    sub = phi[x0:x1, y0:y1, z0:z1]
    if sub.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    rho = (sub * sub).astype(np.float64, copy=False)
    m = float(np.sum(rho, dtype=np.float64))
    if m <= 0.0 or not np.isfinite(m):
        return 0.0, 0.0, 0.0, 0.0
    n = int(phi.shape[0])
    cx = n // 2
    cy = n // 2
    cz = n // 2
    xs = (np.arange(x0, x1, dtype=np.float64) - float(cx))
    ys = (np.arange(y0, y1, dtype=np.float64) - float(cy))
    zs = (np.arange(z0, z1, dtype=np.float64) - float(cz))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    dx = float(np.sum(X * rho) / m)
    dy = float(np.sum(Y * rho) / m)
    dz = float(np.sum(Z * rho) / m)
    r = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    return dx, dy, dz, r

# --- Helper: RMS velocity in a region ---

def _vel_rms(vel: np.ndarray, *, bounds: tuple[int, int, int, int, int, int]) -> float:
    x0, x1, y0, y1, z0, z1 = bounds
    sub = vel[x0:x1, y0:y1, z0:z1]
    if sub.size == 0:
        return 0.0
    vv = (sub * sub).astype(np.float64, copy=False)
    m2 = float(np.mean(vv))
    if not np.isfinite(m2) or m2 < 0.0:
        return 0.0
    return float(np.sqrt(m2))


# --- Helper: max Laplacian magnitude in a region ---
def _lap_abs_max(phi: np.ndarray, *, bounds: tuple[int, int, int, int, int, int]) -> float:
    # Discrete 6-neighbour Laplacian magnitude max within the *interior* of bounds.
    # No wrap. Returns 0.0 if the region is too small.
    x0, x1, y0, y1, z0, z1 = bounds
    if (x1 - x0) < 3 or (y1 - y0) < 3 or (z1 - z0) < 3:
        return 0.0
    c = phi[x0 + 1:x1 - 1, y0 + 1:y1 - 1, z0 + 1:z1 - 1]
    lap = (
        phi[x0 + 2:x1,       y0 + 1:y1 - 1, z0 + 1:z1 - 1]
        + phi[x0:x1 - 2,     y0 + 1:y1 - 1, z0 + 1:z1 - 1]
        + phi[x0 + 1:x1 - 1, y0 + 2:y1,     z0 + 1:z1 - 1]
        + phi[x0 + 1:x1 - 1, y0:y1 - 2,     z0 + 1:z1 - 1]
        + phi[x0 + 1:x1 - 1, y0 + 1:y1 - 1, z0 + 2:z1]
        + phi[x0 + 1:x1 - 1, y0 + 1:y1 - 1, z0:z1 - 2]
        - (6.0 * c)
    )
    m = float(np.nanmax(np.abs(lap)))
    if not np.isfinite(m):
        return 0.0
    return m


def _gyration_radius(phi: np.ndarray, crop_r: int | None = None) -> float:
    # R_g^2 = Σ(r^2 * rho) / Σ(rho), where rho = phi^2.
    # We optionally crop to a cube around the centre to keep n=512 fast.
    n = int(phi.shape[0])
    cx = n // 2
    cy = n // 2
    cz = n // 2

    if crop_r is None:
        x0 = 0
        x1 = n
        y0 = 0
        y1 = n
        z0 = 0
        z1 = n
    else:
        r = int(max(1, crop_r))
        x0 = max(0, cx - r)
        x1 = min(n, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(n, cy + r + 1)
        z0 = max(0, cz - r)
        z1 = min(n, cz + r + 1)

    sub = phi[x0:x1, y0:y1, z0:z1]
    rho = sub * sub
    m = float(np.sum(rho))
    if m <= 0.0 or not np.isfinite(m):
        return 0.0

    xs = np.arange(x0, x1, dtype=np.float32) - np.float32(cx)
    ys = np.arange(y0, y1, dtype=np.float32) - np.float32(cy)
    zs = np.arange(z0, z1, dtype=np.float32) - np.float32(cz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    r2 = X * X + Y * Y + Z * Z

    moment = float(np.sum(r2 * rho))
    if moment < 0.0 or not np.isfinite(moment):
        return 0.0
    return float(np.sqrt(moment / m))


def _gyration_radius_dev(phi: np.ndarray, *, phi0: float, crop_r: int | None = None) -> float:
    # R_g computed on rho_dev = (phi - phi0)^2, so VEV-initialised runs remain interpretable.
    # This avoids the "Rg_ratio≈1" artefact when the crop is dominated by a non-zero vacuum baseline.
    n = int(phi.shape[0])
    cx = n // 2
    cy = n // 2
    cz = n // 2

    if crop_r is None:
        x0 = 0
        x1 = n
        y0 = 0
        y1 = n
        z0 = 0
        z1 = n
    else:
        r = int(max(1, crop_r))
        x0 = max(0, cx - r)
        x1 = min(n, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(n, cy + r + 1)
        z0 = max(0, cz - r)
        z1 = min(n, cz + r + 1)

    sub = phi[x0:x1, y0:y1, z0:z1]
    if sub.size == 0:
        return 0.0

    base = np.float32(phi0)
    dev = sub - base
    rho = dev * dev
    m = float(np.sum(rho))
    if m <= 0.0 or not np.isfinite(m):
        return 0.0

    xs = np.arange(x0, x1, dtype=np.float32) - np.float32(cx)
    ys = np.arange(y0, y1, dtype=np.float32) - np.float32(cy)
    zs = np.arange(z0, z1, dtype=np.float32) - np.float32(cz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    r2 = X * X + Y * Y + Z * Z

    moment = float(np.sum(r2 * rho))
    if moment < 0.0 or not np.isfinite(moment):
        return 0.0
    return float(np.sqrt(moment / m))


def _inject_gaussian_src(src: np.ndarray, sigma: float, amp: float) -> None:
    # src is an acceleration impulse field. Inject a centred 3D Gaussian at t=0.
    if not (np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("soliton: sigma must be > 0")
    if not (np.isfinite(amp) and amp > 0.0):
        raise ValueError("soliton: amp must be > 0")

    n = int(src.shape[0])
    cx = n // 2
    cy = n // 2
    cz = n // 2

    # Windowed injection: compact support to keep it cheap.
    r = int(max(2, int(4.0 * sigma)))
    x0 = max(1, cx - r)
    x1 = min(n - 1, cx + r + 1)
    y0 = max(1, cy - r)
    y1 = min(n - 1, cy + r + 1)
    z0 = max(1, cz - r)
    z1 = min(n - 1, cz + r + 1)

    # Fail-loud: if the window collapses, raise so we don't silently inject nothing.
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        raise ValueError(
            "soliton: injection window collapsed; "
            f"n={n} sigma={float(sigma)} r={r} "
            f"x0={x0} x1={x1} y0={y0} y1={y1} z0={z0} z1={z1}"
        )

    xs = np.arange(x0, x1, dtype=np.float32) - np.float32(cx)
    ys = np.arange(y0, y1, dtype=np.float32) - np.float32(cy)
    zs = np.arange(z0, z1, dtype=np.float32) - np.float32(cz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    d2 = X * X + Y * Y + Z * Z

    s2 = np.float32(sigma * sigma)
    g = np.exp(-0.5 * d2 / s2)
    g = g.astype(np.float32, copy=False)
    src[x0:x1, y0:y1, z0:z1] += np.float32(amp) * g

    # Fail-loud if the injected impulse is all zeros (prevents “E_land is 0” confusion downstream).
    if not np.isfinite(src[x0:x1, y0:y1, z0:z1]).all():
        raise ValueError("soliton: non-finite values produced during src injection")
    if float(np.max(np.abs(src[x0:x1, y0:y1, z0:z1]))) == 0.0:
        raise ValueError(
            "soliton: src injection produced all zeros; "
            f"amp={float(amp)} sigma={float(sigma)} s2={float(s2)}"
        )


def _with_knobs(traffic: Any, k: float, lam: float) -> Any:
    if not np.isfinite(k):
        raise ValueError("soliton: k must be finite")
    if not np.isfinite(lam):
        raise ValueError("soliton: lambda must be finite")
    # k may be negative for symmetry-breaking (double-well) scans.
    if lam < 0.0:
        raise ValueError("soliton: lambda must be >= 0")

    if not dataclasses.is_dataclass(traffic):
        raise ValueError("soliton: traffic params must be a dataclass (TrafficParams)")

    if isinstance(traffic, type):
        raise ValueError("soliton: traffic must be an instance of TrafficParams, not the class")

    # We treat these as vacuum properties (scaffold knobs), not particle constants.
    return dataclasses.replace(cast(Any, traffic), traffic_k=float(k), traffic_lambda=float(lam))


def run_soliton_scan(
    *,
    n: int,
    steps: int,
    sigma: float,
    sigma_start: float | None = None,
    sigma_stop: float | None = None,
    sigma_steps: int | None = None,
    amp: float,
    k: float,
    lambda_start: float,
    lambda_stop: float | None,
    lambda_steps: int,
    sg_k_start: float | None = None,
    sg_k_stop: float | None = None,
    sg_k_steps: int | None = None,
    sg_amp_start: float | None = None,
    sg_amp_stop: float | None = None,
    sg_amp_steps: int | None = None,
    init_vev: bool = False,
    vev_sign: int = 1,
    init_phi: np.ndarray | None = None,
    init_vel: np.ndarray | None = None,
    init_info: dict[str, Any] | None = None,
    soliton_cfg: Any | None = None,
    out_csv: str,
    dump_hdf5_path: str = "",
    dump_sprite: bool = False,
    dump_final_hdf5: bool = True,
    traffic: Any,
    provenance_header: str = "",
    progress_cb: Callable[[int], None] | None = None,
    progress: Callable[[int], None] | None = None,
    evolve_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, Any, int], tuple[np.ndarray, np.ndarray]] = evolve_nonlinear_traffic_steps,
) -> str:
    if int(n) <= 0:
        raise ValueError("soliton: n must be > 0")
    if int(steps) < 10:
        raise ValueError("soliton: steps must be >= 10")

    if int(lambda_steps) < 1:
        raise ValueError("soliton: lambda_steps must be >= 1")
    if not np.isfinite(lambda_start):
        raise ValueError("soliton: lambda_start must be finite")

    # Allow single-point runs: if stop is missing or equals start, force one sample.
    if lambda_stop is None:
        lambda_stop = float(lambda_start)

    if not np.isfinite(lambda_stop):
        raise ValueError("soliton: lambda_stop must be finite")
    if float(lambda_stop) < float(lambda_start):
        raise ValueError("soliton: lambda_stop must be >= lambda_start")

    # Determine mode once; SG uses dedicated sweeps.
    base_mode = str(getattr(traffic, "mode", ""))

    if base_mode == "sine_gordon":
        # Airtight: SG ignores phi^4 interaction; require users to use sg_* sweeps.
        if not (float(lambda_start) == 0.0 and float(lambda_stop) == 0.0 and int(lambda_steps) == 1):
            raise ValueError(
                "soliton: traffic.mode=sine_gordon does not use soliton-lambda-*; "
                "set --soliton-lambda-start 0 --soliton-lambda-stop 0 --soliton-lambda-steps 1 and use --soliton-sg-k-* / --soliton-sg-amp-*"
            )

        # Defaults: if sg_* not provided, treat (k, amp) as single points.
        k0 = float(k) if sg_k_start is None else float(sg_k_start)
        k1 = float(k0) if sg_k_stop is None else float(sg_k_stop)
        ks = 1 if sg_k_steps is None else int(sg_k_steps)

        a0 = float(amp) if sg_amp_start is None else float(sg_amp_start)
        a1 = float(a0) if sg_amp_stop is None else float(sg_amp_stop)
        asteps = 1 if sg_amp_steps is None else int(sg_amp_steps)

        if not np.isfinite(k0) or not np.isfinite(k1):
            raise ValueError("soliton: sg_k_start/sg_k_stop must be finite")
        if ks < 1:
            raise ValueError("soliton: sg_k_steps must be >= 1")
        if float(k1) < float(k0):
            raise ValueError("soliton: sg_k_stop must be >= sg_k_start")
        if float(k1) == float(k0):
            ks = 1
        elif ks == 1:
            raise ValueError("soliton: sg_k_steps=1 requires sg_k_stop == sg_k_start")

        if not np.isfinite(a0) or not np.isfinite(a1):
            raise ValueError("soliton: sg_amp_start/sg_amp_stop must be finite")
        if asteps < 1:
            raise ValueError("soliton: sg_amp_steps must be >= 1")
        if float(a1) < float(a0):
            raise ValueError("soliton: sg_amp_stop must be >= sg_amp_start")
        if float(a1) == float(a0):
            asteps = 1
        elif asteps == 1:
            raise ValueError("soliton: sg_amp_steps=1 requires sg_amp_stop == sg_amp_start")

        # Build SG sweeps.
        if ks == 1:
            sg_ks = np.array([k0], dtype=np.float64)
        else:
            if k0 > 0.0 and k1 > 0.0:
                sg_ks = np.geomspace(k0, k1, ks).astype(np.float64)
            else:
                sg_ks = np.linspace(k0, k1, ks, dtype=np.float64)

        if asteps == 1:
            sg_amps = np.array([a0], dtype=np.float64)
        else:
            if a0 > 0.0 and a1 > 0.0:
                sg_amps = np.geomspace(a0, a1, asteps).astype(np.float64)
            else:
                sg_amps = np.linspace(a0, a1, asteps, dtype=np.float64)

        # Placeholder for non-SG sweep array (unused in SG mode).
        lambdas = np.array([0.0], dtype=np.float64)
    else:
        # Non-SG: preserve existing lambda sweep behaviour.
        if float(lambda_stop) == float(lambda_start):
            lambda_steps = 1
        elif int(lambda_steps) == 1:
            raise ValueError("soliton: lambda_steps=1 requires lambda_stop == lambda_start")

        # Sweep includes endpoints.
        if int(lambda_steps) == 1:
            lambdas = np.array([float(lambda_start)], dtype=np.float64)
        else:
            a0 = float(lambda_start)
            a1 = float(lambda_stop)
            if a0 > 0.0 and a1 > 0.0:
                lambdas = np.geomspace(a0, a1, int(lambda_steps)).astype(np.float64)
            else:
                lambdas = np.linspace(a0, a1, int(lambda_steps), dtype=np.float64)

        sg_ks = np.array([float(k)], dtype=np.float64)
        sg_amps = np.array([float(amp)], dtype=np.float64)

    if int(vev_sign) not in (-1, 1):
        raise ValueError("soliton: vev_sign must be +1 or -1")

    cb = progress_cb if progress_cb is not None else progress

    out_csv = str(out_csv)
    ensure_parent_dir(out_csv)

    dump_hdf5_path = str(dump_hdf5_path or "").strip()
    if dump_hdf5_path != "":
        ensure_parent_dir(dump_hdf5_path)
    pi = float(np.pi)
    eps = 1e-12
    # Terminal aggregation window (SG only): summarise stability over the last W steps.
    # We keep this fixed so 08E remains a cheap mapper, but gains a survivability signal.
    tail_win_default = 512
    # Best-tail viability gate (SG only).
    # These thresholds are intentionally strict: besttail is meant to represent a compact,
    # low-motion, low-drift candidate suitable for sprite extraction.
    # They can be overridden via `params.soliton.sg_besttail_*` when provided by the caller.
    # Localisation gate is evaluated on the besttail snapshot (not the final frame) and uses the full lattice radius.
    # We only want to reject boundary-hits; translation away from the origin is fine for sprite extraction.
    SG_BESTTAIL_MAX_VEL_RMS_MEAN = 0.20
    SG_BESTTAIL_MAX_DE_SHIFTED_PER_STEP_ABS = 1.0e-2
    SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_2PI = 0.01
    SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_PI = 0.20
    SG_BESTTAIL_MIN_PEAK_DEV = 0.20
    SG_BESTTAIL_MAX_CM_R_FRAC_HALF = 0.85
    SG_BESTTAIL_MAX_PEAK_R_FRAC_HALF = 0.90

    def _sg_besttail_override(name: str, default: float) -> float:
        # Accept either a dataclass-like object (preferred: params.soliton) or a dict.
        if soliton_cfg is None:
            return float(default)
        v = None
        if isinstance(soliton_cfg, dict):
            v = soliton_cfg.get(name)
        else:
            v = getattr(soliton_cfg, name, None)
        if v is None:
            return float(default)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(fv):
            return float(default)
        return float(fv)

    if base_mode == "sine_gordon" and soliton_cfg is not None:
        # Names match cli.py wiring: `params.soliton.sg_besttail_*`.
        # Note: localisation overrides are now expressed as fractions of half-grid radius:
        #   sg_besttail_cm_r_frac_half_max, sg_besttail_peak_r_frac_half_max
        SG_BESTTAIL_MAX_VEL_RMS_MEAN = _sg_besttail_override("sg_besttail_vel_rms_mean_max", SG_BESTTAIL_MAX_VEL_RMS_MEAN)
        SG_BESTTAIL_MAX_DE_SHIFTED_PER_STEP_ABS = _sg_besttail_override("sg_besttail_dE_shifted_per_step_abs_max", SG_BESTTAIL_MAX_DE_SHIFTED_PER_STEP_ABS)
        SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_PI = _sg_besttail_override("sg_besttail_frac_abs_phi_gt_pi_max", SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_PI)
        SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_2PI = _sg_besttail_override("sg_besttail_frac_abs_phi_gt_2pi_max", SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_2PI)
        SG_BESTTAIL_MIN_PEAK_DEV = _sg_besttail_override("sg_besttail_peak_dev_min", SG_BESTTAIL_MIN_PEAK_DEV)
        SG_BESTTAIL_MAX_CM_R_FRAC_HALF = _sg_besttail_override("sg_besttail_cm_r_frac_half_max", SG_BESTTAIL_MAX_CM_R_FRAC_HALF)
        SG_BESTTAIL_MAX_PEAK_R_FRAC_HALF = _sg_besttail_override("sg_besttail_peak_r_frac_half_max", SG_BESTTAIL_MAX_PEAK_R_FRAC_HALF)

    # If we're dumping HDF5 or --dump-sprite is enabled, also capture the best (most stable) SG tail snapshot in-memory.
    # This avoids the “end-of-run only” snapshot problem when the most stable moment occurs earlier.
    want_best_tail = (base_mode == "sine_gordon") and bool(dump_sprite)
    want_final_h5 = (str(dump_hdf5_path or "").strip() != "") and bool(dump_final_hdf5)


    # Sigma sweep (optional). If not provided, default to the single `sigma` value.
    ss0 = float(sigma) if sigma_start is None else float(sigma_start)
    ss1 = float(ss0) if sigma_stop is None else float(sigma_stop)
    ssn = 1 if sigma_steps is None else int(sigma_steps)
    if not (np.isfinite(ss0) and np.isfinite(ss1)):
        raise ValueError("soliton: sigma sweep bounds must be finite")
    if ss0 <= 0.0 or ss1 <= 0.0:
        raise ValueError("soliton: sigma sweep bounds must be > 0")
    if ss1 < ss0:
        raise ValueError("soliton: sigma_stop must be >= sigma_start")
    if ss1 == ss0:
        ssn = 1
    else:
        if ssn < 2:
            raise ValueError("soliton: sigma_steps must be >= 2 when sigma_start != sigma_stop")

    sigma_vals = [float(ss0)] if ssn == 1 else [float(x) for x in np.linspace(ss0, ss1, ssn)]

    # Crop radii are computed per-sigma inside the sweep loop.

    t0 = time.perf_counter()

    with open(out_csv, "w", newline="") as f:
        if provenance_header.strip() != "":
            f.write(str(provenance_header))
            if not str(provenance_header).endswith("\n"):
                f.write("\n")

        w = csv.writer(f)
        cols = [
            "idx",
            "lambda",
            "k",
            "sigma",
            "amp",
            "sg_k",
            "sg_amp",
            "init_vev",
            "vev_sign",
            "phi0",
            "steps",
            "traffic_mode",
            "traffic_boundary",
            "traffic_sponge_width",
            "traffic_sponge_strength",
            "traffic_inject",
            "traffic_decay",
            "traffic_gamma",
            "traffic_dt",
            "traffic_c2",
            "crop_r",
            "src_peak",
            "E_land",
            "E_final",
            "E_ratio",
            "Rg_land",
            "Rg_final",
            "Rg_ratio",
            "Rg_dev_land",
            "Rg_dev_final",
            "Rg_dev_ratio",
            "peak_final",
            "peak_final_crop",
            "peak_dev_land",
            "peak_dev_final",
            "peak_dev_ret",
            "peak_land",
            "peak_land_over_pi",
            "peak_final_over_pi",
            "peak_ret",
            "peak_ret_crop",
            "peak_x_land",
            "peak_y_land",
            "peak_z_land",
            "peak_r_land",
            "peak_x_final",
            "peak_y_final",
            "peak_z_final",
            "peak_r_final",
            "peak_move",
            "cm_dx_land",
            "cm_dy_land",
            "cm_dz_land",
            "cm_r_land",
            "cm_dx_final",
            "cm_dy_final",
            "cm_dz_final",
            "cm_r_final",
            "frac_abs_phi_gt_pi_over2_land",
            "frac_abs_phi_gt_pi_land",
            "frac_abs_phi_gt_2pi_land",
            "frac_abs_phi_gt_pi_over2_final",
            "frac_abs_phi_gt_pi_final",
            "frac_abs_phi_gt_2pi_final",
            "outer_shell_abs_ratio_land",
            "outer_shell_abs_ratio_final",
            "E_land_phys",
            "E_final_phys",
            "E_ratio_phys",
            "E_land_phys_shifted",
            "E_final_phys_shifted",
            "E_ratio_phys_shifted",
            "mean_phi_land",
            "mean_phi_final",
            "mean_abs_phi_land",
            "mean_abs_phi_final",
            "mean_phi_outer_land",
            "mean_phi_outer_final",
            "mean_abs_phi_outer_land",
            "mean_abs_phi_outer_final",
            "mean_phi_shell_land",
            "mean_phi_shell_final",
            "mean_abs_phi_shell_land",
            "mean_abs_phi_shell_final",
            "phi_std_shell_land",
            "phi_std_shell_final",
            "delta_outer_shell_land",
            "delta_outer_shell_final",
            "phi_std_outer_land",
            "phi_std_outer_final",
            "phi_p50_outer_land",
            "phi_p50_outer_final",
            "phi_std_land",
            "phi_std_final",
            "phi_p01_land",
            "phi_p01_final",
            "phi_p05_land",
            "phi_p05_final",
            "phi_p16_land",
            "phi_p16_final",
            "phi_p50_land",
            "phi_p50_final",
            "phi_p84_land",
            "phi_p84_final",
            "phi_p95_land",
            "phi_p95_final",
            "phi_p99_land",
            "phi_p99_final",
            "E_land_kin",
            "E_land_grad",
            "E_land_stiff",
            "E_land_phi4",
            "E_land_sine",
            "E_final_kin",
            "E_final_grad",
            "E_final_stiff",
            "E_final_phi4",
            "E_final_sine",
            "vev_theory",
            "mean_phi_final_err",
            "mean_phi_final_abs_err",
            "vel_abs_max_land",
            "vel_abs_max_final",
            "vel_abs_ratio",
            "E_land_kin_frac",
            "E_final_kin_frac",
            "candidate_score",
            "cm_v",
            "peak_v",
            "vel_rms_land",
            "vel_rms_final",
            "vel_rms_ratio",
            "lap_abs_max_land",
            "lap_abs_max_final",
            "lap_abs_ratio",
            "dE_fast",
            "dE_fast_per_step",
            "dE_phys_shifted",
            "dE_phys_shifted_per_step",
            "status",
            "fail_kind",
            "fail_msg",
            "wall_s",
        ]
        if base_mode == "sine_gordon":
            cols += [
                "tail_win",
                "tail_vel_rms_min",
                "tail_vel_rms_mean",
                "tail_vel_rms_max",
                "tail_lap_abs_max",
                "tail_dE_phys_shifted_per_step_mean",
                "tail_dE_phys_shifted_per_step_abs_max",
                "tail_frac_abs_phi_gt_pi_min",
                "tail_frac_abs_phi_gt_pi_max",
                "tail_frac_abs_phi_gt_2pi_min",
                "tail_frac_abs_phi_gt_2pi_max",
                "tail_peak_dev_max",
                "tailp_peak_r_min",
                "tailp_peak_r_mean",
                "tailp_peak_r_max",
                "tailp_vel_rms_mean",
                "tailp_lap_abs_max",
                "tailp_frac_abs_phi_gt_pi_max",
                "tailp_frac_abs_phi_gt_2pi_max",
                "tailp_peak_dev_max",
            ]
            if bool(dump_sprite):
                cols += [
                    "besttail_step",
                    "besttail_score",
                    "besttail_ok",
                    "besttail_dumped",
                    "besttail_h5",
                ]
        w.writerow(cols)

        idx = 0
        for sigma_sweep in sigma_vals:
            # Crop radius for R_g: keep it tied to the injection width.
            # For n=512 this prevents R_g from turning into a full-grid FFT-by-stealth.
            crop_r = int(max(16, int(6.0 * float(sigma_sweep))))
            # Outer probe cube: a larger region to detect “grey goo” (global phase conversion) vs a localised lump.
            crop_r_outer = int(min(int(n // 2 - 2), int(max(crop_r + 8, int(2.0 * float(crop_r))))))
            # Core patch radius for SG sprite harvesting (turning-point quietness). Keep it small to avoid radiation.
            crop_r_core = int(max(8, int(crop_r_outer // 6)))

            for k_sweep in sg_ks.tolist():
                for amp_sweep in sg_amps.tolist():
                    for lam in lambdas.tolist():
                        i = idx
                        idx += 1

                        # Sweep wiring:
                        # - phi^4 / KG scaffold: keep k fixed and sweep lambda.
                        # - sine_gordon: lambda is ignored by the kernel, so interpret the sweep axis as k.
                        if base_mode == "sine_gordon":
                            k_eff = float(k_sweep)
                            lam_eff = 0.0
                            amp_eff = float(amp_sweep)
                        else:
                            k_eff = float(k)
                            lam_eff = float(lam)
                            amp_eff = float(amp)

                        tp = _with_knobs(traffic, k_eff, lam_eff)
                        traffic_mode = str(getattr(tp, "mode", ""))
                        traffic_boundary = str(getattr(tp, "boundary_mode", ""))
                        traffic_sponge_width = int(getattr(tp, "sponge_width", 0))
                        traffic_sponge_strength = float(getattr(tp, "sponge_strength", 0.0))
                        traffic_inject = float(getattr(tp, "inject", 0.0))
                        traffic_decay = float(getattr(tp, "decay", 0.0))
                        traffic_gamma = float(getattr(tp, "gamma", 0.0))
                        traffic_dt = float(getattr(tp, "dt", 0.0))
                        traffic_c2 = float(getattr(tp, "c2", 0.0))

                    # Compute theory VEV and initial field baseline.
                    # For sine_gordon, lam_eff is forced to 0.0 and VEV remains 0.0.
                    if float(lam_eff) > 0.0 and float(k_eff) < 0.0:
                        vev_theory = float(np.sqrt(abs(float(k_eff)) / float(lam_eff)))
                    else:
                        vev_theory = 0.0

                    phi0 = 0.0
                    if bool(init_vev) and vev_theory > 0.0:
                        phi0 = float(int(vev_sign)) * float(vev_theory)

                    # Field buffers.
                    # If an external IC is supplied (e.g. SG "big bang"), use it verbatim and do NOT
                    # inject the Gaussian impulse. This keeps soliton scans usable as a pure-evolution probe.
                    external_ic = (init_phi is not None) or (init_vel is not None)
                    if external_ic and (init_phi is None or init_vel is None):
                        raise ValueError("soliton: init_phi and init_vel must be provided together")

                    if external_ic:
                        if not isinstance(init_phi, np.ndarray) or not isinstance(init_vel, np.ndarray):
                            raise ValueError("soliton: init_phi/init_vel must be numpy arrays")
                        if init_phi.shape != (n, n, n) or init_vel.shape != (n, n, n):
                            raise ValueError(
                                "soliton: init_phi/init_vel shape mismatch; "
                                f"expected {(n, n, n)} got phi={getattr(init_phi, 'shape', None)} vel={getattr(init_vel, 'shape', None)}"
                            )
                        phi = init_phi.astype(np.float32, copy=True)
                        vel = init_vel.astype(np.float32, copy=True)
                        src = np.zeros((n, n, n), dtype=np.float32)
                        src_peak = 0.0
                        stepped = 0
                        # If an IC is supplied, ignore VEV fill (the IC is authoritative).
                    else:
                        phi = np.zeros((n, n, n), dtype=np.float32)
                        vel = np.zeros((n, n, n), dtype=np.float32)
                        src = np.zeros((n, n, n), dtype=np.float32)
                        if phi0 != 0.0:
                            phi.fill(np.float32(phi0))

                        _inject_gaussian_src(src, float(sigma_sweep), float(amp_eff))

                        # Fail-loud early if injection didn't land anything.
                        src_peak_pre = float(np.max(np.abs(src)))
                        if src_peak_pre == 0.0 or not np.isfinite(src_peak_pre):
                            raise ValueError(
                                "soliton: src is empty or non-finite immediately after injection; "
                                f"amp={float(amp_eff)} sigma={float(sigma_sweep)} peak={src_peak_pre:.3e}"
                            )

                        # Land the impulse (step 1), then go passive.
                        src_peak = float(src_peak_pre)
                        phi, vel = evolve_fn(phi, vel, src, tp, 1)
                        src.fill(0.0)
                        if cb is not None:
                            cb(1)
                        stepped = 1

                    E_land = _total_energy(phi, vel)
                    Rg_land = _gyration_radius(phi, crop_r=crop_r)
                    Rg_dev_land = _gyration_radius_dev(phi, phi0=float(phi0), crop_r=crop_r)
                    land_phys = _energy_terms(phi, vel, tp, crop_r=crop_r)
                    E_land_phys = float(land_phys["E_total"])
                    mean_phi_land = float(land_phys.get("mean_phi", 0.0))
                    mean_abs_phi_land = float(land_phys.get("mean_abs_phi", 0.0))
                    phi_std_land = float(land_phys.get("phi_std", 0.0))
                    phi_p01_land = float(land_phys.get("phi_p01", 0.0))
                    phi_p05_land = float(land_phys.get("phi_p05", 0.0))
                    phi_p16_land = float(land_phys.get("phi_p16", 0.0))
                    phi_p50_land = float(land_phys.get("phi_p50", 0.0))
                    phi_p84_land = float(land_phys.get("phi_p84", 0.0))
                    phi_p95_land = float(land_phys.get("phi_p95", 0.0))
                    phi_p99_land = float(land_phys.get("phi_p99", 0.0))
                    E_land_phys_shifted = float(land_phys.get("E_total_shifted", E_land_phys))
                    # --- Outer cube diagnostics for land step ---
                    outer_bounds = _cube_bounds(int(n), crop_r=int(crop_r_outer), scale=1.0)
                    outer_land = _phi_summary(phi, bounds=outer_bounds)
                    mean_phi_outer_land = float(outer_land["mean_phi"])
                    mean_abs_phi_outer_land = float(outer_land["mean_abs_phi"])
                    phi_std_outer_land = float(outer_land["phi_std"])
                    phi_p50_outer_land = float(outer_land["phi_p50"])

                    # --- Shell diagnostics (outside the outer cube) to detect global phase conversion vs local lumps ---
                    tot_sum = float(np.sum(phi, dtype=np.float64))
                    tot_abs = float(np.sum(np.abs(phi), dtype=np.float64))
                    tot_sum2 = float(np.sum((phi * phi).astype(np.float64, copy=False)))
                    tot_count = float(phi.size)

                    outer_m = _phi_moments(phi, bounds=outer_bounds)
                    outer_count = float(outer_m["count"])
                    shell_count = float(max(0.0, tot_count - outer_count))
                    shell_sum = float(tot_sum - float(outer_m["sum"]))
                    shell_abs = float(tot_abs - float(outer_m["sum_abs"]))
                    shell_sum2 = float(tot_sum2 - float(outer_m["sum2"]))

                    if shell_count > 0.0:
                        mean_phi_shell_land = float(shell_sum / shell_count)
                        mean_abs_phi_shell_land = float(shell_abs / shell_count)
                        m2_shell = float(shell_sum2 / shell_count)
                        var_shell = float(max(0.0, m2_shell - mean_phi_shell_land * mean_phi_shell_land))
                        phi_std_shell_land = float(np.sqrt(var_shell))
                    else:
                        mean_phi_shell_land = 0.0
                        mean_abs_phi_shell_land = 0.0
                        phi_std_shell_land = 0.0

                    delta_outer_shell_land = float(mean_phi_outer_land - mean_phi_shell_land)
                    # --- Sine-Gordon / wrap diagnostics (use the same inner crop cube as R_g / energy terms) ---
                    crop_bounds = _cube_bounds(int(n), crop_r=int(crop_r), scale=1.0)
                    cm_dx_land, cm_dy_land, cm_dz_land, cm_r_land = _rho_centroid(phi, bounds=crop_bounds)
                    x0c, x1c, y0c, y1c, z0c, z1c = crop_bounds
                    subc = phi[x0c:x1c, y0c:y1c, z0c:z1c]
                    subc_v = vel[x0c:x1c, y0c:y1c, z0c:z1c]
                    vel_abs_max_land = float(np.max(np.abs(subc_v))) if subc_v.size != 0 else 0.0
                    peak_land = float(np.max(np.abs(subc))) if subc.size != 0 else 0.0
                    peak_dev_land = float(np.max(np.abs(subc - np.float32(phi0)))) if subc.size != 0 else 0.0
                    peak_x_land, peak_y_land, peak_z_land, peak_r_land = _peak_loc_abs(phi)
                    peak_land_over_pi = float(peak_land / pi) if pi > 0.0 else 0.0
                    if subc.size != 0:
                        abs_subc = np.abs(subc)
                        frac_abs_phi_gt_pi_over2_land = float(np.mean(abs_subc > (0.5 * pi)))
                        frac_abs_phi_gt_pi_land = float(np.mean(abs_subc > pi))
                        frac_abs_phi_gt_2pi_land = float(np.mean(abs_subc > (2.0 * pi)))
                    else:
                        frac_abs_phi_gt_pi_over2_land = 0.0
                        frac_abs_phi_gt_pi_land = 0.0
                        frac_abs_phi_gt_2pi_land = 0.0

                    outer_shell_abs_ratio_land = float(mean_abs_phi_outer_land / (mean_abs_phi_shell_land + eps))
                    # --- New: velocity RMS in land step (inner crop cube) ---
                    vel_rms_land = _vel_rms(vel, bounds=crop_bounds)
                    lap_abs_max_land = _lap_abs_max(phi, bounds=crop_bounds)
                    if E_land <= 0.0 or not np.isfinite(E_land):
                        inj = float(getattr(tp, "inject", float("nan")))
                        raise ValueError(
                            "soliton: E_land is 0 after landing step; "
                            f"amp={float(amp_eff)} sigma={float(sigma_sweep)} max|src|={src_peak:.3e} traffic.inject={inj:.3e} "
                            "(check --soliton-amp/--soliton-sigma and --traffic-inject)"
                        )

                    # For multi-point scans we prefer to log and continue on numerical blow-ups.
                    # For single-point runs we keep fail-loud behaviour.
                    is_multi_point = bool((len(sigma_vals) * len(sg_ks) * len(sg_amps) * len(lambdas)) > 1)

                    status = "ok"
                    fail_kind = ""
                    fail_msg = ""

                    # Initialise these outside the evolve try/except so the failure handler never
                    # trips an UnboundLocalError if we blow up immediately.
                    # `stepped` is set above: 1 for Gaussian-land scans, 0 for external IC scans.
                    # (We keep it as an int for progress accounting.)
                    last_good_step = int(stepped)
                    last_good_pmax = float(np.max(np.abs(phi)))
                    last_good_vmax = float(np.max(np.abs(vel)))

                    # SG-only terminal window aggregation.
                    tail_win = 0
                    tail_vel_rms_min = float("nan")
                    tail_vel_rms_mean = float("nan")
                    tail_vel_rms_max = float("nan")
                    tail_lap_abs_max = float("nan")
                    tail_dE_phys_shifted_per_step_mean = float("nan")
                    tail_dE_phys_shifted_per_step_abs_max = float("nan")
                    tail_frac_abs_phi_gt_pi_min = float("nan")
                    tail_frac_abs_phi_gt_pi_max = float("nan")
                    tail_frac_abs_phi_gt_2pi_min = float("nan")
                    tail_frac_abs_phi_gt_2pi_max = float("nan")
                    tail_peak_dev_max = float("nan")
                    # Sprite/besttail outcome variables (always defined for row emission)
                    besttail_step_out = int(stepped)
                    besttail_score_out = float("nan")
                    besttail_ok_out = 0
                    besttail_dumped_out = 0
                    besttail_h5_out = ""
                    tailp_peak_r_min = float("nan")
                    tailp_peak_r_mean = float("nan")
                    tailp_peak_r_max = float("nan")
                    tailp_vel_rms_mean = float("nan")
                    tailp_lap_abs_max = float("nan")
                    tailp_frac_abs_phi_gt_pi_max = float("nan")
                    tailp_frac_abs_phi_gt_2pi_max = float("nan")
                    tailp_peak_dev_max = float("nan")

                    if base_mode == "sine_gordon":
                        tail_win = int(min(int(tail_win_default), max(0, int(steps) - 1)))
                        # Accumulators
                        _tail_count = 0
                        _tail_vel_rms_sum = 0.0
                        _tail_vel_rms_min = float("inf")
                        _tail_vel_rms_max = 0.0
                        _tail_lap_abs_max = 0.0
                        _tail_frac_pi_min = float("inf")
                        _tail_frac_pi_max = 0.0
                        _tail_frac_2pi_min = float("inf")
                        _tail_frac_2pi_max = 0.0
                        _tail_peak_dev_max = 0.0
                        _tail_prev_E = None
                        _tail_dE_sum = 0.0
                        _tail_dE_abs_max = 0.0

                        # Best-tail snapshot capture (SG only). Score is lower-is-better.
                        best_tail_score = float("inf")
                        best_tail_phi: np.ndarray | None = None
                        best_tail_vel: np.ndarray | None = None
                        best_tail_src: np.ndarray | None = None
                        best_tail_vel_abs_max = float("nan")
                        best_tail_step = int(stepped)
                        _tailp_count = 0
                        _tailp_peak_r_sum = 0.0
                        _tailp_peak_r_min = float("inf")
                        _tailp_peak_r_max = 0.0
                        _tailp_vel_rms_sum = 0.0
                        _tailp_lap_abs_max = 0.0
                        _tailp_frac_pi_max = 0.0
                        _tailp_frac_2pi_max = 0.0
                        _tailp_peak_dev_max = 0.0

                    try:
                        # Run in chunks so we can feed the progress bar without doing per-tick Python.

                        remaining = int(steps) - int(stepped)
                        if remaining < 0:
                            remaining = 0
                        chunk = 32
                        while remaining > 0:
                            # For SG sprite hunting, avoid aliasing the breather heartbeat.
                            # In the final tail window, integrate with k_step=1 so we can reliably
                            # sample a turning-point (heartbeat-quiet) frame.
                            in_tail = bool(base_mode == "sine_gordon" and tail_win > 0 and int(stepped) >= (int(steps) - int(tail_win)))
                            k_step = 1 if in_tail else int(chunk if remaining >= chunk else remaining)
                            try:
                                phi, vel = evolve_fn(phi, vel, src, tp, k_step)
                            except ValueError as e:
                                # Fail-loud with useful context. Note: `phi`/`vel` may already contain NaNs,
                                # so we keep last-known-finite magnitudes as well.
                                pmax_now = float(np.nanmax(np.abs(phi)))
                                vmax_now = float(np.nanmax(np.abs(vel)))
                                raise ValueError(
                                    "soliton: non-finite detected during evolve; "
                                    f"idx={int(i)} lambda={float(lam_eff):.6e} k={float(k_eff):.6e} "
                                    f"t={int(stepped)}..{int(stepped + k_step)} of {int(steps)} "
                                    f"peak|phi|={pmax_now:.6e} peak|vel|={vmax_now:.6e} "
                                    f"last_ok_t={int(last_good_step)} last_ok_peak|phi|={float(last_good_pmax):.6e} last_ok_peak|vel|={float(last_good_vmax):.6e} "
                                    f"dt={float(traffic_dt):.3e} c2={float(traffic_c2):.3e} gamma={float(traffic_gamma):.3e} "
                                    f"decay={float(traffic_decay):.3e} inject={float(traffic_inject):.3e} "
                                    "(reduce --soliton-lambda-stop/--soliton-k, or reduce --traffic-dt / --traffic-c2)"
                                ) from e
                            # SG-only tail aggregation update block
                            if base_mode == "sine_gordon" and tail_win > 0:
                                # If we're in the terminal window, accumulate chunk-level stats.
                                # We compute on the inner crop cube for consistency and speed.
                                if int(stepped + k_step) > (int(steps) - int(tail_win)):
                                    vr = _vel_rms(vel, bounds=crop_bounds)
                                    if np.isfinite(vr):
                                        _tail_vel_rms_sum += float(vr)
                                        _tail_vel_rms_min = float(min(_tail_vel_rms_min, float(vr)))
                                        _tail_vel_rms_max = float(max(_tail_vel_rms_max, float(vr)))
                                    la = _lap_abs_max(phi, bounds=crop_bounds)
                                    if np.isfinite(la):
                                        _tail_lap_abs_max = float(max(_tail_lap_abs_max, float(la)))

                                    sub_tail = phi[x0c:x1c, y0c:y1c, z0c:z1c]
                                    if sub_tail.size != 0:
                                        abs_sub = np.abs(sub_tail)
                                        f_pi = float(np.mean(abs_sub > pi))
                                        f_2pi = float(np.mean(abs_sub > (2.0 * pi)))
                                        _tail_frac_pi_min = float(min(_tail_frac_pi_min, f_pi))
                                        _tail_frac_pi_max = float(max(_tail_frac_pi_max, f_pi))
                                        _tail_frac_2pi_min = float(min(_tail_frac_2pi_min, f_2pi))
                                        _tail_frac_2pi_max = float(max(_tail_frac_2pi_max, f_2pi))
                                        # Deviation peak is the SG-relevant amplitude signal.
                                        pkd = float(np.max(np.abs(sub_tail - np.float32(phi0))))
                                        _tail_peak_dev_max = float(max(_tail_peak_dev_max, pkd))

                                    # Peak-centred (moving window) tail diagnostics: robust to translation/drift.
                                    px_t, py_t, pz_t, pr_t = _peak_loc_abs(phi)
                                    _tailp_peak_r_sum += float(pr_t)
                                    _tailp_peak_r_min = float(min(_tailp_peak_r_min, float(pr_t)))
                                    _tailp_peak_r_max = float(max(_tailp_peak_r_max, float(pr_t)))

                                    pb = _cube_bounds_about(int(n), cx=int(px_t), cy=int(py_t), cz=int(pz_t), crop_r=int(crop_r_outer))
                                    vrp = _vel_rms(vel, bounds=pb)
                                    if np.isfinite(vrp):
                                        _tailp_vel_rms_sum += float(vrp)

                                    lap_p = _lap_abs_max(phi, bounds=pb)
                                    if np.isfinite(lap_p):
                                        _tailp_lap_abs_max = float(max(_tailp_lap_abs_max, float(lap_p)))

                                    x0p, x1p, y0p, y1p, z0p, z1p = pb
                                    # Outer peak-centred patch (diagnostics / localisation)
                                    subvp = vel[x0p:x1p, y0p:y1p, z0p:z1p]
                                    subp = phi[x0p:x1p, y0p:y1p, z0p:z1p]

                                    # Core peak-centred patch (harvest stability / heartbeat quietness)
                                    pb_core = _cube_bounds_about(int(n), cx=int(px_t), cy=int(py_t), cz=int(pz_t), crop_r=int(crop_r_core))
                                    x0c2, x1c2, y0c2, y1c2, z0c2, z1c2 = pb_core
                                    subvc = vel[x0c2:x1c2, y0c2:y1c2, z0c2:z1c2]
                                    subc2 = phi[x0c2:x1c2, y0c2:y1c2, z0c2:z1c2]

                                    # Core max |vel| is the key “turning-point” indicator.
                                    vabs_core = float("nan")
                                    if subvc.size != 0:
                                        vabs_core = float(np.max(np.abs(subvc)))
                                    vrc = _vel_rms(vel, bounds=pb_core)
                                    lap_c = _lap_abs_max(phi, bounds=pb_core)

                                    if subp.size != 0:
                                        abs_subp = np.abs(subp)
                                        _tailp_frac_pi_max = float(max(_tailp_frac_pi_max, float(np.mean(abs_subp > pi))))
                                        _tailp_frac_2pi_max = float(max(_tailp_frac_2pi_max, float(np.mean(abs_subp > (2.0 * pi)))))
                                        pkd_p = float(np.max(np.abs(subp - np.float32(phi0))))
                                        _tailp_peak_dev_max = float(max(_tailp_peak_dev_max, pkd_p))

                                    _tailp_count += 1

                                    # Energy drift proxy per step (fixed inner crop cube).
                                    # Important: this must NOT use a moving peak-centred window, otherwise the
                                    # window motion itself looks like “energy drift” and explodes the gate.
                                    # We compute a compact SG energy on the same fixed inner crop used for the
                                    # other tail_* metrics so the drift measures true loss/leakage.
                                    E_inner = float("nan")
                                    subv_tail = vel[x0c:x1c, y0c:y1c, z0c:z1c]
                                    if sub_tail.size != 0 and subv_tail.size != 0:
                                        # Kinetic
                                        E_kin_i = 0.5 * float(np.sum((subv_tail * subv_tail).astype(np.float64, copy=False)))
                                        # Gradient (forward diffs, interior only)
                                        E_grad_i = 0.0
                                        if sub_tail.shape[0] >= 2 and sub_tail.shape[1] >= 2 and sub_tail.shape[2] >= 2:
                                            dx_i = sub_tail[1:, :, :] - sub_tail[:-1, :, :]
                                            dy_i = sub_tail[:, 1:, :] - sub_tail[:, :-1, :]
                                            dz_i = sub_tail[:, :, 1:] - sub_tail[:, :, :-1]
                                            E_grad_i = 0.5 * float(traffic_c2) * float(
                                                np.sum((dx_i * dx_i).astype(np.float64, copy=False))
                                                + np.sum((dy_i * dy_i).astype(np.float64, copy=False))
                                                + np.sum((dz_i * dz_i).astype(np.float64, copy=False))
                                            )
                                        # SG potential
                                        E_pot_i = 0.0
                                        if float(k_eff) != 0.0:
                                            E_pot_i = float(k_eff) * float(np.sum((1.0 - np.cos(sub_tail)).astype(np.float64, copy=False)))
                                        E_inner = float(E_kin_i + E_grad_i + E_pot_i)

                                    if np.isfinite(E_inner):
                                        if _tail_prev_E is not None:
                                            dE_per = float((E_inner - float(_tail_prev_E)) / float(k_step))
                                            _tail_dE_sum += float(dE_per)
                                            _tail_dE_abs_max = float(max(_tail_dE_abs_max, abs(dE_per)))
                                        _tail_prev_E = float(E_inner)

                                    _tail_count += 1

                                    # Capture the best (most stable) tail snapshot in-memory.
                                    # Robust SG besttail score: primarily prefer heartbeat-quiet frames.
                                    # A breather can have low RMS but large instantaneous |vel| spikes; sprites must be harvested
                                    # near a turning point (potential max) so translational kicks actually bite.
                                    # Therefore we weight patch |vel|_max heavily, then RMS/curvature, then drift.
                                    if want_best_tail:
                                        dE_for_score = 0.0
                                        if _tail_prev_E is not None and _tail_count >= 1:
                                            # If we have a drift estimate, reuse the last computed per-step drift magnitude.
                                            dE_for_score = float(_tail_dE_abs_max)
                                        # Score must target the breather core turning-point: use core patch |vel|_max.
                                        vr_s = float(vrc) if ("vrc" in locals() and np.isfinite(vrc)) else float(vr)
                                        la_s = float(lap_c) if ("lap_c" in locals() and np.isfinite(lap_c)) else float(la)
                                        vabs_s = float(vabs_core) if ("vabs_core" in locals() and np.isfinite(vabs_core)) else float("nan")
                                        if not np.isfinite(vabs_s):
                                            vabs_s = 1.0e9
                                        score = float(vabs_s) + (0.25 * float(vr_s)) + (0.02 * float(la_s)) + (50.0 * float(dE_for_score))
                                        if np.isfinite(score) and score < best_tail_score:
                                            best_tail_score = float(score)
                                            best_tail_step = int(stepped + k_step)
                                            # Copy full fields (overhead is acceptable; used for sprite extraction).
                                            best_tail_phi = phi.astype(np.float32, copy=True)
                                            best_tail_vel = vel.astype(np.float32, copy=True)
                                            best_tail_src = src.astype(np.float32, copy=True)
                                            if "vabs_core" in locals() and np.isfinite(vabs_core):
                                                best_tail_vel_abs_max = float(vabs_core)
                                            else:
                                                best_tail_vel_abs_max = float("nan")
                                            # Save for CSV output if needed
                                            besttail_step_out = int(best_tail_step)
                                            besttail_score_out = float(best_tail_score)
                            last_good_step = int(stepped + k_step)
                            last_good_pmax = float(np.max(np.abs(phi)))
                            last_good_vmax = float(np.max(np.abs(vel)))
                            stepped += k_step
                            remaining -= k_step
                            if cb is not None:
                                cb(k_step)

                        E_final = _total_energy(phi, vel)
                        Rg_final = _gyration_radius(phi, crop_r=crop_r)
                        Rg_dev_final = _gyration_radius_dev(phi, phi0=float(phi0), crop_r=crop_r)
                        peak_final = float(np.max(np.abs(phi)))
                        # Peak within the same inner crop cube used for R_g / energy terms.
                        # This avoids boundary/sponge artefacts polluting late-time peak metrics.
                        subc_f = phi[x0c:x1c, y0c:y1c, z0c:z1c]
                        peak_final_crop = float(np.max(np.abs(subc_f))) if subc_f.size != 0 else 0.0
                        peak_dev_final = float(np.max(np.abs(subc_f - np.float32(phi0)))) if subc_f.size != 0 else 0.0
                        peak_x_final, peak_y_final, peak_z_final, peak_r_final = _peak_loc_abs(phi)
                        pdx = float(int(peak_x_final) - int(peak_x_land))
                        pdy = float(int(peak_y_final) - int(peak_y_land))
                        pdz = float(int(peak_z_final) - int(peak_z_land))
                        peak_move = float(np.sqrt(pdx * pdx + pdy * pdy + pdz * pdz))

                        final_phys = _energy_terms(phi, vel, tp, crop_r=crop_r)
                        E_final_phys = float(final_phys["E_total"])
                        mean_phi_final = float(final_phys.get("mean_phi", 0.0))
                        mean_abs_phi_final = float(final_phys.get("mean_abs_phi", 0.0))
                        phi_std_final = float(final_phys.get("phi_std", 0.0))
                        phi_p01_final = float(final_phys.get("phi_p01", 0.0))
                        phi_p05_final = float(final_phys.get("phi_p05", 0.0))
                        phi_p16_final = float(final_phys.get("phi_p16", 0.0))
                        phi_p50_final = float(final_phys.get("phi_p50", 0.0))
                        phi_p84_final = float(final_phys.get("phi_p84", 0.0))
                        phi_p95_final = float(final_phys.get("phi_p95", 0.0))
                        phi_p99_final = float(final_phys.get("phi_p99", 0.0))
                        E_final_phys_shifted = float(final_phys.get("E_total_shifted", E_final_phys))
                        E_ratio = float(E_final / E_land) if E_land > 0.0 else 0.0
                        Rg_ratio = float(Rg_final / Rg_land) if Rg_land > 0.0 else 0.0
                        peak_ret = float(peak_final / peak_land) if peak_land > 0.0 else 0.0
                        peak_ret_crop = float(peak_final_crop / peak_land) if peak_land > 0.0 else 0.0
                        Rg_dev_ratio = float(Rg_dev_final / Rg_dev_land) if Rg_dev_land > 0.0 else 0.0
                        peak_dev_ret = float(peak_dev_final / peak_dev_land) if peak_dev_land > 0.0 else 0.0
                        E_ratio_phys = float(E_final_phys / E_land_phys) if E_land_phys > 0.0 else 0.0
                        E_ratio_phys_shifted = (
                            float(E_final_phys_shifted / E_land_phys_shifted) if E_land_phys_shifted > 0.0 else 0.0
                        )
                        # --- Outer cube diagnostics for final step ---
                        outer_final = _phi_summary(phi, bounds=outer_bounds)
                        mean_phi_outer_final = float(outer_final["mean_phi"])
                        mean_abs_phi_outer_final = float(outer_final["mean_abs_phi"])
                        phi_std_outer_final = float(outer_final["phi_std"])
                        phi_p50_outer_final = float(outer_final["phi_p50"])

                        tot_sum_f = float(np.sum(phi, dtype=np.float64))
                        tot_abs_f = float(np.sum(np.abs(phi), dtype=np.float64))
                        tot_sum2_f = float(np.sum((phi * phi).astype(np.float64, copy=False)))
                        tot_count_f = float(phi.size)

                        outer_m_f = _phi_moments(phi, bounds=outer_bounds)
                        outer_count_f = float(outer_m_f["count"])
                        shell_count_f = float(max(0.0, tot_count_f - outer_count_f))
                        shell_sum_f = float(tot_sum_f - float(outer_m_f["sum"]))
                        shell_abs_f = float(tot_abs_f - float(outer_m_f["sum_abs"]))
                        shell_sum2_f = float(tot_sum2_f - float(outer_m_f["sum2"]))

                        if shell_count_f > 0.0:
                            mean_phi_shell_final = float(shell_sum_f / shell_count_f)
                            mean_abs_phi_shell_final = float(shell_abs_f / shell_count_f)
                            m2_shell_f = float(shell_sum2_f / shell_count_f)
                            var_shell_f = float(max(0.0, m2_shell_f - mean_phi_shell_final * mean_phi_shell_final))
                            phi_std_shell_final = float(np.sqrt(var_shell_f))
                        else:
                            mean_phi_shell_final = 0.0
                            mean_abs_phi_shell_final = 0.0
                            phi_std_shell_final = 0.0

                        delta_outer_shell_final = float(mean_phi_outer_final - mean_phi_shell_final)
                        # --- Sine-Gordon / wrap diagnostics (inner crop cube) ---
                        cm_dx_final, cm_dy_final, cm_dz_final, cm_r_final = _rho_centroid(phi, bounds=crop_bounds)
                        subc_v_f = vel[x0c:x1c, y0c:y1c, z0c:z1c]
                        vel_abs_max_final = float(np.max(np.abs(subc_v_f))) if subc_v_f.size != 0 else 0.0
                        vel_abs_ratio = float(vel_abs_max_final / vel_abs_max_land) if vel_abs_max_land > 0.0 else 0.0
                        E_land_kin = float(land_phys.get("E_kin", 0.0))
                        E_final_kin = float(final_phys.get("E_kin", 0.0))
                        E_land_kin_frac = float(E_land_kin / E_land_phys) if E_land_phys > 0.0 else 0.0
                        E_final_kin_frac = float(E_final_kin / E_final_phys) if E_final_phys > 0.0 else 0.0
                        # Candidate ranking (higher is better): survival * peak retention,
                        # penalised by remaining kinetic stress and global phase drift.
                        candidate_score = float(E_ratio_phys_shifted * peak_ret)
                        candidate_score /= float(1.0 + vel_abs_max_final)
                        candidate_score /= float(1.0 + abs(delta_outer_shell_final))
                        candidate_score /= float(1.0 + abs(mean_phi_shell_final))
                        if not np.isfinite(candidate_score):
                            candidate_score = 0.0
                        peak_final_over_pi = float(peak_final / pi) if pi > 0.0 else 0.0
                        if subc_f.size != 0:
                            abs_subc_f = np.abs(subc_f)
                            frac_abs_phi_gt_pi_over2_final = float(np.mean(abs_subc_f > (0.5 * pi)))
                            frac_abs_phi_gt_pi_final = float(np.mean(abs_subc_f > pi))
                            frac_abs_phi_gt_2pi_final = float(np.mean(abs_subc_f > (2.0 * pi)))
                        else:
                            frac_abs_phi_gt_pi_over2_final = 0.0
                            frac_abs_phi_gt_pi_final = 0.0
                            frac_abs_phi_gt_2pi_final = 0.0

                        outer_shell_abs_ratio_final = float(mean_abs_phi_outer_final / (mean_abs_phi_shell_final + eps))
                        # --- New: velocity RMS in final step (inner crop cube) and ratio ---
                        vel_rms_final = _vel_rms(vel, bounds=crop_bounds)
                        vel_rms_ratio = float(vel_rms_final / vel_rms_land) if vel_rms_land > 0.0 else 0.0
                        lap_abs_max_final = _lap_abs_max(phi, bounds=crop_bounds)
                        lap_abs_ratio = float(lap_abs_max_final / lap_abs_max_land) if lap_abs_max_land > 0.0 else 0.0

                        # mean_phi_final_err and mean_phi_final_abs_err use vev_theory computed above
                        mean_phi_final_err = float(mean_phi_final - vev_theory)
                        mean_phi_final_abs_err = float(abs(mean_phi_final) - vev_theory)

                        # --- New: drift rates and energy change rates ---
                        den = float(max(1, int(steps) - 1))
                        cm_v = float(cm_r_final - cm_r_land) / den
                        peak_v = float(peak_move) / den

                        dE_fast = float(E_final - E_land)
                        dE_fast_per_step = float(dE_fast / den)

                        dE_phys_shifted = float(E_final_phys_shifted - E_land_phys_shifted)
                        dE_phys_shifted_per_step = float(dE_phys_shifted / den)

                        # --- SG terminal window aggregation finalisation ---
                        if base_mode == "sine_gordon" and tail_win > 0 and _tail_count > 0:
                            tail_vel_rms_min = float(_tail_vel_rms_min) if _tail_vel_rms_min != float("inf") else float("nan")
                            tail_vel_rms_mean = float(_tail_vel_rms_sum / float(_tail_count))
                            tail_vel_rms_max = float(_tail_vel_rms_max)
                            tail_lap_abs_max = float(_tail_lap_abs_max)
                            # dE stats only become valid after the second sample.
                            if _tail_count >= 2:
                                tail_dE_phys_shifted_per_step_mean = float(_tail_dE_sum / float(_tail_count - 1))
                                tail_dE_phys_shifted_per_step_abs_max = float(_tail_dE_abs_max)
                            else:
                                tail_dE_phys_shifted_per_step_mean = float("nan")
                                tail_dE_phys_shifted_per_step_abs_max = float("nan")
                            tail_frac_abs_phi_gt_pi_min = float(_tail_frac_pi_min) if _tail_frac_pi_min != float("inf") else float("nan")
                            tail_frac_abs_phi_gt_pi_max = float(_tail_frac_pi_max)
                            tail_frac_abs_phi_gt_2pi_min = float(_tail_frac_2pi_min) if _tail_frac_2pi_min != float("inf") else float("nan")
                            tail_frac_abs_phi_gt_2pi_max = float(_tail_frac_2pi_max)
                            tail_peak_dev_max = float(_tail_peak_dev_max)
                            if _tailp_count > 0:
                                tailp_peak_r_min = float(_tailp_peak_r_min) if _tailp_peak_r_min != float("inf") else float("nan")
                                tailp_peak_r_mean = float(_tailp_peak_r_sum / float(_tailp_count))
                                tailp_peak_r_max = float(_tailp_peak_r_max)
                                tailp_vel_rms_mean = float(_tailp_vel_rms_sum / float(_tailp_count))
                                tailp_lap_abs_max = float(_tailp_lap_abs_max)
                                tailp_frac_abs_phi_gt_pi_max = float(_tailp_frac_pi_max)
                                tailp_frac_abs_phi_gt_2pi_max = float(_tailp_frac_2pi_max)
                                tailp_peak_dev_max = float(_tailp_peak_dev_max)
                            else:
                                tailp_peak_r_min = float("nan")
                                tailp_peak_r_mean = float("nan")
                                tailp_peak_r_max = float("nan")
                                tailp_vel_rms_mean = float("nan")
                                tailp_lap_abs_max = float("nan")
                                tailp_frac_abs_phi_gt_pi_max = float("nan")
                                tailp_frac_abs_phi_gt_2pi_max = float("nan")
                                tailp_peak_dev_max = float("nan")

                    except ValueError as e:
                        status = "fail"
                        fail_kind = "nonfinite"
                        fail_msg = str(e)

                        # Keep progress accounting sane: consume remaining integration work for this point.
                        stepped = int(last_good_step)
                        rem = int(steps) - int(stepped)
                        if rem > 0 and cb is not None:
                            cb(int(rem))

                        if not is_multi_point:
                            raise

                        # Fill final metrics with NaNs for failed points.
                        if base_mode == "sine_gordon":
                            tail_vel_rms_min = float("nan")
                            tail_vel_rms_mean = float("nan")
                            tail_vel_rms_max = float("nan")
                            tail_lap_abs_max = float("nan")
                            tail_dE_phys_shifted_per_step_mean = float("nan")
                            tail_dE_phys_shifted_per_step_abs_max = float("nan")
                            tail_frac_abs_phi_gt_pi_min = float("nan")
                            tail_frac_abs_phi_gt_pi_max = float("nan")
                            tail_frac_abs_phi_gt_2pi_min = float("nan")
                            tail_frac_abs_phi_gt_2pi_max = float("nan")
                            tail_peak_dev_max = float("nan")
                            tailp_peak_r_min = float("nan")
                            tailp_peak_r_mean = float("nan")
                            tailp_peak_r_max = float("nan")
                            tailp_vel_rms_mean = float("nan")
                            tailp_lap_abs_max = float("nan")
                            tailp_frac_abs_phi_gt_pi_max = float("nan")
                            tailp_frac_abs_phi_gt_2pi_max = float("nan")
                            tailp_peak_dev_max = float("nan")
                        # Sprite/besttail outcome variables (always defined for row emission)
                        besttail_step_out = int(stepped)
                        besttail_score_out = float("nan")
                        besttail_ok_out = 0
                        besttail_dumped_out = 0
                        besttail_h5_out = ""
                        E_final = float("nan")
                        Rg_final = float("nan")
                        peak_final = float("nan")
                        peak_final_crop = float("nan")
                        peak_x_final = 0
                        peak_y_final = 0
                        peak_z_final = 0
                        peak_r_final = float("nan")
                        peak_move = float("nan")

                        E_final_phys = float("nan")
                        mean_phi_final = float("nan")
                        mean_abs_phi_final = float("nan")
                        phi_std_final = float("nan")
                        phi_p01_final = float("nan")
                        phi_p05_final = float("nan")
                        phi_p16_final = float("nan")
                        phi_p50_final = float("nan")
                        phi_p84_final = float("nan")
                        phi_p95_final = float("nan")
                        phi_p99_final = float("nan")
                        E_final_phys_shifted = float("nan")

                        E_ratio = float("nan")
                        Rg_ratio = float("nan")
                        Rg_dev_final = float("nan")
                        Rg_dev_ratio = float("nan")
                        peak_ret = float("nan")
                        peak_ret_crop = float("nan")
                        peak_dev_final = float("nan")
                        peak_dev_ret = float("nan")
                        E_ratio_phys = float("nan")
                        E_ratio_phys_shifted = float("nan")

                        mean_phi_outer_final = float("nan")
                        mean_abs_phi_outer_final = float("nan")
                        phi_std_outer_final = float("nan")
                        phi_p50_outer_final = float("nan")
                        mean_phi_shell_final = float("nan")
                        mean_abs_phi_shell_final = float("nan")
                        phi_std_shell_final = float("nan")
                        delta_outer_shell_final = float("nan")

                        cm_dx_final = float("nan")
                        cm_dy_final = float("nan")
                        cm_dz_final = float("nan")
                        cm_r_final = float("nan")

                        vel_abs_max_final = float("nan")
                        vel_abs_ratio = float("nan")
                        E_final_kin = float("nan")
                        E_final_kin_frac = float("nan")
                        candidate_score = float("nan")

                        peak_final_over_pi = float("nan")
                        frac_abs_phi_gt_pi_over2_final = float("nan")
                        frac_abs_phi_gt_pi_final = float("nan")
                        frac_abs_phi_gt_2pi_final = float("nan")

                        outer_shell_abs_ratio_final = float("nan")
                        vel_rms_final = float("nan")
                        vel_rms_ratio = float("nan")
                        lap_abs_max_final = float("nan")
                        lap_abs_ratio = float("nan")

                        mean_phi_final_err = float("nan")
                        mean_phi_final_abs_err = float("nan")

                        cm_v = float("nan")
                        peak_v = float("nan")

                        dE_fast = float("nan")
                        dE_fast_per_step = float("nan")
                        dE_phys_shifted = float("nan")
                        dE_phys_shifted_per_step = float("nan")

                    # --- Optional exports (done before row emission so CSV is a true ledger) ---

                    # 1) Final full-grid snapshot (only when explicitly requested).
                    if want_final_h5 and status == "ok":
                        base_h5 = str(dump_hdf5_path or "").strip()
                        if base_h5 == "":
                            raise ValueError("soliton: dump_final_hdf5 requested but dump_hdf5_path is empty")
                        ensure_parent_dir(base_h5)
                        out_h5 = base_h5
                        if idx > 1:
                            root, ext = os.path.splitext(base_h5)
                            ext2 = ext if ext != "" else ".h5"
                            out_h5 = f"{root}_idx{int(i):04d}{ext2}"
                        load = np.zeros_like(phi)
                        dump_pipeline_state_h5(
                            path=str(out_h5),
                            phi=phi,
                            vel=vel,
                            src=src,
                            load=load,
                            params=tp,
                            step=int(stepped),
                            dt=float(getattr(tp, "dt", 0.0)),
                        )

                    # 2) Sprite asset (SG only) extracted directly from the besttail fields in memory.
                    #    This avoids writing the large *_besttail.h5 full-grid snapshot.
                    if want_best_tail and status == "ok" and best_tail_phi is not None and best_tail_vel is not None and tail_win > 0:
                        # Viability gate (tune later): low tail motion, low drift, limited wrap, non-trivial amplitude, localised.
                        vr_gate = float(tailp_vel_rms_mean) if np.isfinite(tailp_vel_rms_mean) else float(tail_vel_rms_mean)
                        ok_vr = (np.isfinite(vr_gate) and float(vr_gate) <= float(SG_BESTTAIL_MAX_VEL_RMS_MEAN))
                        vabs_gate = float(best_tail_vel_abs_max) if ("best_tail_vel_abs_max" in locals() and np.isfinite(best_tail_vel_abs_max)) else float("nan")
                        if not np.isfinite(vabs_gate):
                            _pxb, _pyb, _pzb, _ = _peak_loc_abs(best_tail_phi)
                            pb_bt = _cube_bounds_about(int(n), cx=int(_pxb), cy=int(_pyb), cz=int(_pzb), crop_r=int(crop_r_outer))
                            x0b, x1b, y0b, y1b, z0b, z1b = pb_bt
                            subvb = best_tail_vel[x0b:x1b, y0b:y1b, z0b:z1b]
                            if subvb.size != 0:
                                vabs_gate = float(np.max(np.abs(subvb)))
                        ok_dE = True
                        wrap2_gate = float(tailp_frac_abs_phi_gt_2pi_max) if np.isfinite(tailp_frac_abs_phi_gt_2pi_max) else float(tail_frac_abs_phi_gt_2pi_max)
                        ok_wrap_2pi = (np.isfinite(wrap2_gate) and float(wrap2_gate) <= float(SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_2PI))
                        wrappi_gate = float(tailp_frac_abs_phi_gt_pi_max) if np.isfinite(tailp_frac_abs_phi_gt_pi_max) else float(tail_frac_abs_phi_gt_pi_max)
                        ok_wrap_pi = (np.isfinite(wrappi_gate) and float(wrappi_gate) <= float(SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_PI))
                        peak_dev_bt = float(np.max(np.abs(best_tail_phi - np.float32(phi0))))
                        amp_gate = float(peak_dev_bt)
                        if np.isfinite(tailp_peak_dev_max):
                            amp_gate = float(max(amp_gate, float(tailp_peak_dev_max)))
                        if np.isfinite(tail_peak_dev_max):
                            amp_gate = float(max(amp_gate, float(tail_peak_dev_max)))
                        ok_amp = (np.isfinite(amp_gate) and float(amp_gate) >= float(SG_BESTTAIL_MIN_PEAK_DEV))
                        cm_dx_bt, cm_dy_bt, cm_dz_bt, cm_r_bt = _rho_centroid(
                            best_tail_phi,
                            bounds=(0, int(n), 0, int(n), 0, int(n)),
                        )
                        _px_bt, _py_bt, _pz_bt, peak_r_bt = _peak_loc_abs(best_tail_phi)
                        half_r = float(max(1, (int(n) // 2) - 2))
                        ok_loc = (
                            np.isfinite(cm_r_bt)
                            and np.isfinite(peak_r_bt)
                            and float(cm_r_bt) <= float(SG_BESTTAIL_MAX_CM_R_FRAC_HALF) * float(half_r)
                            and float(peak_r_bt) <= float(SG_BESTTAIL_MAX_PEAK_R_FRAC_HALF) * float(half_r)
                        )
                        if ok_vr and ok_wrap_2pi and ok_wrap_pi and ok_amp and ok_loc:
                            besttail_ok_out = 1
                            besttail_step_out = int(best_tail_step)
                            besttail_score_out = float(best_tail_score) if np.isfinite(best_tail_score) else float("nan")

                            # Sprite output folder: sibling of the CSV folder, named `_sprites`.
                            csv_dir = os.path.dirname(str(out_csv))
                            parent = os.path.dirname(csv_dir)
                            sprite_dir = os.path.join(parent, "_sprites") if os.path.basename(csv_dir) == "_csv" else os.path.join(csv_dir, "_sprites")
                            os.makedirs(sprite_dir, exist_ok=True)

                            # Unique stem per sweep point.
                            stem0 = os.path.splitext(os.path.basename(str(out_csv)))[0]
                            stem = f"{stem0}_idx{int(i):04d}" if idx > 1 else stem0

                            src_bt = best_tail_src if best_tail_src is not None else np.zeros_like(best_tail_phi)
                            load_bt = np.zeros_like(best_tail_phi)

                            meta0 = {
                                "sg.mode": str(base_mode),
                                "sg.k": float(getattr(tp, "traffic_k", 0.0)),
                                "sg.c2": float(getattr(tp, "c2", 0.0)),
                                "sg.dt": float(getattr(tp, "dt", 0.0)),
                                "sg.gamma": float(getattr(tp, "gamma", 0.0)),
                                "sg.decay": float(getattr(tp, "decay", 0.0)),
                                "boundary.mode": str(getattr(tp, "boundary_mode", "")),
                                "boundary.sponge_width": float(getattr(tp, "sponge_width", 0.0)),
                                "boundary.sponge_strength": float(getattr(tp, "sponge_strength", 0.0)),
                                "source.csv": str(out_csv),
                                "source.idx": float(int(i)),
                                "source.step": float(int(best_tail_step)),
                            }
                            out_sprite_h5, _stats = extract_sprite_from_fields(
                                phi=best_tail_phi,
                                vel=best_tail_vel,
                                src=src_bt,
                                load=load_bt,
                                out_arg=str(sprite_dir),
                                stem=str(stem),
                                step=int(best_tail_step),
                                snapshot_kind="besttail",
                                meta0=meta0,
                                source_snapshot_path="",
                                spec=SpriteExtractSpec(),
                            )

                            besttail_dumped_out = 1
                            besttail_h5_out = str(out_sprite_h5)

                    wall_s = float(time.perf_counter() - t0)

                    row = [
                        int(i),
                        float(lam_eff),
                        float(k_eff),
                        float(sigma_sweep),
                        float(amp_eff),
                        float(k_sweep),
                        float(amp_sweep),
                        int(bool(init_vev)),
                        int(vev_sign),
                        float(phi0),
                        int(steps),
                        str(traffic_mode),
                        str(traffic_boundary),
                        int(traffic_sponge_width),
                        float(traffic_sponge_strength),
                        float(traffic_inject),
                        float(traffic_decay),
                        float(traffic_gamma),
                        float(traffic_dt),
                        float(traffic_c2),
                        int(crop_r),
                        float(src_peak),
                        float(E_land),
                        float(E_final),
                        float(E_ratio),
                        float(Rg_land),
                        float(Rg_final),
                        float(Rg_ratio),
                        float(Rg_dev_land),
                        float(Rg_dev_final),
                        float(Rg_dev_ratio),
                        float(peak_final),
                        float(peak_final_crop),
                        float(peak_dev_land),
                        float(peak_dev_final),
                        float(peak_dev_ret),
                        float(peak_land),
                        float(peak_land_over_pi),
                        float(peak_final_over_pi),
                        float(peak_ret),
                        float(peak_ret_crop),
                        int(peak_x_land),
                        int(peak_y_land),
                        int(peak_z_land),
                        float(peak_r_land),
                        int(peak_x_final),
                        int(peak_y_final),
                        int(peak_z_final),
                        float(peak_r_final),
                        float(peak_move),
                        float(cm_dx_land),
                        float(cm_dy_land),
                        float(cm_dz_land),
                        float(cm_r_land),
                        float(cm_dx_final),
                        float(cm_dy_final),
                        float(cm_dz_final),
                        float(cm_r_final),
                        float(frac_abs_phi_gt_pi_over2_land),
                        float(frac_abs_phi_gt_pi_land),
                        float(frac_abs_phi_gt_2pi_land),
                        float(frac_abs_phi_gt_pi_over2_final),
                        float(frac_abs_phi_gt_pi_final),
                        float(frac_abs_phi_gt_2pi_final),
                        float(outer_shell_abs_ratio_land),
                        float(outer_shell_abs_ratio_final),
                        float(E_land_phys),
                        float(E_final_phys),
                        float(E_ratio_phys),
                        float(E_land_phys_shifted),
                        float(E_final_phys_shifted),
                        float(E_ratio_phys_shifted),
                        float(mean_phi_land),
                        float(mean_phi_final),
                        float(mean_abs_phi_land),
                        float(mean_abs_phi_final),
                        float(mean_phi_outer_land),
                        float(mean_phi_outer_final),
                        float(mean_abs_phi_outer_land),
                        float(mean_abs_phi_outer_final),
                        float(mean_phi_shell_land),
                        float(mean_phi_shell_final),
                        float(mean_abs_phi_shell_land),
                        float(mean_abs_phi_shell_final),
                        float(phi_std_shell_land),
                        float(phi_std_shell_final),
                        float(delta_outer_shell_land),
                        float(delta_outer_shell_final),
                        float(phi_std_outer_land),
                        float(phi_std_outer_final),
                        float(phi_p50_outer_land),
                        float(phi_p50_outer_final),
                        float(phi_std_land),
                        float(phi_std_final),
                        float(phi_p01_land),
                        float(phi_p01_final),
                        float(phi_p05_land),
                        float(phi_p05_final),
                        float(phi_p16_land),
                        float(phi_p16_final),
                        float(phi_p50_land),
                        float(phi_p50_final),
                        float(phi_p84_land),
                        float(phi_p84_final),
                        float(phi_p95_land),
                        float(phi_p95_final),
                        float(phi_p99_land),
                        float(phi_p99_final),
                        float(land_phys["E_kin"]),
                        float(land_phys["E_grad"]),
                        float(land_phys["E_stiff"]),
                        float(land_phys["E_phi4"]),
                        float(land_phys.get("E_sine", 0.0)),
                        float(final_phys["E_kin"]) if status == "ok" else float("nan"),
                        float(final_phys["E_grad"]) if status == "ok" else float("nan"),
                        float(final_phys["E_stiff"]) if status == "ok" else float("nan"),
                        float(final_phys["E_phi4"]) if status == "ok" else float("nan"),
                        float(final_phys.get("E_sine", 0.0)) if status == "ok" else float("nan"),
                        float(vev_theory),
                        float(mean_phi_final_err),
                        float(mean_phi_final_abs_err),
                        float(vel_abs_max_land),
                        float(vel_abs_max_final),
                        float(vel_abs_ratio),
                        float(E_land_kin_frac),
                        float(E_final_kin_frac),
                        float(candidate_score),
                        float(cm_v),
                        float(peak_v),
                        float(vel_rms_land),
                        float(vel_rms_final),
                        float(vel_rms_ratio),
                        float(lap_abs_max_land),
                        float(lap_abs_max_final),
                        float(lap_abs_ratio),
                        float(dE_fast),
                        float(dE_fast_per_step),
                        float(dE_phys_shifted),
                        float(dE_phys_shifted_per_step),
                        str(status),
                        str(fail_kind),
                        str(fail_msg),
                        float(wall_s),
                    ]
                    if base_mode == "sine_gordon":
                        row += [
                            int(tail_win),
                            float(tail_vel_rms_min),
                            float(tail_vel_rms_mean),
                            float(tail_vel_rms_max),
                            float(tail_lap_abs_max),
                            float(tail_dE_phys_shifted_per_step_mean),
                            float(tail_dE_phys_shifted_per_step_abs_max),
                            float(tail_frac_abs_phi_gt_pi_min),
                            float(tail_frac_abs_phi_gt_pi_max),
                            float(tail_frac_abs_phi_gt_2pi_min),
                            float(tail_frac_abs_phi_gt_2pi_max),
                            float(tail_peak_dev_max),
                            float(tailp_peak_r_min),
                            float(tailp_peak_r_mean),
                            float(tailp_peak_r_max),
                            float(tailp_vel_rms_mean),
                            float(tailp_lap_abs_max),
                            float(tailp_frac_abs_phi_gt_pi_max),
                            float(tailp_frac_abs_phi_gt_2pi_max),
                            float(tailp_peak_dev_max),
                        ]
                        if bool(dump_sprite):
                            row += [
                                int(besttail_step_out),
                                float(besttail_score_out),
                                int(besttail_ok_out),
                                int(besttail_dumped_out),
                                str(besttail_h5_out),
                            ]
                    w.writerow(row)
                    f.flush()


    return out_csv
