# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""pipeline.py — micro→index→Shapiro pipeline orchestration

Role
----
This module composes the end-to-end numerical pipeline used by `core.py`:

  microstate/load -> load(x,y,z) -> steady field phi -> index map n(x,y)
  -> ray-tracing regressions (Shapiro analogue calibration)

It contains the heavy numerical pieces extracted from the old monolithic
`core.py`, but remains an orchestration layer (not the lowest-level kernels).

Entry points
------------
- `build_index_from_micro(params, ...)`:
    Builds the 2D index map slice `n_map` and a metrics dict.

- `run_shapiro_mass_lockdown(params, n_map)`:
    Runs a coupling sweep and returns calibration metrics:
      alpha, r2_line, mean_fit_r2, min_fit_r2, per-coupling rows.

- `_ensemble_one(seed_i, base)`:
    Convenience wrapper used by the ensemble mode in `core.py`.

Dependencies (flat module map)
------------------------------
- `lattice.py` / `load.py`   : microstate + micro→load mapping
- `traffic.py`               : load→phi solver (diffuse / telegraph)
- `radial.py`                : radial diagnostics + fit CSVs
- `rays.py`                  : ray tracing + asinh regression

Contracts
---------
- Fail-fast: invalid windows, non-finite fields, and degenerate fits raise.
- No plotting: all visual output lives in `visualiser.py`.
- Flat layout: imported as `from pipeline import build_index_from_micro, ...`.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

from utils import _as_float, _assert_finite, _make_rng

from params import PipelineParams, RayParams

from lattice import lattice_anneal, lattice_init, lattice_init_multiscale
from load import compute_load

from traffic import evolve_traffic

from exporters import dump_pipeline_state_h5
from broadcast import LiveBroadcaster

from radial import (
    dump_radial_csv as _dump_radial_csv,
    dump_radial_fit_csv,
    fit_linear_inv_r,
    fit_powerlaw_inverse,
    radial_profile as _radial_profile,
    slope_scan_powerlaw,
)

from rays import fit_asinh_delay, ray_trace_delay

if TYPE_CHECKING:
    def radial_profile(phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def dump_radial_csv(path: str, r: np.ndarray, mean_phi: np.ndarray, count: np.ndarray, std_phi: np.ndarray, *, provenance: Optional[str] = None) -> None: ...
else:
    radial_profile = _radial_profile
    dump_radial_csv = _dump_radial_csv


# -----------------------------
# Main pipeline
# -----------------------------


def build_index_from_micro(
    params: PipelineParams,
    dump_radial_path: Optional[str] = None,
    dump_radial_fit_path: Optional[str] = None,
    dump_hdf5_path: Optional[str] = None,
    liveview_shm_name: Optional[str] = None,
    progress_cb: Callable[[int], None] | None = None,
    csv_provenance: Optional[str] = None,
    radial_provenance: Optional[str] = None,
    radial_fit_provenance: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    rng = _make_rng(params.seed)

    # Track best 1/r age fit during evolution
    best_inv_r_age = -1
    best_inv_r_err = float("inf")
    best_inv_r_slope = float("nan")
    best_inv_r_r2 = float("nan")

    if params.delta_load:
        n = int(params.lattice.n)
        load = np.zeros((n, n, n), dtype=np.float32)
        c0 = n // 2
        jitter = int(params.delta_jitter)
        margin = int(params.delta_margin)
        if margin < 1:
            raise ValueError("--delta-margin must be >= 1")
        if jitter < 0:
            raise ValueError("--delta-jitter must be >= 0")
        if jitter > 0:
            if (c0 - jitter) < margin or (c0 + jitter) >= (n - margin):
                raise ValueError("--delta-jitter too large for lattice with this --delta-margin")
            dx = int(rng.integers(-jitter, jitter + 1))
            dy = int(rng.integers(-jitter, jitter + 1))
            dz = int(rng.integers(-jitter, jitter + 1))
            x0 = c0 + dx
            y0 = c0 + dy
            z0 = c0 + dz
        else:
            x0 = c0
            y0 = c0
            z0 = c0
        load[x0, y0, z0] = 1.0
        if progress_cb is not None:
            progress_cb(int(params.lattice.steps))
    else:
        init_mode = getattr(params.lattice, "init_mode", "sparse")
        if init_mode == "sparse":
            s = lattice_init(params.lattice, rng)
        elif init_mode == "multiscale":
            s = lattice_init_multiscale(params.lattice, rng)
        else:
            raise ValueError(f"unknown lattice init_mode={init_mode!r}")
        if int(params.lattice.steps) > 0:
            s = lattice_anneal(s, params.lattice, rng, progress_cb=progress_cb)
        else:
            # steps==0 => skip anneal (use init state directly)
            if progress_cb is not None:
                progress_cb(0)
        load = compute_load(s, params.load)

    broadcaster: Optional[LiveBroadcaster] = None
    if bool(getattr(params, "liveview", False)) and int(params.lattice.n) <= 512:
        try:
            # Shared memory is created by core.py; we attach here.
            n_lv = int(params.lattice.n)
            shm_name = str(liveview_shm_name).strip() if liveview_shm_name is not None else ""
            if shm_name == "":
                shm_name = "CAELIX_shm"
            broadcaster = LiveBroadcaster((n_lv, n_lv, n_lv), name=shm_name, create=False, zero_init=False)
        except Exception:
            broadcaster = None

    monitor_refine_until = -1
    monitor_refine_active = False
    monitor_no_improve = 0

    def _monitor_cb(phi_now: np.ndarray, done_steps: int) -> None:
        # Fail-soft: monitoring must not affect the solver.
        nonlocal best_inv_r_age, best_inv_r_err, best_inv_r_slope, best_inv_r_r2, monitor_refine_until, monitor_refine_active, monitor_no_improve
        iters_now = int(done_steps)
        if iters_now <= 0:
            return
        if monitor_refine_until == 0:
            return

        # Adaptive cadence: coarse by default, tighten as we approach |p+1| -> 0.
        base_every = int(getattr(params.traffic, "check_every", 0))
        iters_total = int(getattr(params.traffic, "iters", 0))
        if base_every <= 0:
            # Default cadence: N-aware. Smaller lattices can afford denser sampling and
            # are also more likely to have a narrow optimum in iteration-space.
            n_lat = int(getattr(params.lattice, "n", 0) or 0)
            if n_lat <= 0:
                n_lat = int(params.lattice.n)

            # Target sample count scales ~linearly with 512/N (cap to avoid excess work).
            scale = max(1.0, 512.0 / float(n_lat))
            target_samples = int(round(200.0 * scale))
            target_samples = max(200, min(2000, target_samples))

            if iters_total > 0:
                base_every = max(1, int(iters_total // target_samples))
                # For very short runs, don't let cadence get too coarse.
                if iters_total <= 25000:
                    base_every = min(int(base_every), 25)
            else:
                base_every = 2000

            base_every = min(2000, int(base_every))

        # Tighten thresholds (empirical; tuned for baseline stability).
        ce = int(base_every)
        if math.isfinite(best_inv_r_err):
            if best_inv_r_err < 0.05:
                ce = min(ce, 200)
            if best_inv_r_err < 0.01:
                ce = min(ce, 20)

        # Enter per-step refinement once we're close enough.
        if (not monitor_refine_active) and math.isfinite(best_inv_r_err) and best_inv_r_err < 0.003:
            monitor_refine_active = True
            monitor_no_improve = 0

        if bool(monitor_refine_active):
            ce = 1

        if ce > 1 and (iters_now % ce) != 0:
            if iters_total > 0 and iters_now == iters_total:
                pass
            else:
                return

        try:
            r_now, phi_r_now, _c_now, _s_now = radial_profile(phi_now)

            # Prefer the deoffset fit (phi - B) when possible; it is the better proxy for the true 1/r exponent.
            ff = fit_powerlaw_inverse(phi_r_now, r_now, r_min=float(r_fit_min), r_max=float(r_fit_max))
            p_use = float(ff.slope)
            r2_use = float(ff.r2)

            invr_now = fit_linear_inv_r(phi_r_now, r_now, r_min=float(r_fit_min), r_max=float(r_fit_max))
            b0 = float(invr_now.intercept)

            mask = (r_now >= float(r_fit_min)) & (r_now <= float(r_fit_max)) & (phi_r_now > 0) & (r_now > 0)
            rr = r_now[mask].astype(np.float64)
            yy = phi_r_now[mask].astype(np.float64)
            if rr.size >= 8 and math.isfinite(b0):
                y_de = yy - float(b0)
                m = y_de > 0
                if int(np.count_nonzero(m)) >= 8:
                    x = np.log(rr[m])
                    y = np.log(y_de[m])
                    X = np.vstack([x, np.ones_like(x)]).T
                    p_de, bb = np.linalg.lstsq(X, y, rcond=None)[0]
                    pred = p_de * x + bb
                    ss_res = float(np.sum((y - pred) ** 2))
                    ss_tot = float(np.sum((y - y.mean()) ** 2))
                    r2_de = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                    if math.isfinite(float(p_de)) and math.isfinite(float(r2_de)):
                        p_use = float(p_de)
                        r2_use = float(r2_de)

            if math.isfinite(p_use) and math.isfinite(r2_use):
                err = abs(p_use + 1.0)
                if err < float(best_inv_r_err):
                    best_inv_r_err = float(err)
                    best_inv_r_age = int(iters_now)
                    best_inv_r_slope = float(p_use)
                    best_inv_r_r2 = float(r2_use)
                    if bool(monitor_refine_active):
                        monitor_no_improve = 0
                else:
                    if bool(monitor_refine_active):
                        monitor_no_improve += 1
                        if int(monitor_no_improve) >= 10:
                            # Stop per-step refinement and stop monitoring entirely (we've passed the optimum).
                            monitor_refine_active = False
                            monitor_refine_until = 0
        except Exception:
            pass

    def _live_cb(phi_now: np.ndarray) -> None:
        # Fail-soft: LiveView must not affect the solver.
        if broadcaster is None:
            return
        try:
            broadcaster.update(phi_now)
        except Exception:
            pass

    def _state_cb(phi_now: np.ndarray, done_steps: int) -> None:
        _monitor_cb(phi_now, done_steps)
        _live_cb(phi_now)

    # Move r_fit_min/r_fit_max validation BEFORE evolve_traffic to allow in-run monitoring.
    r_fit_min = float(params.r_fit_min)
    if not math.isfinite(r_fit_min) or r_fit_min <= 0.0:
        raise ValueError("--r-fit-min must be > 0")

    r_fit_max = float(params.r_fit_max)
    if not math.isfinite(r_fit_max) or r_fit_max < 0.0:
        raise ValueError("--r-fit-max must be >= 0")
    if r_fit_max == 0.0:
        r_fit_max = max(6.0, params.lattice.n * 0.35)

    if r_fit_max <= r_fit_min:
        raise ValueError("--r-fit-max must be > --r-fit-min")

    phi = evolve_traffic(load, params.traffic, progress_cb=progress_cb, state_cb=_state_cb)

    # Push a final snapshot and close our handle.
    if broadcaster is not None:
        try:
            broadcaster.update(phi)
        except Exception:
            pass
        try:
            broadcaster.close()
        except Exception:
            pass

    if dump_hdf5_path is not None and str(dump_hdf5_path).strip() != "":
        prov_dict = {
            "seed": int(params.seed),
            "n": int(params.lattice.n),
            "traffic_iters": int(params.traffic.iters),
            "traffic_mode": str(getattr(params.traffic, "mode", "")),
            "k_index": float(params.k_index),
        }
        vel0 = np.zeros_like(phi)
        src0 = np.zeros_like(phi)
        dump_pipeline_state_h5(
            str(dump_hdf5_path),
            phi=phi,
            vel=vel0,
            src=src0,
            load=load,
            params=params,
            provenance=prov_dict,
            step=int(getattr(params.traffic, "iters", 0)),
            dt=float(getattr(params.traffic, "dt", 0.0)),
        )

    r, phi_r, counts, std_r = radial_profile(phi)

    r_fit = fit_powerlaw_inverse(phi_r, r, r_min=r_fit_min, r_max=r_fit_max)
    invr_fit = fit_linear_inv_r(phi_r, r, r_min=r_fit_min, r_max=r_fit_max)
    scan = slope_scan_powerlaw(phi_r, r, r_min=r_fit_min, r_max=r_fit_max)
    mask_used = (r >= r_fit_min) & (r <= r_fit_max) & (phi_r > 0)
    n_fit_pts = int(np.count_nonzero(mask_used))

    rr_used = r[mask_used & (r > 0)].astype(np.float64)
    yy_used = phi_r[mask_used & (r > 0)].astype(np.float64)
    b0 = float(invr_fit.intercept)

    mean_phi_win = float(np.mean(yy_used)) if yy_used.size > 0 else float("nan")
    base_frac = float(np.clip(b0 / mean_phi_win, 0.0, 1.0)) if (
        math.isfinite(b0) and math.isfinite(mean_phi_win) and mean_phi_win > 0
    ) else float("nan")

    de_slope = float("nan")
    de_r2 = float("nan")
    if yy_used.size >= 8:
        y_de = yy_used - b0
        m = y_de > 0
        if int(np.count_nonzero(m)) >= 8:
            x = np.log(rr_used[m])
            y = np.log(y_de[m])
            X = np.vstack([x, np.ones_like(x)]).T
            p, bb = np.linalg.lstsq(X, y, rcond=None)[0]
            pred = p * x + bb
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            de_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            de_slope = float(p)

    if dump_radial_path is not None and str(dump_radial_path).strip() != "":
        prov = radial_provenance if (radial_provenance is not None and str(radial_provenance) != "") else csv_provenance
        dump_radial_csv(str(dump_radial_path), r, phi_r, counts, std_r, provenance=prov)

    if dump_radial_fit_path is not None and str(dump_radial_fit_path).strip() != "":
        rows: List[Dict[str, object]] = []
        rows.append({
            "fit_kind": "powerlaw_loglog",
            "r_min": float(r_fit_min),
            "r_max": float(r_fit_max),
            "n_points": int(n_fit_pts),
            "slope": float(r_fit.slope),
            "intercept": float(r_fit.intercept),
            "r2": float(r_fit.r2),
        })
        rows.append({
            "fit_kind": "powerlaw_scan",
            "r_min": float(r_fit_min),
            "r_max": float(r_fit_max),
            "n_points": int(n_fit_pts),
            "slope": float(scan["median"]),
            "intercept": "",
            "r2": "",
            "scan_used": float(scan["used"]),
            "scan_median": float(scan["median"]),
            "scan_p16": float(scan["p16"]),
            "scan_p84": float(scan["p84"]),
        })
        rows.append({
            "fit_kind": "linear_inv_r",
            "r_min": float(r_fit_min),
            "r_max": float(r_fit_max),
            "n_points": int(n_fit_pts),
            "slope": float(invr_fit.slope),
            "intercept": float(invr_fit.intercept),
            "r2": float(invr_fit.r2),
        })
        rows.append({
            "fit_kind": "base_frac",
            "r_min": float(r_fit_min),
            "r_max": float(r_fit_max),
            "n_points": int(n_fit_pts),
            "slope": "",
            "intercept": "",
            "r2": float(base_frac),
        })
        rows.append({
            "fit_kind": "deoffset_powerlaw",
            "r_min": float(r_fit_min),
            "r_max": float(r_fit_max),
            "n_points": int(n_fit_pts),
            "slope": float(de_slope),
            "intercept": "",
            "r2": float(de_r2),
        })
        # Add tracked best 1/r age fit
        rows.append({
            "fit_kind": "best_inv_r_age",
            "r_min": float(r_fit_min),
            "r_max": float(r_fit_max),
            "n_points": int(n_fit_pts),
            "slope": float(best_inv_r_slope),
            "intercept": "",
            "r2": float(best_inv_r_r2),
            "age_iters": int(best_inv_r_age),
            "err_abs_p_plus_1": float(best_inv_r_err),
        })
        w = (float(r_fit_max) - float(r_fit_min)) * 0.6
        starts = np.linspace(float(r_fit_min), max(float(r_fit_min), float(r_fit_max) - w), 7).astype(np.float64)
        for s in starts:
            a = float(s)
            b = float(s + w)
            try:
                ff = fit_powerlaw_inverse(phi_r, r, r_min=a, r_max=b)
            except ValueError:
                continue
            m = (r >= a) & (r <= b) & (phi_r > 0)
            rows.append({
                "fit_kind": "powerlaw_window",
                "r_min": float(a),
                "r_max": float(b),
                "n_points": int(np.count_nonzero(m)),
                "slope": float(ff.slope),
                "intercept": float(ff.intercept),
                "r2": float(ff.r2),
            })
        prov = radial_fit_provenance if (radial_fit_provenance is not None and str(radial_fit_provenance) != "") else csv_provenance
        dump_radial_fit_csv(str(dump_radial_fit_path), rows, provenance=prov)

    phi_max = float(phi.max())
    if phi_max <= 0:
        raise ValueError("phi is identically zero; load source vanished")

    c = params.lattice.n // 2
    phi2 = (phi[:, :, c] / np.float32(phi_max)).astype(np.float32)

    n_map = np.float32(1.0) + (np.float32(params.k_index) * phi2)
    _assert_finite(n_map, "n_map")

    metrics = {
        "phi_max": phi_max,
        "radial_fit_r_min": float(r_fit_min),
        "radial_fit_r_max": float(r_fit_max),
        "radial_fit_n": float(n_fit_pts),
        "radial_slope": float(r_fit.slope),
        "radial_r2": float(r_fit.r2),
        "radial_inv_r_slope": float(invr_fit.slope),
        "radial_inv_r_intercept": float(invr_fit.intercept),
        "radial_inv_r_r2": float(invr_fit.r2),
        "radial_inv_r_a": float(invr_fit.slope),
        "radial_inv_r_b": float(invr_fit.intercept),
        "radial_base_frac": float(base_frac),
        "radial_deoffset_slope": float(de_slope),
        "radial_deoffset_r2": float(de_r2),
        "radial_slope_scan_used": float(scan["used"]),
        "radial_slope_scan_median": float(scan["median"]),
        "radial_slope_scan_p16": float(scan["p16"]),
        "radial_slope_scan_p84": float(scan["p84"]),
        "best_inv_r_age_iters": float(best_inv_r_age),
        "best_inv_r_slope": float(best_inv_r_slope),
        "best_inv_r_r2": float(best_inv_r_r2),
    }

    if progress_cb is not None:
        progress_cb(250)

    return n_map, metrics


def run_shapiro_mass_lockdown(params: PipelineParams, n_map: np.ndarray) -> Dict[str, object]:
    rp = RayParams(X0=params.X0, ds=params.ds)

    # Coupling sweep (log anchors + mid-scale fill) for a more informative calibration curve.
    couplings = np.array([1, 2, 3, 4, 6, 8, 12, 16], dtype=np.float64)

    H, W = n_map.shape
    half = (min(H, W) / 2.0) - 3.0
    if half <= 4.0:
        raise ValueError("n_map too small for Shapiro sweep; increase --n")

    b_max = float(params.shapiro_b_max)
    if not math.isfinite(b_max) or b_max < 0.0:
        raise ValueError("--shapiro-b-max must be >= 0")
    if b_max == 0.0:
        b_max = float(half)
    if b_max > float(half):
        raise ValueError("--shapiro-b-max exceeds lens half-size")
    if b_max <= 2.0:
        raise ValueError("--shapiro-b-max must be > 2.0")

    b = np.linspace(2.0, b_max, 17).astype(np.float64)

    Ks: List[float] = []
    R2s: List[float] = []
    per: List[Dict[str, float]] = []

    n_delta = n_map - np.float32(1.0)

    for cpl in couplings:
        n_eff = np.float32(1.0) + (np.float32(cpl) * n_delta)
        D = ray_trace_delay(n_eff, b, rp)
        K, _C, r2 = fit_asinh_delay(D, b, params.X0)
        Ks.append(float(K))
        R2s.append(float(r2))
        per.append({"coupling": float(cpl), "K": float(K), "r2_fit": float(r2)})

    Ks_arr = np.array(Ks, dtype=np.float64)

    alpha = float(np.dot(couplings, Ks_arr) / np.dot(couplings, couplings))
    pred = alpha * couplings
    ss_res = float(np.sum((Ks_arr - pred) ** 2))
    ss_tot = float(np.sum((Ks_arr - Ks_arr.mean()) ** 2))
    if ss_tot <= 0.0:
        raise ValueError("r2_line undefined: Ks are degenerate (all equal)")
    r2_line = 1.0 - ss_res / ss_tot

    return {
        "alpha": alpha,
        "r2_line": float(r2_line),
        "mean_fit_r2": float(np.mean(np.array(R2s, dtype=np.float64))),
        "min_fit_r2": float(np.min(np.array(R2s, dtype=np.float64))),
        "shapiro_per_coupling": per,
    }


def _ensemble_one(seed_i: int, base: PipelineParams) -> Dict[str, float]:
    p_i = replace(base, seed=int(seed_i))
    n_map_i, m_i = build_index_from_micro(p_i)
    cal_i = run_shapiro_mass_lockdown(p_i, n_map_i)
    return {
        "seed": float(seed_i),
        "phi_max": float(m_i["phi_max"]),
        "radial_slope": float(m_i["radial_slope"]),
        "radial_r2": float(m_i["radial_r2"]),
        "alpha": float(_as_float(cal_i.get("alpha"), "cal_i.alpha")),
        "r2_line": float(_as_float(cal_i.get("r2_line"), "cal_i.r2_line")),
        "mean_fit_r2": float(_as_float(cal_i.get("mean_fit_r2"), "cal_i.mean_fit_r2")),
        "min_fit_r2": float(_as_float(cal_i.get("min_fit_r2"), "cal_i.min_fit_r2")),
    }