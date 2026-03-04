# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""coulomb.py — signed-source (Coulomb-like) diagnostic on a finite lattice

Role
----
Runs a simple two-charge experiment on the same lattice/traffic machinery used by
the main pipeline, but with *signed* sources. The aim is to measure interaction
energy vs separation under a steady-state diffusion relaxation (Poisson-like
surrogate on a bounded grid).

What it computes
----------------
For each separation `d`, we solve three steady fields:
  - pair field: charges at (x1,c,c) and (x2,c,c) with like/opposite signs
  - self field 1: single +q at x1
  - self field 2: single +q at x2

We report both a "phi-energy" and a "gradient-energy":
  - E_phi   := sum(phi^2)
  - E_grad  := sum(|∇phi|^2) via 6-neighbour edge differences

Interaction energies subtract the two self solves from the pair solve:
  - E_int_phi  := E_pair_phi  - (E_self1_phi  + E_self2_phi)
  - E_int_grad := E_pair_grad - (E_self1_grad + E_self2_grad)

Why the double self solve
-------------------------
On finite domains with absorbing boundaries, self-energy depends on placement.
We compute self-energy at *both* charge positions to reduce boundary artefacts.

Traffic mode constraint
-----------------------
This diagnostic uses diffusion stepping only and is fail-fast if the configured
traffic mode is not `diffuse`.

Contracts
---------
- Requires `params.traffic.mode == "diffuse"`.
- `q` must be finite and non-zero.
- Placement respects `delta_margin` to keep charges away from absorbing faces.
- Output is a CSV written to `--coulomb-out`.

Flat-module layout
------------------
Lives alongside `core.py` and is imported as:
  `from coulomb import run_coulomb_test`

No plotting: visual output (if added later) belongs in `visualiser.py`.
"""

from __future__ import annotations

import math
import os
import csv

from dataclasses import replace
from typing import Dict, List, Tuple, Optional, Callable, cast

import numpy as np
from numba import njit

from params import PipelineParams
from traffic import evolve_diffusion_traffic_steps
from utils import resolve_out_dir


def _grad_energy_6(phi: np.ndarray) -> float:
    """Edge-sum of squared differences over 6-neighbour links (count each undirected edge once)."""
    p = phi.astype(np.float64)
    e = 0.0
    e += float(np.sum((p[1:, :, :] - p[:-1, :, :]) ** 2))
    e += float(np.sum((p[:, 1:, :] - p[:, :-1, :]) ** 2))
    e += float(np.sum((p[:, :, 1:] - p[:, :, :-1]) ** 2))
    return float(e)


@njit(cache=True)
def _nb_phi_energy(phi: np.ndarray) -> float:
    e = 0.0
    n0, n1, n2 = phi.shape
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                v = float(phi[i, j, k])
                e += v * v
    return float(e)


@njit(cache=True)
def _nb_grad_energy_6(phi: np.ndarray) -> float:
    e = 0.0
    n0, n1, n2 = phi.shape
    for i in range(1, n0):
        for j in range(n1):
            for k in range(n2):
                d = float(phi[i, j, k]) - float(phi[i - 1, j, k])
                e += d * d
    for i in range(n0):
        for j in range(1, n1):
            for k in range(n2):
                d = float(phi[i, j, k]) - float(phi[i, j - 1, k])
                e += d * d
    for i in range(n0):
        for j in range(n1):
            for k in range(1, n2):
                d = float(phi[i, j, k]) - float(phi[i, j, k - 1])
                e += d * d
    return float(e)



@njit(cache=True)
def _nb_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    m = 0.0
    n0, n1, n2 = a.shape
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                d = float(a[i, j, k]) - float(b[i, j, k])
                if d < 0.0:
                    d = -d
                if d > m:
                    m = d
    return float(m)


# --- Fit helpers ---
def _r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    if y.size < 2:
        return float("nan")
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((y - yhat) ** 2))
    return float(1.0 - (ss_res / ss_tot))


def _fit_invr(d: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit y ≈ A*(1/d) + B via least squares. Returns (A, B, r2)."""
    d = np.asarray(d, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = 1.0 / d
    X = np.stack([x, np.ones_like(x)], axis=1)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a = float(beta[0])
    b = float(beta[1])
    yhat = (a * x) + b
    r2 = _r2_score(y, yhat)
    return float(a), float(b), float(r2)


def _fit_yukawa_grid(d: np.ndarray, y: np.ndarray, k_min: float, k_max: float, k_steps: int) -> Tuple[float, float, float, float]:
    """Fit y ≈ A*exp(-k d)/d + B by grid-search over k and LS solve for (A,B). Returns (A,B,k,r2)."""
    d = np.asarray(d, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if k_steps < 2:
        k_steps = 2
    ks = np.linspace(float(k_min), float(k_max), int(k_steps), dtype=np.float64)
    best_r2 = -1.0e300
    best_a = float("nan")
    best_b = float("nan")
    best_k = float("nan")

    for k in ks:
        x = np.exp(-k * d) / d
        X = np.stack([x, np.ones_like(x)], axis=1)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0])
        b = float(beta[1])
        yhat = (a * x) + b
        r2 = _r2_score(y, yhat)
        if math.isfinite(r2) and r2 > best_r2:
            best_r2 = float(r2)
            best_a = float(a)
            best_b = float(b)
            best_k = float(k)

    return float(best_a), float(best_b), float(best_k), float(best_r2)


def _solve_self_at(
    phi_self: np.ndarray,
    phi_self_prev: np.ndarray,
    src_self: np.ndarray,
    xp: int,
    c: int,
    q: float,
    tp_fast,
    max_iters_i: int,
    check_i: int,
    tol_i: float,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Tuple[int, float, float, bool, float]:
    src_self[:, :, :] = 0.0
    src_self[xp, c, c] = float(q)
    phi_self[:, :, :] = 0.0
    phi_self_prev[:, :, :] = 0.0
    done = 0
    converged = False
    last_max_delta = float("nan")
    while done < max_iters_i:
        step_chunk = int(min(check_i, max_iters_i - done))
        phi_out = evolve_diffusion_traffic_steps(phi_self, src_self, tp_fast, step_chunk)
        phi_self[:, :, :] = phi_out
        done += step_chunk
        if progress_cb is not None:
            progress_cb(0)
        if tol_i == 0.0:
            continue
        max_delta = float(_nb_max_abs_diff(phi_self, phi_self_prev))
        last_max_delta = float(max_delta)
        if max_delta < tol_i:
            converged = True
            break
        phi_self_prev[:, :, :] = phi_self
    e_phi = float(_nb_phi_energy(phi_self))
    e_grad = float(_nb_grad_energy_6(phi_self))
    return int(done), float(e_phi), float(e_grad), bool(converged), float(last_max_delta)


def run_coulomb_test(
    params: PipelineParams,
    q: float,
    sign_mode: str,
    d_min: int,
    d_max: int,
    d_step: int,
    out_csv: str,
    max_iters: int,
    check_every: int,
    tol: float,
    provenance_header: Optional[str] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> None:
    """Static two-charge test: energy vs separation for like vs opposite signed sources.

    We evolve a signed scalar field under the diffusion traffic update (Poisson-like relaxation).
    Energies (pair + self-subtracted interaction):
      - E_pair_phi   := sum(phi_pair^2)
      - E_pair_grad  := sum(|∇phi_pair|^2) via 6-neighbour edge differences
      - E_self1_phi  := sum(phi_self^2) for a single +q source at (x1,c,c)
      - E_self1_grad := sum(|∇phi_self|^2) for that placement
      - E_self2_phi  := sum(phi_self^2) for a single +q source at (x2,c,c)
      - E_self2_grad := sum(|∇phi_self|^2) for that placement
      - E_self_phi   := E_self1_phi  + E_self2_phi
      - E_self_grad  := E_self1_grad + E_self2_grad
      - E_int_phi    := E_pair_phi  - E_self_phi
      - E_int_grad   := E_pair_grad - E_self_grad

    Notes:
      - We compute self-energy at BOTH charge positions to reduce boundary artefacts on finite domains.
      - Self solves are cold-started per distance to keep each measurement independent.
    """

    if str(params.traffic.mode).strip().lower() != "diffuse":
        raise ValueError("--coulomb requires --traffic-mode diffuse")

    if not math.isfinite(float(q)) or float(q) == 0.0:
        raise ValueError("--coulomb-q must be finite and non-zero")

    sm = str(sign_mode).strip().lower()
    if sm not in ("like", "opposite"):
        raise ValueError("--coulomb-sign must be one of: like, opposite")

    if d_min < 1 or d_max < 1 or d_step < 1:
        raise ValueError("--coulomb-d-* must be >= 1")
    if d_max < d_min:
        raise ValueError("--coulomb-d-max must be >= --coulomb-d-min")

    out_path = str(out_csv).strip()
    if out_path == "":
        raise ValueError("--coulomb-out must be a non-empty path")
    out_path = resolve_out_dir(__file__, out_path)

    n = int(params.lattice.n)
    c = n // 2

    # Keep both charges away from absorbing boundaries.
    margin = int(params.delta_margin)
    safe = int(max(2, margin + 2))
    if (c - safe) < 0 or (c + safe) >= n:
        raise ValueError("coulomb has no legal placement with this --n/--delta-margin")

    # Distance cap: ensure positions always remain within [safe, n-1-safe].
    max_d_pos = int((n - 1 - safe) - safe)
    if d_max > max_d_pos:
        raise ValueError("--coulomb-d-max too large for lattice with this --delta-margin")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cols = [
        "sign",
        "q",
        "d",
        "x1",
        "x2",
        "c",
        "safe",
        "iters_pair",
        "iters_self1",
        "iters_self2",
        "self2_reused",
        "iters_self_logical",
        "iters_self",
        "pair_converged",
        "pair_max_delta",
        "self1_converged",
        "self1_max_delta",
        "self2_converged",
        "self2_max_delta",
        "E_pair_phi",
        "E_pair_grad",
        "E_self1_phi",
        "E_self1_grad",
        "E_self2_phi",
        "E_self2_grad",
        "E_self_phi",
        "E_self_grad",
        "E_int_phi",
        "E_int_grad",
        "E_int_phi_renorm",
        "E_int_grad_renorm",
        "fit_metric",
        "fit_n",
        "fit_d_min",
        "fit_d_max",
        "invr_A",
        "invr_B",
        "invr_r2",
        "invr_yhat",
        "invr_resid",
        "invr_yhat_renorm",
        "invr_resid_renorm",
        "yuk_A",
        "yuk_B",
        "yuk_k",
        "yuk_r2",
        "yuk_half_range",
        "yuk_yhat",
        "yuk_resid",
        "yuk_yhat_renorm",
        "yuk_resid_renorm",
    ]

    phi = np.zeros((n, n, n), dtype=np.float32)
    src = np.zeros((n, n, n), dtype=np.float32)
    phi_self = np.zeros((n, n, n), dtype=np.float32)
    src_self = np.zeros((n, n, n), dtype=np.float32)

    max_iters_i = int(max_iters)
    if max_iters_i <= 0:
        max_iters_i = int(params.traffic.iters)
    if max_iters_i < 1:
        raise ValueError("--coulomb-max-iters must be >= 1 (or 0 to use --traffic-iters)")

    check_i = int(check_every)
    if check_i < 1:
        raise ValueError("--coulomb-check-every must be >= 1")
    if check_i > max_iters_i:
        check_i = max_iters_i

    tol_i = float(tol)
    if not math.isfinite(tol_i) or tol_i < 0.0:
        raise ValueError("--coulomb-tol must be finite and >= 0")

    # Coulomb runs are steady-state relaxations; use max relaxation rate to avoid wasting iterations.
    # This does not affect the main pipeline; it is local to the Coulomb diagnostic.
    tp_fast = replace(params.traffic, rate_rise=1.0, rate_fall=1.0)

    phi_prev = np.zeros_like(phi, dtype=np.float32)
    phi_self_prev = np.zeros_like(phi_self, dtype=np.float32)

    rows: List[Dict[str, object]] = []

    for d in range(int(d_min), int(d_max) + 1, int(d_step)):
        # Place charges along +x/-x around the center.
        x1 = int(c - (d // 2))
        x2 = int(x1 + d)
        if x1 < safe or x2 > (n - 1 - safe):
            raise ValueError("internal: coulomb placement violated safety margin")

        src[:, :, :] = 0.0
        src[x1, c, c] = float(q)
        if sm == "like":
            src[x2, c, c] = float(q)
        else:
            src[x2, c, c] = float(-q)

        if d == int(d_min):
            phi[:, :, :] = 0.0
            phi_self[:, :, :] = 0.0

        # --- Pair run ---
        phi_prev[:, :, :] = phi

        done_pair = 0
        pair_converged = False
        pair_last_max_delta = float("nan")
        while done_pair < max_iters_i:
            step_chunk = int(min(check_i, max_iters_i - done_pair))
            phi = evolve_diffusion_traffic_steps(phi, src, tp_fast, step_chunk)
            done_pair += step_chunk
            if progress_cb is not None:
                progress_cb(0)
            if tol_i == 0.0:
                continue

            max_delta = float(_nb_max_abs_diff(phi, phi_prev))
            pair_last_max_delta = float(max_delta)
            if max_delta < tol_i:
                pair_converged = True
                break
            phi_prev[:, :, :] = phi

        e_pair_phi = float(_nb_phi_energy(phi))
        e_pair_grad = float(_nb_grad_energy_6(phi))

        # --- Self runs (single +q) ---
        # When placements are symmetric about the lattice center (even d), the two self problems are identical.
        # In that case, solve once at x1 and reuse the result for x2 to cut runtime ~2x.
        use_symmetry = bool(int(x1) + int(x2) == 2 * int(c))
        self2_reused = bool(use_symmetry)

        done_self1, e_self1_phi, e_self1_grad, self1_converged, self1_last_max_delta = _solve_self_at(
            phi_self, phi_self_prev, src_self, int(x1), c, float(q), tp_fast, max_iters_i, check_i, tol_i, progress_cb=progress_cb
        )

        if use_symmetry:
            done_self2 = 0
            done_self2_logical = int(done_self1)
            e_self2_phi = float(e_self1_phi)
            e_self2_grad = float(e_self1_grad)
            self2_converged = bool(self1_converged)
            self2_last_max_delta = float(self1_last_max_delta)
        else:
            done_self2, e_self2_phi, e_self2_grad, self2_converged, self2_last_max_delta = _solve_self_at(
                phi_self, phi_self_prev, src_self, int(x2), c, float(q), tp_fast, max_iters_i, check_i, tol_i, progress_cb=progress_cb
            )
            done_self2_logical = int(done_self2)

        done_self = int(done_self1 + done_self2)
        done_self_logical = int(done_self1 + done_self2_logical)
        # Energy is for two self charges even when we reuse the symmetric solve.
        e_self_phi = float(e_self1_phi + e_self2_phi)
        e_self_grad = float(e_self1_grad + e_self2_grad)

        e_int_phi = float(e_pair_phi - e_self_phi)
        e_int_grad = float(e_pair_grad - e_self_grad)

        rec: Dict[str, object] = {
            "sign": sm,
            "q": float(q),
            "d": int(d),
            "x1": int(x1),
            "x2": int(x2),
            "c": int(c),
            "safe": int(safe),
            "iters_pair": int(done_pair),
            "iters_self1": int(done_self1),
            "iters_self2": int(done_self2),
            "self2_reused": int(1 if self2_reused else 0),
            "iters_self_logical": int(done_self_logical),
            "iters_self": int(done_self),
            "pair_converged": int(1 if pair_converged else 0),
            "pair_max_delta": float(pair_last_max_delta),
            "self1_converged": int(1 if bool(self1_converged) else 0),
            "self1_max_delta": float(self1_last_max_delta),
            "self2_converged": int(1 if bool(self2_converged) else 0),
            "self2_max_delta": float(self2_last_max_delta),
            "E_pair_phi": float(e_pair_phi),
            "E_pair_grad": float(e_pair_grad),
            "E_self1_phi": float(e_self1_phi),
            "E_self1_grad": float(e_self1_grad),
            "E_self2_phi": float(e_self2_phi),
            "E_self2_grad": float(e_self2_grad),
            "E_self_phi": float(e_self_phi),
            "E_self_grad": float(e_self_grad),
            "E_int_phi": float(e_int_phi),
            "E_int_grad": float(e_int_grad),
        }

        rows.append(rec)
        if progress_cb is not None:
            progress_cb(1)

        if progress_cb is None:
            print("[coulomb] sign=%s q=%.6g d=%d it_pair=%d it_self=%d (it1=%d it2=%d) E_pair_phi=%.6g E_pair_grad=%.6g E_int_phi=%.6g E_int_grad=%.6g" % (
                sm, float(q), int(d), int(done_pair), int(done_self), int(done_self1), int(done_self2), float(e_pair_phi), float(e_pair_grad), float(e_int_phi), float(e_int_grad),
            ))

    # --- Fit summary (Phase-1 Yukawa vs 1/r) on interaction gradient energy ---
    fit_metric = "E_int_grad"
    ds = np.array([float(cast(int, r["d"])) for r in rows], dtype=np.float64)
    ys = np.array([float(cast(float, r["E_int_grad"])) for r in rows], dtype=np.float64)

    # Renormalise interaction energies by subtracting the far-field value at d_max.
    # This removes the finite-box constant offset so the interaction tends to ~0 at large separation.
    e_int_phi_far = float(cast(float, rows[-1]["E_int_phi"])) if rows else 0.0
    e_int_grad_far = float(cast(float, rows[-1]["E_int_grad"])) if rows else 0.0
    for r in rows:
        r["E_int_phi_renorm"] = float(cast(float, r["E_int_phi"]) - e_int_phi_far)
        r["E_int_grad_renorm"] = float(cast(float, r["E_int_grad"]) - e_int_grad_far)

    # Use a mid-range window to reduce boundary/noise effects at tiny d.
    fit_d_min = float(max(float(d_min), 4.0))
    fit_mask = (ds >= fit_d_min) & np.isfinite(ys)
    ds_fit = ds[fit_mask]
    ys_fit = ys[fit_mask]

    fit_n = int(ds_fit.size)
    fit_d_max = float(np.max(ds_fit)) if fit_n > 0 else float("nan")

    invr_A = float("nan")
    invr_B = float("nan")
    invr_r2 = float("nan")

    yuk_A = float("nan")
    yuk_B = float("nan")
    yuk_k = float("nan")
    yuk_r2 = float("nan")
    yuk_half_range = float("nan")

    if fit_n >= 6:
        invr_A, invr_B, invr_r2 = _fit_invr(ds_fit, ys_fit)
        yuk_A, yuk_B, yuk_k, yuk_r2 = _fit_yukawa_grid(ds_fit, ys_fit, 1.0e-4, 1.0, 400)
        if math.isfinite(yuk_k) and yuk_k > 0.0:
            yuk_half_range = float(math.log(2.0) / yuk_k)

    # Far-field anchor (d_max) for renormalisation and predicted renorm curves.
    d_far = float(np.max(ds)) if ds.size > 0 else float("nan")
    invr_yhat_far = float("nan")
    yuk_yhat_far = float("nan")
    if math.isfinite(invr_A) and math.isfinite(invr_B) and math.isfinite(d_far) and d_far > 0.0:
        invr_yhat_far = float((float(invr_A) * (1.0 / float(d_far))) + float(invr_B))
    if math.isfinite(yuk_A) and math.isfinite(yuk_B) and math.isfinite(yuk_k) and math.isfinite(d_far) and d_far > 0.0:
        yuk_yhat_far = float((float(yuk_A) * (math.exp(-float(yuk_k) * float(d_far)) / float(d_far))) + float(yuk_B))

    if progress_cb is None:
        print("[coulomb-fit] metric=%s n=%d d=[%.0f..%.0f] invr:A=%.6g B=%.6g r2=%.6g yuk:A=%.6g B=%.6g k=%.6g half=%.3f r2=%.6g" % (
            fit_metric, int(fit_n), float(fit_d_min), float(fit_d_max),
            float(invr_A), float(invr_B), float(invr_r2),
            float(yuk_A), float(yuk_B), float(yuk_k), float(yuk_half_range), float(yuk_r2),
        ))

    for r in rows:
        r["fit_metric"] = fit_metric
        r["fit_n"] = int(fit_n)
        r["fit_d_min"] = float(fit_d_min)
        r["fit_d_max"] = float(fit_d_max)
        r["invr_A"] = float(invr_A)
        r["invr_B"] = float(invr_B)
        r["invr_r2"] = float(invr_r2)
        r["yuk_A"] = float(yuk_A)
        r["yuk_B"] = float(yuk_B)
        r["yuk_k"] = float(yuk_k)
        r["yuk_r2"] = float(yuk_r2)
        r["yuk_half_range"] = float(yuk_half_range)

        d_i = float(cast(int, r["d"]))
        y_i = float(cast(float, r[fit_metric]))

        invr_yhat = float("nan")
        invr_resid = float("nan")
        invr_yhat_ren = float("nan")
        invr_resid_ren = float("nan")
        if math.isfinite(invr_A) and math.isfinite(invr_B) and d_i > 0.0:
            invr_yhat = float((float(invr_A) * (1.0 / d_i)) + float(invr_B))
            invr_resid = float(y_i - invr_yhat)
            if math.isfinite(invr_yhat_far):
                invr_yhat_ren = float(invr_yhat - float(invr_yhat_far))

        yuk_yhat = float("nan")
        yuk_resid = float("nan")
        yuk_yhat_ren = float("nan")
        yuk_resid_ren = float("nan")
        if math.isfinite(yuk_A) and math.isfinite(yuk_B) and math.isfinite(yuk_k) and d_i > 0.0:
            yuk_yhat = float((float(yuk_A) * (math.exp(-float(yuk_k) * d_i) / d_i)) + float(yuk_B))
            yuk_resid = float(y_i - yuk_yhat)
            if math.isfinite(yuk_yhat_far):
                yuk_yhat_ren = float(yuk_yhat - float(yuk_yhat_far))

        # Residuals against renormalised observed values.
        if fit_metric == "E_int_grad":
            y_i_ren = float(cast(float, r["E_int_grad_renorm"]))
        else:
            y_i_ren = float(cast(float, r["E_int_phi_renorm"]))

        if math.isfinite(invr_yhat_ren):
            invr_resid_ren = float(y_i_ren - invr_yhat_ren)
        if math.isfinite(yuk_yhat_ren):
            yuk_resid_ren = float(y_i_ren - yuk_yhat_ren)

        r["invr_yhat"] = float(invr_yhat)
        r["invr_resid"] = float(invr_resid)
        r["invr_yhat_renorm"] = float(invr_yhat_ren)
        r["invr_resid_renorm"] = float(invr_resid_ren)
        r["yuk_yhat"] = float(yuk_yhat)
        r["yuk_resid"] = float(yuk_resid)
        r["yuk_yhat_renorm"] = float(yuk_yhat_ren)
        r["yuk_resid_renorm"] = float(yuk_resid_ren)


    with open(out_path, "w", newline="") as f:
        # Provenance header (comment lines). Prefer the run-level header from core.py.
        if provenance_header is not None and str(provenance_header).strip() != "":
            h = str(provenance_header)
            if not h.endswith("\n"):
                h += "\n"
            f.write(h)
            f.write(f"# coulomb_tp_rate_rise={float(tp_fast.rate_rise):.12g}\n")
            f.write(f"# coulomb_tp_rate_fall={float(tp_fast.rate_fall):.12g}\n")
            f.write(f"# fit_metric={str(fit_metric)}\n")
            f.write(f"# fit_d_min_used={float(fit_d_min):.12g}\n")
            f.write(f"# fit_d_max_used={float(fit_d_max):.12g}\n")
            boundary = str(getattr(params.traffic, "boundary", "hard"))
            sponge_w = int(getattr(params.traffic, "sponge_width", getattr(params.traffic, "sponge_w", 0)) or 0)
            sponge_s = float(getattr(params.traffic, "sponge_strength", getattr(params.traffic, "sponge_s", 0.0)) or 0.0)
            f.write(f"# traffic_boundary={boundary}\n")
            f.write(f"# traffic_sponge_width={int(sponge_w)}\n")
            f.write(f"# traffic_sponge_strength={float(sponge_s):.12g}\n")
            f.write(f"# coulomb_renorm_d_far={float(d_far):.12g}\n")
            f.write(f"# coulomb_renorm_E_int_phi_far={float(e_int_phi_far):.12g}\n")
            f.write(f"# coulomb_renorm_E_int_grad_far={float(e_int_grad_far):.12g}\n")
            f.write(f"# coulomb_pred_invr_yhat_far={float(invr_yhat_far):.12g}\n")
            f.write(f"# coulomb_pred_yuk_yhat_far={float(yuk_yhat_far):.12g}\n")
        else:
            f.write(f"# producer=coulomb.py\n")
            f.write(f"# n={int(n)}\n")
            f.write(f"# traffic_mode={str(params.traffic.mode)}\n")
            f.write(f"# traffic_iters={int(params.traffic.iters)}\n")
            f.write(f"# delta_margin={int(params.delta_margin)}\n")
            f.write(f"# coulomb_sign={str(sm)}\n")
            f.write(f"# coulomb_q={float(q):.12g}\n")
            f.write(f"# coulomb_d_min={int(d_min)}\n")
            f.write(f"# coulomb_d_max={int(d_max)}\n")
            f.write(f"# coulomb_d_step={int(d_step)}\n")
            f.write(f"# coulomb_max_iters={int(max_iters_i)}\n")
            f.write(f"# coulomb_check_every={int(check_i)}\n")
            f.write(f"# coulomb_tol={float(tol_i):.12g}\n")
            f.write(f"# coulomb_tp_rate_rise={float(tp_fast.rate_rise):.12g}\n")
            f.write(f"# coulomb_tp_rate_fall={float(tp_fast.rate_fall):.12g}\n")
            f.write(f"# fit_metric={str(fit_metric)}\n")
            f.write(f"# fit_d_min_used={float(fit_d_min):.12g}\n")
            f.write(f"# fit_d_max_used={float(fit_d_max):.12g}\n")
            boundary = str(getattr(params.traffic, "boundary", "hard"))
            sponge_w = int(getattr(params.traffic, "sponge_width", getattr(params.traffic, "sponge_w", 0)) or 0)
            sponge_s = float(getattr(params.traffic, "sponge_strength", getattr(params.traffic, "sponge_s", 0.0)) or 0.0)
            f.write(f"# traffic_boundary={boundary}\n")
            f.write(f"# traffic_sponge_width={int(sponge_w)}\n")
            f.write(f"# traffic_sponge_strength={float(sponge_s):.12g}\n")
            f.write(f"# coulomb_renorm_d_far={float(d_far):.12g}\n")
            f.write(f"# coulomb_renorm_E_int_phi_far={float(e_int_phi_far):.12g}\n")
            f.write(f"# coulomb_renorm_E_int_grad_far={float(e_int_grad_far):.12g}\n")
            f.write(f"# coulomb_pred_invr_yhat_far={float(invr_yhat_far):.12g}\n")
            f.write(f"# coulomb_pred_yuk_yhat_far={float(yuk_yhat_far):.12g}\n")

        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])

    if progress_cb is None:
        print(f"[coulomb] wrote {out_path}")