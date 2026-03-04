# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""visualiser.py — all graphical output (plots now, animations later)

Role
----
This module is the single home for graphical output in CAELIX. It writes the
standard diagnostic figures used by the pipeline and stability benchmarks.

Design rules
------------
- No matplotlib usage in `core.py` / `pipeline.py` / kernel modules.
- Fail-fast: if plotting is requested, matplotlib must be installed.
- Keep output filenames and plot semantics stable (log parsing and notebooks).

Current outputs
---------------
Pipeline:
  - n_map.png                : index map slice (n = 1 + k·phi)
  - n_map_log.png            : optional log10(n - 1) with robust bounds
  - shapiro_asinh_fit.png    : D vs asinh(X0/b) regression plot

Stability:
  - stability_k_sweep.png                : best-case p_full by k
  - stability_k_sweep_mean_surv.png      : best-case mean survival (ticks) by k

Contracts
---------
- All functions take an explicit `out_dir` and will create it if needed.
- Parent directories are created via `utils.ensure_parent_dir`.
- Numerical bounds use `utils.percentile_bounds` (no matplotlib dependencies in utils).

Flat-module layout
------------------
All modules live in the same folder and import locally:
  `from visualiser import plot_all, plot_stability_k_sweep`

Future
------
This file is the intended place for animations (e.g. field evolution, walker
wake), keeping graphical dependencies quarantined.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from utils import ensure_parent_dir, percentile_bounds

from params import PipelineParams, RayParams

from rays import fit_asinh_delay, ray_trace_delay


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _ensure_out_dir(out_dir: str) -> str:
    d = str(out_dir)
    if d.strip() == "":
        raise ValueError("out_dir must be non-empty")
    os.makedirs(d, exist_ok=True)
    return d


def plot_index_maps(
    n_map: np.ndarray,
    out_dir: str,
    *,
    plot_dpi: int,
    scale: float,
    plot_log: bool,
) -> Tuple[str, str]:
    """Write n_map.png and (optionally) n_map_log.png. Returns paths."""
    out_dir = _ensure_out_dir(out_dir)
    plt = _plt()

    plt.figure(figsize=(6.6 * float(scale), 5.4 * float(scale)))
    plt.imshow(n_map, origin="lower")
    plt.colorbar(label="n")
    plt.title("Index map slice (n = 1 + k·phi)")
    p = os.path.join(out_dir, "n_map.png")
    ensure_parent_dir(p)
    plt.tight_layout()
    plt.savefig(p, dpi=int(plot_dpi))

    p_log = ""
    if plot_log:
        dn = (n_map - 1.0).astype(np.float64)
        dn = np.maximum(dn, 1e-12)
        log_dn = np.log10(dn)

        vmin, vmax = percentile_bounds(log_dn, 2.0, 99.5, fallback=(-12.0, 0.0))

        plt.figure(figsize=(6.6 * float(scale), 5.4 * float(scale)))
        plt.imshow(log_dn, origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar(label="log10(n - 1)")
        plt.title("Index perturbation log10(n - 1)")
        p_log = os.path.join(out_dir, "n_map_log.png")
        ensure_parent_dir(p_log)
        plt.tight_layout()
        plt.savefig(p_log, dpi=int(plot_dpi))

    return p, p_log


def plot_shapiro_asinh_fit(
    n_map: np.ndarray,
    params: PipelineParams,
    out_dir: str,
    *,
    plot_dpi: int,
    scale: float,
) -> str:
    """Write shapiro_asinh_fit.png and return the path."""
    out_dir = _ensure_out_dir(out_dir)
    plt = _plt()

    rp = RayParams(X0=params.X0, ds=params.ds)
    b = np.linspace(1.0 / params.eps, 5.0 / params.eps, 17)
    D = ray_trace_delay(n_map, b, rp)

    u = np.arcsinh(params.X0 / b)
    K, C, r2 = fit_asinh_delay(D, b, params.X0)

    plt.figure(figsize=(6.6 * float(scale), 4.6 * float(scale)))
    plt.scatter(u, D, label="data")
    plt.plot(u, C + K * u, linewidth=2.0, label=f"fit: K={K:.4f} R²={r2:.6f}")
    plt.xlabel("asinh(X0/b)")
    plt.ylabel("D = ∫(n-1)ds")
    plt.title("Shapiro analogue (finite-domain asinh form)")
    plt.legend()

    p = os.path.join(out_dir, "shapiro_asinh_fit.png")
    ensure_parent_dir(p)
    plt.tight_layout()
    plt.savefig(p, dpi=int(plot_dpi))

    return p


def plot_all(
    n_map: np.ndarray,
    params: PipelineParams,
    out_dir: str,
    *,
    plot_dpi: int,
    scale: float,
    plot_log: bool,
) -> Tuple[str, str, str]:
    """Convenience wrapper: write all standard plots and return paths."""
    p1, p2 = plot_index_maps(n_map, out_dir, plot_dpi=plot_dpi, scale=scale, plot_log=plot_log)
    p3 = plot_shapiro_asinh_fit(n_map, params, out_dir, plot_dpi=plot_dpi, scale=scale)
    return p1, p2, p3


def plot_stability_k_sweep(
    ks: List[int],
    pfs: List[float],
    means: List[float],
    out_dir: str,
    *,
    plot_dpi: int,
    scale: float,
    ticks_max: float,
) -> Tuple[str, str]:
    """Write stability k-sweep plots.

    Produces:
      - stability_k_sweep.png (best-case p_full by k)
      - stability_k_sweep_mean_surv.png (best-case mean survival by k)
    Returns the two paths.
    """
    out_dir = _ensure_out_dir(out_dir)
    plt = _plt()

    ks_arr = [int(k) for k in ks]
    pfs_arr = [float(v) for v in pfs]
    means_arr = [float(v) for v in means]

    plt.figure(figsize=(6.4 * float(scale), 4.2 * float(scale)))
    plt.plot(ks_arr, pfs_arr, marker="o", linewidth=2.0)
    plt.ylim(-0.02, 1.02)
    plt.xticks(ks_arr)
    plt.xlabel("k (number of face constraints checked)")
    plt.ylabel("best-case p_full")
    plt.title("Stability phase transition by k (best-case over k-of-6 face subsets)")
    p1 = os.path.join(out_dir, "stability_k_sweep.png")
    ensure_parent_dir(p1)
    plt.tight_layout()
    plt.savefig(p1, dpi=int(plot_dpi))

    plt.figure(figsize=(6.4 * float(scale), 4.2 * float(scale)))
    plt.plot(ks_arr, means_arr, marker="o", linewidth=2.0)
    plt.ylim(-0.02, float(ticks_max) * 1.02)
    plt.xticks(ks_arr)
    plt.xlabel("k (number of face constraints checked)")
    plt.ylabel("best-case mean survival (ticks)")
    plt.title("Best-case mean survival by k (face subsets only)")
    p2 = os.path.join(out_dir, "stability_k_sweep_mean_surv.png")
    ensure_parent_dir(p2)
    plt.tight_layout()
    plt.savefig(p2, dpi=int(plot_dpi))

    return p1, p2