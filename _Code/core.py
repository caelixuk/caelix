# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""core.py — experiment dispatch for the CAELIX lattice/field pipeline

Purpose
-------
This file is intentionally thin. It performs startup checks (dependencies, thread defaults),
imports runner modules, and dispatches to the selected experiment.

All CLI wiring lives in `cli.py`: argparse flags, output-path normalisation (never assume cwd),
and mapping `args -> PipelineParams`.

High-level pipeline (single seed)
--------------------------------
  microstate/load -> steady scalar field phi -> index map n(x,y)
  -> ray-tracing regressions (deflection + Shapiro analogue)

Key idea
--------
If the steady scalar field behaves ~ 1/r in 3D, then defining:
  n = 1 + k * phi
should reproduce weak-field signatures:
  theta ∝ 1/b
  D ≈ C + K * asinh(X0/b)
with K proportional to an effective mass scale after calibration.

What lives where (module map)
-----------------------------
CLI + parameter mapping:
    - cli.py            : argparse flags, output path normalisation, args -> PipelineParams
    - experiments.py    : named, ordered experiment presets (wraps core.py)

Core contracts (dataclasses + small helpers):
    - params.py         : dataclasses (PipelineParams, LatticeParams, TrafficParams, ...)
    - utils.py          : shared helpers (rng, finite checks, formatting, paths, timers, thread defaults)

Kernels and diagnostics:
    - lattice.py        : ternary microstate init/anneal helpers
    - load.py           : compute_load (microstate -> load)
    - traffic.py        : diffusion/telegraph solvers (load -> phi)
    - radial.py         : radial profiles + fits + CSV dumps
    - rays.py           : ray tracing + asinh fit helpers

Experiment orchestration:
    - pipeline.py       : build_index_from_micro + Shapiro lockdown (composition)
    - walker.py         : Heavy Walker (moving delta source, wake / Mach probes)
    - collider.py       : two-walker collision (spin dependence; writes CSV)
    - coulomb.py        : signed-source (like/opposite) energy vs separation (optional decay -> Yukawa)
    - stability.py      : face-lock stability benchmarks and subset sweeps

Wave / Interference / Confinement:
    - double_slit.py    : telegraph + hard mask (Dirichlet walls), records detector-line intensity
    - corral.py         : confined geometry resonance sweep (quantisation / mode spectrum)
    - ringdown.py       : passive ringdown / resonance sweep (06B; Dirichlet box; sigma sweep + probe FFT)
    - soliton.py        : Phase 8 scan for long-lived non-linear lumps (08A; sponge boundary; k/lambda sweep + R_g metrics)

Relativity / Timing calibration:
    - isotropy.py       : axis vs diagonal propagation calibration (effective c anisotropy)
    - relativity.py     : light-clock / twin-paradox runner (writes tick/interval CSV)
    - oscillator.py     : driven phase-drift + lensing diagnostics (06B; timing via phase, not ticks)

Visual output:
    - visualiser.py     : all plotting (and later animations)

- Flat-file layout: all modules sit alongside this file.
- Chirality/parity selection for telegraph runs is supported via --chiral-select and --chiral-field.
- `--delta-load` is a pipeline/ensemble switch (micro/load path); walker and collider use moving delta injections internally and do not require it.
- Fail-fast and single-path: no silent coercions, no fallback branches.
- Output policy (implemented in `cli.py`):
  `--out` resolves to the project-local `_Output/` by default.
  Any per-experiment `--*-out` can be:
    * an absolute path, or
    * a bare filename (written under `--out`), or
    * a relative path resolved relative to this module.

Reproducible runs
-----------------
Use `experiments.py` for the definitive presets and suite order:
  python experiments.py --list
  python experiments.py --run 01A_pipeline_baseline (--run 01A will also work for convenience)
  python experiments.py --run-all
Outputs are grouped into category subfolders under `_Output/` (see experiments.py for the canonical layout).
For ad-hoc runs, `core.py --help` still exposes the full CLI surface.
"""

from __future__ import annotations

import os
import csv
import concurrent.futures
import datetime
import math
import sys
import inspect
import re
import h5py
import subprocess
from typing import Dict, List, Tuple, Callable, Optional, Any, cast

import numpy as np

# --- Local module dataclasses ---

from utils import Timer, _as_float, _make_rng, apply_conservative_thread_defaults, ensure_parent_dir, require_dependencies, write_csv_provenance_header

from cli import parse_cli

from params import PipelineParams, StabilityParams, TrafficParams


from plumbing import ProgressBar, _get_log_path, _log_line_only, _maybe_make_progress_bar, _ensure_bundle_dirs, _derive_exp_name


def main() -> None:
    t_run = Timer().start()
    args_for_log: object | None = None
    args: Any | None = None
    params: PipelineParams | None = None
    seed0: int = 0
    viewer_proc: subprocess.Popen[str] | None = None
    live_bcast: Any | None = None
    shm_name: str = ""
    apply_conservative_thread_defaults()
    require_dependencies(
        ("numba", "pip install numba"),
        ("numpy", "pip install numpy"),
        ("matplotlib", "pip install matplotlib"),
    )

    # Import local runner modules *after* thread defaults are applied.
    from pipeline import _ensemble_one, build_index_from_micro, run_shapiro_mass_lockdown
    from walker import _parse_int_list, run_heavy_walker, run_heavy_walker_sweep, walker_work_units, walker_sweep_work_units
    from stability import stability_benchmark, stability_face_subset_sweep
    from coulomb import run_coulomb_test
    from collider import run_collider
    from collidersg import parse_sprites_json, read_sprite_asset_h5, run_collidersg
    from corral import run_quantum_corral_sweep
    from ringdown import run_ringdown_sweep_sigma, ringdown_work_units
    from isotropy import run_isotropy_test, run_isotropy_sigma_sweep
    from relativity import run_light_clock
    from oscillator import OscillatorConfig, LensingConfig, run_gravity_phase_drift, run_lensing_rays, oscillator_work_units
    from double_slit import run_double_slit
    from visualiser import plot_all, plot_stability_k_sweep

    try:
        args, params, seed0 = parse_cli(here_file=__file__)
        args_for_log = args
        _log_line_only(args, "[timing] start=%s" % (datetime.datetime.now().isoformat(timespec="seconds"),))
        # Build a single run identity for any CSV artefacts written by this invocation.
        # Keep this stable across *all* experiment branches (01/02/relativity/etc).
        when_dt = datetime.datetime.now()
        when_iso = when_dt.isoformat(timespec="seconds")
        when_tag = when_dt.strftime("%Y%m%d_%H%M%S")
        cmd = " ".join([sys.executable] + [os.path.abspath(__file__)] + list(sys.argv[1:]))

        # Prefer the run folder stamp as the canonical run_id when available.
        # When invoked via experiments.py, args.out ends with `YYYYMMDD_HHMMSS`.
        out_leaf = os.path.basename(os.path.normpath(str(getattr(args, "out", ""))))
        run_id = str(when_tag)
        if re.fullmatch(r"\d{8}_\d{6}", str(out_leaf)):
            run_id = str(out_leaf)

        # LiveView (single lifecycle per invocation)
        # Create shared memory first so the viewer can always attach.
        if bool(getattr(params, "liveview", False)):
            try:
                from broadcast import LiveBroadcaster

                n_lv = int(getattr(params.lattice, "n", 0))
                if n_lv <= 0:
                    raise ValueError("liveview: invalid lattice.n")

                shm_name = f"CAELIX_{str(run_id)}"
                live_bcast = LiveBroadcaster((n_lv, n_lv, n_lv), name=shm_name, create=True, zero_init=True)

                here_dir = os.path.dirname(os.path.abspath(__file__))
                viewer_path = os.path.join(here_dir, "liveview.py")
                lv_log_path = os.path.join(str(getattr(args, "out", ".")), "_logs", "liveview.log")
                viewer_proc = subprocess.Popen(
                    [
                        sys.executable,
                        viewer_path,
                        "--n",
                        str(n_lv),
                        "--shm-name",
                        str(shm_name),
                        "--log-path",
                        str(lv_log_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                _log_line_only(args, "[liveview] started shm=%s n=%d" % (str(shm_name), int(n_lv)))
            except Exception as e:
                _log_line_only(args, "[liveview] start failed: %s" % (str(e),))
                live_bcast = None
                viewer_proc = None

        def _liveview_kw(fn: Callable[..., Any]) -> dict[str, Any]:
            # Only pass LiveView wiring when we actually started it.
            if not shm_name:
                return {}
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                return {}
            kw: dict[str, Any] = {}
            if "liveview_shm_name" in sig.parameters:
                kw["liveview_shm_name"] = str(shm_name)
            return kw

        # Early-exit: stability benchmark section
        if args.bench_stability:
            if bool(getattr(args, "bench_sweep_k_write_csv", False)) and (not args.bench_sweep_k):
                raise ValueError("--bench-sweep-k-write-csv requires --bench-sweep-k")

            sp = StabilityParams(
                trials=args.bench_trials,
                ticks=args.bench_ticks,
                p_noise=args.bench_p_noise,
                p_center_flip=args.bench_p_center,
            )
            rng = _make_rng(args.seed)
            res = stability_benchmark(rng, sp)

            # Print a stable table (fail-loud, no clever formatting).
            print("[stability-bench] trials=%d ticks=%d p_noise=%.6g p_center=%.6g" % (
                sp.trials, sp.ticks, sp.p_noise, sp.p_center_flip,
            ))

            def _print_row(name: str) -> None:
                ms = res[f"{name}.mean_survival"]
                pf = res[f"{name}.p_full"]
                rpt = res[f"{name}.reads_per_tick"]
                rep = res[f"{name}.repairs_per_tick"]
                print("  %-14s mean_surv=%7.2f  p_full=%6.3f  reads/tick=%6.2f  repairs/tick=%6.3f" % (
                    name, ms, pf, rpt, rep,
                ))

            _print_row("A_planar4")
            _print_row("B_faces6")
            _print_row("C_corners8")
            _print_row("D_faces6_plus2")

            # Paper-ready summary (single-screen): isolate the minimality claim.
            pf_faces6 = float(res["B_faces6.p_full"])
            ms_faces6 = float(res["B_faces6.mean_survival"])
            print("[stability-summary] faces6: p_full=%.3f mean_surv=%.2f reads/tick=%.2f" % (
                pf_faces6, ms_faces6, float(res["B_faces6.reads_per_tick"]),
            ))

            if args.bench_sweep_faces:
                # Best-case among all face-subsets for k=4 and k=5 (no cherry-picking).
                s4 = stability_face_subset_sweep(rng, sp, k=4)
                s5 = stability_face_subset_sweep(rng, sp, k=5)

                print("[stability-sweep] k=4 best_p_full=%.3f best_mean_surv=%.2f best=%s" % (
                    s4.best_p_full, s4.best_mean_survival, s4.best_name,
                ))
                print("[stability-sweep] k=4 worst_p_full=%.3f worst_mean_surv=%.2f worst=%s" % (
                    s4.worst_p_full, s4.worst_mean_survival, s4.worst_name,
                ))

                print("[stability-sweep] k=5 best_p_full=%.3f best_mean_surv=%.2f best=%s" % (
                    s5.best_p_full, s5.best_mean_survival, s5.best_name,
                ))
                print("[stability-sweep] k=5 worst_p_full=%.3f worst_mean_surv=%.2f worst=%s" % (
                    s5.worst_p_full, s5.worst_mean_survival, s5.worst_name,
                ))

                # Minimality statement: compare best-case k<6 to k=6.
                print("[stability-minimality] best(k=5): p_full=%.3f mean_surv=%.2f %s" % (
                    s5.best_p_full, s5.best_mean_survival, s5.best_name,
                ))
                print("[stability-minimality] best(k=4): p_full=%.3f mean_surv=%.2f %s" % (
                    s4.best_p_full, s4.best_mean_survival, s4.best_name,
                ))
                print("[stability-minimality] conclusion: for this 3D face-lock invariant under isotropic noise, k=6 is uniquely sufficient; k<6 is unstable even in its best case.")

            # Optional: full sweep over k=1..6 face subsets for compact monotonicity table.
            if args.bench_sweep_k:
                # Best-case over all k-of-6 face subsets for k=1..6.
                # We re-seed per k for reproducibility and to avoid ordering artefacts.
                print("[stability-k-sweep] best-case stability by k (face subsets only)")
                ks = []
                pfs = []
                means = []
                best_names = []
                for k in range(1, 7):
                    rng_k = _make_rng(int(args.seed) + 1000 * k)
                    sk = stability_face_subset_sweep(rng_k, sp, k=k)
                    ks.append(int(k))
                    pfs.append(float(sk.best_p_full))
                    means.append(float(sk.best_mean_survival))
                    best_names.append(str(sk.best_name))
                    print("  k=%d best_p_full=%.3f best_mean_surv=%.2f best=%s" % (
                        k, sk.best_p_full, sk.best_mean_survival, sk.best_name,
                    ))

                if args.bench_sweep_k_plot:
                    os.makedirs(args.out, exist_ok=True)
                    scale = float(max(1.0, args.plot_scale))
                    p, p2 = plot_stability_k_sweep(
                        ks,
                        pfs,
                        means,
                        str(args.out),
                        plot_dpi=int(args.plot_dpi),
                        scale=scale,
                        ticks_max=float(sp.ticks),
                    )
                    print(f"[stability-k-sweep] wrote {p}")
                    print(f"[stability-k-sweep] wrote {p2}")

                # Write CSV of k-sweep if requested (single path).
                if bool(getattr(args, "bench_sweep_k_write_csv", False)):
                    os.makedirs(args.out, exist_ok=True)
                    csv_path = os.path.join(args.out, "stability_k_sweep.csv")
                    with open(csv_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["k", "best_p_full", "best_mean_surv", "subset", "missing"])
                        for k, pf, ms, name in zip(ks, pfs, means, best_names):
                            s = str(name)
                            if " missing=" in s:
                                subset, missing = s.split(" missing=", 1)
                            else:
                                subset, missing = s, ""
                            w.writerow([int(k), float(pf), float(ms), subset, missing])
                    print(f"[stability-k-sweep] wrote {csv_path}")

            # Minimal claim check: any subset missing ±z should fail versus faces6.
            # We treat success as p_full close to 1.
            print("[stability-bench] expectation: B_faces6 should dominate A_planar4; D adds cost; C is not sufficient for face-lock.")
            return

        # Coulomb test mode (signed-source energy vs separation)
        if bool(args.coulomb):
            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)

            out_csv = str(args.coulomb_out).strip()
            if out_csv == "":
                exp_name = _derive_exp_name(run_dir, default="03")
                out_csv = os.path.join(
                    csv_dir,
                    f"{exp_name}_n{int(params.lattice.n)}_{str(run_id)}.csv",
                )
            else:
                out_csv = str(out_csv)
                ensure_parent_dir(out_csv)

            # Provenance header (match 01/02): embed run identity in the CSV.
            coulomb_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_name),
                    "artefact": "coulomb",
                    "n": str(int(params.lattice.n)),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_iters": str(int(params.traffic.iters)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "coulomb_sign": str(args.coulomb_sign),
                    "coulomb_q": str(float(args.coulomb_q)),
                    "coulomb_d_min": str(int(args.coulomb_d_min)),
                    "coulomb_d_max": str(int(args.coulomb_d_max)),
                    "coulomb_d_step": str(int(args.coulomb_d_step)),
                    "coulomb_max_iters": str(int(args.coulomb_max_iters)),
                    "coulomb_check_every": str(int(args.coulomb_check_every)),
                    "coulomb_tol": str(float(args.coulomb_tol)),
                },
            )

            d_min = int(args.coulomb_d_min)
            d_max = int(args.coulomb_d_max)
            d_step = int(args.coulomb_d_step)
            if d_step <= 0:
                raise ValueError("--coulomb-d-step must be > 0")
            total_units = ((d_max - d_min) // d_step) + 1
            pb = ProgressBar(total_units=int(max(1, total_units)), label="work")
            pb.start()
            try:
                run_coulomb_test(
                    params,
                    q=float(args.coulomb_q),
                    sign_mode=str(args.coulomb_sign),
                    d_min=int(d_min),
                    d_max=int(d_max),
                    d_step=int(d_step),
                    out_csv=str(out_csv),
                    max_iters=int(args.coulomb_max_iters),
                    check_every=int(args.coulomb_check_every),
                    tol=float(args.coulomb_tol),
                    provenance_header=str(coulomb_prov),
                    progress_cb=pb.advance,
                )
            finally:
                pb.finish()
            print("[coulomb] wrote %s" % (str(out_csv),))
            return

        # Collider mode (two moving sources; spin dependence)
        if bool(getattr(args, "collider", False)):
            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)

            # Resolve a concrete log path for this run and reuse it for collider logging.
            # When invoked via experiments.py, this is typically already set to the bundle log.
            n = int(params.lattice.n)
            steps = int(getattr(args, "collider_steps", 0))
            exp_name = _derive_exp_name(run_dir, default="07")
            # Prefer an existing bundle log (experiments.py captures stdout/stderr into one).
            # Only mint a new log if none exists.
            lp = _get_log_path(args)
            if lp == "":
                # If we're running under experiments.py, it will already have created a log
                # in `<run_dir>/_logs/` that includes the run_id. Reuse it.
                try:
                    if os.path.isdir(log_dir):
                        want = str(run_id)
                        cands: list[str] = []
                        for fn in os.listdir(log_dir):
                            if not fn.endswith(".log"):
                                continue
                            if want in fn:
                                cands.append(os.path.join(log_dir, fn))
                        if cands:
                            cands.sort()
                            lp = cands[-1]
                except Exception:
                    lp = ""

            if lp == "":
                lp = os.path.join(log_dir, f"{exp_name}_n{int(n)}_{str(run_id)}.log")

            try:
                setattr(args, "log_path", str(lp))
            except Exception:
                pass

            # Default steps: keep walkers in-bounds for the head-on run.
            # Collider defaults use xA0≈0.375*n, xB0≈0.625*n; scale safe runtime by 1/vx.
            if steps <= 0:
                vx = float(getattr(args, "collider_vx", float(PipelineParams().collider_vx)))
                if vx <= 0.0:
                    raise ValueError("--collider-vx must be > 0")
                margin = 8.0
                steps = max(1, int((0.625 * float(n) - margin) / vx))
                _log_line_only(args, "[collider] default steps=%d (in-bounds for n=%d vx=%.6g)" % (int(steps), int(n), float(vx)))

            # Derive experiment name from the parent folder (e.g. "07A_...").
            # (already set above)

            out_csv = str(getattr(args, "collider_out", "")).strip()
            if out_csv == "":
                out_csv = os.path.join(
                    csv_dir,
                    f"{exp_name}_n{int(n)}_{str(run_id)}.csv",
                )
            else:
                out_csv = str(out_csv)
                ensure_parent_dir(out_csv)

            collider_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_name),
                    "artefact": "collider",
                    "n": str(int(n)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_c2": str(float(params.traffic.c2)),
                    "traffic_gamma": str(float(params.traffic.gamma)),
                    "traffic_dt": str(float(params.traffic.dt)),
                    "traffic_decay": str(float(params.traffic.decay)),
                    "collider_spin_b": str(int(getattr(args, "collider_spin_b", -1))),
                    "collider_vx": str(float(getattr(args, "collider_vx", float(PipelineParams().collider_vx)))),
                    "collider_orbit_radius": str(float(getattr(args, "collider_orbit_radius", float(PipelineParams().collider_orbit_radius)))),
                    "collider_orbit_omega": str(float(getattr(args, "collider_orbit_omega", float(PipelineParams().collider_orbit_omega)))),
                    "collider_steps": str(int(steps)),
                    **({"collider_detectors": str(int(getattr(args, "collider_detectors")))} if hasattr(args, "collider_detectors") else {}),
                    **({"collider_shell": str(int(getattr(args, "collider_shell")))} if hasattr(args, "collider_shell") else {}),
                    **({"collider_shell_stride": str(int(getattr(args, "collider_shell_stride")))} if hasattr(args, "collider_shell_stride") else {}),
                    **({"collider_shell_inner_frac": str(float(getattr(args, "collider_shell_inner_frac")))} if hasattr(args, "collider_shell_inner_frac") else {}),
                    **({"collider_shell_outer_frac": str(float(getattr(args, "collider_shell_outer_frac")))} if hasattr(args, "collider_shell_outer_frac") else {}),
                    **({"collider_hold_steps": str(int(getattr(args, "collider_hold_steps")))} if hasattr(args, "collider_hold_steps") else {}),
                    **({"collider_write_t_rel": str(int(getattr(args, "collider_write_t_rel")))} if hasattr(args, "collider_write_t_rel") else {}),
                    **({"collider_write_ratios": str(int(getattr(args, "collider_write_ratios")))} if hasattr(args, "collider_write_ratios") else {}),
                },
            )

            total_units = int(max(1, int(steps)))
            pb = ProgressBar(total_units=total_units, label="work")
            pb.start()
            try:
                kw: dict[str, Any] = {}
                try:
                    sig_col = inspect.signature(cast(Callable[..., Any], run_collider))
                    if "provenance_header" in sig_col.parameters:
                        kw["provenance_header"] = str(collider_prov)
                    # Progress adapter: collider may call progress_cb(delta) or progress_cb(done, total)
                    last_done = 0
                    def _collider_progress(*a: Any) -> None:
                        nonlocal last_done
                        if not a:
                            return
                        if len(a) == 1:
                            pb.advance(int(a[0]))
                            last_done = int(min(pb.total, pb.done))
                            return
                        # Treat as (done, total)
                        done_i = int(a[0])
                        if done_i < 0:
                            done_i = 0
                        if done_i > pb.total:
                            done_i = pb.total
                        delta = done_i - int(last_done)
                        if delta > 0:
                            pb.advance(int(delta))
                        last_done = done_i

                    if "progress_cb" in sig_col.parameters:
                        kw["progress_cb"] = _collider_progress
                    elif "progress" in sig_col.parameters:
                        kw["progress"] = _collider_progress

                    # Log path: prefer reusing the bundle log when possible, otherwise allow a collider-specific log.
                    if "log_path" in sig_col.parameters:
                        kw["log_path"] = str(lp)
                    elif "log" in sig_col.parameters:
                        kw["log"] = str(lp)
                    # Optional advanced collider features (07C+): only pass if the runner accepts them.
                    # Prefer CLI args (if present), otherwise fall back to PipelineParams attributes.
                    def _maybe_get(name: str, default: Any = None) -> Any:
                        try:
                            return getattr(args, name)
                        except Exception:
                            pass
                        try:
                            return getattr(params, name)
                        except Exception:
                            pass
                        return default

                    opt_names = [
                        # detector / measurement upgrades
                        "collider_detectors",
                        "collider_detector_r",
                        "collider_detector_stride",
                        "collider_detector_inner_frac",
                        "collider_detector_outer_frac",
                        "collider_detector_inner",
                        "collider_detector_outer",
                        # shell calorimetry
                        "collider_shell",
                        "collider_shell_stride",
                        "collider_shell_inner_frac",
                        "collider_shell_outer_frac",
                        "collider_shell_inner",
                        "collider_shell_outer",
                        # post-collision hold / decay
                        "collider_hold",
                        "collider_hold_steps",
                        # richer CSV fields
                        "collider_write_ratios",
                        "collider_write_t_rel",
                    ]

                    for nm in opt_names:
                        if nm in sig_col.parameters:
                            val = _maybe_get(nm, None)
                            if val is not None:
                                kw[nm] = val
                except (TypeError, ValueError):
                    kw = {}

                run_collider(
                    params,
                    spin_b=int(getattr(args, "collider_spin_b", -1)),
                    steps=int(steps),
                    out_dir=str(run_dir),
                    out_csv=str(out_csv),
                    **kw,
                )
            finally:
                pb.finish()

            _log_line_only(args, "[collider] wrote %s" % (str(out_csv),))
            return

        # ColliderSG mode (09) — multi-sprite Sine-Gordon interaction + tracking
        if bool(getattr(args, "collidersg", False)):
            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)

            n = int(params.lattice.n)
            exp_name = _derive_exp_name(run_dir, default="09")
            traffic_dt = float(params.traffic.dt)
            traffic_c2 = float(params.traffic.c2)

            # Required: sprites JSON (either a JSON string or a path to a .json file)
            sprites_src = str(getattr(args, "collidersg_sprites", "")).strip()
            if sprites_src == "":
                raise ValueError("--collidersg requires --collidersg-sprites (JSON string or .json path)")

            # Steps + logging knobs
            steps = int(getattr(args, "collidersg_steps", 2000))
            if steps < 1:
                raise ValueError("--collidersg-steps must be >= 1")
            log_every = int(getattr(args, "collidersg_log_every", 10))
            if log_every < 1:
                raise ValueError("--collidersg-log-every must be >= 1")
            track_r = int(getattr(args, "collidersg_track_r", 16))
            if track_r < 1:
                raise ValueError("--collidersg-track-r must be >= 1")
            peak_thresh = float(getattr(args, "collidersg_peak_thresh", 0.05))
            if not (np.isfinite(peak_thresh) and peak_thresh >= 0.0):
                raise ValueError("--collidersg-peak-thresh must be finite and >= 0")
            phi_abs_every = int(getattr(args, "collidersg_phi_abs_every", 0))
            if phi_abs_every < 0:
                raise ValueError("--collidersg-phi-abs-every must be >= 0")

            # SG stiffness: allow 0/unset (inferred from sprite asset if present).
            sg_k = float(getattr(args, "collidersg_k", 0.0))
            if (not np.isfinite(sg_k)) or (sg_k < 0.0):
                raise ValueError("--collidersg-k must be finite and >= 0 (0 means infer from sprite asset)")

            # Output path
            out_csv = str(getattr(args, "collidersg_out", "")).strip()
            if out_csv == "":
                out_csv = os.path.join(csv_dir, f"{exp_name}_n{int(n)}_{str(run_id)}.csv")
            else:
                out_csv = str(out_csv)
                ensure_parent_dir(out_csv)

            # Boundary hygiene defaults (prefer current traffic params when present)
            boundary_mode = str(getattr(args, "collidersg_boundary", getattr(params.traffic, "boundary_mode", "sponge")))
            sponge_width = int(getattr(args, "collidersg_sponge_width", getattr(params.traffic, "sponge_width", 32)))
            sponge_strength = float(getattr(args, "collidersg_sponge_strength", getattr(params.traffic, "sponge_strength", 0.1)))
            gamma = float(getattr(args, "collidersg_gamma", getattr(params.traffic, "gamma", 0.0)))
            decay = float(getattr(args, "collidersg_decay", getattr(params.traffic, "decay", 0.0)))

            boundary_mode = str(boundary_mode).strip().lower()
            if boundary_mode not in ("open", "zero", "neumann", "sponge"):
                raise ValueError(f"--collidersg-boundary must be one of: open|zero|neumann|sponge (got {boundary_mode!r})")
            if boundary_mode == "sponge":
                if int(sponge_width) <= 0:
                    raise ValueError("TrafficParams.sponge_width must be > 0 when boundary_mode=sponge")
                if (not np.isfinite(float(sponge_strength))) or float(sponge_strength) <= 0.0:
                    raise ValueError("TrafficParams.sponge_strength must be > 0 when boundary_mode=sponge")
            else:
                # Non-sponge modes must not carry sponge params into traffic kernels.
                sponge_width = 0
                sponge_strength = 0.0

            # Optional: spatial k-grid wiring (wire / track experiments).
            # Definitive CLI dest names are:
            #   sg_k_outside, wire_y0/wire_y1, wire_z0/wire_z1, wire_bevel
            k_outside = float(getattr(args, "sg_k_outside", 0.0) or 0.0)
            wire_y0 = int(getattr(args, "wire_y0", -1))
            wire_y1 = int(getattr(args, "wire_y1", -1))
            wire_z0 = int(getattr(args, "wire_z0", -1))
            wire_z1 = int(getattr(args, "wire_z1", -1))
            wire_bevel = int(getattr(args, "wire_bevel", 0))
            wire_geom = str(getattr(args, "wire_geom", "straight") or "straight").strip()
            junction_x = int(getattr(args, "junction_x", -1))
            branch_len = int(getattr(args, "branch_len", 0))
            branch_thick = int(getattr(args, "branch_thick", 2))

            # Parse sprites now so provenance can include the count.
            sprites = parse_sprites_json(sprites_src)

            sprite_asset_h5 = str(getattr(args, "collidersg_sprite_asset", "") or "").strip()
            asset_sg_k = float("nan")
            if sprite_asset_h5 != "":
                try:
                    asset0 = read_sprite_asset_h5(str(sprite_asset_h5))
                    try:
                        asset_sg_k = float((asset0.meta or {}).get("sg.k", float("nan")))
                    except Exception:
                        asset_sg_k = float("nan")
                except Exception:
                    asset_sg_k = float("nan")

            inferred_k = False
            if (not np.isfinite(float(sg_k))) or (abs(float(sg_k)) < 1.0e-12):
                # Prefer inferring from sprite asset when present, otherwise fall back to the
                # global SG stiffness in the TrafficParams (used by kink-wall init).
                if sprite_asset_h5 != "":
                    if not np.isfinite(asset_sg_k):
                        raise ValueError(f"sprite asset missing sg.k attribute: {str(sprite_asset_h5)}")
                    sg_k = float(asset_sg_k)
                    inferred_k = True
                else:
                    tp_k = float(getattr(params.traffic, "k", 0.0))
                    if (not np.isfinite(tp_k)) or (tp_k <= 0.0):
                        raise ValueError("--collidersg-k was unset (0) and no --collidersg-sprite-asset was provided; set --traffic-k (or --collidersg-k)")
                    sg_k = float(tp_k)
                    inferred_k = True

            _log_line_only(
                args,
                "[collidersg] sg_k=%.8g inferred=%d asset_k=%.8g traffic_k=%.8g" % (
                    float(sg_k),
                    int(1 if inferred_k else 0),
                    float(asset_sg_k),
                    float(getattr(params.traffic, "k", 0.0)),
                ),
            )

            collidersg_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_name),
                    "artefact": "collidersg",
                    "n": str(int(n)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "traffic_c2": str(float(traffic_c2)),
                    "traffic_dt": str(float(traffic_dt)),
                    "traffic_gamma": str(float(gamma)),
                    "traffic_decay": str(float(decay)),
                    "boundary_mode": str(boundary_mode),
                    "sponge_width": str(int(sponge_width)),
                    "sponge_strength": str(float(sponge_strength)),
                    "sg_k": str(float(sg_k)),
                    "asset_sg_k": str(float(asset_sg_k)),
                    "sg_k_inferred": str(int(1 if inferred_k else 0)),
                    "steps": str(int(steps)),
                    "log_every": str(int(log_every)),
                    "phi_abs_every": str(int(phi_abs_every)),
                    "track_r": str(int(track_r)),
                    "peak_thresh": str(float(peak_thresh)),
                    "sprite_count": str(int(len(sprites))),
                    "scenario": str(getattr(args, "collidersg_scenario", "")),
                    "sprites_src": ("inline_json" if ("{" in sprites_src or "[" in sprites_src) else sprites_src),
                    "sprite_asset_h5": str(sprite_asset_h5),
                    "sg_k_outside": str(float(k_outside)),
                    "wire_y0": str(int(wire_y0)),
                    "wire_y1": str(int(wire_y1)),
                    "wire_z0": str(int(wire_z0)),
                    "wire_z1": str(int(wire_z1)),
                    "wire_bevel": str(int(wire_bevel)),
                    "wire_geom": str(wire_geom),
                    "junction_x": str(int(junction_x)),
                    "branch_len": str(int(branch_len)),
                    "branch_thick": str(int(branch_thick)),
                },
            )

            pb = ProgressBar(total_units=int(max(1, steps)), label="work")
            pb.start()
            try:
                # Fail-loud: this project expects the runner surface to be stable.
                # Always pass the wire/k-grid knobs so runs cannot silently ignore confinement.
                run_collidersg(
                    n=int(n),
                    steps=int(steps),
                    dt=float(traffic_dt),
                    c2=float(traffic_c2),
                    sg_k=float(sg_k),
                    sg_k_outside=float(k_outside),
                    wire_y0=int(wire_y0),
                    wire_y1=int(wire_y1),
                    wire_z0=int(wire_z0),
                    wire_z1=int(wire_z1),
                    wire_bevel=int(wire_bevel),
                    wire_geom=str(wire_geom),
                    junction_x=int(junction_x),
                    branch_len=int(branch_len),
                    branch_thick=int(branch_thick),
                    sprites=sprites,
                    sprite_asset_h5=str(sprite_asset_h5),
                    out_csv=str(out_csv),
                    log_every=int(log_every),
                    track_r=int(track_r),
                    peak_thresh=float(peak_thresh),
                    phi_abs_every=int(phi_abs_every),
                    boundary_mode=str(boundary_mode),
                    sponge_width=int(sponge_width),
                    sponge_strength=float(sponge_strength),
                    gamma=float(gamma),
                    decay=float(decay),
                    scenario=str(getattr(args, "collidersg_scenario", "")),
                    progress=pb.advance,
                    provenance_header=str(collidersg_prov),
                )
            finally:
                pb.finish()

            _log_line_only(args, "[collidersg] wrote %s" % (str(out_csv),))
            return

        # Double-slit mode (Young interference; hard mask + telegraph)
        if bool(getattr(args, "double_slit", False)):
            ds_steps = int(getattr(args, "ds_steps", 800))
            if ds_steps < 1:
                raise ValueError("--ds-steps must be >= 1")

            out_dir = str(args.out)
            exp_code = _derive_exp_name(out_dir, default="02")

            # Provenance header (match 01-series): embed run identity in the CSV.
            csv_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_code),
                    "artefact": "double_slit",
                    "n": str(int(params.lattice.n)),
                    "ds_steps": str(int(ds_steps)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "lattice_init": str(getattr(params.lattice, "init_mode", "sparse")),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_c2": str(float(params.traffic.c2)),
                    "traffic_gamma": str(float(params.traffic.gamma)),
                    "traffic_dt": str(float(params.traffic.dt)),
                    "traffic_decay": str(float(params.traffic.decay)),
                },
            )

            pb = ProgressBar(total_units=int(ds_steps), label="work")
            pb.start()
            try:
                out_csv = run_double_slit(
                    params,
                    args,
                    out_dir=out_dir,
                    progress_cb=pb.advance,
                    provenance_header=csv_prov,
                )
            finally:
                pb.finish()
            return

        # Soliton scan (08A) — non-linear field sweep (sponge boundary; k/lambda knobs)
        # CLI flag name is `--soliton-scan` (preferred). Accept `--soliton` as an alias
        # because older presets used it and missing the flag silently falls through into the
        # default pipeline path (which is exactly what we do NOT want).
        if bool(getattr(args, "soliton_scan", False)) or bool(getattr(args, "soliton", False)):
            # We want to test for self-sustainment, so prefer sponge (radiation can leave).
            # Soliton runs should follow whatever field dynamics are selected by --traffic-mode.
            traffic_mode = str(getattr(params.traffic, "mode", "")).strip().lower()
            # Strict canonical validation: only allow "nonlinear" or "sine_gordon"
            if traffic_mode not in ("nonlinear", "sine_gordon"):
                raise ValueError("--soliton-scan requires --traffic-mode nonlinear or sine_gordon")

            # Select the evolve kernel for the chosen traffic_mode.
            evolve_fn = None
            if traffic_mode == "nonlinear":
                from traffic import evolve_nonlinear_traffic_steps as evolve_fn
            elif traffic_mode == "sine_gordon":
                from traffic import evolve_sine_gordon_traffic_steps as evolve_fn
            if evolve_fn is None:
                raise ValueError(f"soliton: unsupported traffic_mode={traffic_mode!r}")

            if str(getattr(params.traffic, "boundary_mode", "")).strip().lower() != "sponge":
                raise ValueError("--soliton-scan requires --traffic-boundary sponge")
            if int(getattr(params.traffic, "sponge_width", 0)) <= 0 or float(getattr(params.traffic, "sponge_strength", 0.0)) <= 0.0:
                raise ValueError("--soliton-scan requires sponge enabled (width>0, strength>0)")

            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)
            # Emit the same run headers as other modes so artefact paths are unambiguous.
            lp0 = _get_log_path(args)
            if lp0 != "":
                _log_line_only(args, "[run] log=%s" % (str(lp0),))
            _log_line_only(args, "[run] bundle=%s" % (str(run_dir),))
            _log_line_only(args, "[run] cwd=%s" % (str(os.getcwd()),))

            n = int(params.lattice.n)
            exp_name = _derive_exp_name(run_dir, default="08")

            out_csv = str(getattr(args, "soliton_out", "")).strip()
            if out_csv == "":
                out_csv = os.path.join(
                    csv_dir,
                    f"{exp_name}_n{int(n)}_{str(run_id)}.csv",
                )
            else:
                out_csv = str(out_csv)
                ensure_parent_dir(out_csv)
            _log_line_only(args, "[run] csv=%s" % (str(out_csv),))
            _log_line_only(args, "[soliton] mode=scan")

            # Prefer mapped dataclass params (args -> PipelineParams lives in cli.py),
            # but allow explicit CLI overrides when present.
            sp = getattr(params, "soliton")
            if sp is None:
                raise ValueError("soliton: params.soliton is missing (cli mapping did not populate it)")

            steps = int(getattr(args, "soliton_steps", int(getattr(sp, "steps", 2000))))
            sigma = float(getattr(args, "soliton_sigma", float(getattr(sp, "sigma", 4.0))))
            amp = float(getattr(args, "soliton_amp", float(getattr(sp, "amp", 100.0))))
            sigma_start = float(getattr(args, "soliton_sigma_start", float(getattr(sp, "sigma_start", sigma))))
            sigma_stop = float(getattr(args, "soliton_sigma_stop", float(getattr(sp, "sigma_stop", sigma))))
            sigma_steps = int(getattr(args, "soliton_sigma_steps", int(getattr(sp, "sigma_steps", 1))))

            init_vev = bool(getattr(args, "soliton_init_vev", bool(getattr(sp, "init_vev", False))))
            vev_sign = int(getattr(args, "soliton_vev_sign", int(getattr(sp, "vev_sign", +1))))
            if int(vev_sign) not in (-1, 1):
                raise ValueError("--soliton-vev-sign must be -1 or +1")

            # Mode-specific sweep wiring:
            # - nonlinear: sweep lambda (traffic_interaction), keep k fixed (traffic_potential)
            # - sine_gordon: sweep k (traffic_potential) and/or amp; lambda sweep is not used here
            k_stiff = float(getattr(args, "soliton_k", float(getattr(sp, "k", 0.0))))

            # Defaults for the legacy lambda sweep fields (used only for nonlinear mode).
            lam_start = float(getattr(args, "soliton_lambda_start", float(getattr(sp, "lambda_start", 0.0))))
            raw_lam_stop = getattr(args, "soliton_lambda_stop", None)
            raw_lam_steps = getattr(args, "soliton_lambda_steps", None)

            if traffic_mode == "sine_gordon":
                # For Sine-Gordon we treat lambda as unused; CLI validation should fail-loud if user sets it.
                lam_stop = 0.0
                lam_steps = 1

                sg_k_start = float(getattr(args, "soliton_sg_k_start", float(getattr(sp, "sg_k_start", k_stiff))))
                sg_k_stop = float(getattr(args, "soliton_sg_k_stop", float(getattr(sp, "sg_k_stop", sg_k_start))))
                sg_k_steps = int(getattr(args, "soliton_sg_k_steps", int(getattr(sp, "sg_k_steps", 1))))

                sg_amp_start = float(getattr(args, "soliton_sg_amp_start", float(getattr(sp, "sg_amp_start", amp))))
                sg_amp_stop = float(getattr(args, "soliton_sg_amp_stop", float(getattr(sp, "sg_amp_stop", sg_amp_start))))
                sg_amp_steps = int(getattr(args, "soliton_sg_amp_steps", int(getattr(sp, "sg_amp_steps", 1))))
            else:
                # Nonlinear mode: support single-point lambda runs.
                if raw_lam_stop is None:
                    lam_stop = float(lam_start)
                    lam_steps = 1
                else:
                    lam_stop = float(raw_lam_stop)
                    if raw_lam_steps is None:
                        lam_steps = int(getattr(sp, "lambda_steps", 25))
                    else:
                        lam_steps = int(raw_lam_steps)

                # Dummy values for SG-only knobs.
                sg_k_start = float(k_stiff)
                sg_k_stop = float(k_stiff)
                sg_k_steps = 1
                sg_amp_start = float(amp)
                sg_amp_stop = float(amp)
                sg_amp_steps = 1

            if steps < 10:
                raise ValueError("--soliton-steps must be >= 10")
            if not (np.isfinite(sigma) and sigma > 0.0):
                raise ValueError("--soliton-sigma must be > 0")
            if not np.isfinite(amp):
                raise ValueError("--soliton-amp must be finite")
            if not np.isfinite(k_stiff):
                raise ValueError("--soliton-k must be finite")

            if traffic_mode != "sine_gordon":
                if not (np.isfinite(lam_start) and np.isfinite(lam_stop)):
                    raise ValueError("--soliton-lambda-start/--soliton-lambda-stop must be finite")
                # Allow single-point lambda runs when start == stop.
                if float(lam_start) == float(lam_stop):
                    lam_steps = 1
                else:
                    if lam_steps < 2:
                        raise ValueError("--soliton-lambda-steps must be >= 2 when --soliton-lambda-stop != --soliton-lambda-start")
                if lam_stop < lam_start:
                    raise ValueError("--soliton-lambda-stop must be >= --soliton-lambda-start")
            else:
                # SG-specific sweep validation should already be enforced in cli.py; keep core checks minimal.
                if sg_k_steps < 1:
                    raise ValueError("--soliton-sg-k-steps must be >= 1")
                if sg_amp_steps < 1:
                    raise ValueError("--soliton-sg-amp-steps must be >= 1")

            sol_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_name),
                    "artefact": "soliton_scan",
                    "n": str(int(n)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_c2": str(float(params.traffic.c2)),
                    "traffic_gamma": str(float(params.traffic.gamma)),
                    "traffic_dt": str(float(params.traffic.dt)),
                    "traffic_decay": str(float(params.traffic.decay)),
                    "traffic_boundary": str(getattr(params.traffic, "boundary_mode", "")),
                    "traffic_sponge_width": str(int(getattr(params.traffic, "sponge_width", 0))),
                    "traffic_sponge_strength": str(float(getattr(params.traffic, "sponge_strength", 0.0))),
                    "soliton_k": str(float(k_stiff)),
                    "soliton_lambda_start": str(float(lam_start)),
                    "soliton_lambda_stop": str(float(lam_stop)),
                    "soliton_lambda_steps": str(int(lam_steps)),
                    "soliton_steps": str(int(steps)),
                    "soliton_sigma": str(float(sigma)),
                    "soliton_sigma_start": str(float(sigma_start)),
                    "soliton_sigma_stop": str(float(sigma_stop)),
                    "soliton_sigma_steps": str(int(sigma_steps)),
                    "soliton_amp": str(float(amp)),
                    "soliton_init_vev": str(int(init_vev)),
                    "soliton_vev_sign": str(int(vev_sign)),
                    "soliton_sg_k_start": str(float(sg_k_start)),
                    "soliton_sg_k_stop": str(float(sg_k_stop)),
                    "soliton_sg_k_steps": str(int(sg_k_steps)),
                    "soliton_sg_amp_start": str(float(sg_amp_start)),
                    "soliton_sg_amp_stop": str(float(sg_amp_stop)),
                    "soliton_sg_amp_steps": str(int(sg_amp_steps)),
                },
            )

            # Progress units:
            # - integration work: `steps` per sweep-point (this is what soliton.py reports)
            # - accounting work: post-evolve metrics + CSV write per sweep-point (not currently reported)
            #   We model this as a fixed overhead chunk after each sweep-point completes, so the bar does not
            #   hit 100% and then linger.
            # Sweep points (08): include sigma sweep as an outer axis.
            sweep_points = int(max(1, int(lam_steps)))
            if traffic_mode == "sine_gordon":
                sweep_points = int(max(1, int(sg_k_steps) * int(sg_amp_steps)))
            sweep_points = int(max(1, int(sweep_points) * int(max(1, sigma_steps))))

            # Heuristic: accounting is a noticeable fraction of the cost at moderate/large N.
            # Keep this deterministic and proportional to steps so it scales with longer runs.
            overhead_units = int(max(1, int(round(0.20 * float(max(1, steps))))))

            total_units = int(max(1, int(max(1, (int(steps) + int(overhead_units))) * int(max(1, sweep_points)))))
            pb = ProgressBar(total_units=total_units, label="work")
            pb.start()

            # Wrap soliton progress so we can inject the per-sweep accounting overhead.
            # soliton.py reports only evolve steps (and an initial land tick). We detect sweep-point completion
            # once reported integration reaches `steps`, then add `overhead_units`.
            _sp_accum = 0
            def _soliton_progress(delta: int) -> None:
                nonlocal _sp_accum
                di = int(delta)
                if di <= 0:
                    return
                pb.advance(di)
                _sp_accum += di
                # A sweep-point is complete once we've accounted for >= `steps` integration ticks.
                # Carry any spill into the next sweep-point.
                while _sp_accum >= int(steps) and int(steps) > 0:
                    pb.advance(int(overhead_units))
                    _sp_accum -= int(steps)
            try:
                # Import only when needed to keep core usable if the module is absent.
                from soliton import run_soliton_scan

                # Optional SG initial condition path: expansion.py “big bang” initialisation.
                # SG-only; nonlinear mode remains unchanged.
                use_bigbang = bool(getattr(args, "sg_bigbang", False))
                init_phi = None
                init_vel = None
                init_info = None
                if use_bigbang:
                    if traffic_mode != "sine_gordon":
                        raise ValueError("--sg-bigbang is only valid with --traffic-mode sine_gordon")
                    try:
                        import expansion as _expansion
                        make_bigbang_ic = getattr(_expansion, "make_bigbang_ic", None)
                        if make_bigbang_ic is None or (not callable(make_bigbang_ic)):
                            raise ImportError("sg-bigbang: expansion.py does not define a callable make_bigbang_ic")
                    except Exception as e:
                        raise ImportError("sg-bigbang: failed to load make_bigbang_ic from expansion.py") from e

                    # Pass through any `--sg-bigbang-*` knobs if the CLI exposes them.
                    # We stay signature-aware to keep this fail-loud and single-path.
                    bb_vals: dict[str, Any] = {
                        "n": int(n),
                        "seed": int(getattr(args, "seed", 0)),
                    }

                    # Optional controls (only included when present on args).
                    for nm in (
                        "sg_bigbang_levels",
                        "sg_bigbang_bits",
                        "sg_bigbang_quant",
                        "sg_bigbang_resample",
                        "sg_bigbang_bias",
                        "sg_bigbang_drift",
                        "sg_bigbang_sigma",
                    ):
                        if hasattr(args, nm):
                            v = getattr(args, nm)
                            if v is not None:
                                # Strip argparse empty-string defaults where relevant.
                                if isinstance(v, str) and v.strip() == "":
                                    continue
                                bb_vals[nm.replace("sg_bigbang_", "")] = v

                    try:
                        sig_bb = inspect.signature(cast(Callable[..., Any], make_bigbang_ic))
                        bb_kwargs: dict[str, Any] = {}
                        for k0, v0 in bb_vals.items():
                            if k0 in sig_bb.parameters:
                                bb_kwargs[k0] = v0
                        bb_ret = make_bigbang_ic(**bb_kwargs)
                        if (not isinstance(bb_ret, tuple)) or (len(bb_ret) != 3):
                            raise ValueError("sg-bigbang: make_bigbang_ic must return (phi, vel, info)")
                        init_phi, init_vel, init_info = bb_ret
                    except (TypeError, ValueError):
                        # If we cannot introspect, call with the minimal surface.
                        bb_ret = make_bigbang_ic(
                            n=int(n),
                            seed=int(getattr(args, "seed", 0)),
                        )
                        if (not isinstance(bb_ret, tuple)) or (len(bb_ret) != 3):
                            raise ValueError("sg-bigbang: make_bigbang_ic must return (phi, vel, info)")
                        init_phi, init_vel, init_info = bb_ret

                    # Emit a single stable log line for provenance/debug without bloating CSV.
                    try:
                        if isinstance(init_info, dict) and init_info:
                            keys = sorted([str(k) for k in init_info.keys()])
                            _log_line_only(args, "[soliton] sg_bigbang=1 init_info_keys=%s" % (",".join(keys),))
                        else:
                            _log_line_only(args, "[soliton] sg_bigbang=1")
                    except Exception:
                        pass

                dump_hdf5_path = str(getattr(args, "dump_hdf5", "") or "").strip()
                dump_final_hdf5 = (dump_hdf5_path != "")

                # Sprite-only runs may not request a final snapshot, but soliton.py still needs a base name
                # so it can export *_besttail.h5. We persist the effective base here for the teardown exporter.
                dump_sprite_path = str(getattr(args, "dump_sprite", "") or "").strip()
                dump_hdf5_base_for_sprite = dump_hdf5_path
                if dump_sprite_path != "" and dump_hdf5_base_for_sprite == "":
                    root, _ext = os.path.splitext(str(out_csv))
                    dump_hdf5_base_for_sprite = f"{root}.h5"
                try:
                    setattr(args, "_sprite_hdf5_base", str(dump_hdf5_base_for_sprite))
                except Exception:
                    pass

                # --- Sprite post-processing: CSV gate-fail inference ---
                # If --dump-sprite was requested, wire up besttail gate fail logic for post-processing.
                if dump_sprite_path != "":
                    def _bt_fnum(x):
                        try:
                            return float(x)
                        except Exception:
                            return float("nan")

                    def _bt_fmt_csv_gate(rec: dict[str, str], keys: list[str]) -> str:
                        return ",".join([str(rec.get(k, "")) for k in keys])

                    def _infer_gate_fail(rec: dict[str, str]) -> list[str]:
                        # Best-effort: infer which SG besttail gate(s) likely failed.
                        # Prefer thresholds from the active soliton config (params.soliton); fall back to soliton.py constants if present.
                        bt_ok = str(rec.get("besttail_ok", "")).strip().lower()
                        if bt_ok in ("1", "true", "yes"):
                            return []

                        out: list[str] = []

                        # Prefer core metrics if present; otherwise fall back to peak-patch metrics.
                        vabs = _bt_fnum(rec.get("tailc_vel_abs_max", rec.get("besttail_vel_abs_max", rec.get("tailp_vel_abs_max", ""))))
                        vr = _bt_fnum(rec.get("tailc_vel_rms_mean", rec.get("tailp_vel_rms_mean", rec.get("tail_vel_rms_mean", ""))))
                        de = _bt_fnum(rec.get("tailc_dE_phys_shifted_per_step_abs_max", rec.get("tail_dE_phys_shifted_per_step_abs_max", "")))
                        fpi = _bt_fnum(rec.get("tailp_frac_abs_phi_gt_pi_max", rec.get("tail_frac_abs_phi_gt_pi_max", "")))
                        f2 = _bt_fnum(rec.get("tailp_frac_abs_phi_gt_2pi_max", rec.get("tail_frac_abs_phi_gt_2pi_max", "")))
                        peak_dev = _bt_fnum(rec.get("tailp_peak_dev_max", rec.get("tail_peak_dev_max", "")))
                        cm_r_frac = _bt_fnum(rec.get("tail_cm_r_frac_half", rec.get("tail_cm_r_frac_crop", "")))
                        peak_r_frac = _bt_fnum(rec.get("tail_peak_r_frac_half", rec.get("tail_peak_r_frac_outer", "")))

                        # Thresholds: take from params.soliton when present.
                        # (These match soliton.py override names.)
                        vabs_max = float(getattr(sp, "sg_besttail_vel_abs_max", float("nan")))
                        vr_max = float(getattr(sp, "sg_besttail_vel_rms_mean_max", float("nan")))
                        de_max = float(getattr(sp, "sg_besttail_dE_shifted_per_step_abs_max", float("nan")))
                        fpi_max = float(getattr(sp, "sg_besttail_frac_abs_phi_gt_pi_max", float("nan")))
                        f2_max = float(getattr(sp, "sg_besttail_frac_abs_phi_gt_2pi_max", float("nan")))
                        peak_min = float(getattr(sp, "sg_besttail_peak_dev_min", float("nan")))
                        cm_max = float(getattr(sp, "sg_besttail_cm_r_frac_half_max", getattr(sp, "sg_besttail_cm_r_frac_crop_max", float("nan"))))
                        peak_max = float(getattr(sp, "sg_besttail_peak_r_frac_half_max", getattr(sp, "sg_besttail_peak_r_frac_outer_max", float("nan"))))

                        # Fall back to soliton.py constants only if any threshold is NaN.
                        if (not np.isfinite(vabs_max)) or (not np.isfinite(vr_max)) or (not np.isfinite(de_max)) or (not np.isfinite(fpi_max)) or (not np.isfinite(f2_max)) or (not np.isfinite(peak_min)) or (not np.isfinite(cm_max)) or (not np.isfinite(peak_max)):
                            try:
                                import soliton as _sol
                                if not np.isfinite(vr_max):
                                    vr_max = float(getattr(_sol, "SG_BESTTAIL_MAX_VEL_RMS_MEAN", vr_max))
                                if not np.isfinite(de_max):
                                    de_max = float(getattr(_sol, "SG_BESTTAIL_MAX_DE_SHIFTED_PER_STEP_ABS", de_max))
                                if not np.isfinite(fpi_max):
                                    fpi_max = float(getattr(_sol, "SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_PI", fpi_max))
                                if not np.isfinite(f2_max):
                                    f2_max = float(getattr(_sol, "SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_2PI", f2_max))
                                if not np.isfinite(peak_min):
                                    peak_min = float(getattr(_sol, "SG_BESTTAIL_MIN_PEAK_DEV", peak_min))
                                if not np.isfinite(cm_max):
                                    cm_max = float(getattr(_sol, "SG_BESTTAIL_MAX_CM_R_FRAC_HALF", getattr(_sol, "SG_BESTTAIL_MAX_CM_R_FRAC_CROP", cm_max)))
                                if not np.isfinite(peak_max):
                                    peak_max = float(getattr(_sol, "SG_BESTTAIL_MAX_PEAK_R_FRAC_HALF", getattr(_sol, "SG_BESTTAIL_MAX_PEAK_R_FRAC_OUTER", peak_max)))
                            except Exception:
                                pass

                        if np.isfinite(vabs) and np.isfinite(vabs_max) and float(vabs) > float(vabs_max):
                            out.append(f"vel_abs_max>{vabs_max:g}")
                        if np.isfinite(vr) and np.isfinite(vr_max) and float(vr) > float(vr_max):
                            out.append(f"vel_rms_mean>{vr_max:g}")
                        if np.isfinite(de) and np.isfinite(de_max) and float(de) > float(de_max):
                            out.append(f"dE_shifted_abs>{de_max:g}")
                        if np.isfinite(fpi) and np.isfinite(fpi_max) and float(fpi) > float(fpi_max):
                            out.append(f"frac_|phi|>pi>{fpi_max:g}")
                        if np.isfinite(f2) and np.isfinite(f2_max) and float(f2) > float(f2_max):
                            out.append(f"frac_|phi|>2pi>{f2_max:g}")
                        if np.isfinite(peak_dev) and np.isfinite(peak_min) and float(peak_dev) < float(peak_min):
                            out.append(f"peak_dev<{peak_min:g}")
                        if np.isfinite(cm_r_frac) and np.isfinite(cm_max) and float(cm_r_frac) > float(cm_max):
                            out.append(f"cm_r_frac>{cm_max:g}")
                        if np.isfinite(peak_r_frac) and np.isfinite(peak_max) and float(peak_r_frac) > float(peak_max):
                            out.append(f"peak_r_frac>{peak_max:g}")

                        return out

                call_vals: dict[str, Any] = {
                    "n": int(n),
                    "steps": int(steps),
                    "sigma": float(sigma),
                    "sigma_start": float(sigma_start),
                    "sigma_stop": float(sigma_stop),
                    "sigma_steps": int(sigma_steps),
                    "amp": float(amp),
                    "k": float(k_stiff),
                    "lambda_start": float(lam_start),
                    "lambda_stop": float(lam_stop),
                    "lambda_steps": int(lam_steps),
                    "out_csv": str(out_csv),
                    "dump_hdf5_path": str(dump_hdf5_path),
                    "dump_sprite": bool(getattr(args, "dump_sprite", False)),
                    "dump_final_hdf5": bool(dump_final_hdf5),
                    "traffic": params.traffic,
                    "soliton_cfg": sp,
                    "evolve_fn": evolve_fn,
                    "provenance_header": str(sol_prov),
                    "progress_cb": _soliton_progress,
                    "progress": _soliton_progress,
                    "init_vev": bool(init_vev),
                    "vev_sign": int(vev_sign),
                    "sg_k_start": float(sg_k_start),
                    "sg_k_stop": float(sg_k_stop),
                    "sg_k_steps": int(sg_k_steps),
                    "sg_amp_start": float(sg_amp_start),
                    "sg_amp_stop": float(sg_amp_stop),
                    "sg_amp_steps": int(sg_amp_steps),
                }

                if use_bigbang:
                    call_vals["init_phi"] = init_phi
                    call_vals["init_vel"] = init_vel
                    call_vals["init_info"] = init_info

                _log_line_only(args, "[soliton] init_vev=%d vev_sign=%+d" % (int(init_vev), int(vev_sign)))

                try:
                    sig_fn = inspect.signature(cast(Callable[..., Any], run_soliton_scan))
                    call_kwargs: dict[str, Any] = {}
                    for k0, v0 in call_vals.items():
                        if k0 in sig_fn.parameters:
                            call_kwargs[k0] = v0
                    if use_bigbang:
                        if ("init_phi" not in sig_fn.parameters) or ("init_vel" not in sig_fn.parameters):
                            raise ValueError("soliton: sg-bigbang requested but run_soliton_scan does not accept init_phi/init_vel; wire these into soliton.py")
                    run_soliton_scan(**call_kwargs)
                except (TypeError, ValueError):
                    if use_bigbang:
                        raise ValueError("soliton: sg-bigbang requested but run_soliton_scan signature could not be introspected; cannot safely pass init_phi/init_vel")
                    run_soliton_scan(
                        n=int(n),
                        steps=int(steps),
                        sigma=float(sigma),
                        sigma_start=float(sigma_start),
                        sigma_stop=float(sigma_stop),
                        sigma_steps=int(sigma_steps),
                        amp=float(amp),
                        k=float(k_stiff),
                        lambda_start=float(lam_start),
                        lambda_stop=float(lam_stop),
                        lambda_steps=int(lam_steps),
                        out_csv=str(out_csv),
                        dump_hdf5_path=str(dump_hdf5_path),
                        dump_sprite=bool(getattr(args, "dump_sprite", False)),
                        dump_final_hdf5=bool(dump_final_hdf5),
                        traffic=params.traffic,
                        evolve_fn=evolve_fn,
                        provenance_header=str(sol_prov),
                        progress_cb=_soliton_progress,
                        init_vev=bool(init_vev),
                        vev_sign=int(vev_sign),
                        sg_k_start=float(sg_k_start),
                        sg_k_stop=float(sg_k_stop),
                        sg_k_steps=int(sg_k_steps),
                        sg_amp_start=float(sg_amp_start),
                        sg_amp_stop=float(sg_amp_stop),
                        sg_amp_steps=int(sg_amp_steps),
                    )
            finally:
                pb.finish()

            try:
                setattr(args, "_soliton_out_csv", str(out_csv))
            except Exception:
                pass
            _log_line_only(args, "[soliton] wrote %s" % (str(out_csv),))
            return

        # Quantum corral mode (frequency sweep; confined telegraph field)
        if bool(getattr(args, "corral", False)):
            if str(params.traffic.mode).strip().lower() != "telegraph":
                raise ValueError("--corral requires --traffic-mode telegraph")

            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)

            n = int(params.lattice.n)
            radius = int(getattr(args, "corral_radius", 32))
            geom = str(getattr(args, "corral_geom", "sphere"))
            omega_start = float(getattr(args, "corral_omega_start", 0.10))
            omega_stop = float(getattr(args, "corral_omega_stop", 0.60))
            omega_steps = int(getattr(args, "corral_omega_steps", 50))
            burn = int(getattr(args, "corral_burn", 1000))
            warm = int(getattr(args, "corral_warm", 0))
            burn_frac = float(getattr(args, "corral_burn_frac", 0.90))
            amp = float(getattr(args, "corral_amp", 1.0))
            cx_off = int(getattr(args, "corral_cx_off", 0))
            cy_off = int(getattr(args, "corral_cy_off", 0))
            cz_off = int(getattr(args, "corral_cz_off", 0))
            out_csv = str(getattr(args, "corral_out", "")).strip()

            # Hoist exp_name derivation for consistency and reuse
            exp_name = _derive_exp_name(run_dir, default="04")

            if radius < 2 or radius >= (n // 2):
                raise ValueError("--corral-radius must be in [2, n//2)")
            if not (np.isfinite(omega_start) and np.isfinite(omega_stop)):
                raise ValueError("--corral-omega-start/--corral-omega-stop must be finite")
            if omega_start <= 0.0 or omega_stop <= 0.0 or omega_stop <= omega_start:
                raise ValueError("--corral-omega-stop must be > --corral-omega-start and both > 0")
            if omega_steps < 2:
                raise ValueError("--corral-omega-steps must be >= 2")
            if burn < 1:
                raise ValueError("--corral-burn must be >= 1")
            if warm < 0:
                raise ValueError("--corral-warm must be >= 0")
            if not np.isfinite(burn_frac) or burn_frac <= 0.0 or burn_frac > 1.0:
                raise ValueError("--corral-burn-frac must be in (0, 1]")
            if not np.isfinite(amp):
                raise ValueError("--corral-amp must be finite")

            if out_csv == "":
                out_csv = os.path.join(
                    csv_dir,
                    f"{exp_name}_n{int(n)}_{str(run_id)}.csv",
                )
            else:
                out_csv = str(out_csv)
                ensure_parent_dir(out_csv)

            # Provenance header (match 01/02/03): embed run identity in the CSV.
            corral_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_name),
                    "artefact": "corral",
                    "n": str(int(n)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "lattice_init": str(getattr(params.lattice, "init_mode", "sparse")),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_c2": str(float(params.traffic.c2)),
                    "traffic_gamma": str(float(params.traffic.gamma)),
                    "traffic_dt": str(float(params.traffic.dt)),
                    "traffic_decay": str(float(params.traffic.decay)),
                    "corral_geom": str(geom),
                    "corral_radius": str(int(radius)),
                    "corral_omega_start": str(float(omega_start)),
                    "corral_omega_stop": str(float(omega_stop)),
                    "corral_omega_steps": str(int(omega_steps)),
                    "corral_burn": str(int(burn)),
                    "corral_warm": str(int(warm)),
                    "corral_burn_frac": str(float(burn_frac)),
                    "corral_amp": str(float(amp)),
                    "corral_center_offset": f"{int(cx_off)},{int(cy_off)},{int(cz_off)}",
                },
            )

            pb = ProgressBar(total_units=int(max(1, omega_steps)), label="work")
            pb.start()
            try:
                run_quantum_corral_sweep(
                    n=int(n),
                    radius=int(radius),
                    omega_start=float(omega_start),
                    omega_stop=float(omega_stop),
                    omega_steps=int(omega_steps),
                    burn_in=int(burn),
                    warm_steps=int(warm),
                    out_csv=str(out_csv),
                    traffic=params.traffic,
                    geom=str(geom),
                    burn_frac=float(burn_frac),
                    amp=float(amp),
                    center_offset=(int(cx_off), int(cy_off), int(cz_off)),
                    provenance_header=str(corral_prov),
                    progress_cb=pb.advance,
                )
            finally:
                pb.finish()
            _log_line_only(args, "[corral] wrote %s" % (str(out_csv),))
            return

        # Ringdown sweep (06B) — passive resonance scan (single pulse, Dirichlet box)
        if bool(getattr(args, "ringdown", False)):
            if str(params.traffic.mode).strip().lower() != "telegraph":
                raise ValueError("--ringdown requires --traffic-mode telegraph")

            # For 06B we want a hard box: Dirichlet clamp on faces.
            if str(getattr(params.traffic, "boundary_mode", "")).strip().lower() != "zero":
                raise ValueError("--ringdown requires --traffic-boundary zero")
            if int(getattr(params.traffic, "sponge_width", 0)) != 0 or float(getattr(params.traffic, "sponge_strength", 0.0)) != 0.0:
                raise ValueError("--ringdown requires sponge disabled (width=0, strength=0)")

            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)

            n = int(params.lattice.n)

            # Derive experiment code from the parent folder (e.g. "06B_...").
            exp_name = _derive_exp_name(run_dir, default="06")

            # Output policy:
            # - default: write under the run bundle `_csv/` directory
            # - override: if --ringdown-out is provided, honour it:
            #     * absolute path: use as-is
            #     * bare filename: write under the bundle `_csv/` directory
            #     * relative path: resolve under the run bundle root
            out_csv = str(getattr(args, "ringdown_out", "")).strip()
            if out_csv == "":
                out_csv = os.path.join(
                    csv_dir,
                    f"{exp_name}_n{int(n)}_{str(run_id)}.csv",
                )
            else:
                out_csv = str(out_csv)
                if not os.path.isabs(out_csv):
                    # Bare filename => keep all CSV artefacts under `_csv/`.
                    if (os.path.sep not in out_csv) and (os.path.altsep is None or os.path.altsep not in out_csv):
                        out_csv = os.path.join(csv_dir, out_csv)
                    else:
                        # Relative path with folders => resolve under the run bundle.
                        out_csv = os.path.join(run_dir, out_csv)
                ensure_parent_dir(out_csv)

            # Build params from args (args->PipelineParams mapping lives in cli.py).
            rp = getattr(params, "ringdown")
            steps = int(getattr(rp, "steps", 1000))
            sigma_start = float(getattr(rp, "sigma_start", 1.2))
            sigma_stop = float(getattr(rp, "sigma_stop", 6.0))
            sigma_step = float(getattr(rp, "sigma_step", 0.2))
            probe_window = int(getattr(rp, "probe_window", 500))
            pulse_amp = float(getattr(rp, "pulse_amp", 50.0))

            if steps < 10:
                raise ValueError("--ringdown-steps must be >= 10")
            if not (np.isfinite(sigma_start) and np.isfinite(sigma_stop) and np.isfinite(sigma_step)):
                raise ValueError("--ringdown-sigma-* must be finite")
            if sigma_step <= 0.0:
                raise ValueError("--ringdown-sigma-step must be > 0")
            if sigma_stop < sigma_start:
                raise ValueError("--ringdown-sigma-stop must be >= --ringdown-sigma-start")
            # Allow single-point sigma runs (start == stop). sigma_step remains required > 0.
            if probe_window < 16 or probe_window > steps:
                raise ValueError("--ringdown-probe-window must be in [16, ringdown_steps]")
            if not np.isfinite(pulse_amp):
                raise ValueError("--ringdown-pulse-amp must be finite")

            # Construct the sigma list deterministically.
            sigmas: list[float] = []
            s = float(sigma_start)
            s_stop = float(sigma_stop)
            step_s = float(sigma_step)
            # Include stop if we land exactly; otherwise stop strictly before.
            while s <= (s_stop + 1e-12):
                sigmas.append(float(s))
                s += step_s
            if len(sigmas) < 1:
                raise ValueError("ringdown: sigma sweep produced no values")

            ring_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_name),
                    "artefact": "ringdown_sigma_sweep",
                    "n": str(int(n)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_c2": str(float(params.traffic.c2)),
                    "traffic_gamma": str(float(params.traffic.gamma)),
                    "traffic_dt": str(float(params.traffic.dt)),
                    "traffic_decay": str(float(params.traffic.decay)),
                    "traffic_boundary": str(getattr(params.traffic, "boundary_mode", "")),
                    "ringdown_steps": str(int(steps)),
                    "pulse_amp": str(float(pulse_amp)),
                    "ringdown_norm": "Efinal_over_Eland",
                    "ringdown_note_esrc": "E_src_target is a per-sigma normalisation target: min(unscaled sum(src^2)) across the sweep; src is scaled so sum(src^2)=E_src_target (never amplifies).",
                    "ringdown_note_eland": "E_land is total grid energy after a short post-injection settle (land_tick) and is used to cancel injection coupling; primary stability metric is E_final / E_land.",
                    "ringdown_csv_columns": "sigma,E_src_target,src_scale,src_peak,E_land,E_final,Efinal_over_Eland,Efinal_over_Esrc_target,probe_peak_abs,freq_peak,amp_peak",
                    "ringdown_sigma_start": str(float(sigma_start)),
                    "ringdown_sigma_stop": str(float(sigma_stop)),
                    "ringdown_sigma_step": str(float(sigma_step)),
                    "ringdown_probe_window": str(int(probe_window)),
                    "ringdown_out": str(getattr(args, "ringdown_out", "")),
                },
            )

            pb = ProgressBar(total_units=int(max(1, ringdown_work_units(int(steps), int(len(sigmas))))), label="work")
            pb.start()
            try:
                # Build call kwargs in a signature-aware way (fail-loud, single path).
                call_vals: dict[str, Any] = {
                    "n": int(n),
                    "steps": int(steps),
                    "sigmas": list(sigmas),
                    "sigma_start": float(sigma_start),
                    "sigma_stop": float(sigma_stop),
                    "sigma_step": float(sigma_step),
                    "pulse_amp": float(pulse_amp),
                    "probe_window": int(probe_window),
                    "out_csv": str(out_csv),
                    "traffic": params.traffic,
                    "provenance_header": str(ring_prov),
                    "progress_cb": pb.advance,
                    "progress": pb.advance,
                }

                try:
                    sig_fn = inspect.signature(cast(Callable[..., Any], run_ringdown_sweep_sigma))
                    call_kwargs: dict[str, Any] = {}
                    for k, v in call_vals.items():
                        if k in sig_fn.parameters:
                            call_kwargs[k] = v
                    run_ringdown_sweep_sigma(**call_kwargs)
                except (TypeError, ValueError):
                    # If we cannot introspect, call with the canonical isotropy-style surface.
                    run_ringdown_sweep_sigma(
                        n=int(n),
                        steps=int(steps),
                        sigmas=list(sigmas),
                        pulse_amp=float(pulse_amp),
                        probe_window=int(probe_window),
                        out_csv=str(out_csv),
                        traffic=params.traffic,
                        provenance_header=str(ring_prov),
                        progress_cb=pb.advance,
                    )
            finally:
                pb.finish()

            _log_line_only(args, "[ringdown] wrote %s" % (str(out_csv),))
            return

        # Isotropy calibration (axis vs diagonals)
        if bool(getattr(args, "isotropy", False)):
            if str(params.traffic.mode).strip().lower() != "telegraph":
                raise ValueError("--isotropy requires --traffic-mode telegraph")

            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)

            n = int(params.lattice.n)
            R = int(getattr(args, "iso_R", 0))
            steps = int(getattr(args, "iso_steps", 0))
            sigma = float(getattr(args, "iso_sigma", 0.0))
            amp = float(getattr(args, "iso_amp", 0.0))
            out_csv = str(getattr(args, "iso_out", "")).strip()
            # Sweep arguments
            do_sweep = bool(getattr(args, "iso_sweep", False))
            sigma_start = float(getattr(args, "iso_sigma_start", 1.5))
            sigma_stop = float(getattr(args, "iso_sigma_stop", 8.0))
            sigma_steps = int(getattr(args, "iso_sigma_steps", 20))

            if R <= 0:
                R = max(8, (n // 2) - 12)
            if steps <= 0:
                steps = max(200, int(3 * R))

            if R < 2 or R >= (n // 2):
                raise ValueError("--iso-R must be in [2, n//2)")
            if steps < 10:
                raise ValueError("--iso-steps must be >= 10")
            if not np.isfinite(sigma):
                raise ValueError("--iso-sigma must be finite")
            if not np.isfinite(amp):
                raise ValueError("--iso-amp must be finite")
            if do_sweep:
                if not (np.isfinite(sigma_start) and np.isfinite(sigma_stop)):
                    raise ValueError("--iso-sigma-start/--iso-sigma-stop must be finite")
                if sigma_stop <= sigma_start:
                    raise ValueError("--iso-sigma-stop must be > --iso-sigma-start")
                if sigma_steps < 2:
                    raise ValueError("--iso-sigma-steps must be >= 2")

            # Derive experiment code from the parent folder (e.g. "05A_...").
            exp_name = _derive_exp_name(run_dir, default="05")

            if out_csv == "":
                out_csv = os.path.join(
                    csv_dir,
                    f"{exp_name}_n{int(n)}_{str(run_id)}.csv",
                )
            else:
                out_csv = str(out_csv)
                ensure_parent_dir(out_csv)

            iso_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_name),
                    "artefact": ("isotropy_sigma_sweep" if do_sweep else "isotropy"),
                    "n": str(int(n)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "lattice_init": str(getattr(params.lattice, "init_mode", "sparse")),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_c2": str(float(params.traffic.c2)),
                    "traffic_gamma": str(float(params.traffic.gamma)),
                    "traffic_dt": str(float(params.traffic.dt)),
                    "traffic_decay": str(float(params.traffic.decay)),
                    "iso_R": str(int(R)),
                    "iso_steps": str(int(steps)),
                    "iso_sigma": str(float(sigma)),
                    "iso_sweep": ("1" if do_sweep else "0"),
                    "iso_sigma_start": str(float(sigma_start)),
                    "iso_sigma_stop": str(float(sigma_stop)),
                    "iso_sigma_steps": str(int(sigma_steps)),
                    "iso_amp": str(float(amp)),
                },
            )

            total_units = int(max(1, steps))
            if do_sweep:
                total_units = int(max(1, int(steps) * int(max(1, sigma_steps))))
            pb = ProgressBar(total_units=int(max(1, total_units)), label="work")
            pb.start()
            try:
                kw: dict[str, Any] = {}
                try:
                    fn_iso: Callable[..., Any] = (run_isotropy_sigma_sweep if do_sweep else run_isotropy_test)
                    sig_iso = inspect.signature(cast(Callable[..., Any], fn_iso))
                    if "provenance_header" in sig_iso.parameters:
                        kw["provenance_header"] = str(iso_prov)
                    if "progress_cb" in sig_iso.parameters:
                        kw["progress_cb"] = pb.advance
                    elif "progress" in sig_iso.parameters:
                        kw["progress"] = pb.advance
                except (TypeError, ValueError):
                    kw = {}

                if do_sweep:
                    sigmas = np.linspace(float(sigma_start), float(sigma_stop), int(sigma_steps), dtype=np.float64).tolist()
                    run_isotropy_sigma_sweep(
                        n=n,
                        R=R,
                        steps=steps,
                        sigmas=sigmas,
                        amp=amp,
                        out_csv=str(out_csv),
                        traffic=params.traffic,
                        **kw,
                    )
                else:
                    run_isotropy_test(
                        n=n,
                        R=R,
                        steps=steps,
                        sigma=sigma,
                        amp=amp,
                        out_csv=str(out_csv),
                        traffic=params.traffic,
                        **kw,
                    )
            finally:
                pb.finish()

            print("[isotropy] wrote %s" % (str(out_csv),))
            return

        # Relativity (light clock / twin paradox)
        if bool(getattr(args, "relativity", False)):
            if str(params.traffic.mode).strip().lower() != "telegraph":
                raise ValueError("--relativity requires --traffic-mode telegraph")

            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)

            n = int(params.lattice.n)
            steps = int(getattr(args, "rel_steps", 1600))
            L = int(getattr(args, "rel_L", 48))
            v_frac = float(getattr(args, "rel_v_frac", 0.30))
            v_ref_frac = float(getattr(args, "rel_v_frac_ref", 0.10))
            slab = int(getattr(args, "rel_slab", 1))
            amp = float(getattr(args, "rel_amp", 50.0))
            sigma = float(getattr(args, "rel_sigma", 1.75))
            threshold = float(getattr(args, "rel_threshold", 0.02))
            refractory = int(getattr(args, "rel_refractory", 10))
            rel_detect = str(getattr(args, "rel_detect", "first_cross"))
            rel_start_threshold = float(getattr(args, "rel_start_threshold", 0.01))
            rel_peak_window = int(getattr(args, "rel_peak_window", 12))
            rel_accept_threshold = float(getattr(args, "rel_accept_threshold", -1.0))
            out_csv = str(getattr(args, "rel_out", "")).strip()

            if steps < 10:
                raise ValueError("--rel-steps must be >= 10")
            if L < 2 or L >= (n // 2):
                raise ValueError("--rel-L must be in [2, n//2)")
            if not (np.isfinite(v_frac) and (v_frac > 0.0) and (v_frac < 0.999999)):
                raise ValueError("--rel-v-frac must be in (0, 1)")
            if not (np.isfinite(v_ref_frac) and (v_ref_frac > 0.0) and (v_ref_frac < 0.999999)):
                raise ValueError("--rel-v-frac-ref must be in (0, 1)")
            if slab < 1:
                raise ValueError("--rel-slab must be >= 1")
            if not np.isfinite(amp):
                raise ValueError("--rel-amp must be finite")
            if not (np.isfinite(sigma) and (sigma > 0.0)):
                raise ValueError("--rel-sigma must be > 0")
            if not (np.isfinite(threshold) and (threshold > 0.0)):
                raise ValueError("--rel-threshold must be > 0")
            if refractory < 0:
                raise ValueError("--rel-refractory must be >= 0")
            if rel_detect not in ("first_cross", "window_peak"):
                raise ValueError("--rel-detect must be one of: first_cross, window_peak")
            if not (np.isfinite(rel_start_threshold) and (rel_start_threshold > 0.0) and (rel_start_threshold < 1.0)):
                raise ValueError("--rel-start-threshold must be in (0, 1)")
            if rel_peak_window < 1 or rel_peak_window > 200:
                raise ValueError("--rel-peak-window must be in [1, 200]")
            if not np.isfinite(rel_accept_threshold):
                raise ValueError("--rel-accept-threshold must be finite")
            if rel_accept_threshold >= 1.0:
                raise ValueError("--rel-accept-threshold must be < 1.0 (or <0 to disable)")

            # Derive experiment code from the parent folder (e.g. "06Z_...").
            exp_name = _derive_exp_name(run_dir, default="06")

            if out_csv == "":
                out_csv = os.path.join(
                    csv_dir,
                    f"{exp_name}_n{int(n)}_{str(run_id)}.csv",
                )
            else:
                out_csv = str(out_csv)
                ensure_parent_dir(out_csv)

            # Resolve a concrete log path for this run. Some runners (and some argparse setups)
            # don't expose it under a stable attribute name, so we pin it here for both core
            # and relativity.py logging.
            lp = _get_log_path(args)
            if lp == "":
                lp = os.path.join(log_dir, f"{exp_name}_n{int(n)}_{str(run_id)}.log")
                try:
                    setattr(args, "log_path", str(lp))
                except Exception:
                    pass
            # Keep header lines stable for paper logs (write to run log, not terminal).
            c_eff = float(math.sqrt(float(params.traffic.c2)) * float(params.traffic.dt))
            _log_line_only(args, "[relativity] n=%d steps=%d L=%d v_frac=%.6g v_ref_frac=%.6g c_eff=%.6g slab=%d" % (
                int(n), int(steps), int(L), float(v_frac), float(v_ref_frac), float(c_eff), int(slab),
            ))
            _log_line_only(args, "[relativity] pulse: amp=%.6g sigma=%.6g threshold=%.6g refractory=%d detect=%s start_thr=%.6g peak_window=%d accept_thr=%.6g" % (
                float(amp), float(sigma), float(threshold), int(refractory), str(rel_detect), float(rel_start_threshold), int(rel_peak_window), float(rel_accept_threshold),
            ))
            _log_line_only(args, "[relativity] out_csv=%s" % (str(out_csv),))

            # Map CLI args onto RelativityParams (module owns the schema).
            from relativity import RelativityParams

            v_abs = float(v_frac) * float(c_eff)
            v_ref_abs = float(v_ref_frac) * float(c_eff)
            if float(abs(v_ref_abs)) >= float(c_eff):
                raise ValueError("--rel-v-frac-ref must satisfy |v_ref| < c_eff")
            p_rel = RelativityParams(
                n=int(n),
                steps=int(steps),
                c2=float(params.traffic.c2),
                gamma=float(params.traffic.gamma),
                dt=float(params.traffic.dt),
                decay=float(params.traffic.decay),
                mirror_sep=int(L),
                v=float(v_abs),
                v_ref=float(v_ref_abs),
                slab_half_thickness=max(1, int(slab) // 2),
                pulse_amp=float(amp),
                pulse_sigma=float(sigma),
                detect_threshold=float(threshold),
                refractory=int(refractory),
                detect_mode=str(rel_detect),
                start_threshold=float(rel_start_threshold),
                peak_window=int(rel_peak_window),
                accept_threshold=float(rel_accept_threshold),
                out_csv=str(out_csv),
            )

            # Preflight: exact max safe steps before the moving mirror hits bounds.
            # Align this with relativity.py: max_dx = n - 2*margin - 5; max_steps = floor(max_dx / |v|).
            v_abs_mag = float(abs(float(p_rel.v)))
            margin = int(getattr(p_rel, "margin", 0))
            if margin < 0:
                raise ValueError("relativity: margin must be >= 0")
            max_dx = int(n) - (2 * int(margin)) - 5
            if max_dx < 0:
                max_dx = 0
            if v_abs_mag > 0.0:
                max_steps_motion = int(max(0.0, math.floor(float(max_dx) / float(v_abs_mag))))
                _log_line_only(args, "[relativity] motion_preflight: max_dx=%d v=%.6g max_steps=%d (n=%d margin=%d)" % (
                    int(max_dx), float(v_abs_mag), int(max_steps_motion), int(n), int(margin),
                ))
                if int(steps) > int(max_steps_motion):
                    raise ValueError(
                        "relativity: rel_steps exceeds motion bound "
                        f"(rel_steps={int(steps)} max_steps={int(max_steps_motion)} max_dx={int(max_dx)} v={float(p_rel.v):.6g} n={int(n)} margin={int(margin)}). "
                        "Reduce --rel-steps or --rel-v-frac (or increase --n)."
                    )
            else:
                _log_line_only(args, "[relativity] motion_preflight: v=0 => no motion bound")

            rel_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_name),
                    "artefact": "relativity_lightbox",
                    "n": str(int(n)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "lattice_init": str(getattr(params.lattice, "init_mode", "sparse")),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_c2": str(float(params.traffic.c2)),
                    "traffic_gamma": str(float(params.traffic.gamma)),
                    "traffic_dt": str(float(params.traffic.dt)),
                    "traffic_decay": str(float(params.traffic.decay)),
                    "rel_steps": str(int(steps)),
                    "rel_L": str(int(L)),
                    "rel_v_frac": str(float(v_frac)),
                    "rel_v_frac_ref": str(float(v_ref_frac)),
                    "rel_v_ref": str(float(v_ref_abs)),
                    "rel_v": str(float(v_abs)),
                    "rel_slab": str(int(slab)),
                    "rel_amp": str(float(amp)),
                    "rel_sigma": str(float(sigma)),
                    "rel_threshold": str(float(threshold)),
                    "rel_refractory": str(int(refractory)),
                    "rel_detect": str(rel_detect),
                    "rel_start_threshold": str(float(rel_start_threshold)),
                    "rel_peak_window": str(int(rel_peak_window)),
                    "rel_accept_threshold": str(float(rel_accept_threshold)),
                },
            )

            pb = ProgressBar(total_units=int(max(1, steps)), label="work")
            pb.start()
            try:
                rel_kw: dict[str, Any] = {}
                try:
                    sig_rl = inspect.signature(cast(Callable[..., Any], run_light_clock))
                    if "log_path" in sig_rl.parameters:
                        rel_kw["log_path"] = str(lp)
                    if "log_line" in sig_rl.parameters:
                        def _rl_log_line(s: str) -> None:
                            _log_line_only(args, str(s))
                        rel_kw["log_line"] = _rl_log_line
                    if "quiet" in sig_rl.parameters:
                        rel_kw["quiet"] = True
                    if "verbose" in sig_rl.parameters:
                        rel_kw["verbose"] = False
                except (TypeError, ValueError):
                    rel_kw = {}

                out_path = run_light_clock(
                    p_rel,
                    traffic=params.traffic,
                    provenance_header=str(rel_prov),
                    progress_cb=pb.advance,
                    **rel_kw,
                )
            finally:
                pb.finish()

            _log_line_only(args, "[relativity] wrote %s" % (str(out_path),))
            return

        # Oscillator (06A) — phase-based gravitational time dilation + lensing diagnostics
        if bool(getattr(args, "oscillator", False)):
            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)

            n = int(params.lattice.n)
            if str(params.traffic.mode).strip().lower() != "telegraph":
                raise ValueError("--oscillator requires --traffic-mode telegraph")

            # Derive experiment code from the parent folder (e.g. "06A_...").
            exp_name = _derive_exp_name(run_dir, default="06")

            # Resolve a concrete log path for this run (so _log_line_only writes somewhere).
            lp = _get_log_path(args)
            if lp == "":
                lp = os.path.join(log_dir, f"{exp_name}_n{int(n)}_{str(run_id)}.log")
                try:
                    setattr(args, "log_path", str(lp))
                except Exception:
                    pass
            _log_line_only(args, "[run] log=%s" % (str(lp),))
            _log_line_only(args, "[run] bundle=%s" % (str(run_dir),))
            _log_line_only(args, "[run] cwd=%s" % (str(os.getcwd()),))

            def _resolve_csv_path(raw: str, default_name: str) -> str:
                p = str(raw or "").strip()
                if p == "":
                    p = os.path.join(csv_dir, default_name)
                else:
                    # Honour overrides, but keep bundle hygiene:
                    # - absolute path: use as-is
                    # - bare filename: place under `_csv/`
                    # - relative path with folders: resolve under run bundle root
                    if not os.path.isabs(p):
                        if (os.path.sep not in p) and (os.path.altsep is None or os.path.altsep not in p):
                            p = os.path.join(csv_dir, p)
                        else:
                            p = os.path.join(run_dir, p)
                ensure_parent_dir(p)
                return str(p)

            out_csv = _resolve_csv_path(
                str(getattr(args, "osc_out", "")),
                f"{exp_name}_n{int(n)}_{str(run_id)}.csv",
            )
            out_series_csv = _resolve_csv_path(
                str(getattr(args, "osc_series_out", "")),
                f"{exp_name}_n{int(n)}_{str(run_id)}_series.csv",
            )
            out_lens_csv = _resolve_csv_path(
                str(getattr(args, "osc_lens_out", "")),
                f"{exp_name}_n{int(n)}_{str(run_id)}_lensing.csv",
            )

            _log_line_only(args, "[run] csv=%s" % (str(out_csv),))
            _log_line_only(args, "[run] csv_series=%s" % (str(out_series_csv),))
            _log_line_only(args, "[run] csv_lens=%s" % (str(out_lens_csv),))

            # Oscillator config (use CLI values when provided; otherwise dataclass defaults)
            cfg = OscillatorConfig(
                mass_amp=float(getattr(args, "osc_mass_amp", OscillatorConfig.mass_amp)),
                mass_soften=float(getattr(args, "osc_mass_soften", OscillatorConfig.mass_soften)),
                r_near=int(getattr(args, "osc_r_near", OscillatorConfig.r_near)),
                r_far=int(getattr(args, "osc_r_far", OscillatorConfig.r_far)),
                axis=str(getattr(args, "osc_axis", OscillatorConfig.axis)),
                omega=float(getattr(args, "osc_omega", OscillatorConfig.omega)),
                drive_amp=float(getattr(args, "osc_drive_amp", OscillatorConfig.drive_amp)),
                steps=int(getattr(args, "osc_steps", OscillatorConfig.steps)),
                burn=int(getattr(args, "osc_burn", OscillatorConfig.burn)),
                warm=int(getattr(args, "osc_warm", OscillatorConfig.warm)),
                demod_window=int(getattr(args, "osc_demod_window", OscillatorConfig.demod_window)),
                series_every=int(getattr(args, "osc_sample_every", OscillatorConfig.series_every)),
            )

            cfg_lens = LensingConfig(
                alpha=float(getattr(args, "osc_alpha", LensingConfig.alpha)),
                ray_count=int(getattr(args, "osc_ray_count", LensingConfig.ray_count)),
                ray_span=float(getattr(args, "osc_ray_span", LensingConfig.ray_span)),
                march_steps=int(getattr(args, "osc_march_steps", LensingConfig.march_steps)),
                ds=float(getattr(args, "osc_ds", LensingConfig.ds)),
                x0=float(getattr(args, "osc_x0", LensingConfig.x0)),
                y0=float(getattr(args, "osc_y0", LensingConfig.y0)),
                theta0=float(getattr(args, "osc_theta0", LensingConfig.theta0)),
            )

            do_lens = bool(getattr(args, "osc_lens", True))

            # Diffusion traffic params for building the steady potential (01A-style)
            diff_iters = int(getattr(args, "osc_mass_iters", 20000))
            if diff_iters < 1:
                raise ValueError(f"--osc-mass-iters must be >= 1 (got {int(diff_iters)})")
            diff_inject = float(getattr(args, "osc_mass_inject", 1.0))
            tp0 = params.traffic
            tp_diffuse = TrafficParams(
                mode="diffuse",
                iters=int(diff_iters),
                inject=float(diff_inject),
                rate_rise=float(getattr(tp0, "rate_rise", 0.0)),
                rate_fall=float(getattr(tp0, "rate_fall", 0.0)),
                decay=float(getattr(tp0, "decay", 0.0)),
                c2=float(getattr(tp0, "c2", 0.31)),
                gamma=float(getattr(tp0, "gamma", 0.001)),
                dt=float(getattr(tp0, "dt", 1.0)),
                boundary_mode=str(getattr(tp0, "boundary_mode", "")),
                sponge_width=int(getattr(tp0, "sponge_width", 0)),
                sponge_strength=float(getattr(tp0, "sponge_strength", 0.0)),
            )
            if int(getattr(tp_diffuse, "iters", 0)) < 1:
                raise ValueError(
                    "oscillator: tp_diffuse.iters must be >= 1; "
                    f"got {int(getattr(tp_diffuse, 'iters', 0))} (from --osc-mass-iters={int(diff_iters)})"
                )

            _log_line_only(args, "[oscillator] n=%d steps=%d burn=%d warm=%d axis=%s r_near=%d r_far=%d omega=%.6g drive_amp=%.6g" % (
                int(n), int(cfg.steps), int(cfg.burn), int(cfg.warm), str(cfg.axis), int(cfg.r_near), int(cfg.r_far), float(cfg.omega), float(cfg.drive_amp),
            ))
            _log_line_only(args, "[oscillator] steady_phi: iters=%d inject=%.6g decay=%.6g" % (
                int(tp_diffuse.iters), float(tp_diffuse.inject), float(tp_diffuse.decay),
            ))
            if do_lens:
                _log_line_only(args, "[oscillator] lensing: alpha=%.6g rays=%d span=%.6g march_steps=%d ds=%.6g" % (
                    float(cfg_lens.alpha), int(cfg_lens.ray_count), float(cfg_lens.ray_span), int(cfg_lens.march_steps), float(cfg_lens.ds),
                ))

            osc_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_name),
                    "artefact": "oscillator_phase_drift",
                    "n": str(int(n)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_c2": str(float(params.traffic.c2)),
                    "traffic_gamma": str(float(params.traffic.gamma)),
                    "traffic_dt": str(float(params.traffic.dt)),
                    "traffic_decay": str(float(params.traffic.decay)),
                    "mass_iters": str(int(tp_diffuse.iters)),
                    "mass_inject": str(float(tp_diffuse.inject)),
                    "mass_amp": str(float(cfg.mass_amp)),
                    "mass_boundary": str(getattr(tp_diffuse, "boundary_mode", "")),
                    "mass_sponge_width": str(int(getattr(tp_diffuse, "sponge_width", 0))),
                    "mass_sponge_strength": str(float(getattr(tp_diffuse, "sponge_strength", 0.0))),
                    "axis": str(cfg.axis),
                    "r_near": str(int(cfg.r_near)),
                    "r_far": str(int(cfg.r_far)),
                    "omega": str(float(cfg.omega)),
                    "drive_amp": str(float(cfg.drive_amp)),
                    "steps": str(int(cfg.steps)),
                    "burn": str(int(cfg.burn)),
                    "warm": str(int(cfg.warm)),
                    "demod_window": str(int(cfg.demod_window)),
                    "series_every": str(int(cfg.series_every)),
                },
            )

            lens_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_name),
                    "artefact": "oscillator_lensing",
                    "n": str(int(n)),
                    "alpha": str(float(cfg_lens.alpha)),
                    "ray_count": str(int(cfg_lens.ray_count)),
                    "ray_span": str(float(cfg_lens.ray_span)),
                    "march_steps": str(int(cfg_lens.march_steps)),
                    "ds": str(float(cfg_lens.ds)),
                    "x0": str(float(cfg_lens.x0)),
                    "y0": str(float(cfg_lens.y0)),
                    "theta0": str(float(cfg_lens.theta0)),
                },
            )

            total_units = int(oscillator_work_units(tp_diffuse, cfg, do_lens, cfg_lens))
            pb = ProgressBar(total_units=int(max(1, total_units)), label="work")
            pb.start()
            try:
                def _osc_progress(units: int) -> None:
                    pb.advance(int(units))

                # Build steady potential ONCE (diffusion cost accounted here), then re-use for phase drift + lensing.
                # This avoids double-building the diffusion field and keeps progress accounting honest.
                from oscillator import _build_point_mass_load, _build_steady_potential, _run_gravity_phase_drift_with_bg

                load = _build_point_mass_load(n=int(n), amp=float(cfg.mass_amp))
                phi_bg = _build_steady_potential(
                    n=int(n),
                    load=load,
                    tp_diffuse=tp_diffuse,
                    iters=int(tp_diffuse.iters),
                    progress_cb=_osc_progress,
                )

                # Run phase drift using the precomputed background potential.
                metrics = _run_gravity_phase_drift_with_bg(
                    n=int(n),
                    phi_bg=phi_bg,
                    tp_telegraph=params.traffic,
                    cfg=cfg,
                    out_csv=str(out_csv),
                    out_series_csv=str(out_series_csv),
                    provenance_header=str(osc_prov),
                    progress_cb=_osc_progress,
                    verbose=False,
                )

                # Optional lensing (re-use the same steady potential).
                if do_lens:
                    run_lensing_rays(
                        n=int(n),
                        phi_bg=phi_bg,
                        cfg=cfg_lens,
                        out_csv=str(out_lens_csv),
                        provenance_header=str(lens_prov),
                        progress_cb=_osc_progress,
                        verbose=False,
                    )
            finally:
                pb.finish()

            _log_line_only(args, "[oscillator] wrote %s" % (str(out_csv),))
            print("[oscillator] wrote %s" % (str(out_csv),))
            if do_lens:
                _log_line_only(args, "[oscillator] wrote %s" % (str(out_lens_csv),))
                print("[oscillator] wrote %s" % (str(out_lens_csv),))
            return

        # Heavy-walker sweep mode (Mach-style scan)
        if bool(args.walker_sweep):
            if not bool(args.walker):
                raise ValueError("--walker-sweep requires --walker")

            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)
            n = int(params.lattice.n)
            exp_code = _derive_exp_name(run_dir, default="08")
            if bool(getattr(params, "walker_mach1", False)):
                raise ValueError("--walker-sweep is incompatible with --walker-mach1 (mach1 forces tick iters to 2, making a sweep meaningless)")

            ticks = _parse_int_list(str(args.walker_sweep_ticks), "--walker-sweep-ticks")
            if len(ticks) < 1:
                raise ValueError("--walker-sweep-ticks produced no values")

            out_csv = str(getattr(args, "walker_sweep_out", "")).strip()
            if out_csv == "":
                out_csv = os.path.join(
                    csv_dir,
                    f"{exp_code}_walker_sweep_n{int(n)}_{str(run_id)}.csv",
                )
            else:
                out_csv = str(out_csv)
                ensure_parent_dir(out_csv)

            walker_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_code),
                    "artefact": "walker_tick_sweep",
                    "n": str(int(n)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_c2": str(float(params.traffic.c2)),
                    "traffic_gamma": str(float(params.traffic.gamma)),
                    "traffic_dt": str(float(params.traffic.dt)),
                    "traffic_decay": str(float(params.traffic.decay)),
                    "walker_steps": str(int(getattr(params, "walker_steps", 0))),
                    "walker_hold_steps": str(int(getattr(params, "walker_hold_steps", 0))),
                    "walker_tick_iters_list": ",".join([str(int(t)) for t in ticks]),
                    "walker_probe_r": str(int(getattr(params, "walker_probe_r", 0))),
                    "walker_mach1": ("1" if bool(getattr(params, "walker_mach1", False)) else "0"),
                },
            )

            total_units = int(max(1, walker_sweep_work_units(params, list(ticks))))
            pb = ProgressBar(total_units=int(total_units), label="work")
            pb.start()
            try:
                def _wk_log_line(s: str) -> None:
                    _log_line_only(args, str(s))
                run_heavy_walker_sweep(
                    params,
                    ticks,
                    str(out_csv),
                    provenance_header=str(walker_prov),
                    progress_cb=pb.advance,
                    log_line=_wk_log_line,
                    verbose=False,
                )
            finally:
                pb.finish()

            print("[walker-sweep] wrote %s" % (str(out_csv),))
            return

        # Heavy-walker mode (dynamic lag / inertia test)
        if bool(args.walker):
            run_dir = str(args.out)
            csv_dir, log_dir = _ensure_bundle_dirs(run_dir)
            n = int(params.lattice.n)
            exp_code = _derive_exp_name(run_dir, default="08")

            tick_move = int(getattr(params, "walker_tick_iters", 1))
            tick_hold = int(getattr(params, "walker_hold_tick_iters", tick_move))

            if bool(getattr(params, "walker_mach1", False)):
                tick_move = 2
                tick_hold = 2

            # Determine walker trace CSV path (run-unique) if user did not supply --dump-walker.
            dump_walker_path = str(getattr(args, "dump_walker", "") or "").strip()
            if dump_walker_path == "":
                dump_walker_path = os.path.join(
                    csv_dir,
                    f"{exp_code}_n{int(n)}_{str(run_id)}.csv",
                )

            walker_prov = write_csv_provenance_header(
                producer="CAELIX",
                command=cmd,
                cwd=os.getcwd(),
                python_exe=sys.executable,
                when_iso=when_iso,
                extra={
                    "run_id": str(run_id),
                    "experiment": str(exp_code),
                    "artefact": "walker_trace",
                    "n": str(int(n)),
                    "seed": str(int(getattr(args, "seed", 0))),
                    "traffic_mode": str(params.traffic.mode),
                    "traffic_c2": str(float(params.traffic.c2)),
                    "traffic_gamma": str(float(params.traffic.gamma)),
                    "traffic_dt": str(float(params.traffic.dt)),
                    "traffic_decay": str(float(params.traffic.decay)),
                    "walker_steps": str(int(getattr(params, "walker_steps", 0))),
                    "walker_hold_steps": str(int(getattr(params, "walker_hold_steps", 0))),
                    "walker_tick_iters": str(int(tick_move)),
                    "walker_hold_tick_iters": str(int(tick_hold)),
                    "walker_probe_r": str(int(getattr(params, "walker_probe_r", 0))),
                    "walker_mach1": ("1" if bool(getattr(params, "walker_mach1", False)) else "0"),
                    "dump_walker": str(dump_walker_path),
                },
            )

            total_units = int(max(1, walker_work_units(params, int(tick_move), int(tick_hold))))
            pb = ProgressBar(total_units=int(total_units), label="work")
            pb.start()
            try:
                def _wk_log_line(s: str) -> None:
                    _log_line_only(args, str(s))
                run_heavy_walker(
                    params,
                    provenance_header=str(walker_prov),
                    progress_cb=pb.advance,
                    log_line=_wk_log_line,
                    verbose=False,
                    tick_iters_move_override=int(tick_move),
                    tick_iters_hold_override=int(tick_hold),
                    dump_walker_path=str(dump_walker_path),
                )
            finally:
                pb.finish()

            print("[walker] wrote %s" % (str(dump_walker_path),))
            return

        # Ensemble mode
        if args.ensemble > 0:

            os.makedirs(args.out, exist_ok=True)
            # args.out is already resolved; keep directory creation explicit.
            if params.delta_load and int(params.delta_jitter) <= 0:
                raise ValueError("--ensemble with --delta-load is degenerate unless you set --delta-jitter > 0")

            workers = int(args.ensemble_workers)
            if workers < 0:
                raise ValueError("--ensemble-workers must be >= 0")

            seeds = [int(seed0) + int(i) for i in range(int(args.ensemble))]

            rows: List[Dict[str, float]] = []
            if workers == 0:
                for i, seed_i in enumerate(seeds):
                    row = _ensemble_one(seed_i, params)
                    rows.append(row)
                    print("[ensemble] %2d/%d seed=%d radial_slope=%.4f radial_r2=%.5f alpha=%.6f r2_line=%.6f" % (
                        i + 1, int(args.ensemble), int(seed_i),
                        float(row["radial_slope"]), float(row["radial_r2"]),
                        float(row["alpha"]), float(row["r2_line"]),
                    ))
            else:
                done = 0
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(_ensemble_one, seed_i, params): int(seed_i) for seed_i in seeds}
                    for fut in concurrent.futures.as_completed(futs):
                        seed_i = futs[fut]
                        row = fut.result()
                        rows.append(row)
                        done += 1
                        print("[ensemble] %2d/%d seed=%d radial_slope=%.4f radial_r2=%.5f alpha=%.6f r2_line=%.6f" % (
                            done, int(args.ensemble), int(seed_i),
                            float(row["radial_slope"]), float(row["radial_r2"]),
                            float(row["alpha"]), float(row["r2_line"]),
                        ))

            # Ensure deterministic ordering for downstream diffs/CSV.
            rows.sort(key=lambda r: float(r["seed"]))

            def _mean_std(vals: np.ndarray) -> Tuple[float, float]:
                v = vals.astype(np.float64)
                m = float(np.mean(v))
                if v.size > 1:
                    s = float(np.std(v, ddof=1))
                else:
                    s = 0.0
                return m, s

            arr_slope = np.array([r["radial_slope"] for r in rows], dtype=np.float64)
            arr_rr2 = np.array([r["radial_r2"] for r in rows], dtype=np.float64)
            arr_alpha = np.array([r["alpha"] for r in rows], dtype=np.float64)
            arr_r2l = np.array([r["r2_line"] for r in rows], dtype=np.float64)
            arr_mr2 = np.array([r["mean_fit_r2"] for r in rows], dtype=np.float64)
            arr_nr2 = np.array([r["min_fit_r2"] for r in rows], dtype=np.float64)

            m_slope, s_slope = _mean_std(arr_slope)
            m_rr2, s_rr2 = _mean_std(arr_rr2)
            m_alpha, s_alpha = _mean_std(arr_alpha)
            m_r2l, s_r2l = _mean_std(arr_r2l)
            m_mr2, s_mr2 = _mean_std(arr_mr2)
            m_nr2, s_nr2 = _mean_std(arr_nr2)

            print("[ensemble-summary] n=%d steps=%d traffic=%d rate_rise=%.3f rate_fall=%.3f inject=%.3f decay=%.6f k_index=%.3f X0=%.1f ds=%.3f eps=%.6f" % (
                params.lattice.n, params.lattice.steps,
                params.traffic.iters, params.traffic.rate_rise, params.traffic.rate_fall, params.traffic.inject, params.traffic.decay,
                params.k_index, params.X0, params.ds, params.eps,
            ))
            print("[ensemble-summary] radial_slope mean=%.4f std=%.4f" % (m_slope, s_slope))
            print("[ensemble-summary] radial_r2    mean=%.5f std=%.5f" % (m_rr2, s_rr2))
            print("[ensemble-summary] alpha        mean=%.6f std=%.6f" % (m_alpha, s_alpha))
            print("[ensemble-summary] r2_line      mean=%.6f std=%.6f" % (m_r2l, s_r2l))
            print("[ensemble-summary] mean_fit_r2  mean=%.6f std=%.6f" % (m_mr2, s_mr2))
            print("[ensemble-summary] min_fit_r2   mean=%.6f std=%.6f" % (m_nr2, s_nr2))

            if bool(getattr(args, "ensemble_write_csv", False)):
                csv_path = os.path.join(args.out, "ensemble_metrics.csv")
                ensure_parent_dir(csv_path)
                cols = ["seed", "phi_max", "radial_slope", "radial_r2", "alpha", "r2_line", "mean_fit_r2", "min_fit_r2"]
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(cols)
                    for r in rows:
                        w.writerow([r[c] for c in cols])
                print(f"[ensemble] wrote {csv_path}")

            return

        dump_radial_path = (args.dump_radial if str(args.dump_radial).strip() != "" else None)
        dump_radial_fit_path = (args.dump_radial_fit if str(args.dump_radial_fit).strip() != "" else None)

        # Ensure any requested CSV dump paths are writable (downstream writers do not create folders).
        if dump_radial_path is not None:
            ensure_parent_dir(str(dump_radial_path))
        if dump_radial_fit_path is not None:
            ensure_parent_dir(str(dump_radial_fit_path))
        if str(args.dump_shapiro).strip() != "":
            ensure_parent_dir(str(args.dump_shapiro))

        # Build a single provenance header block for any CSV artefacts written by this run.
        # Uses the shared run identity created immediately after parse_cli().
        csv_prov = write_csv_provenance_header(
            producer="CAELIX",
            command=cmd,
            cwd=os.getcwd(),
            python_exe=sys.executable,
            when_iso=when_iso,
            extra={
                "run_id": str(run_id),
                "n": str(int(params.lattice.n)),
                "steps": str(int(params.lattice.steps)),
                "traffic_iters": str(int(params.traffic.iters)),
                "seed": str(int(args.seed)),
                "lattice_init": str(getattr(params.lattice, "init_mode", "sparse")),
            },
        )

        # Pass provenance through if the downstream runner supports it.
        # Prefer per-artefact headers when available so each CSV carries its own identity.
        radial_prov = csv_prov + "# artefact=radial\n"
        radial_fit_prov = csv_prov + "# artefact=radial_fit\n"
        shapiro_prov = csv_prov + "# artefact=shapiro\n"

        prov_kw: dict[str, Any] = {}
        try:
            sig_b = inspect.signature(cast(Callable[..., Any], build_index_from_micro))

            if "radial_provenance" in sig_b.parameters:
                prov_kw["radial_provenance"] = radial_prov
            if "radial_fit_provenance" in sig_b.parameters:
                prov_kw["radial_fit_provenance"] = radial_fit_prov
            if "shapiro_provenance" in sig_b.parameters:
                prov_kw["shapiro_provenance"] = shapiro_prov

            # Generic fallbacks (single header used for all artefacts).
            if "csv_provenance" in sig_b.parameters:
                prov_kw["csv_provenance"] = csv_prov
            if "provenance" in sig_b.parameters:
                prov_kw["provenance"] = csv_prov
            if "provenance_header" in sig_b.parameters:
                prov_kw["provenance_header"] = csv_prov
            if "csv_header" in sig_b.parameters:
                prov_kw["csv_header"] = csv_prov
        except (TypeError, ValueError):
            prov_kw = {}

        dump_hdf5_path = str(getattr(args, "dump_hdf5", "") or "").strip()

        pb, pb_kw = _maybe_make_progress_bar(params, build_index_from_micro)
        if pb is not None:
            pb.start()

        live_kw: dict[str, Any] = {}
        if str(shm_name).strip() != "":
            live_kw["liveview_shm_name"] = str(shm_name)
        n_map, metrics = build_index_from_micro(
            params,
            dump_radial_path=dump_radial_path,
            dump_radial_fit_path=dump_radial_fit_path,
            dump_hdf5_path=dump_hdf5_path if dump_hdf5_path != "" else None,
            **live_kw,
            **pb_kw,
            **prov_kw,
        )

        if pb is not None:
            pb.finish()

        _log_line_only(args, "[micro->index] phi_max=%.6g radial_slope=%.4f radial_r2=%.5f r_fit=[%.1f,%.1f] n_fit=%d invr_r2=%.5f invr_a=%.4f invr_b=%.4f base_frac=%.3f de_slope=%.4f de_r2=%.5f slope_med=%.4f p16=%.4f p84=%.4f" % (
            metrics["phi_max"], metrics["radial_slope"], metrics["radial_r2"],
            metrics["radial_fit_r_min"], metrics["radial_fit_r_max"], int(metrics["radial_fit_n"]),
            metrics["radial_inv_r_r2"],
            metrics["radial_inv_r_a"], metrics["radial_inv_r_b"], metrics["radial_base_frac"],
            metrics["radial_deoffset_slope"], metrics["radial_deoffset_r2"],
            metrics["radial_slope_scan_median"], metrics["radial_slope_scan_p16"], metrics["radial_slope_scan_p84"],
        ))

        # If a radial-fit CSV was requested, append a single, unambiguous summary row derived
        # from metrics: `phi ≈ A/r + B` and the de-offset exponent on `(phi-B)`.
        # This guards against misleading raw log-log slopes when a background offset is present.
        if dump_radial_fit_path is not None:
            p_fit = str(dump_radial_fit_path)
            try:
                have = set()
                if os.path.exists(p_fit):
                    with open(p_fit, "r", newline="") as f:
                        r = csv.reader(f)
                        header = next(r, None)
                        if header is not None:
                            for row in r:
                                if row and str(row[0]).strip() != "":
                                    have.add(str(row[0]).strip())

                fit_kind = "inv_r_plus_offset_powerlaw"
                if fit_kind not in have:
                    # Base columns expected by radial.dump_radial_fit_csv, plus extra columns.
                    row = {
                        "fit_kind": fit_kind,
                        "r_min": float(metrics.get("radial_fit_r_min", float("nan"))),
                        "r_max": float(metrics.get("radial_fit_r_max", float("nan"))),
                        "n_points": float(metrics.get("radial_fit_n", float("nan"))),
                        # Mirror the inv-r fit into the legacy slope/intercept/r2 slots.
                        "slope": float(metrics.get("radial_inv_r_a", float("nan"))),
                        "intercept": float(metrics.get("radial_inv_r_b", float("nan"))),
                        "r2": float(metrics.get("radial_inv_r_r2", float("nan"))),
                        # New explicit columns.
                        "A": float(metrics.get("radial_inv_r_a", float("nan"))),
                        "B": float(metrics.get("radial_inv_r_b", float("nan"))),
                        "r2_inv_r": float(metrics.get("radial_inv_r_r2", float("nan"))),
                        "p": float(metrics.get("radial_deoffset_slope", float("nan"))),
                        "log_a0": "",
                        "r2_loglog": float(metrics.get("radial_deoffset_r2", float("nan"))),
                    }

                    # Append as a single CSV row matching the current header order when possible.
                    # If the file has no header (unexpected), write one, with provenance header.
                    cols = ["fit_kind", "r_min", "r_max", "n_points", "slope", "intercept", "r2",
                            "A", "B", "r2_inv_r", "p", "log_a0", "r2_loglog"]
                    write_header = (not os.path.exists(p_fit)) or (os.path.getsize(p_fit) == 0)
                    ensure_parent_dir(p_fit)
                    mode = "w" if write_header else "a"
                    when = when_iso
                    cmd = cmd
                    meta = {
                        "tool": "CAELIX",
                        "artefact": "radial_fit_summary",
                        "when": when,
                        "cwd": os.getcwd(),
                        "python": sys.executable,
                        "command": cmd,
                        "n": str(int(params.lattice.n)),
                        "steps": str(int(params.lattice.steps)),
                        "traffic_iters": str(int(params.traffic.iters)),
                        "seed": str(int(args.seed)),
                    }
                    with open(p_fit, mode, newline="") as f:
                        if write_header:
                            f.write(
                                write_csv_provenance_header(
                                    producer="CAELIX",
                                    command=str(meta["command"]),
                                    cwd=str(meta["cwd"]),
                                    python_exe=str(meta["python"]),
                                    when_iso=str(meta["when"]),
                                    extra={
                                        "artefact": str(meta["artefact"]),
                                        "n": str(meta["n"]),
                                        "steps": str(meta["steps"]),
                                        "traffic_iters": str(meta["traffic_iters"]),
                                        "seed": str(meta["seed"]),
                                        "run_id": str(run_id),
                                        "lattice_init": str(getattr(params.lattice, "init_mode", "sparse")),
                                    },
                                )
                            )
                        w = csv.writer(f)
                        if write_header:
                            w.writerow(cols)
                        w.writerow([row.get(c, "") for c in cols])
            except Exception as e:
                # Do not fail the experiment if the optional summary append fails.
                print("[radial-fit] warning: could not append summary row: %s" % (str(e),))

        cal = run_shapiro_mass_lockdown(params, n_map)
        cal_alpha = _as_float(cal.get("alpha"), "cal.alpha")
        cal_r2_line = _as_float(cal.get("r2_line"), "cal.r2_line")
        cal_mean_fit_r2 = _as_float(cal.get("mean_fit_r2"), "cal.mean_fit_r2")
        cal_min_fit_r2 = _as_float(cal.get("min_fit_r2"), "cal.min_fit_r2")

        _log_line_only(args, "[shapiro-lockdown] alpha=%.6f r2_line=%.8f mean_fit_r2=%.8f min_fit_r2=%.8f" % (
            cal_alpha, cal_r2_line, cal_mean_fit_r2, cal_min_fit_r2,
        ))
        per_obj = cal.get("shapiro_per_coupling")
        if not isinstance(per_obj, list):
            raise ValueError("shapiro_per_coupling missing or invalid")
        for rec in per_obj:
            if not isinstance(rec, dict):
                raise ValueError("shapiro_per_coupling contains non-dict entry")
            _log_line_only(args, "[shapiro-cpl] c=%.0f K=%.6g r2=%.6f" % (float(rec["coupling"]), float(rec["K"]), float(rec["r2_fit"])))
        per = per_obj

        if str(args.dump_shapiro).strip() != "":
            p = str(args.dump_shapiro)
            ensure_parent_dir(p)
            with open(p, "w", newline="") as f:
                f.write(
                    write_csv_provenance_header(
                        producer="CAELIX",
                        command=str(cmd),
                        cwd=os.getcwd(),
                        python_exe=sys.executable,
                        when_iso=str(when_iso),
                        extra={
                            "run_id": str(run_id),
                            "artefact": "shapiro",
                            "n": str(int(params.lattice.n)),
                            "steps": str(int(params.lattice.steps)),
                            "traffic_iters": str(int(params.traffic.iters)),
                            "seed": str(int(args.seed)),
                            "lattice_init": str(getattr(params.lattice, "init_mode", "sparse")),
                        },
                    )
                )
                w = csv.writer(f)
                w.writerow(["coupling", "K", "r2_fit"])
                for rec in per:
                    w.writerow([float(rec["coupling"]), float(rec["K"]), float(rec["r2_fit"])])
            print(f"[shapiro-dump] wrote {p}")

        if not args.plot:
            return

        # Scale figure size with lattice n so larger maps remain visually resolved.
        scale = float(max(1.0, args.plot_scale)) * float(max(1.0, args.n / 64.0))

        plot_all(
            n_map,
            params,
            str(args.out),
            plot_dpi=int(args.plot_dpi),
            scale=scale,
            plot_log=bool(args.plot_log),
        )

        if args.plot_log:
            print(f"[plot] wrote {args.out}/n_map.png, {args.out}/n_map_log.png and {args.out}/shapiro_asinh_fit.png")
        else:
            print(f"[plot] wrote {args.out}/n_map.png and {args.out}/shapiro_asinh_fit.png")
    finally:
        if t_run.running:
            t_run.stop()

        # LiveView teardown (owner-side)
        # Order matters: terminate viewer first, then unlink shared memory.
        try:
            if viewer_proc is not None:
                try:
                    viewer_proc.terminate()
                except Exception:
                    pass
                try:
                    viewer_proc.wait(timeout=1.0)
                except Exception:
                    pass
                try:
                    if viewer_proc.poll() is None:
                        viewer_proc.kill()
                except Exception:
                    pass
        finally:
            viewer_proc = None

        try:
            if live_bcast is not None:
                try:
                    # Prefer explicit cleanup (close + unlink) if provided by broadcaster.
                    if hasattr(live_bcast, "cleanup"):
                        live_bcast.cleanup()  # type: ignore[attr-defined]
                    elif hasattr(live_bcast, "close"):
                        live_bcast.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
        finally:
            live_bcast = None

        # Report HDF5 snapshot size after the run completes (avoid progress-bar interleave).
        # Fail-loud: if --dump-hdf5 was requested, the file must exist by now.
        args_run: Any | None = cast(Any | None, args)
        args_log: Any | None = cast(Any | None, args_for_log if args_for_log is not None else args)
        dump_hdf5_path = str(getattr(args_run, "dump_hdf5", "") or "").strip() if args_run is not None else ""
        if dump_hdf5_path != "":
            if not os.path.exists(dump_hdf5_path):
                raise FileNotFoundError(
                    f"--dump-hdf5 was set but no file was written: {dump_hdf5_path} (runner must call the HDF5 exporter and create the _hdf5/ folder)"
                )
            with h5py.File(dump_hdf5_path, "r") as f_h5:
                hb = str(f_h5.attrs.get("export.human_bytes", ""))
                b = int(f_h5.attrs.get("export.bytes", 0))
            msg = f"[exporter] hdf5 snapshot: {dump_hdf5_path} ({hb if hb != '' else str(b) + ' B'})"
            if args_log is not None:
                _log_line_only(args_log, msg)
            print(msg)

        # Optional CLI hook: materialise sprite assets from the soliton CSV ledger.
        # Note: soliton.py may already have written the sprite asset in-memory; in that case we do not re-run extractsprite.
        dump_sprite_path = str(getattr(args_run, "dump_sprite", "") or "").strip() if args_run is not None else ""
        if dump_sprite_path != "":
            # If the user provided a file path (endswith .h5), write exactly there.
            # Otherwise treat it as a directory and let extractsprite choose the filename.
            out_is_file = dump_sprite_path.lower().endswith(".h5")
            if out_is_file:
                ensure_parent_dir(dump_sprite_path)
            else:
                os.makedirs(dump_sprite_path, exist_ok=True)

            # Forward tuning knobs (defaults are set in cli.py).
            ds_radius = int(getattr(args_run, "dump_sprite_radius", 0))
            ds_rel = float(getattr(args_run, "dump_sprite_rel_thresh", 0.08))
            ds_pad = int(getattr(args_run, "dump_sprite_pad", 4))
            ds_maxr = int(getattr(args_run, "dump_sprite_max_radius", 96))
            ds_centre = str(getattr(args_run, "dump_sprite_centre", "") or "").strip()
            ds_comp = str(getattr(args_run, "dump_sprite_compression", "gzip") or "gzip").strip().lower()
            ds_lvl = int(getattr(args_run, "dump_sprite_compression_level", 4))

            if ds_comp not in ("gzip", "lzf", "none"):
                raise ValueError(f"--dump-sprite-compression must be one of: gzip, lzf, none (got {ds_comp!r})")

            # Run extractsprite as an in-repo script to avoid duplicating logic in core.
            script_path = os.path.join(os.path.dirname(__file__), "extractsprite.py")
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"--dump-sprite was set but extractsprite.py was not found: {script_path}")

            # Sweep-safe sprite extraction:
            # - do NOT require every sweep point to dump besttail
            # - extract sprites for points that did dump besttail
            # - fail-loud only if zero sprites were produced

            def _read_all_csv_rows(path: str) -> list[dict[str, str]]:
                p = str(path or "").strip()
                if p == "" or (not os.path.exists(p)):
                    return []
                try:
                    with open(p, "r", newline="") as f:
                        r = csv.reader(f)
                        header: list[str] | None = None
                        rows: list[dict[str, str]] = []
                        for row in r:
                            if not row:
                                continue
                            if row[0].startswith("#"):
                                continue
                            if header is None:
                                header = [str(c).strip() for c in row]
                                continue
                            if header is None:
                                continue
                            rec: dict[str, str] = {}
                            for i, k in enumerate(header):
                                if k == "":
                                    continue
                                v = "" if i >= len(row) else str(row[i]).strip()
                                rec[str(k)] = v
                            rows.append(rec)
                        return rows
                except Exception:
                    return []

            def _is_sprite_asset_path(p: str) -> bool:
                ps = str(p or "").strip()
                if ps == "":
                    return False
                b = os.path.basename(ps)
                if "__sprite_" in b or b.endswith("_sprite.h5"):
                    return True
                if f"{os.path.sep}_sprites{os.path.sep}" in ps:
                    return True
                if os.path.basename(os.path.dirname(ps)) == "_sprites":
                    return True
                return False

            def _fmt_csv_gate(rec: dict[str, str]) -> str:
                # Keep this as a single line: key=value pairs, only if present.
                keys = [
                    "status",
                    "besttail_ok",
                    "besttail_dumped",
                    "besttail_step",
                    "besttail_score",
                    # Insert additional keys after "besttail_score"
                    "besttail_vel_abs_max",
                    "tailc_vel_abs_max",
                    "tailc_vel_rms_mean",
                    "tailc_lap_abs_max",
                    "tailc_dE_phys_shifted_per_step_abs_max",
                    "besttail_h5",
                    "peak_final_over_pi",
                    "peak_r_final",
                    "tail_vel_rms_mean",
                    "tailp_vel_rms_mean",
                    # Insert for backward compatibility after "tailp_vel_rms_mean"
                    "tailp_vel_abs_max",
                    "tail_lap_abs_max",
                    "tailp_lap_abs_max",
                    "tail_dE_phys_shifted_per_step_abs_max",
                    "tail_frac_abs_phi_gt_pi_max",
                    "tailp_frac_abs_phi_gt_pi_max",
                    "tail_frac_abs_phi_gt_2pi_max",
                    "tailp_frac_abs_phi_gt_2pi_max",
                    "tail_peak_dev_max",
                    "tailp_peak_dev_max",
                ]
                parts: list[str] = []
                for k in keys:
                    v = str(rec.get(k, "")).strip()
                    if v != "":
                        parts.append(f"{k}={v}")
                return " ".join(parts)

            def _fnum(s: str) -> float:
                try:
                    return float(str(s).strip())
                except Exception:
                    return float("nan")

            def _infer_gate_fail(rec: dict[str, str]) -> list[str]:
                # Best-effort: infer which SG besttail gate(s) likely failed.
                # This is diagnostics-only: we prefer to import soliton.py constants, but we also
                # carry hard defaults here so the summary is still useful if import fails.
                bt_ok = str(rec.get("besttail_ok", "")).strip().lower()
                if bt_ok in ("1", "true", "yes"):
                    return []

                # Defaults (must match soliton.py defaults; used only when we cannot import).
                _D_VR_MAX = 0.20
                _D_VABS_MAX = 0.50
                _D_DE_MAX = 1.0e-2
                _D_F2_MAX = 0.01
                _D_FPI_MAX = 0.20
                _D_PEAK_MIN = 0.20
                _D_CM_MAX = 0.85
                _D_PEAK_R_MAX = 0.90

                # Try to import soliton constants (preferred).
                try:
                    import soliton as _sol
                except Exception:
                    _sol = None  # type: ignore[assignment]

                def _thr(name: str, default: float) -> float:
                    if _sol is None:
                        return float(default)
                    try:
                        v = getattr(_sol, name)
                        return float(v)
                    except Exception:
                        return float(default)

                vr_max = _thr("SG_BESTTAIL_MAX_VEL_RMS_MEAN", _D_VR_MAX)
                de_max = _thr("SG_BESTTAIL_MAX_DE_SHIFTED_PER_STEP_ABS", _D_DE_MAX)
                fpi_max = _thr("SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_PI", _D_FPI_MAX)
                f2_max = _thr("SG_BESTTAIL_MAX_FRAC_ABS_PHI_GT_2PI", _D_F2_MAX)
                peak_min = _thr("SG_BESTTAIL_MIN_PEAK_DEV", _D_PEAK_MIN)
                cm_max = _thr("SG_BESTTAIL_MAX_CM_R_FRAC_HALF", _thr("SG_BESTTAIL_MAX_CM_R_FRAC_CROP", _D_CM_MAX))
                peak_r_max = _thr("SG_BESTTAIL_MAX_PEAK_R_FRAC_HALF", _thr("SG_BESTTAIL_MAX_PEAK_R_FRAC_OUTER", _D_PEAK_R_MAX))

                out: list[str] = []

                # Prefer core (crop-centred) metrics if present; otherwise fall back to peak-patch metrics.
                vabs = _fnum(rec.get("tailc_vel_abs_max", rec.get("besttail_vel_abs_max", rec.get("tailp_vel_abs_max", ""))))
                vr = _fnum(rec.get("tailc_vel_rms_mean", rec.get("tailp_vel_rms_mean", rec.get("tail_vel_rms_mean", ""))))
                de = _fnum(rec.get("tailc_dE_phys_shifted_per_step_abs_max", rec.get("tail_dE_phys_shifted_per_step_abs_max", "")))
                fpi = _fnum(rec.get("tailp_frac_abs_phi_gt_pi_max", rec.get("tail_frac_abs_phi_gt_pi_max", "")))
                f2 = _fnum(rec.get("tailp_frac_abs_phi_gt_2pi_max", rec.get("tail_frac_abs_phi_gt_2pi_max", "")))
                peak_dev = _fnum(rec.get("tailp_peak_dev_max", rec.get("tail_peak_dev_max", "")))

                if np.isfinite(vr) and np.isfinite(vr_max) and float(vr) > float(vr_max):
                    out.append(f"vel_rms_mean={float(vr):.6g}>{float(vr_max):.6g}")
                if np.isfinite(de) and np.isfinite(de_max) and float(de) > float(de_max):
                    out.append(f"dE_shifted_abs={float(de):.6g}>{float(de_max):.6g}")
                if np.isfinite(fpi) and np.isfinite(fpi_max) and float(fpi) > float(fpi_max):
                    out.append(f"frac_|phi|>pi={float(fpi):.6g}>{float(fpi_max):.6g}")
                if np.isfinite(f2) and np.isfinite(f2_max) and float(f2) > float(f2_max):
                    out.append(f"frac_|phi|>2pi={float(f2):.6g}>{float(f2_max):.6g}")
                if np.isfinite(peak_dev) and np.isfinite(peak_min) and float(peak_dev) < float(peak_min):
                    out.append(f"peak_dev={float(peak_dev):.6g}<{float(peak_min):.6g}")

                # Localisation gates (if present in CSV)
                cm_r_frac = _fnum(rec.get("tail_cm_r_frac_half", rec.get("tail_cm_r_frac_crop", "")))
                peak_r_frac = _fnum(rec.get("tail_peak_r_frac_half", rec.get("tail_peak_r_frac_outer", "")))
                if np.isfinite(cm_r_frac) and np.isfinite(cm_max) and float(cm_r_frac) > float(cm_max):
                    out.append(f"cm_r_frac={float(cm_r_frac):.6g}>{float(cm_max):.6g}")
                if np.isfinite(peak_r_frac) and np.isfinite(peak_r_max) and float(peak_r_frac) > float(peak_r_max):
                    out.append(f"peak_r_frac={float(peak_r_frac):.6g}>{float(peak_r_max):.6g}")

                # If nothing matched but besttail_ok is false, still emit a hint.
                if len(out) == 0:
                    out.append("gate_unknown")

                return out

            def _pick_best_candidate(rows: list[dict[str, str]]) -> dict[str, str]:
                # Choose a representative row for diagnostics: prefer status=ok and minimum besttail_score.
                best: dict[str, str] | None = None
                best_s = float("inf")
                for rec in rows:
                    st = str(rec.get("status", "")).strip().lower()
                    if st not in ("", "ok"):
                        continue
                    s = _fnum(rec.get("besttail_score", ""))
                    if np.isfinite(s) and float(s) < float(best_s):
                        best_s = float(s)
                        best = rec
                if best is not None:
                    return best
                return rows[-1] if len(rows) > 0 else {}

            def _parse_extractsprite_canonical(text: str) -> dict[str, str]:
                # Expected line format (from extractsprite.py):
                #   [extractsprite] phase=<begin|ok|err> rc=<0|1> in=<...> kind=<...> out=<...> ...
                # We parse the first such line we find (in stdout OR stderr).
                for ln in (text or "").splitlines():
                    s = ln.strip()
                    if not s.startswith("[extractsprite]"):
                        continue
                    parts = s.split()
                    out: dict[str, str] = {"line": s}
                    for tok in parts[1:]:
                        if "=" not in tok:
                            continue
                        k, v = tok.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if k != "":
                            out[k] = v
                    return out
                return {}

            def _log_extractsprite_event(prefix: str, canon: dict[str, str], fallback_in: str, fallback_kind: str, fallback_out: str, rc: int | None = None) -> None:
                # Always emit ONE stable line to terminal and (if present) to run log.
                # If extractsprite did not produce a canonical line, we still emit a fully specified one.
                in_h5 = canon.get("in", fallback_in)
                kind = canon.get("kind", fallback_kind)
                out_h5 = canon.get("out", fallback_out)
                phase = canon.get("phase", "")
                phase_s = (f" phase={phase}" if str(phase).strip() != "" else "")
                if rc is None:
                    line = f"{prefix} [extractsprite]{phase_s} in={in_h5} kind={kind} out={out_h5}"
                else:
                    line = f"{prefix} [extractsprite]{phase_s} in={in_h5} kind={kind} out={out_h5} rc={int(rc)}"
                if args_log is not None:
                    _log_line_only(args_log, line)
                print(line, flush=True)

            sol_csv = str(getattr(args_run, "_soliton_out_csv", "") or "").strip() if args_run is not None else ""
            if sol_csv == "" or (not os.path.exists(sol_csv)):
                raise FileNotFoundError(f"--dump-sprite was set but soliton CSV ledger was not found: {sol_csv}")

            sol_rows = _read_all_csv_rows(sol_csv)
            if len(sol_rows) < 1:
                raise ValueError(f"--dump-sprite was set but soliton CSV contains no data rows: {sol_csv}")

            # Determine which points actually dumped a besttail snapshot.
            sol_dumped: list[dict[str, str]] = []
            for rec in sol_rows:
                v = str(rec.get("besttail_dumped", "")).strip().lower()
                if v in ("1", "true", "yes"):
                    sol_dumped.append(rec)

            # If user requested a single output file, only allow single-sprite runs.
            if out_is_file and len(sol_dumped) > 1:
                raise ValueError("--dump-sprite was given as a .h5 file but multiple sweep points dumped besttail; provide a directory path instead")

            ok_count = 0
            fail_count = 0

            for rec in sol_dumped:
                sprite_in_h5 = str(rec.get("besttail_h5", "") or "").strip()
                # New behaviour: besttail_h5 may already be the sprite-asset H5 written by soliton.py.
                if _is_sprite_asset_path(sprite_in_h5) and os.path.exists(sprite_in_h5):
                    ok_count += 1
                    _log_extractsprite_event(
                        prefix="[dumpsprite] OK:",
                        canon={"phase": "ok", "in": str(sprite_in_h5), "kind": "sprite_asset", "out": str(dump_sprite_path)},
                        fallback_in=str(sprite_in_h5),
                        fallback_kind="sprite_asset",
                        fallback_out=str(dump_sprite_path),
                    )
                    continue
                if sprite_in_h5 == "":
                    # Nothing concrete to extract.
                    continue
                if not os.path.exists(sprite_in_h5):
                    fail_count += 1
                    gates = _fmt_csv_gate(rec)
                    fails = _infer_gate_fail(rec)
                    if len(fails) > 0:
                        gates = (gates + " " if gates != "" else "") + ("gate_fail=" + ",".join(fails))
                    note = (" " + gates) if gates != "" else ""
                    _log_extractsprite_event(
                        prefix="[dumpsprite] SKIP:",
                        canon={},
                        fallback_in=str(sprite_in_h5),
                        fallback_kind="ledger_path",
                        fallback_out=str(dump_sprite_path),
                    )
                    if args_log is not None:
                        _log_line_only(args_log, f"[dumpsprite] missing ledger path: {sprite_in_h5}{note}")
                    continue

                cmd_sprite: list[str] = [
                    sys.executable,
                    script_path,
                    "--in", str(sprite_in_h5),
                    "--out", str(dump_sprite_path),
                    "--pad", str(int(ds_pad)),
                    "--max-radius", str(int(ds_maxr)),
                    "--rel-thresh", str(float(ds_rel)),
                    "--compression", str(ds_comp),
                ]
                if ds_comp == "gzip":
                    cmd_sprite += ["--compression-level", str(int(ds_lvl))]
                if int(ds_radius) > 0:
                    cmd_sprite += ["--radius", str(int(ds_radius))]
                if ds_centre != "":
                    cmd_sprite += ["--centre", str(ds_centre)]

                try:
                    cp = subprocess.run(cmd_sprite, capture_output=True, text=True, check=True)
                except subprocess.CalledProcessError as e:
                    fail_count += 1
                    blob = "\n".join([(e.stdout or ""), (e.stderr or "")])
                    canon = _parse_extractsprite_canonical(blob)
                    _log_extractsprite_event(
                        prefix="[dumpsprite] FAILED:",
                        canon=canon,
                        fallback_in=str(sprite_in_h5),
                        fallback_kind="besttail",
                        fallback_out=str(dump_sprite_path),
                        rc=int(getattr(e, "returncode", 1) or 1),
                    )
                    # Non-fatal per-point: continue sweep.
                    continue
                else:
                    ok_count += 1
                    blob = "\n".join([(cp.stdout or ""), (cp.stderr or "")])
                    canon = _parse_extractsprite_canonical(blob)
                    _log_extractsprite_event(
                        prefix="[dumpsprite] OK:",
                        canon=canon,
                        fallback_in=str(sprite_in_h5),
                        fallback_kind="besttail",
                        fallback_out=str(dump_sprite_path),
                    )

            if ok_count <= 0:
                # Fail-loud only if nothing was produced at all.
                # Include best candidate row gate summary and inferred fails for quick triage.
                cand = _pick_best_candidate(sol_rows)
                gates = _fmt_csv_gate(cand)
                fails = _infer_gate_fail(cand)
                if len(fails) > 0:
                    gates = (gates + " " if gates != "" else "") + ("gate_fail=" + ",".join(fails))
                note = (" " + gates) if gates != "" else ""
                raise FileNotFoundError(
                    f"--dump-sprite was set but no sprites were produced (ok=0 fail={int(fail_count)} dumped={int(len(sol_dumped))}). "
                    f"See: csv={sol_csv}{note}"
                )

        line = "[timing] wall_s=%.3f" % (float(t_run.elapsed_s),)
        if args_log is not None:
            _log_line_only(args_log, line)
        print(line)

if __name__ == "__main__":
    main()
