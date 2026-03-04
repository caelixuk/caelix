# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""experiments.py — reproducible experiment presets for CAELIX

Goal
----
Provide a single, stable entrypoint for running the "definitive" experiments in a sensible order,
without retyping long CLI invocations.

Usage
-----
  python experiments.py --list
  python experiments.py --run 01A_pipeline_baseline

Canonical experiment order
--------------------------
This is the intended “discovery -> complexity -> research value” progression.
(Use `--run-all` to execute this order end-to-end.)

00 - Heavy Walker Experiments
  00A_heavy_walker_test                       — sanity-checks moving-source dynamics (advection/drag signatures).
  00B_heavy_walker_warmstart                  — same as 00A but with a warm start (removes switch-on transient).
  00C_heavy_walker_pacing_fast                — warm-started like 00B, but faster pacing (tick-iters=100) to amplify the wake.
  00D_heavy_walker_hold_decay                 — like 00C, then stop and hold to measure wake relaxation (hold-steps=64).
  00E_heavy_walker_circle_anisotropy          — circle path anisotropy probe (wake vs direction), N=512.
  00F_heavy_walker_damping_strong             — damping contrast: like 00C but with gamma=0.010 to suppress ringing/wake.
  00G_heavy_walker_diffuse_baseline           — diffusion control: same pacing as 00C to separate relaxer lag from telegraph wake.
  00H_heavy_walker_diffuse_decay              — diffusion control with decay=0.010 to bound field mass and measure steady lag.
  00I_heavy_walker_telegraph_decay            — telegraph decay contrast: like 00C but with decay=0.010 to reduce field memory without strong damping.
  00J_heavy_walker_near_mach_hold             — near-Mach pacing (tick-iters=2) with decay=0.010, then hold (tick-iters=200) to measure relaxation.
  00K_heavy_walker_circle_decay_anisotropy    — circle anisotropy probe with telegraph decay=0.010 (bounded memory), N=512.
  00L_heavy_walker_tick_sweep_clean_telegraph — walker tick sweep (2..200) in clean telegraph regime (gamma=0.001, decay=0.010), N=512.

01 - Baseline field + diagnostics (establish the substrate)
  01A_pipeline_baseline                       — discovers the steady-field regime and records the radial law + fit diagnostics.

02 - Interference controls (validate wave-like behaviour)
  02A_double_slit_single_slit_control         — establishes the single-aperture diffraction envelope (the control).
  02B_double_slit_two_slit                    — demonstrates fringe structure beyond the single-slit envelope (interference).

03 - Static interaction proxy (energy vs separation)
  03A_coulomb_like_decay0_hard_boundary       — tests whether like sources produce a repulsive energy-vs-distance curve.
  03B_coulomb_opposite_decay0_hard_boundary   — tests whether opposite sources produce an attractive curve under the same conditions.
  03A_coulomb_like_decay0_sponge_boundary     — tests whether like sources produce a repulsive energy-vs-distance curve.
  03B_coulomb_opposite_decay0_sponge_boundary — tests whether opposite sources produce an attractive curve under the same conditions.

04 - Confinement spectrum (quantisation proxy)
  04A_corral_global_sweep_optimised           — searches for discrete resonance peaks in a confined geometry (mode spectrum) - optimised.
  04B_corral_tight_rescan                     — rescans a narrow omega window around the strongest 04A peak for a sharper mode estimate.

05 - Calibration (reduce grid artefacts before higher claims)
  05A_isotropy_sigma_sweep                    — sigma sweep (single CSV) to quantify anisotropy vs smoothing.
  05B_isotropy_baseline_calibration           — measures axis vs diagonal effective propagation and how smoothing reduces anisotropy.

06 - Relativity proxy (+ wave-carrier diagnostics)
  06A_oscillator_phase_drift_lensing          — phase-drift time dilation vs potential + lensing “money shot” (telegraph).
  06B_ringdown_passive_resonance_sweep        — passive box ringdown (Dirichlet walls) sigma sweep; maps grid eigen-spectrum.
  06B_ringdown_passive_resonance_probe        — passive box ringdown (Dirichlet walls) sigma probe; maps grid eigen-spectrum.
  06D_relativity_twin_paradox_threshold_lock  — lightbox baseline: first-cross + flight-gate produces phase-lock (arrival proxy collapses), telegraph, n=512.
  06E_relativity_twin_paradox_window_peak     — lightbox corrected: windowed peak arrival timing (breaks gate-lock), telegraph, n=512.
  06F_relativity_twin_paradox_xy_sponge       — lightbox corrected + lateral XY sponge (side-wall absorber) to suppress echo pollution, telegraph, n=512.

07 - Particle dynamics (collider)
  07A_collider_up_down_opp_spin               — collider calibration (opposite spin).
  07B_collider_down_down_same_spin            — collider calibration (same spin).
  07C_collider_upgrade_opp_spin               — collider upgrade (opposite spin): detectors + calorimetry shells + ratios.
  07D_collider_upgrade_same_spin              — collider upgrade (same spin): detectors + calorimetry shells + ratios.
  07E_collider_decay_opp_spin                 — collider decay-mode (opp spin): stop injection after impact and measure persistence.
  07F_collider_decay_same_spin                — collider decay-mode (same spin): stop injection after impact and measure persistence.
  07G_collider_scatter_opp_spin               — scattering (opp spin): near-miss impact parameter b>0 with full-vector back-reaction + octant flux.
  07H_collider_scatter_same_spin              — scattering (same spin): near-miss impact parameter b>0 with full-vector back-reaction + octant flux.
  07I_collider_bind_opp_spin                  — binding (opp spin): slow approach + strong back-reaction + impact parameter; look for capture / closed orbit.
  07J_collider_bind_same_spin                 — binding (same spin): same settings; contrast capture vs repulsion.
  07K_collider_nucleus_opp_spin               — nucleus (opp spin): add central stationary source as third-body momentum sink (telegraph).
  07L_collider_nucleus_same_spin              — nucleus (same spin): same as 07K; contrast binding / capture signatures (telegraph).
  07M_collider_hydrogen_halo_soft             — hydrogen: single walker + nucleus + local damping halo (soft) to encourage capture / orbit.
  07N_collider_hydrogen_halo_strong           — hydrogen: single walker + nucleus + stronger halo to probe binding threshold.
  07O_collider_hydrogen_deep_orbit_full_q     — hydrogen (capture): deep orbit (r=32) + full nucleus charge (q=1.0) + aggressive halo.
  07P_collider_hydrogen_deep_orbit_half_q     — hydrogen (capture): deep orbit (r=32) + half nucleus charge (q=0.5) + aggressive halo.
  07Q_collider_hydrogen_goldilocks_long       — hydrogen (verification): long-duration (50k steps) stability test of the 07O captured composite.

08 - Non-linear soliton search
  08A_soliton_scan                            — Klein-Gordon/φ⁴ scaffold sweep (k, λ) looking for long-lived, localised cores (nonlinear).
  08B_symmetry_breaking_baseline              — double-well baseline (k<0, λ>0) to validate symmetry breaking / nucleation dynamics (nonlinear).
  08C_symmetry_breaking_local_probe           — fast local probe near the symmetry-breaking boundary (nonlinear).
  08D_symmetry_breaking_local_probe_vev_init  — local probe initialised at the broken-vacuum VEV (nonlinear).
  08E_sine_gordon_breather_baseline           — Sine-Gordon baseline (sponge): breather/oscillon search on a zero-vacuum canvas (traffic-mode=sine_gordon).
  08F_sine_gordon_breather_stable             — Sine-Gordon stable (sponge): breather/oscillon search on a zero-vacuum canvas (traffic-mode=sine_gordon).
  08G_sine_gordon_breather_extract            — Sine-Gordon stable (sponge): breather/oscillon extraction on a zero-vacuum canvas (traffic-mode=sine_gordon).
  
09 - Sine-Gordon sprite interactions
  09A_collidersg_head_on                      — head-on collision (anti-phase) using the tuned N=192.
  09B_collidersg_orbit                        — orbit collision using the tuned N=192.
  09C_collidersg_molecule                     — molecule collision (anti-phase) using the tuned N=192.
  09D_collidersg_kink_walls                   — planar kink-wall scattering (hard-ball sanity), N=192.
  09E_collidersg_transistor_binary_baseline   — binary transistor - both 0 phase (wire-constrained kinks), N=192.
  09F_collidersg_transistor_binary_phasetest  — binary transistor - 0 phase / PI phase (wire-constrained kinks), N=192.
  09G_collidersg_transistor_calibration       — wire speed calibration (single wall), N=192.
  09H_collidersg_transistor_tjunction         — first T-junction routing probe (single +2π wall), N=192.

Output layout
-------------
By default, outputs are written under `_Output/` using category subfolders so results do not
collapse into a single directory:

  _Output/00_heavy_walker/    — particle dynamics (heavy walker)
  _Output/01_baseline/        — baseline field + diagnostics
  _Output/02_interference/    — interference controls
  _Output/03_interaction/     — interaction proxy (energy vs separation)
  _Output/04_confinement/     — confinement spectrum (corral)
  _Output/05_calibration/     — isotropy calibration
  _Output/06_relativity/      — relativity proxy
  _Output/07_collider/        — particle dynamics (linear collider)
  _Output/08_solitons/        — non-linear soliton search
  _Output/09_sg_collider/     — Sine-Gordon multi-soliton interactions (non-linear collider)


You can override the base output directory by editing `out_root` in `experiment_table()` or by
passing explicit `--*-out` file paths in the individual experiment argv blocks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import datetime
import threading
from dataclasses import dataclass
from typing import Dict, List

from utils import ensure_parent_dir, wallclock_iso


@dataclass(frozen=True)
class Experiment:
    name: str
    summary: str
    argv: List[str]


def _repo_root() -> str:
    # Run from the suite folder so relative paths (e.g. `_Output/...`) are stable.
    return os.path.dirname(os.path.abspath(__file__))


def _core_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "core.py")


# --- Run id and argv rewriting helpers ---

def _make_run_id() -> str:
    # Local timestamp for human readability and stable file grouping.
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _extract_n(argv: List[str]) -> str:
    for i, tok in enumerate(argv):
        if tok == "--n" and i + 1 < len(argv):
            v = str(argv[i + 1]).strip()
            if v != "":
                return v
    return "NA"


def _rewrite_out_and_outputs(exp: Experiment, argv: List[str], *, run_id: str) -> tuple[List[str], str, str]:
    """Return (argv2, run_dir, n_str).

    - Rewrites `--out` to a run-bundle directory: <out_base>/<experiment>/<run_id>/
    - Rewrites known output CSV args to land inside that run_dir with timestamps and n-tag.
    """
    argv2 = list(argv)
    out_base = _extract_out_dir(argv2)
    n_str = _extract_n(argv2)

    run_dir = os.path.join(out_base, exp.name, run_id)

    # Replace the `--out` value with the run_dir.
    for i, tok in enumerate(argv2):
        if tok == "--out" and i + 1 < len(argv2):
            argv2[i + 1] = run_dir
            break

    def _set_csv(flag: str, stem: str) -> None:
        for j, t in enumerate(argv2):
            if t == flag and j + 1 < len(argv2):
                argv2[j + 1] = os.path.join(run_dir, "_csv", f"{exp.name}_{stem}_n{n_str}_{run_id}.csv")
                return

    # Canonical diagnostics.
    _set_csv("--dump-radial", "radial")
    _set_csv("--dump-radial-fit", "radial_fit")
    _set_csv("--dump-shapiro", "shapiro")

    # Other experiment outputs (keep stems stable).
    coul_sign = ""
    for j, t in enumerate(argv2):
        if t == "--coulomb-sign" and j + 1 < len(argv2):
            coul_sign = str(argv2[j + 1]).strip().lower()
            break
    if coul_sign == "":
        coul_sign = "na"
    _set_csv("--coulomb-out", f"coulomb_{coul_sign}")

    _set_csv("--corral-out", "corral")
    _set_csv("--iso-out", "isotropy")
    _set_csv("--rel-out", "relativity")
    _set_csv("--osc-out", "oscillator")
    _set_csv("--ringdown-out", "ringdown")
    _set_csv("--collider-out", "collider")
    _set_csv("--soliton-out", "soliton")
    _set_csv("--collidersg-out", "collidersg")

    return argv2, run_dir, n_str


def _extract_out_dir(argv: List[str]) -> str:
    for i, tok in enumerate(argv):
        if tok == "--out" and i + 1 < len(argv):
            v = str(argv[i + 1]).strip()
            if v != "":
                return v
    return "_Output/_misc"


def _format_log_header(exp: Experiment, *, cwd: str, cmd: List[str]) -> str:
    lines: List[str] = []
    lines.append("[CAELIX] experiment=%s" % (exp.name,))
    lines.append("[CAELIX] when=%s" % (wallclock_iso(),))
    lines.append("[CAELIX] what=%s" % (exp.summary,))
    lines.append("[CAELIX] cwd=%s" % (cwd,))
    lines.append("[CAELIX] python=%s" % (sys.executable,))
    lines.append("[CAELIX] command=%s" % (" ".join(cmd),))

    def _sh_quote(tok: str) -> str:
        t = str(tok)
        if t == "":
            return "''"
        if any(c.isspace() for c in t) or any(c in t for c in ['"', "'", "\\", "`", "$", "!", "(", ")", "{", "}", "[", "]", ";", "&", "|", "<", ">", "?"]):
            # Simple single-quote shell quoting, escape embedded single quotes.
            return "'" + t.replace("'", "'\\''") + "'"
        return t

    cmd_ml = " \\\n  ".join(_sh_quote(t) for t in cmd)
    lines.append("[CAELIX] command_ml=\\n  %s" % (cmd_ml,))

    lines.append("")
    return "\n".join(lines)


def _write_both(fp, s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()
    fp.write(s)
    fp.flush()


def _pump_lines(pipe, on_line) -> None:
    for line in pipe:
        on_line(line)


def _run_core(exp: Experiment) -> int:
    run_id = _make_run_id()
    argv2, run_dir, n_str = _rewrite_out_and_outputs(exp, exp.argv, run_id=run_id)

    cmd = [sys.executable, "-u", _core_path(), *argv2]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Run inside the suite folder so relative imports and outputs are stable.
    cwd = _repo_root()

    log_name = f"{exp.name}_n{n_str}_{run_id}.log"
    log_path = os.path.join(run_dir, "_logs", log_name)
    ensure_parent_dir(log_path)

    header = _format_log_header(exp, cwd=cwd, cmd=cmd)

    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write(header)
        fp.flush()

        _write_both(fp, "[run] log=%s\n" % (log_path,))
        _write_both(fp, "[run] bundle=%s\n" % (run_dir,))
        _write_both(fp, "[run] cwd=%s\n" % (cwd,))

        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        assert p.stderr is not None

        t_out = threading.Thread(
            target=_pump_lines,
            args=(p.stdout, lambda ln: _write_both(fp, ln)),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_pump_lines,
            args=(p.stderr, lambda ln: (sys.stderr.write(ln), sys.stderr.flush())),
            daemon=True,
        )
        t_out.start()
        t_err.start()

        rc = int(p.wait())
        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)
        if rc != 0:
            _write_both(fp, "[run] rc=%d\n" % (rc,))
        return int(rc)


def _exp(name: str, summary: str, argv: List[str]) -> Experiment:
    return Experiment(name=name, summary=summary, argv=argv)


def experiment_table() -> Dict[str, Experiment]:
    # Default output root. Each category writes into its own subfolder so the suite
    # does not collapse into a single giant directory.
    out_root = "_Output"
    out_00 = os.path.join(out_root, "00_heavy_walker")
    out_01 = os.path.join(out_root, "01_baseline")
    out_02 = os.path.join(out_root, "02_interference")
    out_03 = os.path.join(out_root, "03_interaction")
    out_04 = os.path.join(out_root, "04_confinement")
    out_05 = os.path.join(out_root, "05_calibration")
    out_06 = os.path.join(out_root, "06_relativity")
    out_07 = os.path.join(out_root, "07_collider")
    out_08 = os.path.join(out_root, "08_solitons")
    out_09 = os.path.join(out_root, "09_sg_collider")

    out_ZZ = os.path.join(out_root, "ZZ_code_testing")

    exps: List[Experiment] = [
        _exp(
            "00A_heavy_walker_test",
            "Heavy Walker test (moving source dynamics).",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "telegraph",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--walker",
                "--walker-steps", "64",
                "--walker-tick-iters", "200",
            ],
        ),
        _exp(
            "00B_heavy_walker_warmstart",
            "Heavy Walker warm-started test (moving source dynamics).",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "telegraph",
                "--traffic-iters", "200",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--walker",
                "--walker-steps", "64",
                "--walker-tick-iters", "200",
            ],
        ),
        _exp(
            "00C_heavy_walker_pacing_fast",
            "Heavy Walker pacing contrast (warm-started; faster move pacing to amplify wake).",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "telegraph",
                "--traffic-iters", "200",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--walker",
                "--walker-steps", "64",
                "--walker-tick-iters", "100",
            ],
        ),
        _exp(
            "00D_heavy_walker_hold_decay",
            "Heavy Walker hold/decay: fast pacing move phase then hold to measure wake relaxation.",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "telegraph",
                "--traffic-iters", "200",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--walker",
                "--walker-steps", "64",
                "--walker-tick-iters", "100",
                "--walker-hold-steps", "64",
                "--walker-hold-tick-iters", "200",
            ],
        ),
        _exp(
            "00E_heavy_walker_circle_anisotropy",
            "Heavy Walker circle-path anisotropy probe: measure wake/lag as direction rotates (telegraph).",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "telegraph",
                "--traffic-iters", "200",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--walker",
                "--walker-path", "circle",
                "--walker-circle-radius", "8",
                "--walker-circle-period", "0",
                "--walker-probe-r", "8",
                "--walker-steps", "256",
                "--walker-tick-iters", "100",
            ],
        ),
        _exp(
            "00F_heavy_walker_damping_strong",
            "Heavy Walker damping contrast: linear path, fast pacing, higher gamma to suppress ringing/wake.",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "telegraph",
                "--traffic-iters", "200",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.010",
                "--traffic-decay", "0.0",
                "--walker",
                "--walker-steps", "64",
                "--walker-tick-iters", "100",
            ],
        ),
        _exp(
            "00G_heavy_walker_diffuse_baseline",
            "Heavy Walker diffusion baseline: linear path, fast pacing, diffuse mode control for lag vs telegraph wake.",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "diffuse",
                "--traffic-iters", "200",
                "--traffic-decay", "0.0",
                "--walker",
                "--walker-steps", "64",
                "--walker-tick-iters", "100",
            ],
        ),
        _exp(
            "00H_heavy_walker_diffuse_decay",
            "Heavy Walker diffusion control with decay: linear path, fast pacing, bounded steady regime.",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "diffuse",
                "--traffic-iters", "200",
                "--traffic-decay", "0.010",
                "--walker",
                "--walker-steps", "64",
                "--walker-tick-iters", "100",
            ],
        ),
        _exp(
            "00I_heavy_walker_telegraph_decay",
            "Heavy Walker telegraph decay contrast: fast pacing, weak damping, add decay to reduce field memory.",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "telegraph",
                "--traffic-iters", "200",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.010",
                "--walker",
                "--walker-steps", "64",
                "--walker-tick-iters", "100",
            ],
        ),
        _exp(
            "00J_heavy_walker_near_mach_hold",
            "Heavy Walker near-Mach stress test: tick-iters=2 move phase with decay, then slow hold to measure relaxation.",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "telegraph",
                "--traffic-iters", "200",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.010",
                "--walker",
                "--walker-steps", "64",
                "--walker-tick-iters", "2",
                "--walker-hold-steps", "64",
                "--walker-hold-tick-iters", "200",
            ],
        ),
        _exp(
            "00K_heavy_walker_circle_decay_anisotropy",
            "Heavy Walker circle-path anisotropy probe with telegraph decay: measure wake modulation with bounded memory.",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "telegraph",
                "--traffic-iters", "200",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.010",
                "--walker",
                "--walker-path", "circle",
                "--walker-circle-radius", "8",
                "--walker-circle-period", "0",
                "--walker-probe-r", "8",
                "--walker-steps", "256",
                "--walker-tick-iters", "100",
            ],
        ),
        _exp(
            "00L_heavy_walker_tick_sweep_clean_telegraph",
            "Heavy Walker tick sweep: pacing curve in clean telegraph regime (gamma=0.001, decay=0.010).",
            [
                "--out", out_00,
                "--n", "512",
                "--traffic-mode", "telegraph",
                "--traffic-iters", "200",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.010",
                "--walker",
                "--walker-steps", "64",
                "--walker-tick-iters", "100",
                "--walker-sweep",
                "--walker-sweep-ticks", "2,4,8,16,32,64,100,200",
            ],
        ),
        _exp(
            "01A_pipeline_baseline_hard_boundary",
            "Steady-field pipeline + Shapiro lockdown (single seed).",
            [
                "--out", out_01,
                "--n", "512",
                "--steps", "0",
                "--traffic-iters", "250000",
                "--traffic-rate", "0.10",
                "--traffic-inject", "1.0",
                "--traffic-decay", "0.0",
                "--k-index", "1.0",
                "--X0", "8000",
                "--ds", "1.0",
                "--eps", "0.005",
                "--delta-load",
                "--dump-radial", "radial.csv",
                "--dump-radial-fit", "radial_fit.csv",
                "--dump-shapiro", "shapiro.csv",
            ],
        ),
        _exp(
            "01B_pipeline_baseline_sponge_boundary",
            "Steady-field pipeline + Shapiro lockdown (single seed).",
            [
                "--out", out_01,
                "--n", "512",
                "--steps", "0",
                "--traffic-iters", "250000",
                "--traffic-rate", "0.10",
                "--traffic-inject", "1.0",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--k-index", "1.0",
                "--X0", "8000",
                "--ds", "1.0",
                "--eps", "0.005",
                "--delta-load",
                "--dump-radial", "radial.csv",
                "--dump-radial-fit", "radial_fit.csv",
                "--dump-shapiro", "shapiro.csv",
            ],
        ),
        _exp(
            "02A_double_slit_single_slit_control",
            "Single-slit control.",
            [
                "--out", out_02,
                "--double-slit",
                "--ds-single-slit",
                "--ds-source", "sine",
                "--ds-omega", "0.80",
                "--ds-gun-speed", "0.0",
                "--ds-burn", "600",
                "--ds-window", "200",
                "--ds-sample-every", "4",
                "--ds-amp", "1000.0",
                "--traffic-mode", "telegraph",
                "--traffic-gamma", "0.0001",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--n", "512",
                "--ds-wall-x", "256",
                "--ds-detector-x", "470",
                "--ds-gun-stop", "246",
                "--ds-steps", "5000",
                "--ds-slit-width", "8",
                "--ds-sample-every", "1",
            ],
        ),
        _exp(
            "02B_double_slit_two_slit",
            "Young double-slit interference (telegraph, sine drive).",
            [
                "--out", out_02,
                "--double-slit",
                "--ds-source", "sine",
                "--ds-omega", "0.80",
                "--ds-gun-speed", "0.0",
                "--ds-burn", "600",
                "--ds-window", "200",
                "--ds-sample-every", "4",
                "--ds-amp", "1000.0",
                "--traffic-mode", "telegraph",
                "--traffic-gamma", "0.0001",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--n", "512",
                "--ds-wall-x", "256",
                "--ds-detector-x", "470",
                "--ds-gun-stop", "246",
                "--ds-steps", "5000",
                "--ds-slit-sep", "64",
                "--ds-slit-width", "8",
                "--ds-sample-every", "1",
            ],
        ),
        _exp(
            "03A_coulomb_like_decay0_hard_boundary",
            "Coulomb proxy (like charges), decay=0.0.",
            [
                "--out", out_03,
                "--coulomb",
                "--n", "512",
                "--traffic-mode", "diffuse",
                "--traffic-iters", "10000",
                "--coulomb-max-iters", "10000",
                "--coulomb-check-every", "200",
                "--coulomb-tol", "1e-4",
                "--traffic-decay", "0.0",
                "--coulomb-sign", "like",
                "--coulomb-d-min", "4",
                "--coulomb-d-max", "48",
                "--coulomb-d-step", "4",
            ],
        ),
        _exp(
            "03B_coulomb_opposite_decay0_hard_boundary",
            "Coulomb proxy (opposite charges), decay=0.0.",
            [
                "--out", out_03,
                "--coulomb",
                "--n", "512",
                "--traffic-mode", "diffuse",
                "--traffic-iters", "10000",
                "--coulomb-max-iters", "10000",
                "--coulomb-check-every", "200",
                "--coulomb-tol", "1e-4",
                "--traffic-decay", "0.0",
                "--coulomb-sign", "opposite",
                "--coulomb-d-min", "4",
                "--coulomb-d-max", "48",
                "--coulomb-d-step", "4",
            ],
        ),
        _exp(
            "03C_coulomb_like_decay0_sponge_boundary",
            "Coulomb proxy (like charges), decay=0.0.",
            [
                "--out", out_03,
                "--coulomb",
                "--n", "512",
                "--traffic-mode", "diffuse",
                "--traffic-iters", "10000",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--coulomb-max-iters", "10000",
                "--coulomb-check-every", "200",
                "--coulomb-tol", "1e-4",
                "--traffic-decay", "0.0",
                "--coulomb-sign", "like",
                "--coulomb-d-min", "4",
                "--coulomb-d-max", "48",
                "--coulomb-d-step", "4",
            ],
        ),
        _exp(
            "03D_coulomb_opposite_decay0_sponge_boundary",
            "Coulomb proxy (opposite charges), decay=0.0.",
            [
                "--out", out_03,
                "--coulomb",
                "--n", "512",
                "--traffic-mode", "diffuse",
                "--traffic-iters", "10000",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--coulomb-max-iters", "10000",
                "--coulomb-check-every", "200",
                "--coulomb-tol", "1e-4",
                "--traffic-decay", "0.0",
                "--coulomb-sign", "opposite",
                "--coulomb-d-min", "4",
                "--coulomb-d-max", "48",
                "--coulomb-d-step", "4",
            ],
        ),
        _exp(
            "04A_corral_global_sweep_optimised",
            "Quantum corral: optimised sweep (telegraph).",
            [
                "--out", out_04,
                "--corral",
                "--corral-geom", "sphere",
                "--traffic-mode", "telegraph",
                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.25",
                "--traffic-gamma", "0.001",
                "--traffic-dt", "1.0",
                "--traffic-decay", "0.0",
                "--corral-radius", "80",
                "--corral-omega-start", "0.01",
                "--corral-omega-stop", "0.62",
                "--corral-omega-steps", "1024",
                "--corral-burn", "4000",
                "--corral-burn-frac", "0.8",
                "--corral-warm", "300",
            ],
        ),
        _exp(
            "04B_corral_tight_rescan",
            "Quantum corral: tight omega rescan around the strongest peak (telegraph).",
            [
                "--out", out_04,
                "--corral",
                "--corral-geom", "sphere",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.25",
                "--traffic-gamma", "0.001",
                "--traffic-dt", "1.0",
                "--traffic-decay", "0.0",
                "--corral-radius", "56",
                "--corral-omega-start", "0.202",
                "--corral-omega-stop", "0.242",
                "--corral-omega-steps", "201",
                "--corral-burn", "1200",
                "--corral-burn-frac", "0.75",
                "--corral-warm", "300",
            ],
        ),
        _exp(
            "05A_isotropy_sigma_sweep",
            "Isotropy sigma sweep (telegraph), single CSV over sigma range (c2=0.31).",
            [
                "--out", out_05,
                "--isotropy",
                "--iso-sweep",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--iso-R", "128",
                "--iso-steps", "512",
                "--iso-amp", "50",
                "--iso-sigma-start", "1.0",
                "--iso-sigma-stop", "8.0",
                "--iso-sigma-steps", "64",
            ],
        ),
        _exp(
            "05B_isotropy_baseline_calibration",
            "Isotropy calibration (telegraph), single-point at sigma=5.6 (c2=0.31).",
            [
                "--out", out_05,
                "--isotropy",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--iso-R", "80",
                "--iso-steps", "260",
                "--iso-sigma", "5.6",
                "--iso-amp", "50",
            ],
        ),
        _exp(
            "06A_oscillator_phase_drift_lensing",
            "Gravitational time dilation via phase drift (oscillators) + lensing preview (telegraph), n=512.",
            [
                "--out", out_06,
                "--oscillator",

                # Background potential build (diffuse)
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",

                "--osc-mass-iters", "220512",
                "--osc-mass-inject", "2.0",

                # Phase drift measurement (telegraph)
                "--osc-steps", "4000",
                "--osc-burn", "800",
                "--osc-warm", "400",
                "--osc-sample-every", "1",
                "--osc-omega", "0.15",
                "--osc-amp", "25.0",

                # Probe radii
                "--osc-r-near", "16",
                "--osc-r-far", "160",

                # Optional lensing pass
                "--osc-lens",
                "--osc-ray-count", "128",
                "--osc-ray-step", "0.75",
                "--osc-ray-max", "640",
            ],
        ),
        _exp(
            "06B_ringdown_passive_resonance_sweep",
            "Passive ringdown sigma sweep (Dirichlet box): single pulse, evolve, record E_final and dominant FFT peak.",
            [
                "--out", out_06,
                "--ringdown",

                # Passive box: telegraph + hard walls.
                "--traffic-mode", "telegraph",
                "--traffic-boundary", "zero",
                "--traffic-sponge-width", "0",
                "--traffic-sponge-strength", "0.0",

                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",

                # Ringdown sweep.
                "--ringdown-steps", "4096",
                "--ringdown-amp", "500",
                "--ringdown-probe-window", "2048",

                # Sigma range (tune later; start with a broad scan).
                "--ringdown-sigma-start", "1.5",
                "--ringdown-sigma-stop", "8.0",
                "--ringdown-sigma-step", "0.5",
            ],
        ),
        _exp(
            "06C_ringdown_passive_resonance_probe",
            "Passive ringdown sigma probe (Dirichlet box): single pulse, evolve, record E_final and dominant FFT peak.",
            [
                "--out", out_06,
                "--ringdown",

                # Passive box: telegraph + hard walls.
                "--traffic-mode", "telegraph",
                "--traffic-boundary", "zero",
                "--traffic-sponge-width", "0",
                "--traffic-sponge-strength", "0.0",

                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",

                # Ringdown sweep.
                "--ringdown-steps", "8192",
                "--ringdown-amp", "500",
                "--ringdown-probe-window", "4096",

                # Sigma range (tune later; start with a broad scan).
                "--ringdown-sigma-start", "6.0",
                "--ringdown-sigma-stop", "6.0",
                "--ringdown-sigma-step", "1.0",
            ],
        ),
        _exp(
            "06D_relativity_twin_paradox_threshold_lock",
            "lightbox baseline: first-cross + flight-gate produces phase-lock (arrival proxy collapses), telegraph, n=512.",
            [
                "--out", out_06,
                "--relativity",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--rel-L", "48",
                "--rel-v-frac", "0.3",
                "--rel-v-frac-ref", "0.1",
                "--rel-steps", "1500",
                "--rel-slab", "8",
                "--rel-amp", "500",
                "--rel-sigma", "2.25",
                "--rel-threshold", "0.01",
                "--rel-refractory", "10"
            ],
        ),
        _exp(
            "06E_relativity_twin_paradox_window_peak",
            "lightbox corrected: windowed peak arrival timing (breaks gate-lock), telegraph, n=512.",
            [
                "--out", out_06,
                "--relativity",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--rel-L", "48",
                "--rel-v-frac", "0.3",
                "--rel-v-frac-ref", "0.1",
                "--rel-steps", "1500",
                "--rel-slab", "8",
                "--rel-amp", "500",
                "--rel-sigma", "2.25",
                "--rel-threshold", "0.01",
                "--rel-refractory", "10",
                "--rel-detect", "window_peak",
                "--rel-start-threshold", "0.01",
                "--rel-peak-window", "12",
                "--rel-accept-threshold", "-1.0"
            ],
        ),
        _exp(
            "06F_relativity_twin_paradox_xy_sponge",
            "lightbox corrected + lateral XY sponge (suppresses echo pollution), telegraph, n=512.",
            [
                "--out", out_06,
                "--relativity",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "8",
                "--traffic-sponge-strength", "1.0",
                "--traffic-sponge-axes", "xy",
                "--rel-L", "48",
                "--rel-v-frac", "0.3",
                "--rel-v-frac-ref", "0.1",
                "--rel-steps", "2000",
                "--rel-slab", "8",
                "--rel-amp", "500",
                "--rel-sigma", "2.25",
                "--rel-threshold", "0.01",
                "--rel-refractory", "10",
                "--rel-detect", "window_peak",
                "--rel-start-threshold", "0.01",
                "--rel-peak-window", "12",
                "--rel-accept-threshold", "-1.0"
            ],
        ),
        _exp(
            "07A_collider_up_down_opp_spin",
            "Collider sanity run (B spin = -1), telegraph.",
            [
                "--out", out_07,
                "--collider",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--collider-spin-b", "-1",
                "--collider-vx", "0.15",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",
                "--collider-steps", "2000",
            ],
        ),
        _exp(
            "07B_collider_down_down_same_spin",
            "Collider sanity run (B spin = +1), telegraph.",
            [
                "--out", out_07,
                "--collider",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--collider-spin-b", "1",
                "--collider-vx", "0.15",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",
                "--collider-steps", "2000",
            ],
        ),
        _exp(
            "07C_collider_upgrade_opp_spin",
            "Collider upgrade run (opp spin) for detector/calorimetry features (same dynamics as 07A).",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.18",
                "--collider-shell1-outer-frac", "0.22",
                "--collider-shell2-inner-frac", "0.22",
                "--collider-shell2-outer-frac", "0.26",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--collider-spin-b", "-1",
                "--collider-vx", "0.15",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",
                "--collider-steps", "2000",
            ],
        ),
        _exp(
            "07D_collider_upgrade_same_spin",
            "Collider upgrade run (same spin) for detector/calorimetry features (same dynamics as 07B).",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.18",
                "--collider-shell1-outer-frac", "0.22",
                "--collider-shell2-inner-frac", "0.22",
                "--collider-shell2-outer-frac", "0.26",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--collider-spin-b", "1",
                "--collider-vx", "0.15",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",
                "--collider-steps", "2000",
            ],
        ),
        _exp(
            "07E_collider_decay_opp_spin",
            "Collider decay-mode (opp spin): detectors + calorimetry, then stop injection post-impact to test persistence.",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.18",
                "--collider-shell1-outer-frac", "0.22",
                "--collider-shell2-inner-frac", "0.22",
                "--collider-shell2-outer-frac", "0.26",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--collider-spin-b", "-1",
                "--collider-vx", "0.15",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",
                "--collider-steps", "3000",

                # Decay / hold: stop driving post-impact.
                "--collider-hold",
                "--collider-hold-steps", "1500",
                "--collider-hold-grace", "50",

                # Stage 3: back-reaction (field-responsive motion).
                "--collider-backreact",
                "--collider-backreact-k", "0.020",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
            ],
        ),
        _exp(
            "07F_collider_decay_same_spin",
            "Collider decay-mode (same spin): detectors + calorimetry, then stop injection post-impact to test persistence.",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.18",
                "--collider-shell1-outer-frac", "0.22",
                "--collider-shell2-inner-frac", "0.22",
                "--collider-shell2-outer-frac", "0.26",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",
                "--collider-spin-b", "1",
                "--collider-vx", "0.15",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",
                "--collider-steps", "3000",

                # Decay / hold: stop driving post-impact.
                "--collider-hold",
                "--collider-hold-steps", "1500",
                "--collider-hold-grace", "50",

                # Stage 3: back-reaction (field-responsive motion).
                "--collider-backreact",
                "--collider-backreact-k", "0.020",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
            ],
        ),
        _exp(
            "07G_collider_scatter_opp_spin",
            "Scattering (opp spin): near-miss impact parameter b>0 with full-vector back-reaction + octant flux (no hold).",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.18",
                "--collider-shell1-outer-frac", "0.22",
                "--collider-shell2-inner-frac", "0.22",
                "--collider-shell2-outer-frac", "0.26",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Opposite spin.
                "--collider-spin-b", "-1",
                "--collider-vx", "0.15",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",

                # Stage 4: near-miss / scattering geometry.
                "--collider-impact-b", "4.0",
                "--collider-impact-bz", "0.0",

                # Stage 3+: back-reaction (full-vector) + octant flux.
                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.020",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                "--collider-steps", "2500",
            ],
        ),
        _exp(
            "07H_collider_scatter_same_spin",
            "Scattering (same spin): near-miss impact parameter b>0 with full-vector back-reaction + octant flux (no hold).",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.18",
                "--collider-shell1-outer-frac", "0.22",
                "--collider-shell2-inner-frac", "0.22",
                "--collider-shell2-outer-frac", "0.26",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Same spin.
                "--collider-spin-b", "1",
                "--collider-vx", "0.15",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",

                # Stage 4: near-miss / scattering geometry.
                "--collider-impact-b", "4.0",
                "--collider-impact-bz", "0.0",

                # Stage 3+: back-reaction (full-vector) + octant flux.
                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.020",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                "--collider-steps", "2500",
            ],
        ),
        _exp(
            "07I_collider_bind_opp_spin",
            "Binding (opp spin): slow approach + strong coupling; look for capture/closed-orbit instead of escape (telegraph).",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.18",
                "--collider-shell1-outer-frac", "0.22",
                "--collider-shell2-inner-frac", "0.22",
                "--collider-shell2-outer-frac", "0.26",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Opposite spin.
                "--collider-spin-b", "-1",
                "--collider-vx", "0.05",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",

                # Stage 4 geometry (impact parameter).
                "--collider-impact-b", "4.0",
                "--collider-impact-bz", "0.0",

                # Stage 3+/4+: back-reaction (full-vector) + octant flux.
                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.20",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                "--collider-steps", "5000",
            ],
        ),
        _exp(
            "07J_collider_bind_same_spin",
            "Binding (same spin): slow approach + strong coupling; contrast capture vs repulsion (telegraph).",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.18",
                "--collider-shell1-outer-frac", "0.22",
                "--collider-shell2-inner-frac", "0.22",
                "--collider-shell2-outer-frac", "0.26",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Same spin.
                "--collider-spin-b", "1",
                "--collider-vx", "0.05",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",

                # Stage 4 geometry (impact parameter).
                "--collider-impact-b", "4.0",
                "--collider-impact-bz", "0.0",

                # Stage 3+/4+: back-reaction (full-vector) + octant flux.
                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.20",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                "--collider-steps", "5000",
            ],
        ),

        _exp(
            "07K_collider_nucleus_opp_spin",
            "Nucleus (opp spin): add central source (third body) to test assisted capture / orbiting.",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.18",
                "--collider-shell1-outer-frac", "0.22",
                "--collider-shell2-inner-frac", "0.22",
                "--collider-shell2-outer-frac", "0.26",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Opposite spin.
                "--collider-spin-b", "-1",
                "--collider-vx", "0.05",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",

                # Stage 4 geometry (impact parameter).
                "--collider-impact-b", "4.0",
                "--collider-impact-bz", "0.0",

                # Stage 3+/4+: back-reaction (full-vector) + octant flux.
                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.20",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                # Add nucleus.
                "--collider-nucleus",

                "--collider-steps", "5000",
            ],
        ),

        _exp(
            "07L_collider_nucleus_same_spin",
            "Nucleus (same spin): add central source (third body) to test assisted capture / orbiting.",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.18",
                "--collider-shell1-outer-frac", "0.22",
                "--collider-shell2-inner-frac", "0.22",
                "--collider-shell2-outer-frac", "0.26",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Same spin.
                "--collider-spin-b", "1",
                "--collider-vx", "0.05",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",

                # Stage 4 geometry (impact parameter).
                "--collider-impact-b", "4.0",
                "--collider-impact-bz", "0.0",

                # Stage 3+/4+: back-reaction (full-vector) + octant flux.
                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.20",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                # Add nucleus.
                "--collider-nucleus",

                "--collider-steps", "5000",
            ],
        ),
        _exp(
            "07M_collider_hydrogen_halo_soft",
            "Hydrogen (soft halo): nucleus + local damping halo to encourage single-body capture/orbit (telegraph).",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.12",
                "--collider-shell1-outer-frac", "0.16",
                "--collider-shell2-inner-frac", "0.16",
                "--collider-shell2-outer-frac", "0.20",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Hydrogen: single walker (disable B).
                "--collider-no-enable-b",
                # Use same-spin by default (more viscous / “sticky”), but the goal here is nucleus capture.
                "--collider-spin-b", "1",
                "--collider-vx", "0.040",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",

                # Near-miss so the walker can be deflected into a bound orbit rather than a straight plunge.
                "--collider-impact-b", "8.0",
                "--collider-impact-bz", "0.0",

                # Back-reaction (full-vector) so the nucleus can curve the trajectory.
                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.20",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                # Permanent nucleus (anchor).
                "--collider-nucleus",

                # Local damping halo (phenomenological radiative cooling).
                "--collider-halo",
                "--collider-halo-r", "72",
                "--collider-halo-strength", "0.010",

                "--collider-steps", "8000",
            ],
        ),
        _exp(
            "07N_collider_hydrogen_halo_strong",
            "Hydrogen (strong halo): nucleus + stronger local damping halo to probe binding threshold (telegraph).",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.12",
                "--collider-shell1-outer-frac", "0.16",
                "--collider-shell2-inner-frac", "0.16",
                "--collider-shell2-outer-frac", "0.20",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Hydrogen: single walker (disable B).
                "--collider-no-enable-b",
                # Same-spin baseline; contrast later by flipping B if desired.
                "--collider-spin-b", "1",
                "--collider-vx", "0.040",
                "--collider-orbit-radius", "80",
                "--collider-orbit-omega", "0.12",

                "--collider-impact-b", "8.0",
                "--collider-impact-bz", "0.0",

                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.20",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                "--collider-nucleus",

                # Stronger halo damping to force capture if a bound state exists.
                "--collider-halo",
                "--collider-halo-r", "72",
                "--collider-halo-strength", "0.030",

                "--collider-steps", "8000",
            ],
        ),
        _exp(
            "07O_collider_hydrogen_deep_orbit_full_q",
            "Hydrogen (capture): Bound State. Deep orbit (r=32) + full nucleus charge (q=1.0) + aggressive halo capture.",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.12",
                "--collider-shell1-outer-frac", "0.16",
                "--collider-shell2-inner-frac", "0.16",
                "--collider-shell2-outer-frac", "0.20",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Hydrogen: single walker (disable B).
                "--collider-no-enable-b",

                "--collider-vx", "0.05",
                "--collider-orbit-radius", "32",
                "--collider-orbit-omega", "0.12",

                # Head-on geometry (impact b=0): test direct capture under aggressive halo.
                "--collider-impact-b", "0.0",
                "--collider-impact-bz", "0.0",

                # Back-reaction (full-vector).
                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.20",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                # Permanent nucleus (DC) with softened charge.
                "--collider-nucleus",
                "--collider-nucleus-mode", "dc",
                "--collider-nucleus-q", "1.0",

                # Aggressive halo (phenomenological radiative cooling proxy).
                "--collider-halo",
                "--collider-halo-center", "nucleus",
                "--collider-halo-r", "96",
                "--collider-halo-strength", "0.25",
                "--collider-halo-profile", "exp",

                "--collider-steps", "8000",
            ],
        ),
        _exp(
            "07P_collider_hydrogen_deep_orbit_half_q",
            "Hydrogen (capture): Bound State. Deep orbit (r=32) + half nucleus charge (q=0.5) + aggressive halo capture.",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.12",
                "--collider-shell1-outer-frac", "0.16",
                "--collider-shell2-inner-frac", "0.16",
                "--collider-shell2-outer-frac", "0.20",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Hydrogen: single walker (disable B).
                "--collider-no-enable-b",

                "--collider-vx", "0.05",
                "--collider-orbit-radius", "32",
                "--collider-orbit-omega", "0.12",

                # Head-on geometry (impact b=0): test capture with reduced nucleus charge.
                "--collider-impact-b", "0.0",
                "--collider-impact-bz", "0.0",

                # Back-reaction (full-vector).
                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.20",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                # Permanent nucleus (DC) with weaker charge.
                "--collider-nucleus",
                "--collider-nucleus-mode", "dc",
                "--collider-nucleus-q", "0.5",

                # Aggressive halo.
                "--collider-halo",
                "--collider-halo-center", "nucleus",
                "--collider-halo-r", "96",
                "--collider-halo-strength", "0.25",
                "--collider-halo-profile", "exp",

                "--collider-steps", "8000",
            ],
        ),
        _exp(
            "07Q_collider_hydrogen_goldilocks_long",
            "Hydrogen (verification): Long-duration (50k steps) verification of the 07O (deep-orbit, full-q) bound state. Tests long-term stability of the captured composite.",
            [
                "--out", out_07,
                "--collider",
                "--collider-detectors",
                "--collider-shell-stride", "2",
                "--collider-shell1-inner-frac", "0.12",
                "--collider-shell1-outer-frac", "0.16",
                "--collider-shell2-inner-frac", "0.16",
                "--collider-shell2-outer-frac", "0.20",

                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "32",
                "--traffic-sponge-strength", "1.0",

                # Single Walker (Hydrogen)
                "--collider-no-enable-b",
                "--collider-vx", "0.05",
                
                # THE GOLDILOCKS GEOMETRY
                "--collider-orbit-radius", "32", 
                "--collider-orbit-omega", "0.12",
                "--collider-impact-b", "0.0",
                "--collider-impact-bz", "0.0",

                # Back-reaction coupling (trajectory feedback).
                "--collider-backreact",
                "--collider-backreact-axes", "xyz",
                "--collider-backreact-k", "0.20",
                "--collider-backreact-mode", "repel",
                "--collider-backreact-vmax", "0.250",
                "--collider-octants",

                # Nucleus
                "--collider-nucleus",
                "--collider-nucleus-mode", "dc",
                "--collider-nucleus-q", "1.0",

                # Damping halo (phenomenological radiative cooling proxy).
                "--collider-halo",
                "--collider-halo-center", "nucleus",
                "--collider-halo-r", "96",
                "--collider-halo-strength", "0.25",
                "--collider-halo-profile", "exp",

                # THE MARATHON
                "--collider-steps", "50000",
            ],
        ),
        _exp(
            "08A_soliton_scan",
            "Non-linear Klein-Gordon/φ⁴ scan (sponge): sweep λ at fixed k to find a long-lived localised core (nonlinear).",
            [
                "--out", out_08,
                "--soliton",

                # Non-linear field mode.
                "--traffic-mode", "nonlinear",

                # Self-sustainment test: shed radiation must leave the grid.
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "24",
                "--traffic-sponge-strength", "0.2",

                # Baseline grid + integrator.
                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",

                # Soliton scan controls (pass explicitly; do not rely on defaults).
                "--soliton-steps", "2000",
                "--soliton-sigma", "4.0",
                "--soliton-amp", "50.0",

                # Scaffold: vacuum stiffness (k) and phi^4 self-interaction (lambda).
                # Scan high enough to reach the collapse edge; dt reduced to avoid NaN blow-ups.
                # If we still only see dispersion, raise --soliton-k next; if we see NaNs, lower lambda-stop or dt.
                "--soliton-k", "0.1",
                "--soliton-lambda-start", "1e-3",
                "--soliton-lambda-stop", "1e-1",
                "--soliton-lambda-steps", "33",
            ],
        ),
        _exp(
            "08B_symmetry_breaking_baseline",
            "Double-well baseline (k<0, λ>0, sponge): validate symmetry breaking / nucleation; logs far-field drift and φ stats (nonlinear).",
            [
                "--out", out_08,
                "--soliton",

                # Non-linear field mode.
                "--traffic-mode", "nonlinear",

                # Self-sustainment: shed radiation must leave.
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "24",
                "--traffic-sponge-strength", "0.2",

                # Grid + integrator (conservative).
                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",

                # Soliton probe (single-shot per lambda).
                "--soliton-steps", "2000",
                "--soliton-sigma", "4.0",
                "--soliton-amp", "50.0",

                # Double-well scaffold:
                # k is negative (buckling), lambda is positive (bounded quartic).
                "--soliton-k", "-0.1",
                "--soliton-lambda-start", "1e-3",
                "--soliton-lambda-stop", "1e-1",
                "--soliton-lambda-steps", "33",
            ],
        ),
        _exp(
            "08C_symmetry_breaking_local_probe",
            "Fast local probe near the symmetry-breaking boundary (k slightly negative): checks for localised cores vs global nucleation (nonlinear).",
            [
                "--out", out_08,
                "--soliton",

                # Non-linear field mode.
                "--traffic-mode", "nonlinear",

                # Self-sustainment: shed radiation must leave.
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "24",
                "--traffic-sponge-strength", "0.2",

                # Grid + integrator (short run, same base physics).
                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",

                # Soliton probe (single-shot, fast).
                "--soliton-steps", "2000",
                "--soliton-sigma", "4.0",
                "--soliton-amp", "50.0",

                # Near-critical double-well: gentle negative k with a tight lambda band around VEV~O(1).
                "--soliton-k", "-0.001",
                "--soliton-lambda-start", "5e-4",
                "--soliton-lambda-stop", "2e-3",
                "--soliton-lambda-steps", "17",
            ],
        ),
        _exp(
            "08D_symmetry_breaking_local_probe_vev_init",
            "Local probe initialised at the broken-vacuum VEV (init-vev, + sign): tests localised core survival vs background sag/drift (nonlinear).",
            [
                "--out", out_08,
                "--soliton",

                # Non-linear field mode (Phase 8).
                "--traffic-mode", "nonlinear",

                # Self-sustainment: shed radiation must leave.
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "24",
                "--traffic-sponge-strength", "0.2",

                # Grid + integrator (short run, same base physics).
                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",

                # Soliton probe (single-shot, fast).
                "--soliton-steps", "2000",
                "--soliton-sigma", "4.0",
                "--soliton-amp", "50.0",

                # 08D: initialise around the broken-vacuum VEV (new soliton init mode).
                "--soliton-init-vev",
                "--soliton-vev-sign", "+1",

                # Slightly deeper into k<0 than 08C, with a tight lambda window.
                "--soliton-k", "-0.001",
                "--soliton-lambda-start", "5e-4",
                "--soliton-lambda-stop", "2e-3",
                "--soliton-lambda-steps", "17",
            ],
        ),
        _exp(
            "08E_sine_gordon_breather_baseline",
            "Sine-Gordon baseline (sponge, traffic-mode=sine_gordon): sweep SG stiffness k and amp; λ is fail-loud ignored.",
            [
                "--out", out_08,
                "--soliton",

                # Sine-Gordon field mode.
                "--traffic-mode", "sine_gordon",

                # Self-sustainment test: shed radiation must leave the grid.
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "24",
                "--traffic-sponge-strength", "0.2",

                # Baseline grid + integrator.
                "--n", "96",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",

                # Soliton probe.
                "--soliton-steps", "10000",
                "--soliton-sigma-start", "3.2",
                "--soliton-sigma-stop", "3.4",
                "--soliton-sigma-steps", "9",

                # Sine-Gordon sweeps (SG-specific flags).
                # Sweep stiffness k (the SG restoring term) across a sensible band.
                "--soliton-sg-k-start", "0.08",
                "--soliton-sg-k-stop", "0.08",
                "--soliton-sg-k-steps", "1",

                # Fix amplitude at a mid-range value; adjust later if needed.
                "--soliton-sg-amp-start", "7.5",
                "--soliton-sg-amp-stop", "9.0",
                "--soliton-sg-amp-steps", "13",

                # Lambda is ignored by sine_gordon; keep the contract explicit (fail-loud in soliton.py/cli.py).
                "--soliton-lambda-start", "0.0",
                "--soliton-lambda-stop", "0.0",
                "--soliton-lambda-steps", "1",
            ],
        ),
        _exp(
            "08F_sine_gordon_breather_stable",
            "Sine-Gordon stable soliton hunt (sponge, traffic-mode=sine_gordon).",
            [
                "--out", out_08,
                "--soliton",

                # Sine-Gordon field mode.
                "--traffic-mode", "sine_gordon",

                # Self-sustainment test: shed radiation must leave the grid.
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "24",
                "--traffic-sponge-strength", "0.2",

                # Baseline grid + integrator.
                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",

                # Soliton probe.
                "--soliton-steps", "50000",
                "--soliton-sigma", "4.2",

                # Sine-Gordon sweeps (SG-specific flags).
                # Sweep stiffness k (the SG restoring term) across a sensible band.
                "--soliton-sg-k-start", "0.4484",
                "--soliton-sg-k-stop", "0.4484",
                "--soliton-sg-k-steps", "1",

                # Fix amplitude at a mid-range value; adjust later if needed.
                "--soliton-sg-amp-start", "5.0",
                "--soliton-sg-amp-stop", "5.0",
                "--soliton-sg-amp-steps", "1",

                # Lambda is ignored by sine_gordon; keep the contract explicit (fail-loud in soliton.py/cli.py).
                "--soliton-lambda-start", "0.0",
                "--soliton-lambda-stop", "0.0",
                "--soliton-lambda-steps", "1",
            ],
        ),
        _exp(
            "08G_sine_gordon_breather_extract",
            "Sine-Gordon stable soliton extraction (sponge, traffic-mode=sine_gordon).",
            [
                "--out", out_08,
                "--soliton",

                # Sine-Gordon field mode.
                "--traffic-mode", "sine_gordon",

                # Self-sustainment test: shed radiation must leave the grid.
                "--traffic-boundary", "sponge",
                "--traffic-sponge-width", "24",
                "--traffic-sponge-strength", "0.2",

                # Baseline grid + integrator.
                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",

                # Soliton probe.
                "--soliton-steps", "10000",
                "--soliton-sigma", "3.4",

                # Sine-Gordon sweeps (SG-specific flags).
                # Sweep stiffness k (the SG restoring term) across a sensible band.
                "--soliton-sg-k-start", "0.08",
                "--soliton-sg-k-stop", "0.08",
                "--soliton-sg-k-steps", "1",

                # Fix amplitude at a mid-range value; adjust later if needed.
                "--soliton-sg-amp-start", "8.6",
                "--soliton-sg-amp-stop", "8.6",
                "--soliton-sg-amp-steps", "1",

                # Extraction exports (preferred snapshot + sprite extraction).
                "--dump-sprite",

                # Lambda is ignored by sine_gordon; keep the contract explicit (fail-loud in soliton.py/cli.py).
                "--soliton-lambda-start", "0.0",
                "--soliton-lambda-stop", "0.0",
                "--soliton-lambda-steps", "1",
            ],
        ),
        _exp(
            "09A_collidersg_head_on",
            "Sine-Gordon sprite collider — N=192 Optimised",
            [
                "--out", out_09,

                # ColliderSG runner
                "--collidersg",
                "--collidersg-scenario", "A",
                # UPDATE: Head-on velocities: sprite 1 right (+x), sprite 2 left (-x)
                "--collidersg-sprites", "[{\"sid\":1,\"pos\":[60,96,96],\"vel\":[0.4,0.0,0.0],\"phase\":0.0},{\"sid\":2,\"pos\":[132,96,96],\"vel\":[-0.4,0.0,0.0],\"phase\":3.14159}]",
                "--collidersg-sprite-asset", os.path.join(out_08, "08G_sine_gordon_breather_extract", "20260217_062107", "_sprites", "08G_sine_gordon_breather_extract_n192_20260217_062107__sprite_step29505_L017_R008.h5"),
                "--collidersg-steps", "30000",
                "--collidersg-log-every", "1",
                "--collidersg-track-r", "14",
                "--collidersg-peak-thresh", "0.001",
                "--collidersg-phi-abs-every", "100",

                # Base physics
                "--traffic-mode", "sine_gordon",
                "--collidersg-boundary", "sponge",
                "--collidersg-sponge-width", "24",
                "--collidersg-sponge-strength", "0.2",

                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",
            ],
        ),
        _exp(
            "09B_collidersg_orbit",
            "Sine-Gordon sprite collider — orbit / near-miss interaction (N=192 optimised).",
            [
                "--out", out_09,

                # ColliderSG runner
                "--collidersg",
                "--collidersg-scenario", "B",

                # Orbit / near-miss geometry: offset in y to introduce an impact parameter.
                # Sprite 1 moves right (+x), sprite 2 moves left (-x), same phase.
                "--collidersg-sprites", "[{\"sid\":1,\"pos\":[60,91,96],\"vel\":[0.2,0.0,0.0],\"phase\":0.0},{\"sid\":2,\"pos\":[132,101,96],\"vel\":[-0.2,0.0,0.0],\"phase\":0.0}]",

                # Same canonical N=192 sprite asset as 09A.
                "--collidersg-sprite-asset", os.path.join(out_08, "08G_sine_gordon_breather_extract", "20260217_062107", "_sprites", "08G_sine_gordon_breather_extract_n192_20260217_062107__sprite_step29505_L017_R008.h5"),

                "--collidersg-steps", "30000",
                "--collidersg-log-every", "1",
                "--collidersg-track-r", "14",
                "--collidersg-peak-thresh", "0.001",
                "--collidersg-phi-abs-every", "100",

                # Base physics
                "--traffic-mode", "sine_gordon",
                "--collidersg-boundary", "sponge",
                "--collidersg-sponge-width", "24",
                "--collidersg-sponge-strength", "0.2",

                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",
            ],
        ),
        _exp(
            "09C_collidersg_molecule",
            "Sine-Gordon sprite collider — molecule / close-pair interaction (anti-phase), N=192 optimised.",
            [
                "--out", out_09,

                # ColliderSG runner
                "--collidersg",
                "--collidersg-scenario", "C",

                # Close pair (anti-phase) near the centre: intended to form a bound molecule / breathing dimer.
                # Both start at rest; separation is small enough to interact immediately.
                "--collidersg-sprites", "[{\"sid\":1,\"pos\":[76,96,96],\"vel\":[0.0,0.0,0.0],\"phase\":0.0},{\"sid\":2,\"pos\":[116,96,96],\"vel\":[0.0,0.0,0.0],\"phase\":3.14159}]",

                # Same canonical N=192 sprite asset as 09A/09B.
                "--collidersg-sprite-asset", os.path.join(out_08, "08G_sine_gordon_breather_extract", "20260217_062107", "_sprites", "08G_sine_gordon_breather_extract_n192_20260217_062107__sprite_step29505_L017_R008.h5"),

                "--collidersg-steps", "30000",
                "--collidersg-log-every", "1",
                "--collidersg-track-r", "14",
                "--collidersg-peak-thresh", "0.001",
                "--collidersg-phi-abs-every", "100",

                # Base physics
                "--traffic-mode", "sine_gordon",
                "--collidersg-boundary", "sponge",
                "--collidersg-sponge-width", "24",
                "--collidersg-sponge-strength", "0.2",

                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",
            ],
        ),
        _exp(
            "09D_collidersg_kink_walls",
            "Sine-Gordon collider — planar kink-wall scattering (hard-ball sanity), N=192.",
            [
                "--out", out_09,

                # ColliderSG runner
                "--collidersg",
                "--collidersg-scenario", "D",

                # Kink walls: YZ-spanning domain walls moving along ±x. No sprite asset required.
                # Same-charge (phase ~0) is the repulsive / hard-ball baseline.
                "--collidersg-sprites", "[{\"sid\":1,\"kind\":\"kink_wall\",\"pos\":[60,96,96],\"vel\":[0.2,0.0,0.0],\"phase\":0.0},{\"sid\":2,\"kind\":\"kink_wall\",\"pos\":[132,96,96],\"vel\":[-0.2,0.0,0.0],\"phase\":0.0}]",

                "--collidersg-steps", "10000",
                "--collidersg-log-every", "1",
                "--collidersg-track-r", "14",
                "--collidersg-peak-thresh", "0.001",
                "--collidersg-phi-abs-every", "100",
                "--collidersg-k", "0.05",

                # Base physics
                "--traffic-mode", "sine_gordon",
                "--collidersg-boundary", "neumann",

                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",
            ],
        ),
        _exp(
            "09E_collidersg_transistor_binary_baseline",
            "Sine-Gordon collider — binary transistor, N=192.",
            [
                "--out", out_09,

                # ColliderSG runner
                "--collidersg",
                "--collidersg-scenario", "E",

                # Kink walls: YZ-spanning domain walls moving along ±x. No sprite asset required.
                # Same-charge (phase ~0) is the repulsive / hard-ball baseline.
                "--collidersg-sprites", "[{\"sid\":1,\"kind\":\"kink_wall\",\"pos\":[60,96,96],\"vel\":[0.4,0.0,0.0],\"phase\":0.0},{\"sid\":2,\"kind\":\"kink_wall\",\"pos\":[132,96,96],\"vel\":[-0.4,0.0,0.0],\"phase\":0.0}]",

                "--collidersg-steps", "5000",
                "--collidersg-log-every", "1",
                "--collidersg-track-r", "6",
                "--collidersg-peak-thresh", "0.001",
                "--collidersg-phi-abs-every", "100",
                "--collidersg-k", "0.05",
                "--collidersg-k-outside", "5.0",
                "--collidersg-wire-y0", "90",
                "--collidersg-wire-y1", "102",
                "--collidersg-wire-z0", "90",
                "--collidersg-wire-z1", "102",
                "--collidersg-wire-bevel", "0",

                # Base physics
                "--traffic-mode", "sine_gordon",
                "--collidersg-boundary", "neumann",

                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",
            ],
        ),
        _exp(
            "09F_collidersg_transistor_binary_phasetest",
            "Sine-Gordon collider — binary transistor, N=192.",
            [
                "--out", out_09,

                # ColliderSG runner
                "--collidersg",
                "--collidersg-scenario", "E",

                # Kink walls: YZ-spanning domain walls moving along ±x. No sprite asset required.
                # Different-charge (phase ~0 / PI)
                "--collidersg-sprites", "[{\"sid\":1,\"kind\":\"kink_wall\",\"pos\":[60,96,96],\"vel\":[0.4,0.0,0.0],\"phase\":0.0},{\"sid\":2,\"kind\":\"kink_wall\",\"pos\":[132,96,96],\"vel\":[-0.4,0.0,0.0],\"phase\":3.14159265359}]",

                "--collidersg-steps", "5000",
                "--collidersg-log-every", "1",
                "--collidersg-track-r", "6",
                "--collidersg-peak-thresh", "0.001",
                "--collidersg-phi-abs-every", "100",
                "--collidersg-k", "0.05",
                "--collidersg-k-outside", "5.0",
                "--collidersg-wire-y0", "90",
                "--collidersg-wire-y1", "102",
                "--collidersg-wire-z0", "90",
                "--collidersg-wire-z1", "102",
                "--collidersg-wire-bevel", "0",

                # Base physics
                "--traffic-mode", "sine_gordon",
                "--collidersg-boundary", "neumann",

                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",
            ],
        ),
        _exp(
            "09G_collidersg_transistor_calibration",
            "Sine-Gordon collider — wire speed calibration (single wall), N=192.",
            [
                "--out", out_09,

                # ColliderSG runner
                "--collidersg",
                "--collidersg-scenario", "E",

                # Single kink wall in a confined wire: measure realised vx before reflection.
                "--collidersg-sprites", "[{\"sid\":1,\"kind\":\"kink_wall\",\"pos\":[60,96,96],\"vel\":[0.52,0.0,0.0],\"phase\":0.0}]",

                # Stop before boundary reflection (~4300 steps observed in 09E/09F).
                "--collidersg-steps", "1500",
                "--collidersg-log-every", "1",
                "--collidersg-track-r", "6",
                "--collidersg-peak-thresh", "0.001",
                "--collidersg-phi-abs-every", "100",
                "--collidersg-k", "0.08",
                "--collidersg-k-outside", "5.0",
                "--collidersg-wire-y0", "90",
                "--collidersg-wire-y1", "102",
                "--collidersg-wire-z0", "90",
                "--collidersg-wire-z1", "102",
                "--collidersg-wire-bevel", "0",

                # Base physics
                "--traffic-mode", "sine_gordon",
                "--collidersg-boundary", "neumann",

                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",
            ],
        ),
        _exp(
            "09H_collidersg_transistor_tjunction",
            "Sine-Gordon collider — first T-junction routing probe (single +2π wall), N=192.",
            [
                "--out", out_09,

                # ColliderSG runner
                "--collidersg",
                "--collidersg-scenario", "E",

                # Single kink wall injected into a T-junction wire: observe reflect / split / route.
                "--collidersg-sprites", "[{\"sid\":1,\"kind\":\"kink_wall\",\"pos\":[60,96,96],\"vel\":[0.52,0.0,0.0],\"phase\":0.0},{\"sid\":2,\"kind\":\"kink_wall\",\"pos\":[120,60,96],\"vel\":[0.0,0.0,0.0],\"phase\":0.0,\"track_only\":true}]",

                # Long enough to reach the junction and show post-interaction behaviour (avoid boundary).
                "--collidersg-steps", "2500",
                "--collidersg-log-every", "1",
                "--collidersg-track-r", "6",
                "--collidersg-peak-thresh", "0.001",
                "--collidersg-phi-abs-every", "100",
                "--collidersg-k", "0.08",
                "--collidersg-k-outside", "5.0",
                "--collidersg-wire-y0", "90",
                "--collidersg-wire-y1", "102",
                "--collidersg-wire-z0", "90",
                "--collidersg-wire-z1", "102",
                "--collidersg-wire-bevel", "0",
                "--collidersg-wire-geom", "t_junction",
                "--collidersg-junction-x", "120",
                "--collidersg-branch-len", "48",
                "--collidersg-branch-thick", "2",

                # Base physics
                "--traffic-mode", "sine_gordon",
                "--collidersg-boundary", "neumann",

                "--n", "192",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "0.05",
                "--traffic-gamma", "0.0",
                "--traffic-decay", "0.0",
            ],
        ),
        _exp(
            "ZZZ_pipeline_baseline_visual_test",
            "Steady-field pipeline + Shapiro lockdown (single seed).",
            [
                "--out", out_ZZ,
                "--n", "128",
                "--steps", "0",
                "--traffic-iters", "100000",
                "--traffic-rate", "0.10",
                "--traffic-inject", "1.0",
                "--traffic-decay", "0.0",
                "--k-index", "1.0",
                "--X0", "8000",
                "--ds", "1.0",
                "--eps", "0.005",
                "--delta-load",
                "--dump-radial", "radial.csv",
                "--dump-radial-fit", "radial_fit.csv",
                "--dump-shapiro", "shapiro.csv",
                "--dump-hdf5",
                "--liveview",
            ],
        ),
        _exp(
            "ZZ6_relativity_twin_paradox",
            "Relativity light-clock / twin paradox (telegraph), n=512.",
            [
                "--out", out_ZZ,
                "--relativity",
                "--traffic-mode", "telegraph",
                "--n", "512",
                "--traffic-iters", "1",
                "--traffic-c2", "0.31",
                "--traffic-dt", "1.0",
                "--traffic-gamma", "0.001",
                "--traffic-decay", "0.0",
                "--rel-L", "48",
                "--rel-v-frac", "0.3",
                "--rel-v-frac-ref", "0.1",
                "--rel-steps", "1500",
                "--rel-slab", "8",
                "--rel-amp", "500",
                "--rel-sigma", "2.25",
                "--rel-threshold", "0.01",
                "--rel-refractory", "10"
            ],
        ),
    ]

    tab: Dict[str, Experiment] = {}
    for e in exps:
        if e.name in tab:
            raise ValueError("duplicate experiment name: %s" % (e.name,))
        tab[e.name] = e
    return tab


def _print_list(tab: Dict[str, Experiment]) -> None:
    names = sorted(tab.keys())
    max_name = 0
    for name in names:
        if len(name) > max_name:
            max_name = len(name)
    # Keep a stable look while avoiding over-wide padding.
    col_w = max(28, min(44, max_name + 2))

    cat_labels = {
        "00": "00 - Heavy Walker",
        "01": "01 - Baseline Field + Diagnostics",
        "02": "02 - Interference Controls",
        "03": "03 - Static Interaction Proxy",
        "04": "04 - Confinement Spectrum",
        "05": "05 - Calibration",
        "06": "06 - Relativity Proxy",
        "07": "07 - Particle Dynamics",
        "08": "08 - Non-Linear Soliton Search",
        "09": "09 - Sine-Gordon Sprite Interactions",

        "ZZ": "ZZ - Code Testing",
    }

    last_cat = ""
    print("")
    for name in names:
        cat = name[:2]
        if cat != last_cat:
            if last_cat != "":
                print("")
            print(cat_labels.get(cat, cat))
            last_cat = cat
        e = tab[name]
        print(f"  {e.name:<{col_w}}  {e.summary}")

    # Ensure a trailing newline for nicer terminal output.
    print("")


def _resolve_experiment_name(token: str, tab: Dict[str, Experiment]) -> str:
    t = str(token).strip()
    if t == "":
        raise ValueError("empty experiment selector")
    if t in tab:
        return t

    # Allow shorthand codes like "01A" that map to a unique "01A_*" experiment.
    prefix = t + "_"
    cands = [k for k in tab.keys() if k.startswith(prefix)]
    if len(cands) == 1:
        return cands[0]
    if len(cands) == 0:
        raise ValueError("unknown experiment: %s (use --list)" % (t,))

    cands_sorted = sorted(cands)
    raise ValueError(
        "ambiguous experiment selector: %s matches %s" % (t, ", ".join(cands_sorted))
    )


def _run_one(name: str) -> int:
    tab = experiment_table()
    resolved = _resolve_experiment_name(name, tab)
    e = tab[resolved]
    return _run_core(e)


def _run_all(names: List[str]) -> int:
    tab = experiment_table()
    rc = 0
    for name in names:
        resolved = _resolve_experiment_name(name, tab)
        print("\n[experiment] %s — %s" % (resolved, tab[resolved].summary))
        r = _run_core(tab[resolved])
        if r != 0:
            rc = r
            print("[experiment] FAILED name=%s rc=%d" % (name, int(r)))
            break
    return int(rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", type=str, default="")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument(
        "--suite",
        type=str,
        default="core",
        choices=["core"],
        help="named suite order (currently: core)",
    )

    args = ap.parse_args()
    tab = experiment_table()

    if bool(args.list):
        _print_list(tab)
        return

    if str(args.run).strip() != "":
        raise SystemExit(_run_one(str(args.run).strip()))

    if bool(args.run_all):
        if str(args.suite).strip() != "core":
            raise ValueError("unknown suite")
        order = [k for k in sorted(tab.keys()) if not str(k).startswith("ZZ")]
        raise SystemExit(_run_all(order))

    ap.print_help()


if __name__ == "__main__":
    main()
