# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""cli.py — argument parsing + normalisation for CAELIX core

This module owns:
- argparse flag definitions
- output path normalisation (never assume cwd)
- mapping args -> PipelineParams

core.py should stay thin: deps/thread defaults, imports, then dispatch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Iterable

from params import LatticeParams, PipelineParams, RingdownParams, SolitonParams, TrafficParams
from utils import resolve_out_dir, resolve_out_path


def _resolve_optional_out_file(here_file: str, out_dir: str, p: Any) -> str:
    v = str(p).strip()
    if v == "":
        return ""
    if os.path.isabs(v):
        return v
    if ("/" not in v) and ("\\" not in v):
        return os.path.join(str(out_dir), v)
    return resolve_out_path(here_file, v)


def _normalise_out_flags(args: Any, here_file: str, names: Iterable[str]) -> None:
    out_dir = str(getattr(args, "out"))
    csv_dir = os.path.join(out_dir, "_csv")
    log_dir = os.path.join(out_dir, "_log")

    # Any flag that represents a CSV artefact should default into <out>/_csv
    # when passed as a bare filename (no slashes). Only --log defaults into <out>/_log.
    csv_names = {
        "dump_radial",
        "dump_radial_fit",
        "dump_shapiro",
        "dump_walker",
        "walker_sweep_out",
        "coulomb_out",
        "ds_out",
        "corral_out",
        "iso_out",
        "rel_out",
        "osc_out",
        "ringdown_out",
        "soliton_out",
        "collidersg_out",
        "collider_out",
    }

    for name in names:
        cur = getattr(args, name, "")
        v = str(cur).strip()
        if v != "" and ("/" not in v) and ("\\" not in v):
            if name == "log":
                setattr(args, name, os.path.join(str(log_dir), v))
                continue
            if name in csv_names:
                setattr(args, name, os.path.join(str(csv_dir), v))
                continue
        setattr(args, name, _resolve_optional_out_file(here_file, out_dir, cur))


def build_parser(*, here_file: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(here_file))
    default_out = os.path.join(here, "_Output")
    defaults = PipelineParams()

    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--p-seed", type=float, default=0.0008)
    ap.add_argument("--lattice-init", dest="lattice_init", type=str, default="sparse", choices=["sparse", "multiscale"])
    ap.add_argument("--traffic-iters", type=int, default=30_000)
    ap.add_argument("--traffic-rate", type=float, default=None)
    ap.add_argument("--traffic-rate-rise", type=float, default=None)
    ap.add_argument("--traffic-rate-fall", type=float, default=None)
    ap.add_argument("--traffic-inject", type=float, default=1.0)
    ap.add_argument("--traffic-decay", type=float, default=0.0)
    ap.add_argument(
        "--traffic-boundary",
        dest="traffic_boundary",
        type=str,
        default="zero",
        choices=["open", "zero", "neumann", "sponge"],
    )
    ap.add_argument("--traffic-sponge-width", dest="traffic_sponge_width", type=int, default=0)
    ap.add_argument("--traffic-sponge-strength", dest="traffic_sponge_strength", type=float, default=0.0)
    ap.add_argument("--traffic-sponge-axes", dest="traffic_sponge_axes", type=str, default="xyz")
    ap.add_argument("--traffic-mode", type=str, default="diffuse", choices=["diffuse", "telegraph", "nonlinear", "sine_gordon"])
    ap.add_argument("--traffic-c2", type=float, default=0.25)
    ap.add_argument("--traffic-gamma", type=float, default=0.10)
    ap.add_argument("--traffic-dt", type=float, default=1.0)
    # Optional SG initial-condition path (expansion/big-bang IC). Only used by the SG runtime.
    ap.add_argument("--sg-bigbang", dest="sg_bigbang", action="store_true")
    ap.add_argument("--sg-bigbang-no", dest="sg_bigbang", action="store_false")
    ap.set_defaults(sg_bigbang=False)
    ap.add_argument("--sg-bigbang-levels", dest="sg_bigbang_levels", type=int, default=194)
    ap.add_argument("--sg-bigbang-bits", dest="sg_bigbang_bits", type=int, default=24)
    ap.add_argument("--sg-bigbang-quant", dest="sg_bigbang_quant", type=str, default="nearest", choices=["nearest", "floor", "trunc"])
    ap.add_argument("--sg-bigbang-resample", dest="sg_bigbang_resample", type=str, default="none", choices=["none", "halfshift"])
    # SG-only besttail viability gate tuning (used by soliton SG capture).
    # If left unset (None), soliton.py will use its internal defaults.
    ap.add_argument("--sg-besttail-vel-rms-max", dest="sg_besttail_vel_rms_max", type=float, default=None)
    ap.add_argument("--sg-besttail-de-step-abs-max", dest="sg_besttail_de_step_abs_max", type=float, default=None)
    ap.add_argument("--sg-besttail-wrap-pi-frac-max", dest="sg_besttail_wrap_pi_frac_max", type=float, default=None)
    ap.add_argument("--sg-besttail-wrap-2pi-frac-max", dest="sg_besttail_wrap_2pi_frac_max", type=float, default=None)
    ap.add_argument("--sg-besttail-cm-r-frac-max", dest="sg_besttail_cm_r_frac_max", type=float, default=None)
    ap.add_argument("--sg-besttail-peak-r-frac-max", dest="sg_besttail_peak_r_frac_max", type=float, default=None)
    ap.add_argument("--sg-besttail-peak-dev-min", dest="sg_besttail_peak_dev_min", type=float, default=None)
    # Phase 8: non-linear scaffold knobs (vacuum material properties)
    ap.add_argument("--traffic-k", dest="traffic_k", type=float, default=0.0)
    ap.add_argument("--traffic-lambda", dest="traffic_lambda", type=float, default=0.0)
    ap.add_argument("--k-index", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=6174)

    ap.add_argument("--chiral-select", type=int, default=0, choices=[-1, 0, 1])
    ap.add_argument("--chiral-field", type=str, default="none", choices=["none", "split_x", "split_y", "split_z"])

    ap.add_argument("--X0", type=float, default=2000.0)
    ap.add_argument("--ds", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--delta-load", action="store_true")
    ap.add_argument("--delta-jitter", type=int, default=0)
    ap.add_argument("--delta-margin", type=int, default=6)

    ap.add_argument("--liveview", action="store_true")

    ap.add_argument("--log", type=str, default="")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-dpi", type=int, default=220)
    ap.add_argument("--plot-scale", type=float, default=1.0)
    ap.add_argument("--plot-log", action="store_true")
    ap.add_argument("--out", type=str, default=default_out)
    ap.add_argument("--dump-radial", type=str, default="")
    ap.add_argument("--dump-radial-fit", type=str, default="")
    ap.add_argument("--dump-shapiro", type=str, default="")
    ap.add_argument("--dump-hdf5", nargs="?", const="default", default="")
    ap.add_argument("--dump-sprite", dest="dump_sprite", nargs="?", const="default", default="")

    # Sprite extraction tuning (used by extractsprite.py / core exporter hook)
    ap.add_argument("--dump-sprite-radius", dest="dump_sprite_radius", type=int, default=0)
    ap.add_argument("--dump-sprite-rel-thresh", dest="dump_sprite_rel_thresh", type=float, default=0.08)
    ap.add_argument("--dump-sprite-pad", dest="dump_sprite_pad", type=int, default=4)
    ap.add_argument("--dump-sprite-max-radius", dest="dump_sprite_max_radius", type=int, default=72)
    ap.add_argument("--dump-sprite-centre", dest="dump_sprite_centre", type=str, default="")
    ap.add_argument("--dump-sprite-compression", dest="dump_sprite_compression", type=str, default="gzip", choices=["gzip", "lzf", "none"])
    ap.add_argument("--dump-sprite-compression-level", dest="dump_sprite_compression_level", type=int, default=4)

    ap.add_argument("--r-fit-min", type=float, default=3.0)
    ap.add_argument("--r-fit-max", type=float, default=0.0)
    ap.add_argument("--shapiro-b-max", type=float, default=0.0)

    ap.add_argument("--walker", action="store_true")
    ap.add_argument("--walker-path", type=str, default="linear", choices=["linear", "circle"])
    ap.add_argument("--walker-circle-radius", type=int, default=8)
    ap.add_argument("--walker-circle-period", type=int, default=0)
    ap.add_argument("--walker-center-x", type=int, default=-1)
    ap.add_argument("--walker-center-y", type=int, default=-1)
    ap.add_argument("--walker-center-z", type=int, default=-1)
    ap.add_argument("--walker-probe-r", type=int, default=0)
    ap.add_argument("--walker-steps", type=int, default=64)
    ap.add_argument("--walker-tick-iters", type=int, default=200)
    ap.add_argument("--walker-hold-steps", type=int, default=0)
    ap.add_argument("--walker-hold-tick-iters", type=int, default=200)
    ap.add_argument("--walker-dx", type=int, default=1)
    ap.add_argument("--walker-dy", type=int, default=0)
    ap.add_argument("--walker-dz", type=int, default=0)
    ap.add_argument("--walker-r-local", type=int, default=6)
    ap.add_argument("--walker-hold-inject", type=float, default=1.0)
    ap.add_argument("--dump-walker", type=str, default="")
    ap.add_argument("--walker-sweep", action="store_true")
    ap.add_argument("--walker-sweep-ticks", type=str, default="")
    ap.add_argument("--walker-sweep-out", type=str, default="")
    ap.add_argument("--walker-mach1", action="store_true")

    ap.add_argument("--coulomb", action="store_true")
    ap.add_argument("--coulomb-q", type=float, default=1.0)
    ap.add_argument("--coulomb-sign", type=str, default="like", choices=["like", "opposite"])
    ap.add_argument("--coulomb-d-min", type=int, default=2)
    ap.add_argument("--coulomb-d-max", type=int, default=32)
    ap.add_argument("--coulomb-d-step", type=int, default=2)
    ap.add_argument("--coulomb-out", dest="coulomb_out", type=str, default="")
    ap.add_argument("--coulomb-max-iters", type=int, default=0)
    ap.add_argument("--coulomb-check-every", type=int, default=200)
    ap.add_argument("--coulomb-tol", type=float, default=1e-4)

    ap.add_argument("--collider", action="store_true")
    ap.add_argument("--collider-spin-b", type=int, default=-1, choices=[-1, 1])
    ap.add_argument("--collider-vx", type=float, default=float(defaults.collider_vx))
    ap.add_argument("--collider-orbit-radius", type=float, default=float(defaults.collider_orbit_radius))
    ap.add_argument("--collider-orbit-omega", type=float, default=float(defaults.collider_orbit_omega))
    ap.add_argument("--collider-steps", type=int, default=0)
    ap.add_argument("--collider-out", type=str, default="")
    ap.add_argument("--collider-out-dir", dest="collider_out_dir", type=str, default="")

    ap.add_argument("--collider-detectors", dest="collider_detectors", action="store_true")
    ap.add_argument("--collider-no-detectors", dest="collider_detectors", action="store_false")
    ap.set_defaults(collider_detectors=False)

    ap.add_argument("--collider-shell-stride", dest="collider_shell_stride", type=int, default=2)
    ap.add_argument("--collider-shell1-inner-frac", dest="collider_shell1_inner_frac", type=float, default=0.18)
    ap.add_argument("--collider-shell1-outer-frac", dest="collider_shell1_outer_frac", type=float, default=0.22)
    ap.add_argument("--collider-shell2-inner-frac", dest="collider_shell2_inner_frac", type=float, default=0.22)
    ap.add_argument("--collider-shell2-outer-frac", dest="collider_shell2_outer_frac", type=float, default=0.26)

    ap.add_argument("--collider-hold", dest="collider_hold", action="store_true")
    ap.add_argument("--collider-no-hold", dest="collider_hold", action="store_false")
    ap.set_defaults(collider_hold=False)
    ap.add_argument("--collider-hold-grace", dest="collider_hold_grace", type=int, default=int(defaults.collider_hold_grace_steps))
    ap.add_argument("--collider-hold-steps", dest="collider_hold_steps", type=int, default=int(defaults.collider_hold_steps))
    ap.add_argument("--collider-center-ball-r", dest="collider_center_ball_r", type=int, default=int(defaults.collider_center_ball_r))
    # Stage 3 back-reaction arguments
    ap.add_argument("--collider-backreact", dest="collider_backreact", action="store_true")
    ap.add_argument("--collider-no-backreact", dest="collider_backreact", action="store_false")
    ap.set_defaults(collider_backreact=bool(getattr(defaults, "collider_backreact", False)))
    ap.add_argument("--collider-backreact-k", dest="collider_backreact_k", type=float, default=float(getattr(defaults, "collider_backreact_k", 0.0)))
    ap.add_argument(
        "--collider-backreact-mode",
        dest="collider_backreact_mode",
        type=str,
        default=str(getattr(defaults, "collider_backreact_mode", "repel")),
        choices=["repel", "attract"],
    )
    ap.add_argument("--collider-backreact-vmax", dest="collider_backreact_vmax", type=float, default=float(getattr(defaults, "collider_backreact_vmax", 0.25)))

    # Stage 4: scattering (impact parameter) + optional angular calorimetry
    ap.add_argument("--collider-impact-b", dest="collider_impact_b", type=float, default=float(getattr(defaults, "collider_impact_b", 0.0)))
    ap.add_argument("--collider-impact-bz", dest="collider_impact_bz", type=float, default=float(getattr(defaults, "collider_impact_bz", 0.0)))
    ap.add_argument(
        "--collider-backreact-axes",
        dest="collider_backreact_axes",
        type=str,
        default=str(getattr(defaults, "collider_backreact_axes", "x")),
        choices=["x", "xyz"],
    )
    ap.add_argument("--collider-octants", dest="collider_octants", action="store_true")
    ap.add_argument("--collider-no-octants", dest="collider_octants", action="store_false")
    ap.set_defaults(collider_octants=bool(getattr(defaults, "collider_octants", False)))

    # Stage 6: nucleus (third-body anchor)
    ap.add_argument("--collider-nucleus", dest="collider_nucleus", action="store_true")
    ap.add_argument("--collider-no-nucleus", dest="collider_nucleus", action="store_false")
    ap.set_defaults(collider_nucleus=bool(getattr(defaults, "collider_nucleus", False)))
    ap.add_argument("--collider-nucleus-q", dest="collider_nucleus_q", type=float, default=float(getattr(defaults, "collider_nucleus_q", 1.0)))
    ap.add_argument(
        "--collider-nucleus-mode",
        dest="collider_nucleus_mode",
        type=str,
        default=str(getattr(defaults, "collider_nucleus_mode", "dc")),
        choices=["dc", "sin"],
    )
    ap.add_argument("--collider-nucleus-omega", dest="collider_nucleus_omega", type=float, default=float(getattr(defaults, "collider_nucleus_omega", float(getattr(defaults, "collider_orbit_omega", 0.12)))))

    ap.add_argument("--collider-nucleus-phase", dest="collider_nucleus_phase", type=float, default=float(getattr(defaults, "collider_nucleus_phase", 0.0)))

    # Stage 7: binding proxy (local damping halo)
    ap.add_argument("--collider-enable-b", dest="collider_enable_b", action="store_true")
    ap.add_argument("--collider-no-enable-b", dest="collider_enable_b", action="store_false")
    ap.set_defaults(collider_enable_b=bool(getattr(defaults, "collider_enable_b", True)))

    ap.add_argument("--collider-halo", dest="collider_halo", action="store_true")
    ap.add_argument("--collider-no-halo", dest="collider_halo", action="store_false")
    ap.set_defaults(collider_halo=bool(getattr(defaults, "collider_halo", False)))


    ap.add_argument("--collider-halo-r", dest="collider_halo_r", type=int, default=int(getattr(defaults, "collider_halo_r", 0)))
    ap.add_argument("--collider-halo-strength", dest="collider_halo_strength", type=float, default=float(getattr(defaults, "collider_halo_strength", 0.0)))
    ap.add_argument(
        "--collider-halo-profile",
        dest="collider_halo_profile",
        type=str,
        default=str(getattr(defaults, "collider_halo_profile", "linear")),
        choices=["linear", "quadratic", "exp"],
    )
    ap.add_argument(
        "--collider-halo-center",
        dest="collider_halo_center",
        type=str,
        default=str(getattr(defaults, "collider_halo_center", "nucleus")),
        choices=["nucleus", "collision"],
    )

    # 09: Sine-Gordon multi-sprite collider (collidersg.py)
    ap.add_argument("--collidersg", action="store_true")
    ap.add_argument("--collidersg-sprites", dest="collidersg_sprites", type=str, default="")
    ap.add_argument("--collidersg-scenario", dest="collidersg_scenario", type=str, default="", choices=["", "A", "B", "C", "D", "E"])
    ap.add_argument("--collidersg-steps", dest="collidersg_steps", type=int, default=2000)
    ap.add_argument("--collidersg-log-every", dest="collidersg_log_every", type=int, default=10)
    ap.add_argument("--collidersg-phi-abs-every", dest="collidersg_phi_abs_every", type=int, default=0)
    ap.add_argument("--collidersg-track-r", dest="collidersg_track_r", type=int, default=16)
    ap.add_argument("--collidersg-peak-thresh", dest="collidersg_peak_thresh", type=float, default=0.05)
    ap.add_argument(
        "--collidersg-k",
        dest="collidersg_k",
        type=float,
        default=0.0,
        help="Sine-Gordon k for collidersg. If 0/unset, inferred from the sprite asset (sg.k HDF5 attribute).",
    )
    ap.add_argument(
        "--collidersg-k-outside",
        dest="sg_k_outside",
        type=float,
        default=0.0,
        help="Optional outside-k for collidersg wire confinement. If >0 and a wire box is provided, a spatial k-grid is built: k=collidersg_k inside the wire and ramps to sg_k_outside outside.",
    )
    ap.add_argument("--collidersg-wire-y0", dest="wire_y0", type=int, default=-1)
    ap.add_argument("--collidersg-wire-y1", dest="wire_y1", type=int, default=-1)
    ap.add_argument("--collidersg-wire-z0", dest="wire_z0", type=int, default=-1)
    ap.add_argument("--collidersg-wire-z1", dest="wire_z1", type=int, default=-1)
    ap.add_argument("--collidersg-wire-bevel", dest="wire_bevel", type=int, default=0)
    ap.add_argument(
        "--collidersg-wire-geom",
        dest="wire_geom",
        type=str,
        default="straight",
        choices=["straight", "t_junction", "y_junction", "or_junction"],
        help="Wire geometry for collidersg confinement. 'straight' uses the rectangular wire box; 't_junction' carves a simple T-junction at junction_x; 'y_junction' builds a symmetric Y-junction mask feeding into the trunk; 'or_junction' builds an OR-style impact junction with a shared dump cavity.",
    )
    ap.add_argument("--collidersg-junction-x", dest="junction_x", type=int, default=-1)
    ap.add_argument("--collidersg-branch-len", dest="branch_len", type=int, default=0)
    ap.add_argument("--collidersg-branch-thick", dest="branch_thick", type=int, default=2)
    ap.add_argument("--collidersg-dump-len", dest="dump_len", type=int, default=24)
    ap.add_argument("--collidersg-dump-throat", dest="dump_throat", type=int, default=2)
    ap.add_argument("--collidersg-dump-y-pad", dest="dump_y_pad", type=int, default=0)

    ap.add_argument("--collidersg-out", dest="collidersg_out", type=str, default="")
    ap.add_argument("--collidersg-sprite-asset", dest="collidersg_sprite_asset", type=str, default="")

    ap.add_argument(
        "--collidersg-boundary",
        dest="collidersg_boundary",
        type=str,
        default="sponge",
        choices=["open", "zero", "neumann", "sponge"],
    )
    ap.add_argument("--collidersg-sponge-width", dest="collidersg_sponge_width", type=int, default=32)
    ap.add_argument("--collidersg-sponge-strength", dest="collidersg_sponge_strength", type=float, default=0.1)
    ap.add_argument("--collidersg-gamma", dest="collidersg_gamma", type=float, default=0.0)
    ap.add_argument("--collidersg-decay", dest="collidersg_decay", type=float, default=0.0)

    ap.add_argument("--double-slit", action="store_true")
    ap.add_argument("--ds-steps", type=int, default=800)
    ap.add_argument("--ds-wall-x", type=int, default=0)
    ap.add_argument("--ds-slit-sep", type=int, default=16)
    ap.add_argument("--ds-slit-width", type=int, default=4)
    ap.add_argument("--ds-gun-start", type=int, default=10)
    ap.add_argument("--ds-gun-speed", type=float, default=0.5)
    ap.add_argument("--ds-gun-stop", type=int, default=0)
    ap.add_argument("--ds-detector-x", type=int, default=0)
    ap.add_argument("--ds-line-z", type=int, default=0)
    ap.add_argument("--ds-single-slit", action="store_true")
    ap.add_argument("--ds-burn", type=int, default=150)
    ap.add_argument("--ds-sample-every", type=int, default=1)
    ap.add_argument("--ds-dump-samples", dest="ds_dump_samples", action="store_true")
    ap.add_argument("--ds-window", type=int, default=80)
    ap.add_argument("--ds-out", dest="ds_out", type=str, default="")
    ap.add_argument("--ds-source", type=str, default="sine", choices=["dc", "sine", "square"])
    ap.add_argument("--ds-amp", type=float, default=1.0)
    ap.add_argument("--ds-omega", type=float, default=0.0)
    ap.add_argument("--ds-half-period", type=int, default=0)
    ap.add_argument("--ds-verbose", dest="ds_verbose", action="store_true")

    ap.add_argument("--corral", action="store_true")
    ap.add_argument("--corral-radius", type=int, default=32)
    ap.add_argument("--corral-omega-start", type=float, default=0.10)
    ap.add_argument("--corral-omega-stop", type=float, default=0.60)
    ap.add_argument("--corral-omega-steps", type=int, default=50)
    ap.add_argument("--corral-burn", type=int, default=1000)
    ap.add_argument("--corral-warm", type=int, default=0)
    ap.add_argument("--corral-out", dest="corral_out", type=str, default="")
    ap.add_argument("--corral-geom", dest="corral_geom", type=str, default="sphere")
    ap.add_argument("--corral-burn-frac", dest="corral_burn_frac", type=float, default=0.90)
    ap.add_argument("--corral-amp", dest="corral_amp", type=float, default=1.0)
    ap.add_argument("--corral-cx-off", dest="corral_cx_off", type=int, default=0)
    ap.add_argument("--corral-cy-off", dest="corral_cy_off", type=int, default=0)
    ap.add_argument("--corral-cz-off", dest="corral_cz_off", type=int, default=0)

    ap.add_argument("--isotropy", action="store_true")
    ap.add_argument("--iso-R", type=int, default=80)
    ap.add_argument("--iso-steps", type=int, default=260)
    ap.add_argument("--iso-sigma", type=float, default=3.5)
    ap.add_argument("--iso-amp", type=float, default=50.0)
    ap.add_argument("--iso-out", dest="iso_out", type=str, default="")

    ap.add_argument("--iso-sweep", dest="iso_sweep", action="store_true")
    ap.add_argument("--iso-sigma-start", dest="iso_sigma_start", type=float, default=1.5)
    ap.add_argument("--iso-sigma-stop", dest="iso_sigma_stop", type=float, default=8.0)
    ap.add_argument("--iso-sigma-steps", dest="iso_sigma_steps", type=int, default=20)

    ap.add_argument("--relativity", action="store_true")
    ap.add_argument("--rel-steps", type=int, default=1600)
    ap.add_argument("--rel-L", type=int, default=48)
    ap.add_argument("--rel-v-frac", type=float, default=0.30)
    ap.add_argument("--rel-v-frac-ref", type=float, default=0.10)
    ap.add_argument("--rel-slab", type=int, default=8)
    ap.add_argument("--rel-amp", type=float, default=500.0)
    ap.add_argument("--rel-sigma", type=float, default=1.75)
    ap.add_argument("--rel-threshold", type=float, default=0.08)
    ap.add_argument("--rel-refractory", type=int, default=10)
    ap.add_argument("--rel-detect", dest="rel_detect", type=str, default="first_cross", choices=["first_cross", "window_peak"])
    ap.add_argument("--rel-start-threshold", dest="rel_start_threshold", type=float, default=0.01)
    ap.add_argument("--rel-peak-window", dest="rel_peak_window", type=int, default=12)
    ap.add_argument("--rel-accept-threshold", dest="rel_accept_threshold", type=float, default=-1.0)
    ap.add_argument("--rel-out", dest="rel_out", type=str, default="")

    # 06A: gravitational time dilation via phase drift + optional lensing
    ap.add_argument("--oscillator", action="store_true")
    ap.add_argument("--osc-out", dest="osc_out", type=str, default="")

    # 06B: passive ringdown resonance sweep (single injection, no feedback)
    ap.add_argument("--ringdown", action="store_true")
    ap.add_argument("--ringdown-steps", type=int, default=1000)
    ap.add_argument("--ringdown-amp", dest="ringdown_amp", type=float, default=50.0)
    ap.add_argument("--ringdown-sigma-start", dest="ringdown_sigma_start", type=float, default=1.2)
    ap.add_argument("--ringdown-sigma-stop", dest="ringdown_sigma_stop", type=float, default=6.0)
    ap.add_argument("--ringdown-sigma-step", dest="ringdown_sigma_step", type=float, default=0.2)
    ap.add_argument("--ringdown-probe-window", dest="ringdown_probe_window", type=int, default=500)
    ap.add_argument("--ringdown-out", dest="ringdown_out", type=str, default="")

    # 08A: soliton/oscillon scan (non-linear telegraph)
    ap.add_argument("--soliton", action="store_true")
    ap.add_argument("--soliton-steps", dest="soliton_steps", type=int, default=2000)
    ap.add_argument("--soliton-sigma", dest="soliton_sigma", type=float, default=4.0)
    ap.add_argument("--soliton-sigma-start", dest="soliton_sigma_start", type=float, default=None)
    ap.add_argument("--soliton-sigma-stop", dest="soliton_sigma_stop", type=float, default=None)
    ap.add_argument("--soliton-sigma-steps", dest="soliton_sigma_steps", type=int, default=None)
    ap.add_argument("--soliton-amp", dest="soliton_amp", type=float, default=100.0)
    # Sensible scan defaults: non-zero scaffold to avoid accidental flatline runs.
    ap.add_argument("--soliton-k", dest="soliton_k", type=float, default=1.0e-2)
    ap.add_argument("--soliton-lambda-start", dest="soliton_lambda_start", type=float, default=1.0e-6)
    ap.add_argument("--soliton-lambda-stop", dest="soliton_lambda_stop", type=float, default=None)
    ap.add_argument("--soliton-lambda-steps", dest="soliton_lambda_steps", type=int, default=20)
    ap.add_argument("--soliton-out", dest="soliton_out", type=str, default="")
    # 08E+: sine-gordon specific sweeps (optional; used when --traffic-mode sine_gordon)
    ap.add_argument("--soliton-sg-k-start", dest="soliton_sg_k_start", type=float, default=None)
    ap.add_argument("--soliton-sg-k-stop", dest="soliton_sg_k_stop", type=float, default=None)
    ap.add_argument("--soliton-sg-k-steps", dest="soliton_sg_k_steps", type=int, default=None)
    ap.add_argument("--soliton-sg-amp-start", dest="soliton_sg_amp_start", type=float, default=None)
    ap.add_argument("--soliton-sg-amp-stop", dest="soliton_sg_amp_stop", type=float, default=None)
    ap.add_argument("--soliton-sg-amp-steps", dest="soliton_sg_amp_steps", type=int, default=None)
    # 08B/08C: symmetry-breaking scaffold helpers
    ap.add_argument("--soliton-init-vev", dest="soliton_init_vev", action="store_true")
    ap.add_argument("--soliton-no-init-vev", dest="soliton_init_vev", action="store_false")
    ap.set_defaults(soliton_init_vev=False)
    ap.add_argument("--soliton-vev-sign", dest="soliton_vev_sign", type=int, default=1, choices=[-1, 1])

    # steady potential (diffuse build) used as background index field
    ap.add_argument("--osc-mass-iters", type=int, default=20000)
    ap.add_argument("--osc-mass-inject", type=float, default=1.0)

    # dynamic phase measurement (telegraph)
    ap.add_argument("--osc-steps", type=int, default=2000)
    ap.add_argument("--osc-burn", type=int, default=400)
    ap.add_argument("--osc-warm", type=int, default=0)
    ap.add_argument("--osc-sample-every", type=int, default=1)
    ap.add_argument("--osc-omega", type=float, default=0.15)
    ap.add_argument("--osc-amp", dest="osc_drive_amp", type=float, default=25.0)

    # probe placement (radii from center)
    ap.add_argument("--osc-r-near", type=int, default=24)
    ap.add_argument("--osc-r-far", type=int, default=120)

    # optional lensing pass (ray march through n(r) field)
    ap.add_argument("--osc-lens", dest="osc_lens", action="store_true")
    ap.add_argument("--osc-no-lens", dest="osc_lens", action="store_false")
    ap.set_defaults(osc_lens=True)
    ap.add_argument("--osc-ray-count", type=int, default=256)
    ap.add_argument("--osc-ray-step", dest="osc_ds", type=float, default=0.75)
    ap.add_argument("--osc-ray-max", dest="osc_march_steps", type=int, default=640)

    ap.add_argument("--ensemble", type=int, default=0)
    ap.add_argument("--ensemble-seed0", type=int, default=6174)
    ap.add_argument("--ensemble-workers", type=int, default=0)
    ap.add_argument("--ensemble-write-csv", dest="ensemble_write_csv", action="store_true")

    ap.add_argument("--bench-stability", action="store_true")
    ap.add_argument("--bench-trials", type=int, default=256)
    ap.add_argument("--bench-ticks", type=int, default=600)
    ap.add_argument("--bench-p-noise", type=float, default=0.0020)
    ap.add_argument("--bench-p-center", type=float, default=0.0005)
    ap.add_argument("--bench-sweep-faces", action="store_true")
    ap.add_argument("--bench-sweep-k", action="store_true")
    ap.add_argument("--bench-sweep-k-plot", action="store_true")
    ap.add_argument("--bench-sweep-k-write-csv", dest="bench_sweep_k_write_csv", action="store_true")

    return ap


def parse_cli(argv: list[str] | None = None, *, here_file: str) -> tuple[argparse.Namespace, PipelineParams, int]:
    ap = build_parser(here_file=here_file)
    defaults = PipelineParams()
    args = ap.parse_args(argv)

    args.out = resolve_out_dir(here_file, str(args.out))

    # HDF5 snapshot output: keep relative paths under <out>/..., including subfolders.
    dh5 = str(getattr(args, "dump_hdf5", "") or "").strip()
    if dh5 == "default":
        # Match bundle artefact naming: <run_id>_n<N>_<timestamp>.h5
        out_dir = str(args.out)
        ts = os.path.basename(os.path.normpath(out_dir))
        rid = os.path.basename(os.path.dirname(os.path.normpath(out_dir)))
        if rid == "" or rid == ts:
            rid = "run"
        fname = f"{rid}_n{int(args.n)}_{ts}.h5"
        dh5 = os.path.join(out_dir, "_hdf5", fname)
    elif dh5 != "" and (not os.path.isabs(dh5)):
        dh5 = os.path.join(str(args.out), dh5)
    args.dump_hdf5 = dh5

    # Sprite asset output: default writes to <out>/_sprites/ (final filename chosen later using export.step).
    ds = str(getattr(args, "dump_sprite", "") or "").strip()
    if ds == "default":
        ds = os.path.join(str(args.out), "_sprites")
    elif ds != "" and (not os.path.isabs(ds)):
        ds = os.path.join(str(args.out), ds)
    args.dump_sprite = ds

    # Note: --dump-sprite does not require --dump-hdf5.
    # If a *_besttail.h5 export is produced during the run, core can extract from that.
    # If no besttail export exists, core may fall back to an end-of-run snapshot (only available
    # when --dump-hdf5 was requested). Keeping this optional avoids confusing “mandatory final
    # snapshot” behaviour when users only want besttail-based extraction.

    # Basic validation for sprite tuning flags.
    if int(getattr(args, "dump_sprite_radius", 0)) < 0:
        raise ValueError("--dump-sprite-radius must be >= 0")
    if float(getattr(args, "dump_sprite_rel_thresh", 0.08)) <= 0.0:
        raise ValueError("--dump-sprite-rel-thresh must be > 0")
    if int(getattr(args, "dump_sprite_pad", 4)) < 0:
        raise ValueError("--dump-sprite-pad must be >= 0")
    if int(getattr(args, "dump_sprite_max_radius", 96)) <= 0:
        raise ValueError("--dump-sprite-max-radius must be > 0")
    if int(getattr(args, "dump_sprite_compression_level", 4)) < 0:
        raise ValueError("--dump-sprite-compression-level must be >= 0")

    _normalise_out_flags(
        args,
        here_file,
        names=(
            "log",
            "dump_radial",
            "dump_radial_fit",
            "dump_shapiro",
            "dump_hdf5",
            "dump_sprite",
            "dump_walker",
            "walker_sweep_out",
            "coulomb_out",
            "ds_out",
            "corral_out",
            "collider_out",
            "collider_out_dir",
            "iso_out",
            "rel_out",
            "osc_out",
            "ringdown_out",
            "soliton_out",
            "collidersg_out",
        ),
    )

    # collidersg sprite asset (input): resolve relative paths against the project root.
    sa = str(getattr(args, "collidersg_sprite_asset", "") or "").strip()
    if sa != "" and (not os.path.isabs(sa)):
        sa = resolve_out_path(here_file, sa)
    args.collidersg_sprite_asset = sa
    # Canonicalise soliton lambda_stop for downstream params mapping.
    # This must exist even when --soliton is not active (core may still build PipelineParams).
    _sol_lstop_raw = getattr(args, "soliton_lambda_stop", None)
    if _sol_lstop_raw is None:
        _sol_lstop_raw = getattr(args, "soliton_lambda_start", float(getattr(defaults.soliton, "lambda_start")))
    soliton_lambda_stop_f: float = float(_sol_lstop_raw)

    if bool(getattr(args, "soliton", False)):
        s_steps = int(getattr(args, "soliton_steps"))
        if s_steps <= 0:
            raise ValueError("--soliton-steps must be > 0")
        s_sigma = float(getattr(args, "soliton_sigma"))
        if s_sigma <= 0.0:
            raise ValueError("--soliton-sigma must be > 0")
        ss0 = getattr(args, "soliton_sigma_start")
        ss1 = getattr(args, "soliton_sigma_stop")
        ssn = getattr(args, "soliton_sigma_steps")
        any_s = (ss0 is not None) or (ss1 is not None) or (ssn is not None)
        if any_s:
            if ss0 is None or ss1 is None or ssn is None:
                raise ValueError("--soliton-sigma-start/--soliton-sigma-stop/--soliton-sigma-steps must be provided together")
            s0 = float(ss0)
            s1 = float(ss1)
            sn = int(ssn)
            if s0 <= 0.0 or s1 <= 0.0:
                raise ValueError("--soliton-sigma-start and --soliton-sigma-stop must be > 0")
            if s1 < s0:
                raise ValueError("--soliton-sigma-stop must be >= --soliton-sigma-start")
            if s1 == s0:
                if sn != 1:
                    raise ValueError("--soliton-sigma-steps must be 1 when --soliton-sigma-start == --soliton-sigma-stop")
                args.soliton_sigma_steps = 1
            else:
                if sn < 2:
                    raise ValueError("--soliton-sigma-steps must be >= 2 when --soliton-sigma-start != --soliton-sigma-stop")
        else:
            args.soliton_sigma_start = float(s_sigma)
            args.soliton_sigma_stop = float(s_sigma)
            args.soliton_sigma_steps = 1
        s_amp = float(getattr(args, "soliton_amp"))
        if s_amp <= 0.0:
            raise ValueError("--soliton-amp must be > 0")
        s_k = float(getattr(args, "soliton_k"))
        if not math.isfinite(s_k):
            raise ValueError("--soliton-k must be finite")
        s_l0 = float(getattr(args, "soliton_lambda_start"))
        s_l1_raw = getattr(args, "soliton_lambda_stop")
        s_l1 = None if s_l1_raw is None else float(s_l1_raw)
        s_ln = int(getattr(args, "soliton_lambda_steps"))
        if s_l0 < 0.0:
            raise ValueError("--soliton-lambda-start must be >= 0")
        if s_l1 is not None and s_l1 < 0.0:
            raise ValueError("--soliton-lambda-stop must be >= 0")
        if s_l1 is not None and s_l1 < s_l0:
            raise ValueError("--soliton-lambda-stop must be >= --soliton-lambda-start")
        if s_ln < 1:
            raise ValueError("--soliton-lambda-steps must be >= 1")

        # Single-point lambda runs: if stop is omitted OR start==stop, force steps=1.
        if (s_l1 is None) or (s_l1 == s_l0):
            args.soliton_lambda_stop = s_l0
            args.soliton_lambda_steps = 1
        else:
            # True sweep: require at least 2 points.
            if s_ln < 2:
                raise ValueError("--soliton-lambda-steps must be >= 2 when --soliton-lambda-start != --soliton-lambda-stop")

        # Canonical, non-optional lambda_stop (Pylance aid).
        if getattr(args, "soliton_lambda_stop") is None:
            raise ValueError("--soliton-lambda-stop must be resolved (internal error)")
        soliton_lambda_stop_f: float = float(getattr(args, "soliton_lambda_stop"))

        tmode = str(getattr(args, "traffic_mode"))
        if tmode not in ("nonlinear", "sine_gordon"):
            raise ValueError("--soliton requires --traffic-mode nonlinear or sine_gordon")
        # SG big-bang IC is only meaningful in SG mode.
        if bool(getattr(args, "sg_bigbang", False)) and tmode != "sine_gordon":
            raise ValueError("--sg-bigbang requires --traffic-mode sine_gordon")
        if bool(getattr(args, "sg_bigbang", False)):
            lv = int(getattr(args, "sg_bigbang_levels"))
            bits = int(getattr(args, "sg_bigbang_bits"))
            if lv <= 0:
                raise ValueError("--sg-bigbang-levels must be > 0")
            if bits <= 0:
                raise ValueError("--sg-bigbang-bits must be > 0")
        # SG besttail thresholds are SG-only (used for SG candidate/capture gating).
        bt_names = (
            "sg_besttail_vel_rms_max",
            "sg_besttail_de_step_abs_max",
            "sg_besttail_wrap_pi_frac_max",
            "sg_besttail_wrap_2pi_frac_max",
            "sg_besttail_cm_r_frac_max",
            "sg_besttail_peak_r_frac_max",
            "sg_besttail_peak_dev_min",
        )
        any_bt = any(getattr(args, n, None) is not None for n in bt_names)
        if any_bt and tmode != "sine_gordon":
            raise ValueError("--sg-besttail-* flags require --traffic-mode sine_gordon")
        if any_bt:
            for n in bt_names:
                v = getattr(args, n, None)
                if v is None:
                    continue
                fv = float(v)
                if not math.isfinite(fv):
                    raise ValueError(f"--{n.replace('_', '-')} must be finite")
                # Fractional thresholds must be in [0, 1].
                if n.endswith("_frac_max"):
                    if fv < 0.0 or fv > 1.0:
                        raise ValueError(f"--{n.replace('_', '-')} must be in [0, 1]")
                else:
                    if fv < 0.0:
                        raise ValueError(f"--{n.replace('_', '-')} must be >= 0")
        # k sign rules depend on the physics mode.
        # - nonlinear: allow k to be negative (double-well / symmetry breaking)
        # - sine_gordon: require k >= 0
        if tmode == "sine_gordon" and s_k < 0.0:
            raise ValueError("--soliton-k must be >= 0 for --traffic-mode sine_gordon")
        if tmode == "sine_gordon":
            # Sine-Gordon does not use the (k,lambda) scaffold in the same way.
            # We treat soliton_lambda_* as non-applicable for SG and fail-loud if a sweep is attempted.
            # Single-point lambda is tolerated (and normalised) so existing experiment wiring doesn't need to care.
            if s_l1 is None:
                # stop omitted => treat as single point
                args.soliton_lambda_stop = s_l0
                args.soliton_lambda_steps = 1
            else:
                # if user tries to sweep lambda under SG, fail-loud
                if s_l1 != s_l0:
                    raise ValueError("--traffic-mode sine_gordon: soliton lambda sweep is not supported; use --soliton-sg-* sweeps")
                if int(getattr(args, "soliton_lambda_steps")) != 1:
                    raise ValueError("--traffic-mode sine_gordon: --soliton-lambda-steps must be 1 (lambda is ignored)")

            sg_k0 = getattr(args, "soliton_sg_k_start")
            sg_k1 = getattr(args, "soliton_sg_k_stop")
            sg_kn = getattr(args, "soliton_sg_k_steps")
            sg_a0 = getattr(args, "soliton_sg_amp_start")
            sg_a1 = getattr(args, "soliton_sg_amp_stop")
            sg_an = getattr(args, "soliton_sg_amp_steps")

            any_k = (sg_k0 is not None) or (sg_k1 is not None) or (sg_kn is not None)
            any_a = (sg_a0 is not None) or (sg_a1 is not None) or (sg_an is not None)

            # Canonicalise SG sweeps. If SG sweep flags are omitted, we run a single-point
            # experiment using `--soliton-k` and `--soliton-amp`.
            if any_k:
                if sg_k0 is None or sg_k1 is None or sg_kn is None:
                    raise ValueError("--soliton-sg-k-start/--soliton-sg-k-stop/--soliton-sg-k-steps must be provided together")
                k0 = float(sg_k0)
                k1 = float(sg_k1)
                kn = int(sg_kn)
                if k0 < 0.0 or k1 < 0.0:
                    raise ValueError("--soliton-sg-k-start and --soliton-sg-k-stop must be >= 0")
                if k1 < k0:
                    raise ValueError("--soliton-sg-k-stop must be >= --soliton-sg-k-start")
                if k1 == k0:
                    if kn != 1:
                        raise ValueError("--soliton-sg-k-steps must be 1 when --soliton-sg-k-start == --soliton-sg-k-stop")
                    args.soliton_sg_k_steps = 1
                else:
                    if kn < 2:
                        raise ValueError("--soliton-sg-k-steps must be >= 2 when --soliton-sg-k-start != --soliton-sg-k-stop")
            else:
                if s_k < 0.0:
                    raise ValueError("--soliton-k must be >= 0 for --traffic-mode sine_gordon")
                args.soliton_sg_k_start = float(s_k)
                args.soliton_sg_k_stop = float(s_k)
                args.soliton_sg_k_steps = 1

            if any_a:
                if sg_a0 is None or sg_a1 is None or sg_an is None:
                    raise ValueError("--soliton-sg-amp-start/--soliton-sg-amp-stop/--soliton-sg-amp-steps must be provided together")
                a0 = float(sg_a0)
                a1 = float(sg_a1)
                an = int(sg_an)
                if a0 <= 0.0 or a1 <= 0.0:
                    raise ValueError("--soliton-sg-amp-start and --soliton-sg-amp-stop must be > 0")
                if a1 < a0:
                    raise ValueError("--soliton-sg-amp-stop must be >= --soliton-sg-amp-start")
                if a1 == a0:
                    if an != 1:
                        raise ValueError("--soliton-sg-amp-steps must be 1 when --soliton-sg-amp-start == --soliton-sg-amp-stop")
                    args.soliton_sg_amp_steps = 1
                else:
                    if an < 2:
                        raise ValueError("--soliton-sg-amp-steps must be >= 2 when --soliton-sg-amp-start != --soliton-sg-amp-stop")
            else:
                # Default to the single-point amplitude value.
                args.soliton_sg_amp_start = float(s_amp)
                args.soliton_sg_amp_stop = float(s_amp)
                args.soliton_sg_amp_steps = 1
        if int(getattr(args, "soliton_vev_sign")) not in (-1, 1):
            raise ValueError("--soliton-vev-sign must be -1 or 1")

    # Fail-loud: mach1 forces tick iters to 2/2, so a tick sweep is meaningless.
    if bool(getattr(args, "walker_sweep", False)) and bool(getattr(args, "walker_mach1", False)):
        raise ValueError("--walker-sweep is incompatible with --walker-mach1 (mach1 forces tick iters to 2, making a sweep meaningless)")

    # Fail-loud: sweep requires explicit tick list.
    if bool(getattr(args, "walker_sweep", False)) and str(getattr(args, "walker_sweep_ticks", "")).strip() == "":
        raise ValueError("--walker-sweep requires --walker-sweep-ticks")

    if int(args.ensemble) < 0:
        raise ValueError("--ensemble must be >= 0")
    if int(args.ensemble) > 0 and bool(args.plot):
        raise ValueError("--plot is not supported with --ensemble; run a single seed for plotting")

    if bool(getattr(args, "oscillator", False)):
        osc_steps = int(getattr(args, "osc_steps"))
        osc_burn = int(getattr(args, "osc_burn"))
        osc_warm = int(getattr(args, "osc_warm"))
        osc_every = int(getattr(args, "osc_sample_every"))
        if osc_steps <= 0:
            raise ValueError("--osc-steps must be > 0")
        if osc_burn < 0:
            raise ValueError("--osc-burn must be >= 0")
        if osc_burn >= osc_steps:
            raise ValueError("--osc-burn must be < --osc-steps")
        if osc_warm < 0:
            raise ValueError("--osc-warm must be >= 0")
        if osc_every <= 0:
            raise ValueError("--osc-sample-every must be >= 1")
        osc_omega = float(getattr(args, "osc_omega"))
        if osc_omega <= 0.0:
            raise ValueError("--osc-omega must be > 0")
        # Nyquist guard: keep wavelength comfortably above voxel scale.
        # With c_eff ~ 0.55, omega ~ 0.35 implies lambda ~ 10 vox.
        if osc_omega > 0.35:
            raise ValueError("--osc-omega too high for lattice stability; keep <= 0.35")
        osc_mass_iters = int(getattr(args, "osc_mass_iters"))
        osc_mass_inject = float(getattr(args, "osc_mass_inject"))
        if osc_mass_iters < 1:
            raise ValueError("--osc-mass-iters must be >= 1")
        if osc_mass_inject <= 0.0:
            raise ValueError("--osc-mass-inject must be > 0")
        r_near = int(getattr(args, "osc_r_near"))
        r_far = int(getattr(args, "osc_r_far"))
        if r_near <= 0 or r_far <= 0:
            raise ValueError("--osc-r-near and --osc-r-far must be > 0")
        if r_near >= r_far:
            raise ValueError("--osc-r-near must be < --osc-r-far")
        if bool(getattr(args, "osc_lens", True)):
            if int(getattr(args, "osc_ray_count")) <= 0:
                raise ValueError("--osc-ray-count must be > 0")
            if float(getattr(args, "osc_ds")) <= 0.0:
                raise ValueError("--osc-ray-step must be > 0")
            if int(getattr(args, "osc_march_steps")) <= 0:
                raise ValueError("--osc-ray-max must be > 0")

    if bool(getattr(args, "ringdown", False)):
        rd_steps = int(getattr(args, "ringdown_steps"))
        if rd_steps <= 0:
            raise ValueError("--ringdown-steps must be > 0")
        amp = float(getattr(args, "ringdown_amp"))
        if amp <= 0.0:
            raise ValueError("--ringdown-amp must be > 0")
        s0 = float(getattr(args, "ringdown_sigma_start"))
        s1 = float(getattr(args, "ringdown_sigma_stop"))
        ds = float(getattr(args, "ringdown_sigma_step"))
        if s0 <= 0.0 or s1 <= 0.0:
            raise ValueError("--ringdown-sigma-start and --ringdown-sigma-stop must be > 0")
        if s1 < s0:
            raise ValueError("--ringdown-sigma-stop must be >= --ringdown-sigma-start")
        if ds <= 0.0:
            raise ValueError("--ringdown-sigma-step must be > 0")
        pw = int(getattr(args, "ringdown_probe_window"))
        if pw < 64:
            raise ValueError("--ringdown-probe-window must be >= 64")

    if bool(getattr(args, "relativity", False)):
        r_steps = int(getattr(args, "rel_steps"))
        if r_steps <= 0:
            raise ValueError("--rel-steps must be > 0")
        L = int(getattr(args, "rel_L"))
        if L <= 0:
            raise ValueError("--rel-L must be > 0")
        slab = int(getattr(args, "rel_slab"))
        if slab <= 0:
            raise ValueError("--rel-slab must be > 0")
        amp = float(getattr(args, "rel_amp"))
        if amp <= 0.0:
            raise ValueError("--rel-amp must be > 0")
        sig = float(getattr(args, "rel_sigma"))
        if sig <= 0.0:
            raise ValueError("--rel-sigma must be > 0")
        thr = float(getattr(args, "rel_threshold"))
        if (not math.isfinite(thr)) or (thr <= 0.0) or (thr >= 1.0):
            raise ValueError("--rel-threshold must be a fraction in (0, 1)")
        ref = int(getattr(args, "rel_refractory"))
        if ref < 0:
            raise ValueError("--rel-refractory must be >= 0")

        mode = str(getattr(args, "rel_detect"))
        if mode not in ("first_cross", "window_peak"):
            raise ValueError("--rel-detect must be one of: first_cross, window_peak")

        st = float(getattr(args, "rel_start_threshold"))
        if (not math.isfinite(st)) or (st <= 0.0) or (st >= 1.0):
            raise ValueError("--rel-start-threshold must be a fraction in (0, 1)")

        pw = int(getattr(args, "rel_peak_window"))
        if pw < 1 or pw > 200:
            raise ValueError("--rel-peak-window must be in [1, 200]")

        at = float(getattr(args, "rel_accept_threshold"))
        if (not math.isfinite(at)):
            raise ValueError("--rel-accept-threshold must be finite")
        if at >= 1.0:
            raise ValueError("--rel-accept-threshold must be < 1.0 (or <0 to disable)")

    if bool(getattr(args, "collider", False)):
        c_steps = int(getattr(args, "collider_steps"))
        if c_steps <= 0:
            raise ValueError("--collider-steps must be > 0")
        vx = float(getattr(args, "collider_vx"))
        if vx <= 0.0:
            raise ValueError("--collider-vx must be > 0")
        orad = float(getattr(args, "collider_orbit_radius"))
        if orad <= 0.0:
            raise ValueError("--collider-orbit-radius must be > 0")
        omeg = float(getattr(args, "collider_orbit_omega"))
        if omeg <= 0.0:
            raise ValueError("--collider-orbit-omega must be > 0")

        if bool(getattr(args, "collider_nucleus", False)):
            nq = float(getattr(args, "collider_nucleus_q"))
            if nq <= 0.0:
                raise ValueError("--collider-nucleus-q must be > 0")
            nmode = str(getattr(args, "collider_nucleus_mode"))
            if nmode not in ("dc", "sin"):
                raise ValueError("--collider-nucleus-mode must be one of: dc, sin")
            if nmode == "sin":
                nomega = float(getattr(args, "collider_nucleus_omega"))
                if nomega <= 0.0:
                    raise ValueError("--collider-nucleus-omega must be > 0")

        if bool(getattr(args, "collider_halo", False)):
            halo_r = int(getattr(args, "collider_halo_r"))
            halo_strength = float(getattr(args, "collider_halo_strength"))
            if halo_r <= 0:
                raise ValueError("--collider-halo-r must be > 0 when --collider-halo is set")
            if halo_strength <= 0.0:
                raise ValueError("--collider-halo-strength must be > 0 when --collider-halo is set")
            halo_profile = str(getattr(args, "collider_halo_profile"))
            if halo_profile not in ("linear", "quadratic", "exp"):
                raise ValueError("--collider-halo-profile must be one of: linear, quadratic, exp")
            halo_center = str(getattr(args, "collider_halo_center"))
            if halo_center not in ("nucleus", "collision"):
                raise ValueError("--collider-halo-center must be one of: nucleus, collision")

    # collidersg validation (fail-loud)
    if bool(getattr(args, "collidersg", False)):
        # Avoid ambiguous dual-collider runs.
        if bool(getattr(args, "collider", False)):
            raise ValueError("use either --collider OR --collidersg, not both")

        # collidersg assumes the SG physics path.
        tmode = str(getattr(args, "traffic_mode"))
        if tmode != "sine_gordon":
            raise ValueError("--collidersg requires --traffic-mode sine_gordon")

        sprites_src = str(getattr(args, "collidersg_sprites", "")).strip()
        scenario = str(getattr(args, "collidersg_scenario", "")).strip().upper()

        asset_h5 = str(getattr(args, "collidersg_sprite_asset", "") or "").strip()
        if asset_h5 != "" and (not os.path.exists(asset_h5)):
            raise ValueError(f"--collidersg-sprite-asset not found: {asset_h5}")

        if sprites_src == "":
            raise ValueError("--collidersg requires --collidersg-sprites (JSON string or .json path)")

        c_steps = int(getattr(args, "collidersg_steps"))
        if c_steps <= 0:
            raise ValueError("--collidersg-steps must be > 0")

        le = int(getattr(args, "collidersg_log_every"))
        if le <= 0:
            raise ValueError("--collidersg-log-every must be >= 1")
        pae = int(getattr(args, "collidersg_phi_abs_every"))
        if pae < 0:
            raise ValueError("--collidersg-phi-abs-every must be >= 0")

        tr = int(getattr(args, "collidersg_track_r"))
        if tr <= 1:
            raise ValueError("--collidersg-track-r must be > 1")

        pt = float(getattr(args, "collidersg_peak_thresh"))
        if (not math.isfinite(pt)) or pt < 0.0:
            raise ValueError("--collidersg-peak-thresh must be finite and >= 0")

        ksg = float(getattr(args, "collidersg_k"))
        if (not math.isfinite(ksg)) or ksg < 0.0:
            raise ValueError("--collidersg-k must be finite and >= 0 (0 means infer from sprite asset)")

        # Wire confinement: validate --collidersg-k-outside (sg_k_outside) and wire box parameters
        k_out = float(getattr(args, "sg_k_outside"))
        if (not math.isfinite(k_out)) or k_out < 0.0:
            raise ValueError("--collidersg-k-outside must be finite and >= 0")

        wy0 = int(getattr(args, "wire_y0"))
        wy1 = int(getattr(args, "wire_y1"))
        wz0 = int(getattr(args, "wire_z0"))
        wz1 = int(getattr(args, "wire_z1"))
        wb = int(getattr(args, "wire_bevel"))
        if wb < 0:
            raise ValueError("--collidersg-wire-bevel must be >= 0")

        wg = str(getattr(args, "wire_geom", "straight") or "straight").strip()
        if wg not in ("straight", "t_junction", "y_junction", "or_junction"):
            raise ValueError("--collidersg-wire-geom must be one of: straight, t_junction, y_junction, or_junction")

        jx = int(getattr(args, "junction_x"))
        bl = int(getattr(args, "branch_len"))
        bt = int(getattr(args, "branch_thick"))
        dl = int(getattr(args, "dump_len", 24))
        dtw = int(getattr(args, "dump_throat", 2))
        dyp = int(getattr(args, "dump_y_pad", 0))
        if dl < 0:
            raise ValueError("--collidersg-dump-len must be >= 0")
        if dtw <= 0:
            raise ValueError("--collidersg-dump-throat must be > 0")
        if dyp < 0:
            raise ValueError("--collidersg-dump-y-pad must be >= 0")
        if bt <= 0:
            raise ValueError("--collidersg-branch-thick must be > 0")

        any_wire = (wy0 >= 0) or (wy1 >= 0) or (wz0 >= 0) or (wz1 >= 0)
        if any_wire:
            if wy0 < 0 or wy1 < 0 or wz0 < 0 or wz1 < 0:
                raise ValueError("--collidersg-wire-y0/--collidersg-wire-y1/--collidersg-wire-z0/--collidersg-wire-z1 must be provided together")
            n = int(getattr(args, "n"))
            if wy0 < 0 or wy1 <= wy0 or wy1 > n:
                raise ValueError("--collidersg-wire-y0/--collidersg-wire-y1 must satisfy 0 <= y0 < y1 <= n")
            if wz0 < 0 or wz1 <= wz0 or wz1 > n:
                raise ValueError("--collidersg-wire-z0/--collidersg-wire-z1 must satisfy 0 <= z0 < z1 <= n")
            if k_out <= 0.0:
                raise ValueError("--collidersg-k-outside must be > 0 when a wire box is specified")

        # Geometry-specific requirements.
        if wg == "t_junction":
            if not any_wire:
                raise ValueError("--collidersg-wire-geom t_junction requires a wire box: --collidersg-wire-y0/--collidersg-wire-y1/--collidersg-wire-z0/--collidersg-wire-z1")
            n = int(getattr(args, "n"))
            if wb != 0:
                raise ValueError("--collidersg-wire-geom t_junction requires --collidersg-wire-bevel 0")
            if jx < 0 or jx >= n:
                raise ValueError("--collidersg-junction-x must satisfy 0 <= junction_x < n")
            if bl <= 0:
                raise ValueError("--collidersg-branch-len must be > 0 for t_junction")

        if wg == "y_junction":
            if not any_wire:
                raise ValueError("--collidersg-wire-geom y_junction requires a wire box: --collidersg-wire-y0/--collidersg-wire-y1/--collidersg-wire-z0/--collidersg-wire-z1")
            n = int(getattr(args, "n"))
            if wb != 0:
                raise ValueError("--collidersg-wire-geom y_junction requires --collidersg-wire-bevel 0")
            if jx < 0 or jx >= n:
                raise ValueError("--collidersg-junction-x must satisfy 0 <= junction_x < n")
            if bl <= 0:
                raise ValueError("--collidersg-branch-len must be > 0 for y_junction")

        if wg == "or_junction":
            if not any_wire:
                raise ValueError("--collidersg-wire-geom or_junction requires a wire box: --collidersg-wire-y0/--collidersg-wire-y1/--collidersg-wire-z0/--collidersg-wire-z1")
            n = int(getattr(args, "n"))
            if wb != 0:
                raise ValueError("--collidersg-wire-geom or_junction requires --collidersg-wire-bevel 0")
            if jx < 0 or jx >= n:
                raise ValueError("--collidersg-junction-x must satisfy 0 <= junction_x < n")
            if bl <= 0:
                raise ValueError("--collidersg-branch-len must be > 0 for or_junction")
            if dl > 0 and (jx + bt + dl) > n:
                raise ValueError("--collidersg-dump-len too large: junction_x + branch_thick + dump_len must be <= n")

        bnd = str(getattr(args, "collidersg_boundary"))
        if bnd not in ("open", "zero", "neumann", "sponge"):
            raise ValueError("--collidersg-boundary must be one of: open, zero, neumann, sponge")

        sw = int(getattr(args, "collidersg_sponge_width"))
        if sw < 0:
            raise ValueError("--collidersg-sponge-width must be >= 0")

        ss = float(getattr(args, "collidersg_sponge_strength"))
        if (not math.isfinite(ss)) or ss < 0.0:
            raise ValueError("--collidersg-sponge-strength must be finite and >= 0")
        if bnd == "sponge" and sw <= 0:
            raise ValueError("--collidersg-sponge-width must be > 0 when --collidersg-boundary sponge")

        cg = float(getattr(args, "collidersg_gamma"))
        if (not math.isfinite(cg)) or cg < 0.0:
            raise ValueError("--collidersg-gamma must be finite and >= 0")

        cd = float(getattr(args, "collidersg_decay"))
        if (not math.isfinite(cd)) or cd < 0.0:
            raise ValueError("--collidersg-decay must be finite and >= 0")

    if bool(getattr(args, "collider", False)) and bool(getattr(args, "collider_detectors", False)):
        stride = int(getattr(args, "collider_shell_stride"))
        if stride < 1:
            raise ValueError("--collider-shell-stride must be >= 1")

        f1i = float(getattr(args, "collider_shell1_inner_frac"))
        f1o = float(getattr(args, "collider_shell1_outer_frac"))
        f2i = float(getattr(args, "collider_shell2_inner_frac"))
        f2o = float(getattr(args, "collider_shell2_outer_frac"))

        for v, name in (
            (f1i, "--collider-shell1-inner-frac"),
            (f1o, "--collider-shell1-outer-frac"),
            (f2i, "--collider-shell2-inner-frac"),
            (f2o, "--collider-shell2-outer-frac"),
        ):
            if v <= 0.0 or v >= 0.5:
                raise ValueError(f"{name} must be in (0, 0.5)")

        if f1o < f1i:
            raise ValueError("--collider-shell1-outer-frac must be >= --collider-shell1-inner-frac")
        if f2o < f2i:
            raise ValueError("--collider-shell2-outer-frac must be >= --collider-shell2-inner-frac")

    if bool(getattr(args, "collider", False)) and bool(getattr(args, "collider_hold", False)):
        hold_grace = int(getattr(args, "collider_hold_grace"))
        hold_steps = int(getattr(args, "collider_hold_steps"))
        center_r = int(getattr(args, "collider_center_ball_r"))
        if hold_grace < 0:
            raise ValueError("--collider-hold-grace must be >= 0")
        if hold_steps <= 0:
            raise ValueError("--collider-hold-steps must be > 0 when --collider-hold is set")
        if center_r < 1:
            raise ValueError("--collider-center-ball-r must be >= 1")
        if not bool(getattr(args, "collider_detectors", False)):
            raise ValueError("--collider-hold requires --collider-detectors")
    # Back-reaction validation
    if bool(getattr(args, "collider", False)) and bool(getattr(args, "collider_backreact", False)):
        k = float(getattr(args, "collider_backreact_k"))
        if k <= 0.0:
            raise ValueError("--collider-backreact-k must be > 0 when --collider-backreact is set")
        vmax = float(getattr(args, "collider_backreact_vmax"))
        if vmax <= 0.0:
            raise ValueError("--collider-backreact-vmax must be > 0")
        mode = str(getattr(args, "collider_backreact_mode"))
        if mode not in ("repel", "attract"):
            raise ValueError("--collider-backreact-mode must be one of: repel, attract")

    if bool(getattr(args, "collider", False)):
        axes = str(getattr(args, "collider_backreact_axes", "x"))
        if axes not in ("x", "xyz"):
            raise ValueError("--collider-backreact-axes must be one of: x, xyz")
        if bool(getattr(args, "collider_octants", False)) and not bool(getattr(args, "collider_detectors", False)):
            raise ValueError("--collider-octants requires --collider-detectors")

    seed0 = int(args.ensemble_seed0)
    if int(args.ensemble) > 0 and int(args.ensemble_seed0) == 6174 and int(args.seed) != 6174:
        seed0 = int(args.seed)
    if int(args.ensemble) == 0:
        seed0 = int(args.seed)

    tr = args.traffic_rate
    tr_rise = args.traffic_rate_rise
    tr_fall = args.traffic_rate_fall

    if (tr_rise is None) != (tr_fall is None):
        raise ValueError("--traffic-rate-rise and --traffic-rate-fall must be provided together")

    if tr is not None and tr_rise is not None:
        raise ValueError("use either --traffic-rate OR (--traffic-rate-rise and --traffic-rate-fall), not both")

    if tr_rise is None:
        if tr is None:
            rate_rise = float(TrafficParams().rate_rise)
            rate_fall = float(TrafficParams().rate_fall)
        else:
            rate_rise = float(tr)
            rate_fall = float(tr)
    else:
        rate_rise = float(tr_rise)
        rate_fall = float(tr_fall)

    tmode = str(getattr(args, "traffic_mode"))
    if tmode in ("nonlinear", "sine_gordon"):
        k = float(getattr(args, "traffic_k"))
        if not math.isfinite(k):
            raise ValueError("--traffic-k must be finite")
        if tmode == "sine_gordon" and k < 0.0:
            raise ValueError("--traffic-k must be >= 0 for --traffic-mode sine_gordon")
        if tmode == "nonlinear":
            lam = float(getattr(args, "traffic_lambda"))
            if (not math.isfinite(lam)) or lam < 0.0:
                raise ValueError("--traffic-lambda must be finite and >= 0")
    # Global traffic boundary validation
    tb = str(getattr(args, "traffic_boundary"))
    tax = str(getattr(args, "traffic_sponge_axes", "xyz")).strip().lower()
    if tb not in ("open", "zero", "neumann", "sponge"):
        raise ValueError("--traffic-boundary must be one of: open, zero, neumann, sponge")
    if tb == "sponge":
        tw = int(getattr(args, "traffic_sponge_width"))
        if tw <= 0:
            raise ValueError("--traffic-sponge-width must be > 0 when --traffic-boundary sponge")
        if tax == "":
            raise ValueError("--traffic-sponge-axes must be non-empty when --traffic-boundary sponge")
        for ch in tax:
            if ch not in ("x", "y", "z"):
                raise ValueError("--traffic-sponge-axes must contain only x/y/z")
    if tb != "sponge":
        if tax not in ("", "xyz"):
            raise ValueError("--traffic-sponge-axes requires --traffic-boundary sponge")
    # Canonical SG sweep values (always concrete, never None)
    sg_k_start: float = float(getattr(args, "soliton_sg_k_start")) if getattr(args, "soliton_sg_k_start") is not None else float(getattr(defaults.soliton, "sg_k_start"))
    sg_k_stop: float = float(getattr(args, "soliton_sg_k_stop")) if getattr(args, "soliton_sg_k_stop") is not None else float(getattr(defaults.soliton, "sg_k_stop"))
    sg_k_steps: int = int(getattr(args, "soliton_sg_k_steps")) if getattr(args, "soliton_sg_k_steps") is not None else int(getattr(defaults.soliton, "sg_k_steps"))
    sg_amp_start: float = float(getattr(args, "soliton_sg_amp_start")) if getattr(args, "soliton_sg_amp_start") is not None else float(getattr(defaults.soliton, "sg_amp_start"))
    sg_amp_stop: float = float(getattr(args, "soliton_sg_amp_stop")) if getattr(args, "soliton_sg_amp_stop") is not None else float(getattr(defaults.soliton, "sg_amp_stop"))
    sg_amp_steps: int = int(getattr(args, "soliton_sg_amp_steps")) if getattr(args, "soliton_sg_amp_steps") is not None else int(getattr(defaults.soliton, "sg_amp_steps"))

    params = PipelineParams(
        seed=int(args.seed),
        lattice=LatticeParams(n=int(args.n), steps=int(args.steps), p_seed=float(args.p_seed), init_mode=str(getattr(args, "lattice_init"))),
        traffic=TrafficParams(
            iters=int(args.traffic_iters),
            mode=str(args.traffic_mode),
            rate_rise=float(rate_rise),
            rate_fall=float(rate_fall),
            c2=float(args.traffic_c2),
            gamma=float(args.traffic_gamma),
            dt=float(args.traffic_dt),
            inject=float(args.traffic_inject),
            decay=float(args.traffic_decay),
            boundary_mode=str(getattr(args, "traffic_boundary")),
            sponge_width=int(getattr(args, "traffic_sponge_width")),
            sponge_strength=float(getattr(args, "traffic_sponge_strength")),
            sponge_axes=str(getattr(args, "traffic_sponge_axes")),
            traffic_k=float(getattr(args, "traffic_k")),
            traffic_lambda=float(getattr(args, "traffic_lambda")),
        ),
        k_index=float(args.k_index),
        r_fit_min=float(args.r_fit_min),
        r_fit_max=float(args.r_fit_max),
        X0=float(args.X0),
        ds=float(args.ds),
        eps=float(args.eps),
        shapiro_b_max=float(args.shapiro_b_max),
        delta_load=bool(args.delta_load),
        delta_jitter=int(args.delta_jitter),
        delta_margin=int(args.delta_margin),
        liveview=bool(getattr(args, "liveview", False)),
        walker=bool(args.walker),
        walker_path=str(args.walker_path),
        walker_mach1=bool(args.walker_mach1),
        walker_circle_radius=int(args.walker_circle_radius),
        walker_circle_period=int(args.walker_circle_period),
        walker_center_x=int(args.walker_center_x),
        walker_center_y=int(args.walker_center_y),
        walker_center_z=int(args.walker_center_z),
        walker_probe_r=int(args.walker_probe_r),
        walker_steps=int(args.walker_steps),
        walker_tick_iters=int(args.walker_tick_iters),
        walker_hold_steps=int(args.walker_hold_steps),
        walker_hold_tick_iters=int(args.walker_hold_tick_iters),
        walker_dx=int(args.walker_dx),
        walker_dy=int(args.walker_dy),
        walker_dz=int(args.walker_dz),
        walker_r_local=int(args.walker_r_local),
        walker_hold_inject=float(args.walker_hold_inject),
        dump_walker=str(args.dump_walker),
        collider_vx=float(args.collider_vx),
        collider_orbit_radius=float(args.collider_orbit_radius),
        collider_orbit_omega=float(args.collider_orbit_omega),
        collider_spin_b=int(getattr(args, "collider_spin_b")),
        collider_steps=int(getattr(args, "collider_steps")),
        collider_detectors=bool(getattr(args, "collider_detectors", False)),
        collider_shell_stride=int(getattr(args, "collider_shell_stride", 2)),
        collider_shell1_inner_frac=float(getattr(args, "collider_shell1_inner_frac", 0.18)),
        collider_shell1_outer_frac=float(getattr(args, "collider_shell1_outer_frac", 0.22)),
        collider_shell2_inner_frac=float(getattr(args, "collider_shell2_inner_frac", 0.22)),
        collider_shell2_outer_frac=float(getattr(args, "collider_shell2_outer_frac", 0.26)),
        collider_hold=bool(getattr(args, "collider_hold", False)),
        collider_hold_grace_steps=int(getattr(args, "collider_hold_grace", int(defaults.collider_hold_grace_steps))),
        collider_hold_steps=int(getattr(args, "collider_hold_steps", int(defaults.collider_hold_steps))),
        collider_center_ball_r=int(getattr(args, "collider_center_ball_r", int(defaults.collider_center_ball_r))),
        collider_backreact=bool(getattr(args, "collider_backreact", bool(getattr(defaults, "collider_backreact", False)))),
        collider_backreact_k=float(getattr(args, "collider_backreact_k", float(getattr(defaults, "collider_backreact_k", 0.0)))),
        collider_backreact_mode=str(getattr(args, "collider_backreact_mode", str(getattr(defaults, "collider_backreact_mode", "repel")))),
        collider_backreact_vmax=float(getattr(args, "collider_backreact_vmax", float(getattr(defaults, "collider_backreact_vmax", 0.25)))),
        collider_impact_b=float(getattr(args, "collider_impact_b", float(getattr(defaults, "collider_impact_b", 0.0)))),
        collider_impact_bz=float(getattr(args, "collider_impact_bz", float(getattr(defaults, "collider_impact_bz", 0.0)))),
        collider_backreact_axes=str(getattr(args, "collider_backreact_axes", str(getattr(defaults, "collider_backreact_axes", "x")))),
        collider_octants=bool(getattr(args, "collider_octants", bool(getattr(defaults, "collider_octants", False)))),
        collider_nucleus=bool(getattr(args, "collider_nucleus", bool(getattr(defaults, "collider_nucleus", False)))),
        collider_nucleus_q=float(getattr(args, "collider_nucleus_q", float(getattr(defaults, "collider_nucleus_q", 1.0)))),
        collider_nucleus_mode=str(getattr(args, "collider_nucleus_mode", str(getattr(defaults, "collider_nucleus_mode", "dc")))),
        collider_nucleus_omega=float(getattr(args, "collider_nucleus_omega", float(getattr(defaults, "collider_nucleus_omega", float(getattr(defaults, "collider_orbit_omega", 0.12)))))),
        collider_nucleus_phase=float(getattr(args, "collider_nucleus_phase", float(getattr(defaults, "collider_nucleus_phase", 0.0)))),
        collider_enable_b=bool(getattr(args, "collider_enable_b", bool(getattr(defaults, "collider_enable_b", True)))),
        collider_halo=bool(getattr(args, "collider_halo", bool(getattr(defaults, "collider_halo", False)))),
        collider_halo_r=int(getattr(args, "collider_halo_r", int(getattr(defaults, "collider_halo_r", 0)))),
        collider_halo_strength=float(getattr(args, "collider_halo_strength", float(getattr(defaults, "collider_halo_strength", 0.0)))),
        collider_halo_profile=str(getattr(args, "collider_halo_profile", str(getattr(defaults, "collider_halo_profile", "linear")))),
        collider_halo_center=str(getattr(args, "collider_halo_center", str(getattr(defaults, "collider_halo_center", "nucleus")))),
        ringdown=RingdownParams(
            steps=int(getattr(args, "ringdown_steps", int(getattr(defaults.ringdown, "steps", 1000)))),
            pulse_amp=float(getattr(args, "ringdown_amp", float(getattr(defaults.ringdown, "pulse_amp", 50.0)))),
            sigma_start=float(getattr(args, "ringdown_sigma_start", float(getattr(defaults.ringdown, "sigma_start", 1.2)))),
            sigma_stop=float(getattr(args, "ringdown_sigma_stop", float(getattr(defaults.ringdown, "sigma_stop", 6.0)))),
            sigma_step=float(getattr(args, "ringdown_sigma_step", float(getattr(defaults.ringdown, "sigma_step", 0.2)))),
            probe_window=int(getattr(args, "ringdown_probe_window", int(getattr(defaults.ringdown, "probe_window", 500)))),
        ),
        soliton=SolitonParams(
            steps=int(getattr(args, "soliton_steps", int(getattr(defaults.soliton, "steps", 2000)))),
            sigma=float(getattr(args, "soliton_sigma", float(getattr(defaults.soliton, "sigma", 4.0)))),
            sigma_start=float(
                getattr(defaults.soliton, "sigma_start", 0.0)
                if getattr(args, "soliton_sigma_start", None) is None
                else getattr(args, "soliton_sigma_start")
            ),
            sigma_stop=float(
                getattr(defaults.soliton, "sigma_stop", 0.0)
                if getattr(args, "soliton_sigma_stop", None) is None
                else getattr(args, "soliton_sigma_stop")
            ),
            sigma_steps=int(
                getattr(defaults.soliton, "sigma_steps", 1)
                if getattr(args, "soliton_sigma_steps", None) is None
                else getattr(args, "soliton_sigma_steps")
            ),
            amp=float(getattr(args, "soliton_amp", float(getattr(defaults.soliton, "amp", 100.0)))),
            k=float(getattr(args, "soliton_k", float(getattr(defaults.soliton, "k", 0.0)))),
            lambda_start=float(getattr(args, "soliton_lambda_start", float(getattr(defaults.soliton, "lambda_start", 0.0)))),
            lambda_stop=float(soliton_lambda_stop_f),
            lambda_steps=int(getattr(args, "soliton_lambda_steps", int(getattr(defaults.soliton, "lambda_steps", 20)))),
            init_vev=bool(getattr(args, "soliton_init_vev", False)),
            vev_sign=int(getattr(args, "soliton_vev_sign", 1)),
            sg_k_start=sg_k_start,
            sg_k_stop=sg_k_stop,
            sg_k_steps=sg_k_steps,
            sg_amp_start=sg_amp_start,
            sg_amp_stop=sg_amp_stop,
            sg_amp_steps=sg_amp_steps,
        ),
    )

    # Attach optional SG besttail thresholds onto params.soliton (no constructor changes required).
    # Downstream code should read via getattr(params.soliton, "sg_besttail_*", None) and fall back
    # to internal defaults when None.
    for n in (
        "sg_besttail_vel_rms_max",
        "sg_besttail_de_step_abs_max",
        "sg_besttail_wrap_pi_frac_max",
        "sg_besttail_wrap_2pi_frac_max",
        "sg_besttail_cm_r_frac_max",
        "sg_besttail_peak_r_frac_max",
        "sg_besttail_peak_dev_min",
    ):
        v = getattr(args, n, None)
        if v is not None:
            setattr(params.soliton, n, float(v))

    return args, params, int(seed0)
