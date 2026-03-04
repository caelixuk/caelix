# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""double_slit.py — Young-style interference experiment (telegraph + hard wall mask)

This module owns the double-slit runner:
- Builds a Dirichlet wall mask with one or two slits.
- Injects a moving point source (“gun”).
- Evolves the telegraph field.
- Samples intensity on a detector line and writes a CSV.

Notes:
- This experiment requires telegraph mode.
- Chirality/parity selection is supported via (chiral_select, chiral_field).
- Output paths are caller-resolved; if out_csv is empty we write a stable default under out_dir.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, List, Optional

import numpy as np

from params import PipelineParams
from traffic import evolve_telegraph_traffic_steps
from utils import ensure_parent_dir


def run_double_slit(
    params: PipelineParams,
    args: Any,
    *,
    out_dir: str,
    progress_cb: Optional[Callable[[int], None]] = None,
    provenance_header: str = "",
) -> str:
    """Run the double-slit experiment using CLI-like args.

    `args` is expected to provide the ds_* and chiral_* attributes defined in core.py.
    Returns the output CSV path.
    """

    if str(params.traffic.mode).strip().lower() != "telegraph":
        raise ValueError("--double-slit requires --traffic-mode telegraph")

    n = int(params.lattice.n)

    def _extract_run_id(ph: str) -> str:
        rid = ""
        for ln in str(ph).splitlines():
            s = str(ln).strip()
            if s.startswith("#"):
                s = s[1:].strip()
            if s.startswith("run_id="):
                rid = s.split("=", 1)[1].strip()
                break
        return rid

    def _get(name: str, default: Any) -> Any:
        return getattr(args, name, default)

    # Boundary-aware detector bounds (avoid sampling inside the sponge).
    # Prefer explicit CLI args when present; fall back to params.traffic.
    boundary_mode = str(_get("traffic_boundary", getattr(params.traffic, "boundary", "hard"))).strip().lower()
    sponge_width = int(_get("traffic_sponge_width", getattr(params.traffic, "sponge_width", getattr(params.traffic, "sponge_w", 0))) or 0)
    sponge_strength = float(_get("traffic_sponge_strength", getattr(params.traffic, "sponge_strength", getattr(params.traffic, "sponge_s", 0.0))) or 0.0)

    det_y0 = 0
    det_y1 = n
    if boundary_mode == "sponge" and sponge_width > 0:
        det_y0 = int(sponge_width)
        det_y1 = int(n - sponge_width)
        if not (0 <= det_y0 < det_y1 <= n):
            raise ValueError("double-slit: invalid sponge detector crop")

    steps = int(_get("ds_steps", 800))
    wall_x = int(_get("ds_wall_x", 0))
    slit_sep = int(_get("ds_slit_sep", 16))
    slit_width = int(_get("ds_slit_width", 4))
    gun_start = int(_get("ds_gun_start", 10))
    gun_speed = float(_get("ds_gun_speed", 0.5))
    gun_stop = int(_get("ds_gun_stop", 0))
    detector_x = int(_get("ds_detector_x", 0))
    line_z = int(_get("ds_line_z", 0))
    single_slit = bool(_get("ds_single_slit", False))
    burn = int(_get("ds_burn", 150))
    sample_every = int(_get("ds_sample_every", 1))
    window = int(_get("ds_window", 80))
    out_csv = str(_get("ds_out", "")).strip()
    verbose = bool(_get("ds_verbose", False))
    dump_samples = bool(_get("ds_dump_samples", False))

    src_mode = str(_get("ds_source", "dc")).strip().lower()
    src_amp = float(_get("ds_amp", 1.0))
    src_omega = float(_get("ds_omega", 0.0))
    src_half_period = int(_get("ds_half_period", 0))

    chiral_select = int(_get("chiral_select", 0))
    chiral_field = str(_get("chiral_field", "none")).strip().lower()

    # Chirality validation
    if chiral_select not in (-1, 0, 1):
        raise ValueError("--chiral-select must be one of -1, 0, 1")
    if chiral_field not in ("none", "split_x", "split_y", "split_z"):
        raise ValueError("--chiral-field must be one of: none, split_x, split_y, split_z")
    if chiral_select != 0 and chiral_field == "none":
        raise ValueError("--chiral-field is required when --chiral-select is non-zero")

    if wall_x <= 0:
        wall_x = n // 2
    if detector_x <= 0:
        # Prefer a detector plane well inside the non-sponge region.
        if boundary_mode == "sponge" and sponge_width > 0:
            detector_x = n - int(sponge_width) - 33
        else:
            detector_x = n - 10
    if gun_stop <= 0:
        gun_stop = wall_x - 10

    if not (0 < wall_x < n - 1):
        raise ValueError("--ds-wall-x out of bounds")
    if not (0 < detector_x < n - 1):
        raise ValueError("--ds-detector-x out of bounds")

    if boundary_mode == "sponge" and sponge_width > 0:
        x_max = n - int(sponge_width) - 1
        x_min = int(sponge_width)
        if detector_x > x_max:
            detector_x = x_max
        if detector_x < x_min:
            detector_x = x_min

    if not (0 <= gun_start < n):
        raise ValueError("--ds-gun-start out of bounds")
    if gun_stop < 0:
        raise ValueError("--ds-gun-stop must be >= 0")
    if slit_sep < 1 or slit_width < 1:
        raise ValueError("--ds-slit-sep and --ds-slit-width must be >= 1")
    if sample_every < 1:
        raise ValueError("--ds-sample-every must be >= 1")
    if window < 1:
        raise ValueError("--ds-window must be >= 1")
    if src_mode not in ("dc", "sine", "square"):
        raise ValueError("--ds-source must be one of: dc, sine, square")
    if not np.isfinite(src_amp):
        raise ValueError("--ds-amp must be finite")
    if src_mode == "sine":
        if not np.isfinite(src_omega) or src_omega <= 0.0:
            raise ValueError("--ds-omega must be > 0 for --ds-source sine")
    if src_mode == "square":
        if src_half_period <= 0:
            raise ValueError("--ds-half-period must be >= 1 for --ds-source square")

    phi = np.zeros((n, n, n), dtype=np.float32)
    vel = np.zeros((n, n, n), dtype=np.float32)
    src = np.zeros((n, n, n), dtype=np.float32)
    mask = np.zeros((n, n, n), dtype=np.int8)

    chirality: Optional[np.ndarray] = None
    if chiral_select != 0:
        chirality = np.empty((n, n, n), dtype=np.int8)
        if chiral_field == "split_x":
            mid = n // 2
            chirality[:mid, :, :] = -1
            chirality[mid:, :, :] = 1
        elif chiral_field == "split_y":
            mid = n // 2
            chirality[:, :mid, :] = -1
            chirality[:, mid:, :] = 1
        elif chiral_field == "split_z":
            mid = n // 2
            chirality[:, :, :mid] = -1
            chirality[:, :, mid:] = 1
        else:
            raise ValueError("unreachable chiral_field")

    cy = n // 2
    if line_z <= 0:
        cz = n // 2
    else:
        cz = int(line_z)
    if not (0 <= cz < n):
        raise ValueError("--ds-line-z out of bounds")

    # Wall
    mask[wall_x, :, :] = 1

    y1_start = 0
    y1_end = 0
    y2_start = 0
    y2_end = 0

    if single_slit:
        # Single slit centerd at cy.
        y1_start = cy - (slit_width // 2)
        y1_end = y1_start + slit_width
        y2_start = 0
        y2_end = 0

        if y1_start < 1 or y1_end > n - 2:
            raise ValueError("double-slit: slit out of bounds")

        mask[wall_x, y1_start:y1_end, :] = 0

        if verbose:
            print("[double-slit] n=%d steps=%d wall_x=%d detector_x=%d cz=%d" % (n, steps, wall_x, detector_x, cz))
            print("[double-slit] single slit: y=[%d:%d]" % (y1_start, y1_end))
    else:
        # Two slits (gaps) in y, spanning all z.
        y1_start = cy - (slit_sep // 2) - slit_width
        y1_end = cy - (slit_sep // 2)
        y2_start = cy + (slit_sep // 2)
        y2_end = cy + (slit_sep // 2) + slit_width

        if y1_start < 1 or y2_end > n - 2:
            raise ValueError("double-slit: slits out of bounds")

        mask[wall_x, y1_start:y1_end, :] = 0
        mask[wall_x, y2_start:y2_end, :] = 0

        if verbose:
            print("[double-slit] n=%d steps=%d wall_x=%d detector_x=%d cz=%d" % (n, steps, wall_x, detector_x, cz))
            print("[double-slit] slits: y=[%d:%d] and y=[%d:%d]" % (y1_start, y1_end, y2_start, y2_end))

    if verbose:
        print("[double-slit] gun: start=%d stop=%d speed=%.6g" % (gun_start, gun_stop, gun_speed))
        if src_mode == "dc":
            print("[double-slit] src: mode=dc amp=%.6g" % (float(src_amp),))
        elif src_mode == "sine":
            print("[double-slit] src: mode=sine amp=%.6g omega=%.6g" % (float(src_amp), float(src_omega)))
        else:
            print("[double-slit] src: mode=square amp=%.6g half_period=%d" % (float(src_amp), int(src_half_period)))

        if chiral_select == 0:
            print("[double-slit] chirality: off")
        else:
            print("[double-slit] chirality: select=%d field=%s" % (int(chiral_select), str(chiral_field)))

    samples_t: List[int] = []
    samples_i: List[np.ndarray] = []
    for t in range(steps):
        src.fill(0.0)
        x = gun_start + int(float(t) * gun_speed)
        if x < gun_stop:
            if src_mode == "dc":
                s = float(src_amp)
            elif src_mode == "sine":
                s = float(src_amp) * float(math.sin(float(src_omega) * float(t)))
            else:
                # Square wave: flip sign every half-period ticks.
                if ((int(t) // int(src_half_period)) % 2) == 0:
                    s = float(src_amp)
                else:
                    s = -float(src_amp)
            src[x, cy, cz] = float(s)

        phi, vel = evolve_telegraph_traffic_steps(
            phi,
            vel,
            src,
            params.traffic,
            1,
            mask=mask,
            chirality=chirality,
            chiral_select=int(chiral_select),
        )

        if progress_cb is not None:
            progress_cb(1)

        if t >= burn and ((t - burn) % sample_every == 0):
            line = phi[detector_x, det_y0:det_y1, cz].astype(np.float64)
            samples_t.append(int(t))
            samples_i.append(line * line)

        if verbose and (t % 50) == 0:
            print("[double-slit] step=%d phi_max=%.6g" % (t, float(np.max(phi))))

    if len(samples_i) == 0:
        raise ValueError("double-slit produced no samples; reduce --ds-burn or increase --ds-steps")

    w = int(window)
    if w > len(samples_i):
        w = len(samples_i)

    # Average the last w sampled detector lines.
    pattern = np.mean(np.stack(samples_i[-w:], axis=0), axis=0)

    # Window timestep bounds (inclusive) for the averaged pattern.
    t_last_used = int(samples_t[-1])
    t_first_used = int(samples_t[-w])
    window_span = int(t_last_used - t_first_used)

    # Simple sanity stats.
    pattern_sum = float(np.sum(pattern))
    pattern_max = float(np.max(pattern))
    phi_max_last = float(np.max(phi))

    # Use the experiment folder name as the filename stem (matches log naming).
    stem = os.path.basename(os.path.dirname(os.path.normpath(str(out_dir))))
    if stem == "":
        stem = "02_double_slit"

    run_id = _extract_run_id(provenance_header)

    npz_path = ""
    if dump_samples:
        if run_id == "":
            raise ValueError("double-slit: provenance_header must include run_id when --ds-dump-samples is enabled")

        npz_dir = os.path.join(str(out_dir), "_npz")
        npz_name = "%s_n%d_steps%d_%s.npz" % (str(stem), int(n), int(steps), str(run_id))
        npz_path = os.path.join(npz_dir, npz_name)
        ensure_parent_dir(npz_path)

        # Save intensity cube as float32 to keep size down: shape=(samples, det_y_len)
        cube = np.stack([a.astype(np.float32, copy=False) for a in samples_i], axis=0)
        t_arr = np.asarray(samples_t, dtype=np.int32)
        y_arr = np.arange(det_y0, det_y1, dtype=np.int32)

        # Store provenance/header as bytes for portability.
        ph_bytes = np.asarray([str(provenance_header)], dtype=object)

        np.savez_compressed(
            npz_path,
            intensity=cube,
            t=t_arr,
            y=y_arr,
            det_y0=np.int32(det_y0),
            det_y1=np.int32(det_y1),
            detector_x=np.int32(detector_x),
            cz=np.int32(cz),
            n=np.int32(n),
            steps=np.int32(steps),
            burn=np.int32(burn),
            sample_every=np.int32(sample_every),
            window=np.int32(w),
            boundary_mode=np.asarray([str(boundary_mode)], dtype=object),
            sponge_width=np.int32(sponge_width),
            sponge_strength=np.float32(sponge_strength),
            provenance_header=ph_bytes,
        )

    if out_csv == "":
        if run_id == "":
            raise ValueError("double-slit: provenance_header must include run_id when ds_out is not set")

        csv_dir = os.path.join(str(out_dir), "_csv")
        name = "%s_n%d_%s.csv" % (str(stem), int(n), str(run_id))
        out_csv = os.path.join(csv_dir, name)
    else:
        out_csv = str(out_csv)

    ensure_parent_dir(out_csv)

    with open(out_csv, "w", encoding="utf-8") as f:
        if str(provenance_header).strip() != "":
            f.write(str(provenance_header))
            if not str(provenance_header).endswith("\n"):
                f.write("\n")
        f.write("# double-slit\n")
        f.write("# n=%d steps=%d wall_x=%d detector_x=%d cz=%d cy=%d\n" % (n, steps, wall_x, detector_x, cz, cy))
        f.write(
            "# slits: slit_sep=%d slit_width=%d single_slit=%d y1=[%d:%d] y2=[%d:%d] z=[0:%d)\n" % (
                slit_sep,
                slit_width,
                (1 if single_slit else 0),
                y1_start,
                y1_end,
                y2_start,
                y2_end,
                n,
            )
        )
        f.write("# detector: line_y at x=%d, z=%d; intensity=phi^2\n" % (detector_x, cz))
        f.write("# detector_y=[%d:%d) boundary=%s sponge_width=%d sponge_strength=%.9g\n" % (det_y0, det_y1, str(boundary_mode), int(sponge_width), float(sponge_strength)))
        f.write("# gun: start=%d stop=%d speed=%.9g\n" % (gun_start, gun_stop, gun_speed))
        f.write("# src: mode=%s amp=%.9g omega=%.9g half_period=%d\n" % (str(src_mode), float(src_amp), float(src_omega), int(src_half_period)))
        f.write("# chirality: select=%d field=%s\n" % (int(chiral_select), str(chiral_field)))
        f.write("# sampling: burn=%d sample_every=%d samples=%d window=%d\n" % (burn, sample_every, len(samples_i), w))
        f.write("# dump_samples=%d\n" % (1 if dump_samples else 0))
        if npz_path != "":
            f.write("# npz=%s\n" % (str(npz_path),))
        # section:pattern_mean
        f.write("# section:pattern_mean\n")
        f.write("y,y_off,intensity,intensity_norm\n")
        denom = pattern_sum if pattern_sum > 0.0 else 1.0
        for yi, y in enumerate(range(det_y0, det_y1)):
            inten = float(pattern[yi])
            f.write("%d,%d,%.12g,%.12g\n" % (y, int(y - cy), inten, float(inten / denom)))

        if dump_samples:
            # section:samples
            f.write("# section:samples\n")
            f.write("t,sample_idx,y,intensity\n")
            for i in range(len(samples_i)):
                t_s = int(samples_t[i])
                line_i = samples_i[i]
                for yi, y in enumerate(range(det_y0, det_y1)):
                    f.write("%d,%d,%d,%.12g\n" % (t_s, i, y, float(line_i[yi])))

        # section:sample_metrics
        f.write("# section:sample_metrics\n")
        f.write("t,sample_idx,max_intensity,sum_intensity,centroid_y,width_rms\n")
        y_idx = np.arange(det_y0, det_y1, dtype=np.float64)
        for i in range(len(samples_i)):
            t_s = int(samples_t[i])
            line_i = samples_i[i].astype(np.float64)
            s = float(np.sum(line_i))
            m = float(np.max(line_i))
            if s > 0.0:
                cy_w = float(np.sum(y_idx * line_i) / s)
                w2 = float(np.sum(((y_idx - cy_w) ** 2) * line_i) / s)
                wrms = float(math.sqrt(max(0.0, w2)))
            else:
                cy_w = float("nan")
                wrms = float("nan")
            f.write("%d,%d,%.12g,%.12g,%.12g,%.12g\n" % (t_s, i, m, s, cy_w, wrms))

    if verbose:
        print("[double-slit] out_csv=%s" % (out_csv,))
        print("[double-slit] intensity: max=%.6g sum=%.6g" % (float(pattern_max), float(pattern_sum)))

    return str(out_csv)