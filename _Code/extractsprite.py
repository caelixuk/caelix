# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""extractsprite.py — Sprite extraction utilities + CLI"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
import sys
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SpriteExtractSpec:
    """Specification for extracting a sprite patch."""

    # Either provide a fixed cube radius, or auto-estimate from thresholds.
    radius: Optional[int] = None

    # Auto-estimation: include voxels with |phi| >= peak_abs * rel_thresh.
    rel_thresh: float = 0.08

    # Extra margin added on top of the estimated radius (or fixed radius).
    pad: int = 4

    # Hard safety cap on radius during estimation (fail-loud if exceeded).
    max_radius: int = 72

    # Optional explicit centre (x,y,z). If None, use argmax(|phi|).
    centre_xyz: Optional[Tuple[int, int, int]] = None


def _require_3d(name: str, arr: np.ndarray) -> None:
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape={arr.shape}")


def find_sprite_centre(phi: np.ndarray) -> Tuple[int, int, int]:
    """Return centre as argmax(|phi|)."""

    _require_3d("phi", phi)
    k = int(np.argmax(np.abs(phi)))
    x, y, z = np.unravel_index(k, phi.shape)
    return int(x), int(y), int(z)


def estimate_sprite_radius(
    phi: np.ndarray,
    *,
    centre_xyz: Tuple[int, int, int],
    rel_thresh: float,
    pad: int,
    max_radius: int,
) -> int:
    """Estimate a cube radius that contains the sprite.

    We treat the sprite as a *smeared* object across many voxels.

    Algorithm:
      - Compute peak_abs = max(|phi|).
      - Threshold = peak_abs * rel_thresh.
      - Find the farthest voxel (Chebyshev distance from centre) within that thresholded support.
      - Return that radius + pad.

    This is fast and stable for a single isolated sprite. Fail-loud if the inferred support
    exceeds max_radius (usually means: multiple objects, too much radiation, or threshold too low).
    """

    _require_3d("phi", phi)
    if not (0.0 < rel_thresh < 1.0):
        raise ValueError(f"rel_thresh must be in (0,1), got {rel_thresh}")
    if pad < 0:
        raise ValueError(f"pad must be >=0, got {pad}")
    if max_radius < 4:
        raise ValueError(f"max_radius must be >=4, got {max_radius}")

    cx, cy, cz = centre_xyz
    peak = float(np.max(np.abs(phi)))
    if peak <= 0.0:
        raise ValueError("phi has zero amplitude; cannot extract sprite")
    thr = peak * float(rel_thresh)

    mask = np.abs(phi) >= thr
    if not bool(np.any(mask)):
        raise ValueError("threshold produced empty support; increase rel_thresh")

    xs, ys, zs = np.nonzero(mask)
    supp = int(xs.size)
    n_tot = int(phi.size)
    supp_frac = float(supp) / float(n_tot) if n_tot > 0 else 0.0
    # Expose support fraction for canonical logs.
    support_frac = float(supp_frac)

    dx = np.abs(xs.astype(np.int32) - int(cx))
    dy = np.abs(ys.astype(np.int32) - int(cy))
    dz = np.abs(zs.astype(np.int32) - int(cz))
    r = int(np.max(np.maximum(np.maximum(dx, dy), dz)))

    r2 = r + int(pad)
    if r2 > int(max_radius):
        raise ValueError(
            f"estimated radius {r2} exceeds max_radius={max_radius}; "
            f"support_voxels={supp} support_frac={supp_frac:.6g}; "
            f"likely not a single isolated sprite (thr={thr:.6g}, peak={peak:.6g}, rel_thresh={rel_thresh:.6g}, pad={int(pad)})"
        )

    return r2


def extract_cube(arr: np.ndarray, *, centre_xyz: Tuple[int, int, int], radius: int) -> np.ndarray:
    """Extract a (2R+1)^3 cube centred on centre_xyz (fail-loud if out of bounds)."""

    _require_3d("arr", arr)
    if radius < 1:
        raise ValueError(f"radius must be >=1, got {radius}")

    n0, n1, n2 = arr.shape
    if n0 != n1 or n0 != n2:
        raise ValueError(f"expected cubic grid, got {arr.shape}")
    n = n0

    cx, cy, cz = centre_xyz
    x0, x1 = cx - radius, cx + radius + 1
    y0, y1 = cy - radius, cy + radius + 1
    z0, z1 = cz - radius, cz + radius + 1

    if x0 < 0 or y0 < 0 or z0 < 0 or x1 > n or y1 > n or z1 > n:
        raise ValueError(f"patch out of bounds: centre={centre_xyz}, radius={radius}, n={n}")

    return arr[x0:x1, y0:y1, z0:z1].copy()


def extract_sprite_patch(
    *,
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    load: np.ndarray,
    spec: SpriteExtractSpec,
) -> tuple[Tuple[int, int, int], int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract a sprite patch from in-memory arrays.

    Returns:
      (centre_xyz, radius, phi_patch, vel_patch, src_patch, load_patch, stats)
    """

    _require_3d("phi", phi)
    _require_3d("vel", vel)
    _require_3d("src", src)
    _require_3d("load", load)

    if phi.shape != vel.shape or phi.shape != src.shape or phi.shape != load.shape:
        raise ValueError(f"state arrays must have identical shapes, got phi={phi.shape} vel={vel.shape} src={src.shape} load={load.shape}")

    centre = spec.centre_xyz or find_sprite_centre(phi)

    stats_support_frac = float("nan")

    if spec.radius is not None:
        r = int(spec.radius)
        if r < 1:
            raise ValueError(f"radius must be >=1, got {r}")
        r = r + int(spec.pad)
        if r > int(spec.max_radius):
            raise ValueError(f"radius+pad={r} exceeds max_radius={spec.max_radius}")
    else:
        r = estimate_sprite_radius(
            phi,
            centre_xyz=centre,
            rel_thresh=float(spec.rel_thresh),
            pad=int(spec.pad),
            max_radius=int(spec.max_radius),
        )
        peak = float(np.max(np.abs(phi)))
        thr = peak * float(spec.rel_thresh)
        mask = np.abs(phi) >= thr
        supp = int(np.count_nonzero(mask))
        n_tot = int(phi.size)
        stats_support_frac = float(supp) / float(n_tot) if n_tot > 0 else 0.0

    phi_p = extract_cube(phi, centre_xyz=centre, radius=r).astype(np.float32, copy=False)
    vel_p = extract_cube(vel, centre_xyz=centre, radius=r).astype(np.float32, copy=False)
    src_p = extract_cube(src, centre_xyz=centre, radius=r).astype(np.float32, copy=False)
    load_p = extract_cube(load, centre_xyz=centre, radius=r).astype(np.float32, copy=False)

    stats: dict[str, Any] = {
        "centre_xyz": [int(centre[0]), int(centre[1]), int(centre[2])],
        "radius": int(r),
        "L": int(phi_p.shape[0]),
        "phi_abs_max_full": float(np.max(np.abs(phi))),
        "phi_abs_max_patch": float(np.max(np.abs(phi_p))),
        "vel_abs_max_patch": float(np.max(np.abs(vel_p))),
        "rel_thresh": float(spec.rel_thresh),
        "pad": int(spec.pad),
        "max_radius": int(spec.max_radius),
        "support_frac": float(stats_support_frac),
    }

    return centre, r, phi_p, vel_p, src_p, load_p, stats


def read_full_state_h5(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read a full-state snapshot HDF5 file.

    Required datasets: phi, vel, src, load.
    """

    if not str(path).strip():
        raise ValueError("path is empty")

    import h5py

    with h5py.File(path, "r") as f:
        for k in ("phi", "vel", "src", "load"):
            if k not in f:
                raise ValueError(f"missing dataset '{k}' in {path}")
        phi = np.asarray(f["phi"], dtype=np.float32)
        vel = np.asarray(f["vel"], dtype=np.float32)
        src = np.asarray(f["src"], dtype=np.float32)
        load = np.asarray(f["load"], dtype=np.float32)
        meta = {str(k): f.attrs[k] for k in f.attrs.keys()}

    return phi, vel, src, load, meta


def write_sprite_asset_h5(
    path: str,
    *,
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    load: np.ndarray,
    meta: dict[str, Any],
    compression: str = "gzip",
    compression_level: int = 4,
) -> str:
    """Write an extracted sprite patch asset.

    Datasets: phi, vel, src, load (all float32).
    Attributes include export.schema and a JSON stats blob.
    """

    if not str(path).strip():
        raise ValueError("path is empty")

    import h5py

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with h5py.File(path, "w") as f:
        comp = str(compression or "").strip().lower()
        if comp in ("", "none", "off", "false", "0"):
            comp2: str | None = None
        else:
            comp2 = comp

        # h5py expects `compression=None` for no compression.
        # `lzf` does not accept `compression_opts`.
        kwargs: dict[str, Any] = {"shuffle": True}
        if comp2 is not None:
            kwargs["compression"] = comp2
            if comp2 == "gzip":
                kwargs["compression_opts"] = int(compression_level)

        f.create_dataset("phi", data=np.asarray(phi, dtype=np.float32), **kwargs)
        f.create_dataset("vel", data=np.asarray(vel, dtype=np.float32), **kwargs)
        f.create_dataset("src", data=np.asarray(src, dtype=np.float32), **kwargs)
        f.create_dataset("load", data=np.asarray(load, dtype=np.float32), **kwargs)

        for k, v in meta.items():
            kk = str(k)
            # Never allow callers to overwrite the sprite schema.
            if kk == "export.schema":
                continue
            try:
                f.attrs[kk] = v
            except Exception:
                pass

        f.attrs["export.schema"] = "dlft.sprite.v1"

    return path


# --- Public in-memory entry point (no full-state H5 required)
def extract_sprite_from_fields(
    *,
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    load: np.ndarray,
    out_arg: str,
    stem: str,
    step: Any = None,
    snapshot_kind: str = "besttail",
    source_snapshot_path: str = "",
    spec: Optional[SpriteExtractSpec] = None,
    meta0: Optional[dict[str, Any]] = None,
    compression: str = "gzip",
    compression_level: int = 4,
) -> tuple[str, dict[str, Any]]:
    """Extract + write a sprite asset directly from in-memory arrays.

    Returns: (out_h5_path, stats)

    Notes:
      - This does NOT write any full-grid snapshots.
      - `out_arg` may be a directory or a full filename.
      - `stem` is used to name the output when `out_arg` is a directory.
    """

    spec2 = spec if spec is not None else SpriteExtractSpec()

    centre2, r, phi_p, vel_p, src_p, load_p, stats = extract_sprite_patch(
        phi=phi,
        vel=vel,
        src=src,
        load=load,
        spec=spec2,
    )

    out_h5 = _auto_sprite_out_path_from_stem(out_arg, stem=str(stem), stats=stats, step=step)

    meta: dict[str, Any] = {}
    if meta0 is not None:
        for k, v in meta0.items():
            kk = str(k)
            if kk.startswith("export."):
                continue
            meta[kk] = v

    # Source provenance (best-effort; may be empty for pure in-memory callers).
    meta["source.snapshot.path"] = str(source_snapshot_path)
    meta["source.snapshot.kind"] = str(snapshot_kind)

    meta["sprite.stats.json"] = json.dumps(stats, sort_keys=True)
    meta["sprite.centre_xyz"] = json.dumps([int(centre2[0]), int(centre2[1]), int(centre2[2])])
    meta["sprite.radius"] = int(r)

    write_sprite_asset_h5(
        out_h5,
        phi=phi_p,
        vel=vel_p,
        src=src_p,
        load=load_p,
        meta=meta,
        compression=str(compression),
        compression_level=int(compression_level),
    )

    return out_h5, stats


def _parse_xyz(s: str) -> Optional[Tuple[int, int, int]]:
    t = str(s or "").strip()
    if not t:
        return None
    parts = [p.strip() for p in t.split(",")]
    if len(parts) != 3:
        raise ValueError("centre must be 'x,y,z'")
    return int(parts[0]), int(parts[1]), int(parts[2])


# --- Output path helper for sprite extraction
def _auto_sprite_out_path(out_arg: str, in_h5: str, *, stats: dict[str, Any], meta0: dict[str, Any]) -> str:
    """Resolve output path.

    If out_arg is a directory (exists as a dir) or ends with a path separator, build a deterministic
    filename using the input stem and extracted stats.
    """

    out_arg2 = str(out_arg or "").strip()
    if out_arg2 == "":
        raise ValueError("--out is empty")

    is_dir = os.path.isdir(out_arg2) or out_arg2.endswith(os.sep) or out_arg2.endswith("/")
    if not is_dir:
        return out_arg2

    os.makedirs(out_arg2, exist_ok=True)

    step = meta0.get("export.step", None)
    step_s = "NA"
    try:
        if step is not None:
            step_s = str(int(step))
    except Exception:
        step_s = "NA"

    L = int(stats.get("L", 0) or 0)
    r = int(stats.get("radius", 0) or 0)

    stem = os.path.splitext(os.path.basename(str(in_h5)))[0]
    name = f"{stem}__sprite_step{step_s}_L{L:03d}_R{r:03d}.h5"
    return os.path.join(out_arg2, name)

# --- Output path helper for in-memory extraction (no input snapshot path)
def _auto_sprite_out_path_from_stem(out_arg: str, *, stem: str, stats: dict[str, Any], step: Any = None) -> str:
    out_arg2 = str(out_arg or "").strip()
    if out_arg2 == "":
        raise ValueError("out_arg is empty")

    is_dir = os.path.isdir(out_arg2) or out_arg2.endswith(os.sep) or out_arg2.endswith("/")
    if not is_dir:
        return out_arg2

    os.makedirs(out_arg2, exist_ok=True)

    step_s = "NA"
    try:
        if step is not None and str(step) != "":
            step_s = str(int(step))
    except Exception:
        step_s = "NA"

    L = int(stats.get("L", 0) or 0)
    r = int(stats.get("radius", 0) or 0)

    stem2 = str(stem or "sprite").strip()
    if stem2 == "":
        stem2 = "sprite"

    name = f"{stem2}__sprite_step{step_s}_L{L:03d}_R{r:03d}.h5"
    return os.path.join(out_arg2, name)

# --- Helper to infer snapshot kind (besttail/end) from filename
def _infer_snapshot_kind(in_h5: str) -> str:
    stem = os.path.splitext(os.path.basename(str(in_h5)))[0].lower()
    if stem.endswith("_besttail") or "_besttail_" in stem or "_besttail" in stem:
        return "besttail"
    return "end"


def _peek_flag(argv: list[str], name: str) -> str:
    # Best-effort flag peeker for canonical logs when argparse fails.
    try:
        for i, a in enumerate(argv):
            if a == name and i + 1 < len(argv):
                return str(argv[i + 1])
            if a.startswith(name + "="):
                return str(a.split("=", 1)[1])
    except Exception:
        pass
    return ""


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="extractsprite", add_help=True)
    ap.add_argument("--in", dest="in_h5", required=True, help="Input full-state snapshot HDF5 (must contain phi,vel,src,load)")
    ap.add_argument("--out", dest="out_h5", required=True, help="Output sprite asset HDF5")

    ap.add_argument("--centre", default="", help="Optional centre 'x,y,z' (default: argmax(|phi|))")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--radius", type=int, default=None, help="Fixed radius R (cube is (2R+1)^3). pad is still applied.")
    g.add_argument("--rel-thresh", type=float, default=0.08, help="Auto radius: include voxels with |phi| >= peak*rel_thresh")

    ap.add_argument("--pad", type=int, default=4, help="Extra margin added to estimated/fixed radius")
    ap.add_argument("--max-radius", type=int, default=96, help="Fail if inferred radius exceeds this")

    ap.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"])
    ap.add_argument("--compression-level", type=int, default=4)

    argv2 = list(argv) if argv is not None else list(sys.argv[1:])
    in_peek = _peek_flag(argv2, "--in")
    out_peek = _peek_flag(argv2, "--out")
    snap_kind_peek = _infer_snapshot_kind(in_peek) if str(in_peek).strip() != "" else "unknown"
    out_hint = str(out_peek)

    try:
        ns = ap.parse_args(argv)
    except SystemExit:
        # argparse exits on parse errors; still emit the canonical line for core.py.
        print(f"[extractsprite] phase=err rc=2 in={in_peek} kind={snap_kind_peek} out={out_hint}", flush=True)
        raise

    if int(ns.compression_level) < 0:
        raise ValueError("--compression-level must be >= 0")

    snap_kind = _infer_snapshot_kind(ns.in_h5)

    # Canonical entry line (core.py may scrape this).
    print(f"[extractsprite] phase=begin in={ns.in_h5} kind={snap_kind} out={out_hint}", flush=True)

    try:
        phi, vel, src, load, meta0 = read_full_state_h5(ns.in_h5)

        centre = _parse_xyz(ns.centre)

        spec = SpriteExtractSpec(
            radius=ns.radius,
            rel_thresh=float(ns.rel_thresh),
            pad=int(ns.pad),
            max_radius=int(ns.max_radius),
            centre_xyz=centre,
        )

        centre2, r, phi_p, vel_p, src_p, load_p, stats = extract_sprite_patch(
            phi=phi,
            vel=vel,
            src=src,
            load=load,
            spec=spec,
        )

        out_h5 = _auto_sprite_out_path(ns.out_h5, ns.in_h5, stats=stats, meta0=meta0)

        # Keep snapshot provenance, but do NOT copy snapshot export.* keys (bytes/schema/etc) into the sprite asset.
        meta: dict[str, Any] = {}
        for k, v in meta0.items():
            kk = str(k)
            if kk.startswith("export."):
                continue
            meta[kk] = v

        # Explicit source export provenance.
        meta["source.export.schema"] = meta0.get("export.schema", "")
        meta["source.export.when_iso"] = meta0.get("export.when_iso", "")
        meta["source.export.step"] = meta0.get("export.step", "")
        meta["source.snapshot.path"] = str(ns.in_h5)
        meta["source.snapshot.kind"] = str(snap_kind)

        meta["sprite.stats.json"] = json.dumps(stats, sort_keys=True)
        meta["sprite.centre_xyz"] = json.dumps([int(centre2[0]), int(centre2[1]), int(centre2[2])])
        meta["sprite.radius"] = int(r)

        write_sprite_asset_h5(
            out_h5,
            phi=phi_p,
            vel=vel_p,
            src=src_p,
            load=load_p,
            meta=meta,
            compression=str(ns.compression),
            compression_level=int(ns.compression_level),
        )

        print(
            f"[extractsprite] phase=ok rc=0 in={ns.in_h5} kind={snap_kind} out={out_h5} "
            f"centre={centre2[0]},{centre2[1]},{centre2[2]} radius={int(r)} L={int(phi_p.shape[0])} "
            f"support_frac={float(stats.get('support_frac', float('nan'))):.6g} "
            f"phi_abs_max_full={float(stats.get('phi_abs_max_full', float('nan'))):.6g} "
            f"phi_abs_max_patch={float(stats.get('phi_abs_max_patch', float('nan'))):.6g}",
            flush=True,
        )

        return 0

    except Exception:
        # Always emit the canonical line (core.py parses this). Include rc on error.
        print(f"[extractsprite] phase=err rc=1 in={ns.in_h5} kind={snap_kind} out={out_hint}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())