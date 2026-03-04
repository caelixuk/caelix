# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""exporters.py — Heavy I/O (HDF5 snapshots)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Mapping, Optional, cast

import os
import dataclasses

import h5py
import numpy as np

from utils import ensure_parent_dir, wallclock_iso


def dump_pipeline_state_h5(
    path: str,
    *,
    phi: np.ndarray,
    vel: np.ndarray,
    src: np.ndarray,
    load: np.ndarray,
    params: Any = None,
    provenance: Optional[Dict[str, Any]] = None,
    step: Optional[int] = None,
    dt: Optional[float] = None,
    schema: str = "dlft.snapshot.v1",
    extra_arrays: Optional[Mapping[str, np.ndarray]] = None,
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    """Write a full simulation snapshot to HDF5.

    Datasets (root):
      /phi   : float32 [..]  (required)
      /vel   : float32 [..]  (required; momentum/velocity field)
      /src   : float32 [..]  (required; forcing/source field)
      /load  : float32 [..]  (required; auxiliary / density / load field)

      plus any `extra_arrays` items written as datasets at the file root.

    Metadata:
      Stored as attributes on the file object (`f.attrs`):
        - export.schema          (string)
        - export.when_iso        (ISO timestamp)
        - export.created_unix_s  (float)
        - export.step            (int, if provided)
        - export.dt              (float, if provided)
        - params.*               (flattened dataclass / object)
        - prov.*                 (flattened mapping)
        - params.json            (lossless JSON, if serialisable)
        - prov.json              (lossless JSON, if serialisable)

    Contract:
      - Fail-loud on invalid path / write failures.
      - Overwrites existing file.
      - Does not mutate inputs.
    """
    if path is None or str(path).strip() == "":
        raise ValueError("dump_pipeline_state_h5: path must be a non-empty string")

    ensure_parent_dir(path)

    tmp_path = f"{path}.tmp"

    phi_f32 = np.asarray(phi, dtype=np.float32)
    vel_f32 = np.asarray(vel, dtype=np.float32)
    src_f32 = np.asarray(src, dtype=np.float32)
    load_f32 = np.asarray(load, dtype=np.float32)

    if vel_f32.shape != phi_f32.shape:
        raise ValueError(f"dump_pipeline_state_h5: vel shape {vel_f32.shape} must match phi shape {phi_f32.shape}")
    if src_f32.shape != phi_f32.shape:
        raise ValueError(f"dump_pipeline_state_h5: src shape {src_f32.shape} must match phi shape {phi_f32.shape}")
    if load_f32.shape != phi_f32.shape:
        raise ValueError(f"dump_pipeline_state_h5: load shape {load_f32.shape} must match phi shape {phi_f32.shape}")

    try:
        with h5py.File(tmp_path, "w") as f:
            f.attrs["export.schema"] = str(schema)
            f.attrs["export.when_iso"] = str(wallclock_iso())
            f.attrs["export.created_unix_s"] = float(time.time())
            if step is not None:
                f.attrs["export.step"] = int(step)
            if dt is not None:
                f.attrs["export.dt"] = float(dt)

            _write_array_dataset(
                f,
                name="phi",
                arr=phi_f32,
                compression=compression,
                compression_level=compression_level,
            )

            _write_array_dataset(
                f,
                name="vel",
                arr=vel_f32,
                compression=compression,
                compression_level=compression_level,
            )

            _write_array_dataset(
                f,
                name="src",
                arr=src_f32,
                compression=compression,
                compression_level=compression_level,
            )

            _write_array_dataset(
                f,
                name="load",
                arr=load_f32,
                compression=compression,
                compression_level=compression_level,
            )

            if extra_arrays is not None:
                for k, arr in extra_arrays.items():
                    if k is None or str(k).strip() == "":
                        raise ValueError("dump_pipeline_state_h5: extra_arrays contains an empty key")
                    name = str(k).strip()
                    if name.startswith("/"):
                        name = name[1:]
                    if "/" in name:
                        raise ValueError(f"dump_pipeline_state_h5: extra_arrays key must be a root name without '/', got {k!r}")
                    if name in ("phi", "vel", "src", "load"):
                        raise ValueError(f"dump_pipeline_state_h5: extra_arrays key collides with reserved dataset '{name}'")
                    _write_array_dataset(
                        f,
                        name=name,
                        arr=np.asarray(arr, dtype=np.float32),
                        compression=compression,
                        compression_level=compression_level,
                    )

            if params is not None:
                # Fail-loud if a dataclass *type* (class) is accidentally passed.
                if dataclasses.is_dataclass(params) and isinstance(params, type):
                    raise TypeError("dump_pipeline_state_h5: params must be a dataclass instance, not a dataclass type")

                _write_attrs_from_obj(f, params, prefix="params.")
                try:
                    if dataclasses.is_dataclass(params):
                        params_dc = cast(Any, params)
                        f.attrs["params.json"] = json.dumps(dataclasses.asdict(params_dc), sort_keys=True)
                    elif isinstance(params, dict):
                        f.attrs["params.json"] = json.dumps(params, sort_keys=True)
                except Exception:
                    pass

            if provenance is not None:
                if not isinstance(provenance, dict):
                    raise TypeError("dump_pipeline_state_h5: provenance must be a dict[str, Any]")
                _write_attrs_from_mapping(f, provenance, prefix="prov.")
                try:
                    f.attrs["prov.json"] = json.dumps(provenance, sort_keys=True)
                except Exception:
                    pass

        # Note: do not print here; progress bars write to stdout and will interleave.
        # Store size metadata for downstream tools.
        n_bytes = int(os.path.getsize(tmp_path))
        with h5py.File(tmp_path, "r+") as f2:
            f2.attrs["export.bytes"] = int(n_bytes)
            f2.attrs["export.human_bytes"] = str(_fmt_bytes(int(n_bytes)))
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

    os.replace(tmp_path, path)


def _write_array_dataset(
    f: h5py.File,
    *,
    name: str,
    arr: np.ndarray,
    compression: str,
    compression_level: int,
) -> None:
    if arr.ndim < 1:
        raise ValueError(f"HDF5 export expects an array for '{name}', got shape={arr.shape}")

    # Chunked + compressed to support partial reads for visualisers.
    # gzip is widely compatible; level 4 is a reasonable throughput/size tradeoff.
    comp = str(compression or "").strip().lower()
    if comp in ("", "none", "off", "false", "0"):
        comp2: str | None = None
    else:
        comp2 = comp

    # h5py expects `compression=None` for no compression.
    # `lzf` does not accept `compression_opts`.
    kwargs: dict[str, Any] = {
        "chunks": True,
        "shuffle": True,
        "fletcher32": False,
    }
    if comp2 is not None:
        kwargs["compression"] = comp2
        if comp2 == "gzip":
            kwargs["compression_opts"] = int(compression_level)

    f.create_dataset(
        name,
        data=arr,
        **kwargs,
    )


def _write_attrs_from_mapping(h5_obj: Any, mapping: Dict[str, Any], prefix: str) -> None:
    for k, v in mapping.items():
        if k is None:
            continue
        key = f"{prefix}{str(k)}"
        _set_attr(h5_obj, key, v)


def _write_attrs_from_obj(h5_obj: Any, obj: Any, prefix: str) -> None:
    """Recursively flatten a dataclass/object into HDF5 attributes.

    Rules:
      - keys are dotted paths (e.g., params.traffic.iters)
      - private fields (leading underscore) are ignored
      - dicts are recursed into
      - lists/tuples are stringified (stable, human readable)
      - everything else: int/float/bool stored natively; others stringified
    """
    if obj is None:
        return

    if dataclasses.is_dataclass(obj):
        if isinstance(obj, type):
            raise TypeError("HDF5 export: dataclass type passed where an instance is required")
        for field in dataclasses.fields(obj):
            name = field.name
            if name.startswith("_"):
                continue
            _write_attrs_from_obj(h5_obj, getattr(obj, name), prefix=f"{prefix}{name}.")
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k is None:
                continue
            kk = str(k)
            if kk.startswith("_"):
                continue
            _write_attrs_from_obj(h5_obj, v, prefix=f"{prefix}{kk}.")
        return

    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            _write_attrs_from_obj(h5_obj, v, prefix=f"{prefix}{k}.")
        return

    # Base case
    _set_attr(h5_obj, prefix[:-1], obj)


def _set_attr(h5_obj: Any, key: str, value: Any) -> None:
    if key is None or str(key).strip() == "":
        return

    # HDF5 attributes support scalars and small arrays; for safety we keep it scalar.
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        h5_obj.attrs[str(key)] = value
        return

    if value is None:
        h5_obj.attrs[str(key)] = "None"
        return

    if isinstance(value, (str, bytes)):
        # Ensure bytes become a readable string.
        h5_obj.attrs[str(key)] = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return

    if isinstance(value, (list, tuple)):
        h5_obj.attrs[str(key)] = "[" + ",".join(str(x) for x in value) + "]"
        return

    # Fallback: stringify complex objects (paths, enums, numpy dtypes, etc.)
    h5_obj.attrs[str(key)] = str(value)


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    kb = n / 1024.0
    if kb < 1024.0:
        return f"{kb:.1f} KiB"
    mb = kb / 1024.0
    if mb < 1024.0:
        return f"{mb:.1f} MiB"
    gb = mb / 1024.0
    return f"{gb:.2f} GiB"