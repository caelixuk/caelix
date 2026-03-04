# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""plumbing.py — shared runtime helpers for core.py

This module exists to keep `core.py` thin and editable.
It contains only runtime/CLI plumbing (progress + run-log helpers).
No experiment logic lives here.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
import time
from typing import Any, Callable, Mapping, TextIO, cast

from params import PipelineParams
from utils import ensure_parent_dir


class ProgressBar:
    def __init__(self, total_units: int, label: str = "progress") -> None:
        self.total = int(max(1, total_units))
        self.label = str(label)
        self.done = 0
        self._t0 = 0.0
        self._t_last = 0.0
        # Prefer a real TTY so progress can render even if stdout/stderr are pipes.
        # If we can write to the controlling terminal device, do that.
        self._owns_stream = False
        tty_stream: TextIO | None = None
        try:
            if os.name == "posix":
                # Works when a controlling terminal exists (even if sys.stderr is redirected).
                tty_stream = open("/dev/tty", "w", encoding="utf-8", newline="")
            elif os.name == "nt":
                # Windows console output device.
                tty_stream = open("CONOUT$", "w", encoding="utf-8", newline="")
        except Exception:
            tty_stream = None

        if tty_stream is not None:
            try:
                if bool(getattr(tty_stream, "isatty", lambda: False)()):
                    self._stream = tty_stream
                    self._owns_stream = True
                else:
                    try:
                        tty_stream.close()
                    except Exception:
                        pass
                    tty_stream = None
            except Exception:
                try:
                    tty_stream.close()
                except Exception:
                    pass
                tty_stream = None

        if tty_stream is None:
            candidates: list[TextIO] = []
            s0 = sys.__stderr__
            if s0 is not None:
                candidates.append(s0)
            s1 = sys.__stdout__
            if s1 is not None:
                candidates.append(s1)
            if sys.stderr is not None:
                candidates.append(sys.stderr)
            if sys.stdout is not None:
                candidates.append(sys.stdout)

            chosen: TextIO | None = None
            for s in candidates:
                try:
                    if bool(getattr(s, "isatty", lambda: False)()):
                        chosen = s
                        break
                except Exception:
                    pass
            if chosen is None:
                # Fall back to something non-None; progress will be disabled if not a TTY.
                chosen = candidates[0] if candidates else None
            if chosen is None:
                import io
                chosen = io.StringIO()

            self._stream = chosen

        self._isatty = bool(getattr(self._stream, "isatty", lambda: False)())

    def start(self) -> None:
        t = time.monotonic()
        self._t0 = t
        self._t_last = 0.0
        self._render(force=True)

    def advance(self, units: int) -> None:
        if units <= 0:
            # Heartbeat: allow callers to request a redraw (timer/ETA) without advancing work.
            self._render(force=False)
            return
        self.done = int(min(self.total, self.done + int(units)))
        self._render(force=False)

    def finish(self) -> None:
        self.done = self.total
        self._render(force=True)
        if self._isatty:
            self._stream.write("\n")
            self._stream.flush()
        if getattr(self, "_owns_stream", False):
            try:
                self._stream.close()
            except Exception:
                pass
            self._owns_stream = False

    def _render(self, force: bool) -> None:
        if not self._isatty:
            return
        t = time.monotonic()
        if (not force) and (self._t_last != 0.0) and ((t - self._t_last) < 1.0):
            return
        self._t_last = t
        frac = float(self.done) / float(self.total)
        pct = int(round(100.0 * frac))
        width = 28
        filled = int(round(frac * width))
        bar = ("#" * filled) + ("-" * (width - filled))
        elapsed = t - self._t0

        # Format elapsed time as m:ss (or h:mm:ss for long runs).
        if elapsed >= 3600.0:
            eh = int(elapsed // 3600)
            em = int((elapsed - 3600.0 * eh) // 60)
            es = int(max(0.0, elapsed - 3600.0 * eh - 60.0 * em))
            elapsed_txt = f"{eh:d}:{em:02d}:{es:02d}"
        else:
            em = int(elapsed // 60)
            es = int(max(0.0, elapsed - 60.0 * em))
            elapsed_txt = f"{em:d}:{es:02d}"

        eta_s: float | None = None
        if self.done > 0 and elapsed > 0.0 and self.done < self.total:
            rate = float(self.done) / float(elapsed)
            if rate > 0.0:
                eta_s = float(self.total - self.done) / rate

        if eta_s is None:
            eta_txt = "--"
        else:
            # Keep it compact; minutes/hours only when needed.
            if eta_s >= 3600.0:
                h = int(eta_s // 3600)
                m = int((eta_s - 3600.0 * h) // 60)
                s = int(max(0.0, eta_s - 3600.0 * h - 60.0 * m))
                eta_txt = f"{h:d}:{m:02d}:{s:02d}"
            elif eta_s >= 60.0:
                m = int(eta_s // 60)
                s = int(max(0.0, eta_s - 60.0 * m))
                eta_txt = f"{m:d}:{s:02d}"
            else:
                eta_txt = f"{eta_s:4.0f}s"

        # \r to column 0 + ANSI clear-line to avoid leftovers from previous frames.
        self._stream.write(
            f"\r\x1b[2K[{self.label}] {pct:3d}% |{bar}| {self.done}/{self.total}  t={elapsed_txt}  eta={eta_txt}   "
        )
        self._stream.flush()


def _get_log_path(args: object) -> str:
    """Best-effort: locate a configured run-log path on the args object."""
    cand_names = ("log", "log_path", "run_log", "run_log_path", "log_file", "out_log")
    for nm in cand_names:
        try:
            v = getattr(args, nm)
        except Exception:
            v = None

        if v is None:
            continue

        # argparse often stores paths as pathlib.Path; accept os.PathLike too.
        if isinstance(v, (str, os.PathLike)):
            p = str(v).strip()
            if p != "":
                return p

    return ""


def _log_line_only(args: object, line: str) -> bool:
    """Best-effort: write a line to the run log (if one is configured), and do not print it."""
    log_path = _get_log_path(args)
    if log_path == "":
        return False

    try:
        ensure_parent_dir(log_path)
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            f.write(str(line).rstrip("\n") + "\n")
        return True
    except Exception:
        return False


def _maybe_make_progress_bar(params: PipelineParams, fn: object) -> tuple[ProgressBar | None, dict[str, Any]]:
    if not callable(fn):
        return None, {}
    try:
        sig = inspect.signature(cast(Callable[..., Any], fn))
    except (TypeError, ValueError):
        return None, {}

    if ("progress_cb" not in sig.parameters) and ("progress" not in sig.parameters):
        return None, {}

    # Work-unit model (01-series baseline): we only visualise the long-running solver stage.
    # The microstate initialisation/anneal phase can be large and often reports progress in a
    # single big jump (e.g. ~200k units), which makes the bar look broken. So we *ignore* the
    # first `lattice.steps` units reported by the pipeline and start counting once that phase
    # is complete. This keeps the bar monotone and avoids a misleading 0%→67% jump.
    skip_units = int(max(0, int(params.lattice.steps)))
    total_units = int(max(1, int(params.traffic.iters) + 250))
    pb = ProgressBar(total_units=total_units, label="work")

    key = "progress_cb" if "progress_cb" in sig.parameters else "progress"

    def _progress_cb(units: int) -> None:
        nonlocal skip_units
        u = int(units)
        if u <= 0:
            return
        if skip_units > 0:
            if u <= skip_units:
                skip_units -= u
                return
            u -= skip_units
            skip_units = 0
        pb.advance(u)

    return pb, {key: _progress_cb}


def _ensure_bundle_dirs(run_dir: str) -> tuple[str, str]:
    """Create and return (csv_dir, log_dir) under a run bundle directory."""
    rd = str(run_dir)
    csv_dir = os.path.join(rd, "_csv")
    log_dir = os.path.join(rd, "_logs")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return csv_dir, log_dir


# Consistent artefact naming helpers

def _get_str_attr(args: object, name: str) -> str:
    try:
        v = getattr(args, name)
    except Exception:
        return ""
    if v is None:
        return ""
    if isinstance(v, (str, os.PathLike)):
        return str(v)
    return ""


def _set_str_attr(args: object, name: str, value: str) -> bool:
    try:
        setattr(args, name, str(value))
        return True
    except Exception:
        return False
def _default_log_path(run_dir: str, n: int, run_id: str, default: str = "00") -> str:
    """Default log path under a run bundle.

    Naming contract: <exp_name>_n<N>_<run_id>.log, where exp_name is the experiment folder name.
    """
    _, log_dir = _ensure_bundle_dirs(run_dir)
    exp_name = _derive_exp_name(run_dir, default=str(default))
    return os.path.join(log_dir, f"{exp_name}_n{int(n)}_{str(run_id)}.log")


def _default_csv_path(run_dir: str, n: int, run_id: str, default: str = "00", tag: str = "") -> str:
    """Default CSV path under a run bundle.

    Naming contract: <exp_name>[_<tag>]_n<N>_<run_id>.csv, where exp_name is the experiment folder name.
    `tag` is optional and should be short (e.g. "series", "lensing").
    """
    csv_dir, _ = _ensure_bundle_dirs(run_dir)
    exp_name = _derive_exp_name(run_dir, default=str(default))
    t = str(tag).strip()
    if t != "":
        return os.path.join(csv_dir, f"{exp_name}_{t}_n{int(n)}_{str(run_id)}.csv")
    return os.path.join(csv_dir, f"{exp_name}_n{int(n)}_{str(run_id)}.csv")


def _mint_default_log(args: object, run_dir: str, n: int, run_id: str, default: str = "00") -> str:
    """If args has an empty log path attribute, set it to the default bundle log path and return it."""
    # Common arg attribute names used across modules.
    for nm in ("log", "log_path", "run_log", "run_log_path", "log_file", "out_log"):
        cur = _get_str_attr(args, nm).strip()
        if cur == "":
            lp = _default_log_path(run_dir, n=n, run_id=run_id, default=str(default))
            if _set_str_attr(args, nm, lp):
                return lp
        else:
            return cur
    # No known attribute present; still return the default path for callers that want it.
    return _default_log_path(run_dir, n=n, run_id=run_id, default=str(default))


def _mint_default_csv(args: object, attr_name: str, run_dir: str, n: int, run_id: str, default: str = "00", tag: str = "") -> str:
    """If args.<attr_name> is an empty string, set it to the default bundle CSV path and return it."""
    cur = _get_str_attr(args, attr_name).strip()
    if cur != "":
        return cur
    cp = _default_csv_path(run_dir, n=n, run_id=run_id, default=str(default), tag=str(tag))
    _set_str_attr(args, attr_name, cp)
    return cp



def _derive_exp_code(run_dir: str, default: str) -> str:
    """Derive the experiment code from the parent folder name (e.g. 06C_... -> 06C)."""
    parent = os.path.basename(os.path.dirname(os.path.normpath(str(run_dir))))
    if parent == "":
        return str(default)
    if "_" not in parent:
        return str(default)
    return parent.split("_", 1)[0]


def _derive_exp_name(run_dir: str, default: str = "00") -> str:
    """Derive the full experiment name from a run bundle directory.

    experiments.py typically creates:
      <out_base>/<exp_name>/<run_id>
    where run_id matches YYYYMMDD_HHMMSS.

    If we can see that structure, return <exp_name> (the parent folder).
    Otherwise, fall back to the short experiment code (e.g. 06C) or `default`.
    """
    rd = os.path.normpath(str(run_dir))
    leaf = os.path.basename(rd)
    if re.fullmatch(r"\d{8}_\d{6}", leaf):
        parent = os.path.basename(os.path.dirname(rd))
        if parent != "":
            return str(parent)
    # Ad-hoc runs (no <exp_name>/<run_id> bundle). Keep legacy behaviour.
    return str(_derive_exp_code(rd, default=str(default)))


def _filtered_kwargs(fn: object, candidates: Mapping[str, Any]) -> dict[str, Any]:
    """Return a dict containing only kwargs accepted by `fn` (signature-aware, fail-soft)."""
    if not callable(fn):
        return {}
    try:
        sig = inspect.signature(cast(Callable[..., Any], fn))
    except (TypeError, ValueError):
        return {}
    out: dict[str, Any] = {}
    for k, v in candidates.items():
        if k in sig.parameters:
            out[k] = v
    return out