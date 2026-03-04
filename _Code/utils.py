# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""utils.py — small shared helpers (keep it boring)

Role
----
This module holds only light-weight utilities shared across the CAELIX flat
module set. It is safe to import from anywhere.

What belongs here
-----------------
- Fail-fast assertions and small coercions:
    - `_assert_finite(arr, name)`
    - `_as_float(x, name)`
- Deterministic RNG construction:
    - `_make_rng(seed)`
- Log-friendly numeric formatting:
    - `fmt_g(x, sig=6)` general format
    - `fmt_f(x, ndp=6)` fixed format
- Plot bounds helpers (pure numpy, no matplotlib):
    - `percentile_bounds(arr, lo, hi, fallback=...)`
- Path normalisation (no filesystem touch except mkdir helper):
    - `norm_path(path)`
    - `here_dir(file_path)`
    - `resolve_out_dir(base_file, out_dir)`
    - `resolve_out_path(base_file, out_path)`
    - `resolve_out_file(base_file, out_dir, filename)`
    - `ensure_parent_dir(path)`
- Lightweight time stamps:
    - `wallclock_iso()`

Design constraints
------------------
- No compute kernels.
- No project imports.
- No matplotlib (graphics lives in `visualiser.py`).

Flat-module layout
------------------
All modules live in the same folder and import locally, e.g.
  `from utils import _assert_finite, ensure_parent_dir`
"""

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, TextIO, Tuple, Union, cast

import numpy as np


def _assert_finite(arr: np.ndarray, name: str) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")


def _as_float(x: object, name: str) -> float:
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    raise TypeError(f"{name} is not a float-like value: {type(x)}")


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def fmt_g(x: object, sig: int = 6) -> str:
    """Stable, log-friendly float formatting.

    - Finite floats use general format with `sig` significant digits.
    - NaN/Inf are emitted as 'nan'/'inf'/'-inf'.
    """
    if isinstance(x, (int, float, np.floating)):
        v = float(x)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return format(v, f".{int(sig)}g")
    return str(x)


def fmt_f(x: object, ndp: int = 6) -> str:
    """Fixed-point float formatting with finite checks."""
    if isinstance(x, (int, float, np.floating)):
        v = float(x)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return format(v, f".{int(ndp)}f")
    return str(x)


def percentile_bounds(
    arr: np.ndarray,
    lo: float = 2.0,
    hi: float = 99.5,
    *,
    fallback: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Robust (vmin, vmax) bounds for plotting.

    Returns percentile bounds when finite and ordered; otherwise falls back to min/max;
    if still invalid, uses `fallback` when provided, else (-12.0, 0.0).
    """
    if arr.size == 0:
        return fallback if fallback is not None else (-12.0, 0.0)

    vmin = float(np.percentile(arr, float(lo)))
    vmax = float(np.percentile(arr, float(hi)))

    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))

    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmax <= vmin:
        return fallback if fallback is not None else (-12.0, 0.0)

    return vmin, vmax


def norm_path(path: str) -> str:
    """Normalise a path string without touching the filesystem."""
    p = str(path).strip()
    if p == "":
        return ""
    return os.path.normpath(os.path.expanduser(p))


def here_dir(file_path: str) -> str:
    """Directory of a file path, resolved to an absolute path."""
    return os.path.dirname(os.path.abspath(str(file_path)))


def ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory of `path` exists (single-path, no fallbacks)."""
    p = norm_path(path)
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)


def resolve_out_dir(base_file: str, out_dir: str) -> str:
    """Resolve an output directory.

    - If `out_dir` is empty, returns empty.
    - If `out_dir` is absolute, normalises it.
    - Otherwise resolves relative to the directory containing `base_file`.
    """
    o = norm_path(out_dir)
    if o == "":
        return ""
    _reject_project_prefixed_relative(base_file, o)
    if os.path.isabs(o):
        return o
    return os.path.normpath(os.path.join(here_dir(base_file), o))


def _reject_project_prefixed_relative(base_file: str, p: str) -> None:
    """Fail-fast if a relative path redundantly includes the project folder.

    This prevents accidental duplication:

    Rule: if `p` is relative and starts with either:
      - "Projects/<basename>/" or
      - "<basename>/"
    where basename is the directory name containing `base_file`.
    """
    if p == "" or os.path.isabs(p):
        return
    base = os.path.basename(here_dir(base_file))
    p0 = str(p).replace("\\", "/").lstrip("./")
    a = os.path.normpath(os.path.join("Projects", base))
    b = os.path.normpath(base)
    p_norm = os.path.normpath(p0)
    if p_norm == a or p_norm.startswith(a + os.sep) or p_norm == b or p_norm.startswith(b + os.sep):
        raise ValueError(
            "output path must be relative to the project dir; do not include 'Projects/%s/' or '%s/' (use '_Output/...' or an absolute path)" % (base, base)
        )


def resolve_out_path(base_file: str, out_path: str) -> str:
    """Resolve an output file path.

    - If `out_path` is empty, returns empty.
    - If `out_path` is absolute, normalises it.
    - Otherwise resolves relative to the directory containing `base_file`.

    Also rejects redundant project-prefixed relative paths (fail-fast).
    """
    o = norm_path(out_path)
    if o == "":
        return ""
    _reject_project_prefixed_relative(base_file, o)
    if os.path.isabs(o):
        return o
    return os.path.normpath(os.path.join(here_dir(base_file), o))


def resolve_out_file(base_file: str, out_dir: str, filename: str) -> str:
    """Resolve an output file under an output directory."""
    d = resolve_out_dir(base_file, out_dir)
    f = norm_path(filename)
    if f == "":
        raise ValueError("filename must be non-empty")
    return os.path.normpath(os.path.join(d, f))


def now_s() -> float:
    """Monotonic wall-clock time in seconds."""
    return float(time.perf_counter())


def wallclock_iso() -> str:
    """Local wallclock timestamp (human-readable, log-friendly)."""
    t = time.localtime(time.time())
    return time.strftime("%Y-%m-%d %H:%M:%S", t)


# --- Timestamped log path helpers ---

def wallclock_compact() -> str:
    """Local wallclock timestamp suitable for filenames (YYYYMMDD_HHMMSS)."""
    t = time.localtime(time.time())
    return time.strftime("%Y%m%d_%H%M%S", t)


def _slugify_stem(stem: str) -> str:
    """Make a conservative filename stem (ASCII-ish, no spaces)."""
    s = str(stem).strip()
    if s == "":
        return "run"
    s = s.replace(" ", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def make_run_log_path(out_dir: str, stem: str, *, ext: str = ".log") -> str:
    """Create a timestamped log path under `<out_dir>/_logs/`.

    Example:
        make_run_log_path("_Output/01_baseline", "01A_pipeline")
        -> ".../_Output/01_baseline/_logs/01A_pipeline_20260131_175329.log"

    The returned path is resolved relative to the current working directory if `out_dir` is relative.
    """
    d0 = norm_path(out_dir)
    if d0 == "":
        raise ValueError("out_dir must be non-empty")
    stem0 = _slugify_stem(stem)
    ext0 = str(ext)
    if not ext0.startswith("."):
        ext0 = "." + ext0
    ts = wallclock_compact()
    p = os.path.join(d0, "_logs", f"{stem0}_{ts}{ext0}")
    return os.path.normpath(p)



@dataclass
class Timer:
    """Simple monotonic timer (no printing).

    Usage:
        t = Timer().start()
        ...
        seconds = t.stop_s()

    Or as a context manager:
        with Timer() as t:
            ...
        seconds = t.elapsed_s
    """

    t0: float = field(default=0.0)
    t1: float = field(default=0.0)
    running: bool = field(default=False)

    def start(self) -> "Timer":
        self.t0 = now_s()
        self.t1 = 0.0
        self.running = True
        return self

    def stop(self) -> "Timer":
        if not self.running:
            raise ValueError("Timer.stop() called when not running")
        self.t1 = now_s()
        self.running = False
        return self

    def stop_s(self) -> float:
        """Stop the timer and return elapsed seconds."""
        if not self.running:
            raise ValueError("Timer.stop_s() called when not running")
        self.t1 = now_s()
        self.running = False
        return float(self.t1 - self.t0)

    @property
    def elapsed_s(self) -> float:
        if self.running:
            return now_s() - self.t0
        if self.t1 <= 0.0:
            return 0.0
        return self.t1 - self.t0

    def lap_s(self) -> float:
        """Return elapsed seconds since start without stopping."""
        if not self.running:
            raise ValueError("Timer.lap_s() called when not running")
        return now_s() - self.t0

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.running:
            self.stop()


# --- Heartbeat progress bar (stderr-only, no log pollution) ---

def estimate_cycles_baseline(*, steps: int, traffic_iters: int, diag_units: int = 250) -> int:
    """Estimate planned work units for the baseline pipeline run.

    This is an internal progress currency, not CPU cycles.

    Policy (MV1):
    - anneal units ~= `steps`
    - traffic units ~= `traffic_iters`
    - diagnostics are a small fixed tail (`diag_units`) so progress doesn't hit 100% early
    """
    s = int(steps)
    t = int(traffic_iters)
    d = int(diag_units)
    if s < 0 or t < 0 or d < 0:
        raise ValueError("cycle estimate inputs must be non-negative")
    return int(s + t + d)


@dataclass
class ProgressBar:
    """Single-line progress bar for long runs.

    - Writes to `sys.__stderr__` so it remains purely visual and avoids tee-to-log.
    - Emits at most once per `rate_s` seconds.
    - Uses carriage return '\r' to update a single terminal line.
    """

    total_units: int
    label: str = ""
    rate_s: float = 1.0
    stream: TextIO = field(default_factory=lambda: cast(TextIO, sys.__stderr__ if sys.__stderr__ is not None else sys.stderr))

    _t0: float = field(default=0.0, init=False, repr=False)
    _last_emit: float = field(default=0.0, init=False, repr=False)
    _last_len: int = field(default=0, init=False, repr=False)
    _done_units: float = field(default=0.0, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def start(self) -> "ProgressBar":
        self._t0 = now_s()
        self._last_emit = 0.0
        self._last_len = 0
        self._done_units = 0.0
        self._finished = False
        return self

    @property
    def elapsed_s(self) -> float:
        if self._t0 <= 0.0:
            return 0.0
        return now_s() - self._t0

    def update(self, done_units: float, *, phase: str = "") -> None:
        """Update progress state; renders at a 1 Hz (or `rate_s`) heartbeat."""
        if self._finished:
            return
        tot = float(max(1, int(self.total_units)))
        du = float(done_units)
        if du < 0.0:
            du = 0.0
        if du > tot:
            du = tot
        self._done_units = du

        t = now_s()
        if self._last_emit > 0.0 and (t - self._last_emit) < float(self.rate_s):
            return
        self._last_emit = t
        self._render(phase=str(phase))

    def finish(self, *, phase: str = "") -> None:
        """Force 100% and terminate the progress line with a newline."""
        if self._finished:
            return
        self._done_units = float(max(0, int(self.total_units)))
        self._render(phase=str(phase), force=True)
        try:
            self.stream.write("\n")
            self.stream.flush()
        except Exception:
            pass
        self._finished = True

    def _render(self, *, phase: str, force: bool = False) -> None:
        tot = float(max(1, int(self.total_units)))
        frac = float(self._done_units) / tot
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0

        width = 28
        filled = int(round(frac * width))
        if filled < 0:
            filled = 0
        if filled > width:
            filled = width
        bar = "=" * filled + "-" * (width - filled)

        el = self.elapsed_s
        rate = (float(self._done_units) / el) if el > 0.5 else 0.0
        eta_s = ((tot - float(self._done_units)) / rate) if rate > 0.0 else -1.0

        def _mmss(sec: float) -> str:
            if sec < 0.0 or not math.isfinite(sec):
                return "--:--"
            s = int(round(sec))
            m = s // 60
            s2 = s % 60
            return "%02d:%02d" % (m, s2)

        p = "%5.1f%%" % (100.0 * frac)
        lbl = (str(self.label).strip() + " ") if str(self.label).strip() != "" else ""
        ph = ("phase=%s " % str(phase)) if str(phase).strip() != "" else ""
        line = "[progress] %s|%s| %s elapsed=%s eta=%s" % (
            p,
            bar,
            (lbl + ph).strip(),
            _mmss(el),
            _mmss(eta_s),
        )

        if force:
            line = line.replace("--:--", "00:00")

        pad_n = max(0, int(self._last_len) - len(line))
        try:
            self.stream.write("\r" + line + (" " * pad_n))
            self.stream.flush()
        except Exception:
            return
        self._last_len = len(line)


def make_weighted_progress_callback(
    bar: ProgressBar,
    *,
    phase_weights: Dict[str, int],
) -> Callable[[str, int, int], None]:
    """Build a simple weighted progress callback.

    The callback signature is: cb(phase, i, n)

    - `phase` must exist in `phase_weights`.
    - `i` is current count (0..n)
    - `n` is total for the phase (>0)

    The bar is updated as a single shared progress line.
    """

    weights = {str(k): int(v) for k, v in dict(phase_weights).items()}
    ordered = list(weights.keys())
    prefix_weight: Dict[str, int] = {}
    acc = 0
    for k in ordered:
        prefix_weight[k] = acc
        acc += int(weights[k])

    total = int(acc) if int(acc) > 0 else int(max(1, int(bar.total_units)))
    bar.total_units = int(total)

    def _cb(phase: str, i: int, n: int) -> None:
        ph = str(phase)
        if ph not in weights:
            return
        nn = int(n)
        if nn <= 0:
            return
        ii = int(i)
        if ii < 0:
            ii = 0
        if ii > nn:
            ii = nn
        base = float(prefix_weight[ph])
        w = float(weights[ph])
        done = base + (w * (float(ii) / float(nn)))
        bar.update(done, phase=ph)

    return _cb


class _TeeTextIO:
    """A tiny text stream wrapper that writes to two underlying streams."""

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def write(self, s: str) -> int:
        n = self._a.write(s)
        self._b.write(s)
        return int(n)

    def flush(self) -> None:
        self._a.flush()
        self._b.flush()

    def isatty(self) -> bool:
        try:
            return bool(self._a.isatty())
        except Exception:
            return False


@dataclass
class TeeStdStreams:
    """Context manager that tees stdout/stderr to a file while preserving terminal output."""

    log_path: str
    mode: str = "w"
    tee_stdout: bool = True
    tee_stderr: bool = True
    header: Optional[str] = None
    _fp: Optional[TextIO] = field(default=None, init=False, repr=False)
    _old_out: Optional[TextIO] = field(default=None, init=False, repr=False)
    _old_err: Optional[TextIO] = field(default=None, init=False, repr=False)

    def __enter__(self) -> "TeeStdStreams":
        p = norm_path(self.log_path)
        if p == "":
            raise ValueError("log_path must be non-empty")
        ensure_parent_dir(p)
        self._fp = cast(TextIO, open(p, str(self.mode), encoding="utf-8"))
        if self.header is not None and str(self.header) != "":
            h = str(self.header)
            if not h.endswith("\n"):
                h += "\n"
            self._fp.write(h)
            self._fp.flush()
        self._old_out = sys.stdout
        self._old_err = sys.stderr
        if bool(self.tee_stdout):
            sys.stdout = _TeeTextIO(self._old_out, self._fp)
        if bool(self.tee_stderr):
            sys.stderr = _TeeTextIO(self._old_err, self._fp)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old_out is not None:
            sys.stdout = self._old_out
        if self._old_err is not None:
            sys.stderr = self._old_err
        if self._fp is not None:
            try:
                self._fp.flush()
            except Exception:
                pass
            try:
                self._fp.close()
            except Exception:
                pass


def tee_to_file(
    log_path: str,
    *,
    mode: str = "w",
    tee_stdout: bool = True,
    tee_stderr: bool = True,
    header: Optional[str] = None,
) -> TeeStdStreams:
    """Convenience factory for `TeeStdStreams`."""
    return TeeStdStreams(
        str(log_path),
        mode=str(mode),
        tee_stdout=bool(tee_stdout),
        tee_stderr=bool(tee_stderr),
        header=header,
    )


# --- Apple Metal capability probe ---

@dataclass
class AppleMetalInfo:
    """Host capability snapshot for Apple/Metal checks.

    This is a probe helper: it should not raise for normal "not available" cases.
    """

    is_macos: bool
    cpu_brand: str
    is_apple_silicon: bool
    perf_cores: int
    eff_cores: int
    metal_supported: bool
    metal_summary: str


def probe_apple_metal() -> AppleMetalInfo:
    """Probe macOS CPU/GPU Metal capability without side effects.

    Returns a snapshot describing whether the host is macOS, whether the CPU looks like
    Apple Silicon, and whether the primary GPU reports Metal support.

    This function is intended to be used from an if/else and should not raise for
    typical "not supported" environments.
    """

    def _sysctl(key: str) -> str:
        out = subprocess.check_output(["sysctl", "-n", key], text=True)
        return out.strip()

    is_macos = (sys.platform == "darwin")
    if not is_macos:
        return AppleMetalInfo(
            is_macos=False,
            cpu_brand="",
            is_apple_silicon=False,
            perf_cores=-1,
            eff_cores=-1,
            metal_supported=False,
            metal_summary="non-macOS",
        )

    cpu_brand = ""
    is_apple = False
    perf = -1
    eff = -1
    try:
        cpu_brand = _sysctl("machdep.cpu.brand_string")
        is_apple = ("Apple" in cpu_brand)
    except Exception:
        cpu_brand = ""
        is_apple = False

    try:
        perf = int(_sysctl("hw.perflevel0.physicalcpu"))
        eff = int(_sysctl("hw.perflevel1.physicalcpu"))
    except Exception:
        perf = -1
        eff = -1

    metal_supported = False
    metal_summary = ""
    try:
        out = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
        txt = str(out)
        if "Metal:" in txt:
            for line in txt.splitlines():
                if "Metal:" in line:
                    metal_summary = line.strip()
                    if "Supported" in line:
                        metal_supported = True
                    break
        if metal_summary == "":
            metal_summary = "Metal: unknown"
    except Exception:
        metal_supported = False
        metal_summary = "Metal: probe failed"

    return AppleMetalInfo(
        is_macos=True,
        cpu_brand=str(cpu_brand),
        is_apple_silicon=bool(is_apple),
        perf_cores=int(perf),
        eff_cores=int(eff),
        metal_supported=bool(metal_supported),
        metal_summary=str(metal_summary),
    )


# --- Portable conservative thread-tuning helper ---

def apply_conservative_thread_defaults() -> None:
    """Apply conservative thread defaults on any host.

    Purpose: avoid oversubscription when numpy/BLAS/numexpr/numba are used together.

    Policy:
    - Cap Numba threads at 8 (cache-friendly sweet spot).
    - On Apple Silicon, prefer performance cores (ignore efficiency cores).
    - Pin BLAS/OpenMP-style libraries to 1 thread.

    This function must be called before importing numba for the env vars to take effect.
    """

    cpu_n = int(os.cpu_count() or 0)

    perf_cores = -1
    info = probe_apple_metal()
    if info.is_macos and info.is_apple_silicon and int(info.perf_cores) > 0:
        perf_cores = int(info.perf_cores)

    if perf_cores > 0:
        numba_threads = min(8, perf_cores)
    else:
        numba_threads = min(8, cpu_n if cpu_n > 0 else 8)

    defaults = {
        "NUMBA_NUM_THREADS": str(int(numba_threads)),
        "VECLIB_MAXIMUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    for k, v in defaults.items():
        os.environ[k] = v

    print(
        "[threads] conservative cpu=%d perf=%d numba=%s vecLib=%s" % (
            int(cpu_n),
            int(perf_cores),
            defaults["NUMBA_NUM_THREADS"],
            defaults["VECLIB_MAXIMUM_THREADS"],
        )
    )


# --- Fail-fast dependency helpers ---

def require_dependency(module_name: str, *, install_hint: str = "") -> object:
    """Fail-fast import for required dependencies.

    Returns the imported module object.

    `install_hint` is appended verbatim (kept short) to help the user fix the environment.
    """
    try:
        return importlib.import_module(str(module_name))
    except Exception as e:
        hint = (" " + str(install_hint).strip()) if str(install_hint).strip() != "" else ""
        raise RuntimeError(f"missing dependency: {module_name}.{hint} ({e})")


def require_dependencies(*specs: Union[str, Tuple[str, str]]) -> None:
    """Fail-fast check for a set of required imports.

    Each spec is either:
      - a module name string, or
      - a (module_name, install_hint) tuple.
    """
    for spec in specs:
        if isinstance(spec, tuple):
            if len(spec) != 2:
                raise ValueError("require_dependencies tuple spec must be (module_name, install_hint)")
            name, hint = spec
            require_dependency(str(name), install_hint=str(hint))
        else:
            require_dependency(str(spec))


def write_csv_provenance_header(
    *,
    producer: str,
    command: str,
    cwd: str,
    python_exe: str,
    when_iso: str,
    experiment: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Build a lightweight provenance header for CSV artefacts.

    Returns a string containing `# key=value` lines so the metadata remains
    CSV-comment-friendly. The caller should write it at the top of the file.

    Keep this minimal: the file already contains the numeric evidence.
    """
    lines: list[str] = []
    lines.append(f"# producer={producer}")
    lines.append(f"# when={when_iso}")
    if experiment is not None and str(experiment).strip() != "":
        lines.append(f"# experiment={experiment}")
    lines.append(f"# cwd={cwd}")
    lines.append(f"# python={python_exe}")
    lines.append(f"# command={command}")
    if extra is not None:
        for k, v in extra.items():
            ks = str(k).strip().replace(" ", "_")
            if ks == "":
                continue
            lines.append(f"# {ks}={v}")
    return "\n".join(lines) + "\n"