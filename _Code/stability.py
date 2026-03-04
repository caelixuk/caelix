# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""stability.py — face-lock stability benchmarks and subset sweeps

Role
----
Contains small, controlled experiments used to test a specific claim about the
3D ternary microstate: that maintaining a stable "particle" (center excitation
with clean faces) requires enforcing all 6 face-neighbour constraints.

Two experiments
---------------
1) `stability_benchmark()`
   Compares a few hand-picked neighbour-check protocols (planar4, faces6,
   corners8, etc.) under isotropic shell noise.

2) `stability_face_subset_sweep()`
   Sweeps all k-of-6 face subsets and reports best/worst stability. This is the
   non-hand-wavy version: for k<6 every subset misses at least one face.

Noise and maintenance model
---------------------------
- Noise corrupts sites in the 26-neighbour shell to ±1 with prob `p_noise`.
- The center may be knocked to 0 with prob `p_center_flip`.
- A protocol scans its chosen offsets and repairs any non-zero neighbour back to
  0 each tick.
- The particle "exists" only while all 6 faces are zero; otherwise it decays.

Outputs
-------
- `stability_benchmark()` returns a flat dict of metrics.
- `stability_face_subset_sweep()` returns a `FaceSweepSummary` dataclass.

Flat-module layout
------------------
Lives alongside `core.py` and is imported as:
  `from stability import stability_benchmark, stability_face_subset_sweep`

No plotting: visual output (if invoked) is handled by `visualiser.py`.
"""

from __future__ import annotations

import itertools
from typing import Dict, Tuple

import numpy as np

from params import FaceSweepSummary, StabilityParams


# -----------------------------
# Stability benchmark (derive neighbour overhead)
# -----------------------------


def _shell_offsets_26() -> Tuple[Tuple[int, int, int], ...]:
    offs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offs.append((dx, dy, dz))
    return tuple(offs)


def stability_benchmark(rng: np.random.Generator, sp: StabilityParams) -> Dict[str, float]:
    """Empirical test: what neighbour-check set is minimally sufficient to maintain a stable
    'particle' (center +1 with intact 6-face adjacency) under random annealing noise.

    The invariant we demand for "existence" is strict and 3D:
      - center is +1
      - all 6 face neighbours are 0

    Noise model:
      - each tick, any site in the 26-neighbour shell may be corrupted to ±1 with prob p_noise
      - center may be flipped to 0 with prob p_center_flip

    Maintenance model:
      - protocol scans its chosen neighbour offsets
      - if scanned neighbour is non-zero, it repairs it back to 0
      - after repairs, if any of the 6 faces is non-zero, the particle has decayed (stop)
      - otherwise, center is restored to +1

    This deliberately tests the claim: 6 face checks are the minimum bounding constraints.
    """

    # Protocol definitions.
    faces6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    planar4 = faces6[:4]
    corners8 = (
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
    )
    faces6_plus2 = faces6 + ((1, 1, 1), (-1, -1, -1))

    protocols = {
        "A_planar4": planar4,
        "B_faces6": faces6,
        "C_corners8": corners8,
        "D_faces6_plus2": faces6_plus2,
    }

    shell = _shell_offsets_26()

    def _run_trials(offsets: Tuple[Tuple[int, int, int], ...]) -> Tuple[float, float, float, float]:
        """Return (mean_survival, p_full, reads_per_tick, repairs_per_tick) across trials."""

        def run_one() -> Tuple[int, int, int]:
            # Returns: (survived_ticks, reads, repairs)
            # Reads counted as neighbour reads per tick (len(offsets)).
            n = 9  # tiny local workspace around the center
            c = n // 2
            g = np.zeros((n, n, n), dtype=np.int8)
            g[c, c, c] = 1

            reads = 0
            repairs = 0

            for t in range(sp.ticks):
                # Apply noise to the 26-shell.
                for dx, dy, dz in shell:
                    if rng.random() < sp.p_noise:
                        g[c + dx, c + dy, c + dz] = int(rng.choice(np.array([-1, 1], dtype=np.int8)))

                # center may be knocked to 0.
                if rng.random() < sp.p_center_flip:
                    g[c, c, c] = 0

                # Maintenance: scan + repair.
                reads += len(offsets)
                for dx, dy, dz in offsets:
                    if g[c + dx, c + dy, c + dz] != 0:
                        g[c + dx, c + dy, c + dz] = 0
                        repairs += 1

                # Existence invariant: all 6 faces must be clean.
                if (
                    g[c + 1, c, c] != 0 or g[c - 1, c, c] != 0 or
                    g[c, c + 1, c] != 0 or g[c, c - 1, c] != 0 or
                    g[c, c, c + 1] != 0 or g[c, c, c - 1] != 0
                ):
                    return t, reads, repairs

                # Restore center.
                g[c, c, c] = 1

            return sp.ticks, reads, repairs

        surv = np.empty(sp.trials, dtype=np.float64)
        reads = np.empty(sp.trials, dtype=np.float64)
        repairs = np.empty(sp.trials, dtype=np.float64)

        for i in range(sp.trials):
            st, rd, rpairs = run_one()
            surv[i] = st
            reads[i] = rd
            repairs[i] = rpairs

        mean_surv = float(np.mean(surv))
        p_full = float(np.mean(surv >= sp.ticks))
        reads_per_tick = float(np.mean(reads) / max(1.0, mean_surv))
        repairs_per_tick = float(np.mean(repairs) / max(1.0, mean_surv))
        return mean_surv, p_full, reads_per_tick, repairs_per_tick

    rows = []
    for name, offs in protocols.items():
        mean_surv, p_full, reads_per_tick, repairs_per_tick = _run_trials(offs)
        rows.append((name, mean_surv, p_full, reads_per_tick, repairs_per_tick))

    # Summarise into a compact dict (print formatting happens in main).
    out: Dict[str, float] = {}
    for name, mean_surv, p_full, reads_per_tick, repairs_per_tick in rows:
        out[f"{name}.mean_survival"] = mean_surv
        out[f"{name}.p_full"] = p_full
        out[f"{name}.reads_per_tick"] = reads_per_tick
        out[f"{name}.repairs_per_tick"] = repairs_per_tick

    return out


# -----------------------------
# Face-subset sweep for stability
# -----------------------------


def stability_face_subset_sweep(rng: np.random.Generator, sp: StabilityParams, k: int) -> FaceSweepSummary:
    """Sweep all k-of-6 face-neighbour subsets and report best/worst stability.

    This is the bulletproof version of the '6 is minimal' claim:
      - For k < 6, every subset is missing at least one face constraint.
      - Under isotropic noise, missing constraints should show up as decay.

    We report the best-case subset to avoid cherry-picking accusations.
    """

    if k < 1 or k > 6:
        raise ValueError("k must be in [1,6]")

    faces6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    shell = _shell_offsets_26()

    face_label = {
        (1, 0, 0): "+x",
        (-1, 0, 0): "-x",
        (0, 1, 0): "+y",
        (0, -1, 0): "-y",
        (0, 0, 1): "+z",
        (0, 0, -1): "-z",
    }

    def subset_to_str(offsets: Tuple[Tuple[int, int, int], ...]) -> str:
        labs = [face_label[o] for o in offsets]
        # Stable ordering for readability.
        order = {"+x": 0, "-x": 1, "+y": 2, "-y": 3, "+z": 4, "-z": 5}
        labs.sort(key=lambda s: order[s])
        return "[" + ",".join(labs) + "]"

    def missing_to_str(offsets: Tuple[Tuple[int, int, int], ...]) -> str:
        have = set(offsets)
        miss = [face_label[o] for o in faces6 if o not in have]
        order = {"+x": 0, "-x": 1, "+y": 2, "-y": 3, "+z": 4, "-z": 5}
        miss.sort(key=lambda s: order[s])
        return "[" + ",".join(miss) + "]" if miss else "[]"

    def run_trials(offsets: Tuple[Tuple[int, int, int], ...]) -> Tuple[float, float]:
        # Return (mean_survival, p_full)
        n = 9
        c = n // 2

        surv = np.empty(sp.trials, dtype=np.float64)

        for i in range(sp.trials):
            g = np.zeros((n, n, n), dtype=np.int8)
            g[c, c, c] = 1

            for t in range(sp.ticks):
                for dx, dy, dz in shell:
                    if rng.random() < sp.p_noise:
                        g[c + dx, c + dy, c + dz] = int(rng.choice(np.array([-1, 1], dtype=np.int8)))

                if rng.random() < sp.p_center_flip:
                    g[c, c, c] = 0

                for dx, dy, dz in offsets:
                    if g[c + dx, c + dy, c + dz] != 0:
                        g[c + dx, c + dy, c + dz] = 0

                if (
                    g[c + 1, c, c] != 0 or g[c - 1, c, c] != 0 or
                    g[c, c + 1, c] != 0 or g[c, c - 1, c] != 0 or
                    g[c, c, c + 1] != 0 or g[c, c, c - 1] != 0
                ):
                    surv[i] = t
                    break

                g[c, c, c] = 1
            else:
                surv[i] = sp.ticks

        mean_surv = float(np.mean(surv))
        p_full = float(np.mean(surv >= sp.ticks))
        return mean_surv, p_full

    best_name = ""
    best_p_full = -1.0
    best_mean = -1.0

    worst_name = ""
    worst_p_full = 2.0
    worst_mean = 1e18

    for offs in itertools.combinations(faces6, k):
        mean_surv, p_full = run_trials(tuple(offs))

        name = subset_to_str(tuple(offs)) + " missing=" + missing_to_str(tuple(offs))

        if p_full > best_p_full or (p_full == best_p_full and mean_surv > best_mean):
            best_p_full = p_full
            best_mean = mean_surv
            best_name = name

        if p_full < worst_p_full or (p_full == worst_p_full and mean_surv < worst_mean):
            worst_p_full = p_full
            worst_mean = mean_surv
            worst_name = name

    return FaceSweepSummary(
        k=k,
        best_name=best_name,
        best_p_full=float(best_p_full),
        best_mean_survival=float(best_mean),
        worst_name=worst_name,
        worst_p_full=float(worst_p_full),
        worst_mean_survival=float(worst_mean),
    )