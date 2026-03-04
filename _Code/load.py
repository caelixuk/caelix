# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

"""load.py — microstate → non-negative load field

Role
----
`compute_load()` maps the ternary microstate `s ∈ {-1,0,+1}` onto a
non-negative scalar load field `load ≥ 0`.

Interpretation
--------------
Load is a simple non-negative proxy for local defect/interface density:
  - absolute activity term: |s|
  - symmetric 6-neighbour mismatch term: for each axis-aligned edge, the mismatch |s(i)-s(j)| is counted once and shared 50/50 between the two incident voxels.

Boundary semantics
------------------
Neighbour mismatch uses hard (non-periodic) boundaries: we use `np.roll` for
speed but explicitly zero the wrapped face so edge contributions do not wrap. This produces hard (non-periodic) boundaries while keeping a centred (unbiased) stencil.

Contracts
---------
- Input: `s` shape (n,n,n), dtype integer-ish; values expected in {-1,0,+1}.
- Output: `load` shape (n,n,n), dtype float32, finite everywhere.

Flat-module layout
------------------
Lives alongside `core.py` and is imported as:
  `from load import compute_load`
"""

from __future__ import annotations

import numpy as np

from params import LoadParams
from utils import _assert_finite


def compute_load(s: np.ndarray, lp: LoadParams) -> np.ndarray:
    """Compute a non-negative 'processing load' scalar field from the ternary microstate."""
    load = np.zeros_like(s, dtype=np.float32)

    # abs cost
    load += lp.w_abs * np.abs(s).astype(np.float32)

    # mismatch cost: count neighbour differences (hard boundaries)
    # Symmetric edge-sharing: each mismatch across an edge contributes equally to both voxels.
    for axis in (0, 1, 2):
        rolled = np.roll(s, -1, axis=axis)
        diff = np.abs(s.astype(np.int16) - rolled.astype(np.int16)).astype(np.float32)

        # roll uses periodic boundaries; we must zero the wrapped face (no wrap contributions).
        if axis == 0:
            diff[-1, :, :] = 0.0
        elif axis == 1:
            diff[:, -1, :] = 0.0
        else:
            diff[:, :, -1] = 0.0

        # Share each edge mismatch between the two incident voxels.
        load += 0.5 * lp.w_mismatch * diff

        diff_back = np.roll(diff, 1, axis=axis)
        if axis == 0:
            diff_back[0, :, :] = 0.0
        elif axis == 1:
            diff_back[:, 0, :] = 0.0
        else:
            diff_back[:, :, 0] = 0.0

        load += 0.5 * lp.w_mismatch * diff_back

    _assert_finite(load, "load")
    return load