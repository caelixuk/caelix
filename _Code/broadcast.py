# CAELIX — Constructive Algorithmics for Emergent Lattice Interaction eXperiments
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Alan Ball

# broadcast.py — Shared Memory Broadcaster

from __future__ import annotations

import atexit
from typing import Optional, Tuple

import numpy as np
from multiprocessing import shared_memory


class LiveBroadcaster:
    """Producer-side shared memory broadcaster.

    Notes:
      - The producer allocates (or attaches to) a named shared memory block.
      - We keep the buffer as float32, shape=(n,n,n).
      - We only unlink if we created the block (owner semantics).
      - On POSIX, unlinking removes the name but readers with an open handle continue.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        *,
        name: str = "CAELIX_shm",
        create: bool = True,
        zero_init: bool = True,
    ) -> None:
        if len(shape) != 3:
            raise ValueError(f"LiveBroadcaster: shape must be 3D, got {shape}")
        if any(int(x) <= 0 for x in shape):
            raise ValueError(f"LiveBroadcaster: invalid shape {shape}")

        self.name = str(name)
        self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        self.dtype = np.float32
        self._owner = False

        n_bytes = int(np.prod(self.shape)) * int(np.dtype(self.dtype).itemsize)
        if n_bytes <= 0:
            raise ValueError("LiveBroadcaster: computed shared memory size is invalid")

        shm: Optional[shared_memory.SharedMemory] = None
        if create:
            try:
                try:
                    shm = shared_memory.SharedMemory(
                        create=True, size=n_bytes, name=self.name, track=False  # type: ignore[call-arg]
                    )  # py3.12+
                except TypeError:
                    shm = shared_memory.SharedMemory(create=True, size=n_bytes, name=self.name)
                self._owner = True
            except FileExistsError:
                # Non-owner attachment: avoid resource_tracker cleanup warnings on Python 3.12+.
                try:
                    shm = shared_memory.SharedMemory(name=self.name, track=False)  # py3.12+  # type: ignore[call-arg]
                except TypeError:
                    shm = shared_memory.SharedMemory(name=self.name)
                self._owner = False
        else:
            # Non-owner attachment: avoid resource_tracker cleanup warnings on Python 3.12+.
            try:
                shm = shared_memory.SharedMemory(name=self.name, track=False)  # py3.12+  # type: ignore[call-arg]
            except TypeError:
                shm = shared_memory.SharedMemory(name=self.name)
            self._owner = False

        self.shm = shm

        self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        if zero_init:
            self.buffer.fill(0.0)

        atexit.register(self.close)

    def update(self, data: np.ndarray) -> None:
        """Copy `data` into the shared buffer.

        `data` must be broadcast-compatible with (n,n,n) and convertible to float32.
        """
        if self.shm is None:
            raise RuntimeError("LiveBroadcaster.update called after close")
        arr = np.asarray(data, dtype=self.dtype)
        if arr.shape != self.shape:
            raise ValueError(
                f"LiveBroadcaster.update: shape mismatch {arr.shape} != {self.shape}"
            )
        np.copyto(self.buffer, arr, casting="no")

    def close(self) -> None:
        """Close (and if owner, unlink) the shared memory block."""
        shm = getattr(self, "shm", None)
        if shm is None:
            return
        try:
            shm.close()
        finally:
            if self._owner:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    # Already unlinked.
                    pass
                except PermissionError:
                    # Some platforms may restrict unlink while readers are attached.
                    pass
        self.shm = None

    def __enter__(self) -> "LiveBroadcaster":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()