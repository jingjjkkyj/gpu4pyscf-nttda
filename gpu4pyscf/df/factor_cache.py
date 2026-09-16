# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Bounded, solve-local cache for a fixed right density factor."""

import cupy as cp

from gpu4pyscf.__config__ import num_devices
from gpu4pyscf.lib.cupy_helper import get_avail_mem


class FixedFactorCache:
    """Cache ``B[L,i,j] R[j,k]`` without modifying the DF object.

    The owner must keep ``factor`` immutable and clear this cache after its
    orbital solve. Integral rebuilds (including reset and range separation)
    have separate entries. Blocks are indexed by auxiliary *ranges*, so a
    change of DF block size cannot return the wrong integral slice.
    """

    def __init__(self, factor, max_bytes=4 * 1024**3):
        self.factor = factor
        self.max_bytes = max_bytes
        self._entries = {}
        self._bytes = {}
        self.hits = 0
        self.misses = 0
        self.peak_bytes = 0

    @property
    def nbytes(self):
        return sum(self._bytes.values())

    def _entry(self, dfobj, device):
        key = (dfobj, device)
        entry = self._entries.get(key)
        integrals = dfobj._cderi[device]
        if entry is None or entry[0] is not dfobj.intopt or entry[1] is not integrals:
            if entry is not None:
                self._bytes[device] -= sum(x[0].nbytes for x in entry[2].values())
            entry = (dfobj.intopt, integrals, {})
            self._entries[key] = entry
        return entry[2]

    def get(self, dfobj, device, start, size):
        item = self._entry(dfobj, device).get((start, size))
        if item is None:
            self.misses += 1
            return None
        value, ready = item
        cp.cuda.get_current_stream().wait_event(ready)
        self.hits += 1
        return value

    def put(self, dfobj, device, start, value):
        used = self._bytes.get(device, 0)
        # Divide the cap across devices; leave working space for XC and J/K.
        if (used + value.nbytes > self.max_bytes // num_devices
                or value.nbytes > get_avail_mem() // 4):
            return
        blocks = self._entry(dfobj, device)
        key = (start, len(value))
        if key in blocks:
            return
        stored = value.copy()
        ready = cp.cuda.Event()
        ready.record()
        blocks[key] = (stored, ready)
        self._bytes[device] = used + stored.nbytes
        self.peak_bytes = max(self.peak_bytes, self.nbytes)

    def clear(self):
        self._entries.clear()
        self._bytes.clear()
