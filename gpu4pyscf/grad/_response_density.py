# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Exact occupied-factor densities for ensemble orbital response."""

import cupy as cp

from gpu4pyscf.df.factor_cache import FixedFactorCache
from gpu4pyscf.lib.cupy_helper import tag_array


class OrbitalRotationDensity:
    """Own fixed orbitals and DF contractions for one orbital solve.

    For real antisymmetric kappa, C[kappa,n]C.T = L R.T + R L.T,
    where R=C[:,n!=0] and L=(C kappa[:,n!=0]) n[n!=0].
    No occupation threshold, eigendecomposition or rank truncation is used.
    Dense AO values are still supplied to XC and non-DF response routines.
    """

    def __init__(self, orbitals, occupation):
        self.orbitals = cp.array(orbitals, copy=True)
        self.occupation = cp.array(occupation, copy=True)
        self.occupied = cp.flatnonzero(self.occupation != 0)
        self.right = self.orbitals[:, self.occupied].copy()
        self.cache = FixedFactorCache(self.right)

    def __call__(self, rotation):
        rotation = cp.asarray(rotation)
        c = self.orbitals
        n = self.occupation
        if (c.dtype != cp.float64 or rotation.dtype != cp.float64
                or not self.occupied.size):
            return c @ (rotation * (n[None, :] - n[:, None])) @ c.conj().T
        left = (c @ rotation[..., self.occupied]) * n[self.occupied]
        directed = left @ self.right.T
        density = directed + directed.swapaxes(-1, -2)
        # Keep the fixed side 2-D even for a single density: DF can reuse it
        # across all batch members as well as across Krylov iterations.
        return tag_array(
            density, factor_l=left.reshape(-1, c.shape[0], self.occupied.size),
            factor_r=self.right, symmetrize=1, _df_factor_cache=self.cache,
        )

    def clear(self):
        self.cache.clear()
