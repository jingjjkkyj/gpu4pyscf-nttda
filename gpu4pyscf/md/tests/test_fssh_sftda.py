# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

import numpy as np
from pyscf import gto

from gpu4pyscf.md.fssh_sftda import FSSH_SFTDA, select_sf_singlet_manifold


class FakeMF:
    e_tot = -10.0


class FakeTD:
    e = np.array([0.00, 0.10, 0.20])
    xy = [object(), object(), object()]

    def spin_square(self, state=None):
        values = np.array([2.0, 0.0, 0.1])
        if state is None:
            return values
        return values[state]


class FakeSCF:
    def __init__(self, mol):
        self.mol = mol


class FakeTDObject:
    nstates = 3
    verbose = 0

    def __init__(self, mol):
        self.mol = mol
        self._scf = FakeSCF(mol)


class KnownValues(unittest.TestCase):
    def test_select_sf_singlet_manifold_uses_gpu_zero_based_roots(self):
        selection = select_sf_singlet_manifold(
            FakeMF(), FakeTD(), states=[1, 2], n_lowest=3
        )

        self.assertEqual(selection.triplet_root, 0)
        self.assertEqual(selection.state_map, {1: 1, 2: 2})
        self.assertTrue(np.allclose(selection.energies_for_states([1, 2]),
                                    [-9.9, -9.8]))

    def test_fssh_sftda_rejects_zero_state_label(self):
        mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            unit="Angstrom",
            spin=2,
            verbose=0,
        )

        with self.assertRaisesRegex(ValueError, "1-based"):
            FSSH_SFTDA(FakeTDObject(mol), states=[0, 1])


if __name__ == "__main__":
    unittest.main()

