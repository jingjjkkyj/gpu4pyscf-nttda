# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest import mock

import cupy as cp
import numpy as np

import gpu4pyscf.sftda.nttda as nttda


class FakeMol:
    spin = 2


class FakeMF:
    def __init__(self):
        self.mol = FakeMol()
        self.verbose = 0
        self.stdout = None
        self.max_memory = 321
        self.mo_occ = np.array([2.0, 1.0, 1.0, 0.0])


class KnownValues(unittest.TestCase):
    def test_spin_lowered_reference_is_open_open_identity(self):
        reference = nttda._spin_lowered_reference_vector(
            nocc=3, nvir=3, nopen=2,
        ).reshape(3, 3)
        expected = np.array([
            [0.0, 0.0, 0.0],
            [1.0 / np.sqrt(2.0), 0.0, 0.0],
            [0.0, 1.0 / np.sqrt(2.0), 0.0],
        ])
        self.assertTrue(np.allclose(reference, expected))
        self.assertAlmostEqual(np.linalg.norm(reference), 1.0)

    def test_reference_character_removes_exactly_one_root(self):
        reference, order = nttda._select_physical_root_order(
            energies=np.array([-0.1, 0.0, 5e-8, 0.2]),
            reference_overlaps=np.array([0.0, 0.0, 1.0, 0.0]),
            nstates=3,
            overlap_tol=0.8,
            energy_tol=1e-6,
        )
        self.assertEqual(reference, 2)
        self.assertEqual(order.tolist(), [0, 1, 3])

    def test_ambiguous_reference_character_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "reference-root overlap"):
            nttda._select_physical_root_order(
                energies=np.array([0.0, 0.1]),
                reference_overlaps=np.array([0.7, 0.7]),
                nstates=1,
                overlap_tol=0.8,
                energy_tol=1e-6,
            )

    def test_kernel_keeps_physical_zero_root_and_passes_max_memory(self):
        mf = FakeMF()
        td = nttda.NTTDA(mf)
        td.nstates = 3
        td.conv_tol = 1e-9
        td.gen_vind_sfd = lambda: (
            lambda vectors: vectors,
            cp.arange(9, dtype=float),
        )

        reference = nttda._spin_lowered_reference_vector(3, 3, 2)
        physical = np.eye(9)[[0, 1, 2]]
        vectors = np.vstack((physical[:2], reference, physical[2:]))
        captured = {}

        def fake_eigh(*args, **kwargs):
            captured.update(kwargs)
            return (
                np.ones(4, dtype=bool),
                np.array([-0.1, 0.0, 5e-8, 0.2]),
                cp.asarray(vectors),
            )

        with mock.patch.object(nttda, "lr_eigh", side_effect=fake_eigh):
            energies, _xy = td.kernel()

        self.assertEqual(captured["max_memory"], 321)
        self.assertTrue(np.allclose(energies, [-0.1, 0.0, 0.2]))
        self.assertEqual(td.converged.tolist(), [True, True, True])


if __name__ == "__main__":
    unittest.main()
