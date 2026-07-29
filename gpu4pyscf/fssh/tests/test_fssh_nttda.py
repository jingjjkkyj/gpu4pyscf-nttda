# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import types
import unittest

import numpy as np
from pyscf import gto

from gpu4pyscf.fssh import FSSH_NTTDA
from gpu4pyscf.fssh.fssh import FSSH, PES


class FakeSCF:
    def __init__(self, mol):
        self.mol = mol
        self.converged = False


class FakeNTTDA:
    def __init__(self, mol):
        self.mol = mol
        self._scf = FakeSCF(mol)
        self.deltaS = -1
        self.e = None
        self.xy = None


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='H 0 0 0; H 0 0 0.74',
            basis='sto-3g',
            unit='Angstrom',
            verbose=0,
        )

    def test_public_driver_overrides_pes_contract(self):
        self.assertIsNot(FSSH_NTTDA.evaluate_pes, FSSH.evaluate_pes)

        driver = FSSH_NTTDA(FakeNTTDA(self.mol), states=[2, 1])
        calls = []

        def fake_electronic(
                self, position, cur_state=None, with_nacv=True):
            calls.append((cur_state, with_nacv))
            natm = self.mol.natm
            return (
                np.array([-1.0, -1.1]),
                np.zeros((natm, 3)),
                np.zeros((2, 2, natm, 3)),
            )

        driver.calc_electronic = types.MethodType(fake_electronic, driver)
        pes = driver.evaluate_pes(
            self.mol.atom_coords(), cur_state=1, with_nacv=False,
        )

        self.assertIsInstance(pes, PES)
        self.assertEqual(calls, [(1, False)])
        self.assertEqual(pes.force.shape, (self.mol.natm, 3))

    def test_state_labels_must_be_positive_and_unique(self):
        td = FakeNTTDA(self.mol)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            FSSH_NTTDA(td, states=[])
        with self.assertRaisesRegex(ValueError, '1-based'):
            FSSH_NTTDA(td, states=[0, 1])
        with self.assertRaisesRegex(ValueError, 'unique'):
            FSSH_NTTDA(td, states=[1, 1])

    def test_descending_state_order_and_explicit_options_are_supported(self):
        driver = FSSH_NTTDA(
            FakeNTTDA(self.mol),
            states=[np.int64(2), np.int64(1)],
            use_etfs=False,
            root_tracking_buffer=2,
            cphf_max_cycle=77,
        )

        self.assertEqual(driver.states, [2, 1])
        self.assertEqual(driver.cur_state, 2)
        self.assertFalse(driver.use_etfs)
        self.assertEqual(driver.nstates_solver, 4)
        self.assertEqual(driver.cphf_max_cycle, 77)

    def test_invalid_root_tracking_buffer_is_rejected(self):
        td = FakeNTTDA(self.mol)
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ValueError, 'root_tracking_buffer'):
                    FSSH_NTTDA(
                        td, states=[1, 2], root_tracking_buffer=value,
                    )

    def test_invalid_cphf_max_cycle_is_rejected(self):
        td = FakeNTTDA(self.mol)
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'cphf_max_cycle'):
                    FSSH_NTTDA(
                        td, states=[1, 2], cphf_max_cycle=value,
                    )


if __name__ == '__main__':
    unittest.main()
