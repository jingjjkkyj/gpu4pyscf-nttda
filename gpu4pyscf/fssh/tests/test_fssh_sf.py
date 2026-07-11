# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import tempfile
import unittest
import types
import pickle
from pathlib import Path

import numpy as np
from pyscf import gto

from gpu4pyscf.fssh import FSSH_SF
from gpu4pyscf.fssh.sf_state_selection import select_sf_singlet_manifold


class FakeDerivative:
    cphf_max_cycle = 50
    cphf_conv_tol = 1e-6


class FakeSCF:
    def __init__(self, mol):
        self.mol = mol


class FakeSpinFlipTD:
    nstates = 4
    extype = 1
    collinear = "mcol"
    collinear_samples = 10
    verbose = 0

    def __init__(self, mol):
        self.mol = mol
        self._scf = FakeSCF(mol)

    def Gradients(self):
        return FakeDerivative()

    def NAC(self):
        return FakeDerivative()


class FakeMF:
    e_tot = -10.0


class FakeRoots:
    e = np.array([0.00, 0.10, 0.20, 0.30])
    xy = [object(), object(), object(), object()]

    def spin_square(self, state=None):
        values = np.array([2.0, 0.0, 0.1, 0.2])
        return values if state is None else values[state]


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            unit="Angstrom",
            verbose=0,
        )

    def test_public_driver_keeps_bscc_configuration_convention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FSSH_SF(
                FakeSpinFlipTD(self.mol),
                states=[1, 2],
                dt=0.25,
                nsteps=5,
                output_dir=tmpdir,
                cphf_options={"max_cycle": 77, "conv_tol": 2e-8},
            )

            self.assertAlmostEqual(driver.dt / 41.34137, 0.25)
            self.assertEqual(driver.nsteps, 5)
            self.assertEqual(driver.output_dir, Path(tmpdir))
            self.assertEqual(driver.cphf_max_cycle, 77)
            self.assertAlmostEqual(driver.cphf_conv_tol, 2e-8)

    def test_sf_state_selection_uses_gpu_zero_based_spin_square(self):
        selection = select_sf_singlet_manifold(
            FakeMF(), FakeRoots(), states=[1, 2], n_lowest=4
        )

        self.assertEqual(selection.triplet_root, 0)
        self.assertEqual(selection.state_map, {1: 1, 2: 2})
        self.assertTrue(
            np.allclose(selection.energies_for_states([1, 2]), [-9.9, -9.8])
        )

    def test_zero_based_sf_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "1-based"):
            FSSH_SF(FakeSpinFlipTD(self.mol), states=[0, 1])

    def test_complete_five_step_dynamics_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = FSSH_SF(
                FakeSpinFlipTD(self.mol),
                states=[1, 2],
                dt=0.5,
                nsteps=5,
                output_dir=tmpdir,
            )

            def fake_electronic(self, position):
                natm = self.tddft.mol.natm
                return (
                    np.array([-1.0, -0.9]),
                    np.zeros((natm, 3)),
                    np.zeros((2, 2, natm, 3)),
                )

            driver.calc_electronic = types.MethodType(fake_electronic, driver)
            velocity = np.full((self.mol.natm, 3), 10.0)
            coefficient = np.array([1.0, 0.0], dtype=complex)
            _, final_velocity, final_coefficient = driver.kernel(
                position=self.mol.atom_coords(unit="Bohr"),
                velocity=velocity,
                coefficient=coefficient,
            )

            with (Path(tmpdir) / "checkpoint.pkl").open("rb") as handle:
                checkpoint = pickle.load(handle)
            frames = sum(
                line.strip() == str(self.mol.natm)
                for line in (Path(tmpdir) / "trajectory.xyz").read_text().splitlines()
            )
            self.assertEqual(checkpoint["step"], 5)
            self.assertEqual(frames, 6)
            self.assertTrue(np.allclose(final_velocity, velocity))
            self.assertTrue(
                np.allclose(np.abs(final_coefficient) ** 2, np.abs(coefficient) ** 2)
            )


if __name__ == "__main__":
    unittest.main()
