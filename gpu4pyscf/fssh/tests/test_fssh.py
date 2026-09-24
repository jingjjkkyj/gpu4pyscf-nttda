# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from pyscf import gto

from gpu4pyscf.fssh.fssh import FSSH, HoppingProbabilityError, PES


class DeterministicHopFSSH(FSSH):
    """Minimal electronic model for exercising the nuclear FSSH integrator."""

    def __init__(self, mol):
        super().__init__(mol, states=[0, 1])
        self.mass = np.ones(mol.natm)
        self.dt = 1.0
        self.nsteps = 1
        self.decoherence = False
        self.save_force = True
        self.calls = []

    def evaluate_pes(self, position, cur_state, with_nacv=True):
        self.calls.append((cur_state, with_nacv, np.array(position, copy=True)))
        force_value = 2.0 if cur_state == 0 else 6.0
        nacv = np.zeros((2, 2, self.mol.natm, 3))
        nacv[0, 1, :, 0] = 0.25
        nacv[1, 0, :, 0] = -0.25
        return PES(
            energy=np.array([0.0, 0.0]),
            force=np.full((self.mol.natm, 3), force_value),
            nacv=nacv,
        )

    def update_coefficient(self, coeffs, energy, nact):
        return coeffs

    def evaluate_hopping(self, coeffs, nact, cur_state):
        return 1

    def rescale_velocity(self, hop_index, cur_state, energy, velocity, d_vec):
        return True, velocity


class OversizedProbabilityFSSH(FSSH):
    """Model whose coarse electronic step requires probability substeps."""

    def __init__(self, mol):
        super().__init__(mol, states=[0, 1])
        self.mass = np.ones(mol.natm)
        self.dt = 1.0
        self.nsteps = 1
        self.decoherence = False
        self.save_force = True
        self.random_calls = 0

    def evaluate_pes(self, position, cur_state, with_nacv=True):
        nacv = np.zeros((2, 2, self.mol.natm, 3))
        nacv[0, 1, :, 0] = 0.75
        nacv[1, 0, :, 0] = -0.75
        return PES(
            energy=np.zeros(2),
            force=np.zeros((self.mol.natm, 3)),
            nacv=nacv,
        )

    def update_coefficient(self, coeffs, energy, nact, dt=None):
        return np.array(coeffs, copy=True)

    def random_uniform(self):
        self.random_calls += 1
        return 0.99


class NestedOversizedProbabilityFSSH(OversizedProbabilityFSSH):
    """Model that also rejects the initially planned electronic substeps."""

    def __init__(self, mol):
        super().__init__(mol)
        self.probability_calls = 0

    def compute_hopping_probability(
            self, coeffs, nact, cur_state, dt=None):
        self.probability_calls += 1
        electronic_dt = self.dt if dt is None else dt
        if electronic_dt > 0.05:
            raise HoppingProbabilityError(1.5)
        return np.zeros(len(self.states))


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom="H 0 0 0",
            basis="sto-3g",
            unit="Bohr",
            spin=1,
            verbose=0,
        )

    def test_successful_hop_uses_target_force_immediately(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = DeterministicHopFSSH(self.mol)
            driver.filename = str(Path(tmpdir) / "trajectory.h5")

            _, velocity, _ = driver.kernel(
                position=np.zeros((1, 3)),
                velocity=np.zeros((1, 3)),
                coefficient=np.array([1.0, 0.0], dtype=complex),
            )

            self.assertEqual([(x[0], x[1]) for x in driver.calls], [
                (0, False),
                (0, True),
                (1, False),
            ])
            self.assertTrue(np.allclose(velocity, 4.0))
            with h5py.File(driver.filename, "r") as handle:
                self.assertEqual(int(handle["1/cur_state"][()]), 1)
                self.assertTrue(np.allclose(handle["1/force"], 6.0))
                self.assertAlmostEqual(float(handle["1/nacv"][0, 1, 0, 0]), 0.25)

    def test_active_state_population_must_not_vanish(self):
        driver = FSSH(self.mol, states=[0, 1])
        with self.assertRaisesRegex(RuntimeError, "active-state population"):
            driver.compute_hopping_probability(
                np.array([0.0, 1.0], dtype=complex),
                np.zeros((2, 2)),
                cur_state=0,
            )

    def test_total_hopping_probability_above_one_is_rejected(self):
        driver = FSSH(self.mol, states=[0, 1, 2])
        driver.dt = 1.0
        coeffs = np.array([1.0, 1.0, 1.0], dtype=complex)
        nact = np.array([
            [0.0, 0.4, 0.3],
            [-0.4, 0.0, 0.0],
            [-0.3, 0.0, 0.0],
        ])
        with self.assertRaisesRegex(RuntimeError, "sum to"):
            driver.compute_hopping_probability(coeffs, nact, cur_state=0)

    def test_kernel_retries_oversized_probability_with_electronic_substeps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = OversizedProbabilityFSSH(self.mol)
            driver.filename = str(Path(tmpdir) / "trajectory.h5")
            coefficient = np.full(2, 1 / np.sqrt(2), dtype=complex)

            driver.kernel(
                position=np.zeros((1, 3)),
                velocity=np.array([[1.0, 0.0, 0.0]]),
                coefficient=coefficient,
            )

            self.assertEqual(driver._last_electronic_substeps, 15)
            self.assertEqual(driver.random_calls, 15)
            self.assertEqual(driver.cur_step, 1)
            with h5py.File(driver.filename, "r") as handle:
                self.assertIn("1", handle)

    def test_invalid_substeps_are_split_before_random_sampling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            driver = NestedOversizedProbabilityFSSH(self.mol)
            driver.filename = str(Path(tmpdir) / "trajectory.h5")
            coefficient = np.full(2, 1 / np.sqrt(2), dtype=complex)

            driver.kernel(
                position=np.zeros((1, 3)),
                velocity=np.array([[1.0, 0.0, 0.0]]),
                coefficient=coefficient,
            )

            self.assertEqual(driver._last_electronic_substeps, 30)
            self.assertEqual(driver.probability_calls, 46)
            self.assertEqual(driver.random_calls, 30)
            self.assertEqual(driver.cur_step, 1)

    def test_active_state_is_never_a_hop_target(self):
        driver = FSSH(self.mol, states=[0, 1])
        driver.dt = 1.0
        probability = driver.compute_hopping_probability(
            np.array([1.0, 1.0], dtype=complex),
            np.array([[0.2, 0.1], [-0.1, 0.0]]),
            cur_state=0,
        )
        self.assertEqual(probability[0], 0.0)


if __name__ == "__main__":
    unittest.main()
