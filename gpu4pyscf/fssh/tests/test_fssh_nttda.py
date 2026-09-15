# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import types
import shutil
import tempfile
import unittest
from pathlib import Path

import cupy as cp
import h5py
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
                self, position, cur_state=None, with_nacv=True,
                with_frame=True):
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
        self.assertEqual(driver.state_ordering, 'energy')

    def test_energy_ordering_keeps_solver_order_and_aligns_phase(self):
        driver = FSSH_NTTDA(
            FakeNTTDA(self.mol), states=[1, 2], root_overlap_tol=0.4,
        )
        td = types.SimpleNamespace(
            e=np.array([0.1, 0.2]),
            xy=[
                (np.array([[1.0, 0.0]]), 0),
                (np.array([[0.0, 1.0]]), 0),
            ],
            converged=np.array([True, True]),
        )
        tracking = {
            'C': np.eye(2),
            'occ': np.array([1.0, 0.0]),
            'xy_p': [
                np.array([[0.6, 0.8]]),
                np.array([[0.8, -0.6]]),
            ],
            's_occ': np.eye(1),
            's_vir': np.eye(2),
        }

        driver._track_roots(self.mol, td, tracking)

        self.assertTrue(np.allclose(td.e, [0.1, 0.2]))
        self.assertEqual(driver.root_assignment, [0, 1])
        self.assertTrue(np.allclose(td.xy[0][0], [[1.0, 0.0]]))
        self.assertTrue(np.allclose(td.xy[1][0], [[0.0, -1.0]]))
        self.assertTrue(np.allclose(driver.root_overlaps, [0.6, 0.6]))

    def test_overlap_ordering_is_explicit_opt_in(self):
        driver = FSSH_NTTDA(
            FakeNTTDA(self.mol),
            states=[1, 2],
            state_ordering='overlap',
            root_overlap_tol=0.4,
        )
        td = types.SimpleNamespace(
            e=np.array([0.1, 0.2]),
            xy=[
                (np.array([[1.0, 0.0]]), 0),
                (np.array([[0.0, 1.0]]), 0),
            ],
            converged=np.array([True, True]),
        )
        tracking = {
            'C': np.eye(2),
            'occ': np.array([1.0, 0.0]),
            'xy_p': [
                np.array([[0.6, 0.8]]),
                np.array([[0.8, -0.6]]),
            ],
            's_occ': np.eye(1),
            's_vir': np.eye(2),
        }

        driver._track_roots(self.mol, td, tracking)

        self.assertTrue(np.allclose(td.e, [0.2, 0.1]))
        self.assertEqual(driver.root_assignment, [1, 0])

    def test_invalid_root_tracking_buffer_is_rejected(self):
        td = FakeNTTDA(self.mol)
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ValueError, 'root_tracking_buffer'):
                    FSSH_NTTDA(
                        td, states=[1, 2], root_tracking_buffer=value,
                    )

    def test_invalid_state_ordering_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'state_ordering'):
            FSSH_NTTDA(
                FakeNTTDA(self.mol),
                states=[1, 2],
                state_ordering='character',
            )

    def test_invalid_cphf_max_cycle_is_rejected(self):
        td = FakeNTTDA(self.mol)
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'cphf_max_cycle'):
                    FSSH_NTTDA(
                        td, states=[1, 2], cphf_max_cycle=value,
                    )

    def test_only_dfj_surface_is_rejected(self):
        td = FakeNTTDA(self.mol)
        td._scf.only_dfj = True
        with self.assertRaisesRegex(NotImplementedError, 'only_dfj'):
            FSSH_NTTDA(td, states=[1, 2])

    def test_checkpoint_restores_electronic_gauge_cache_and_rng(self):
        from gpu4pyscf.dft import roks
        from gpu4pyscf.sftda import NTTDA

        mf = roks.ROKS(self.mol, xc='B3LYP')
        mf.conv_tol = 1e-10
        mf.mo_coeff = cp.eye(2)
        mf.mo_occ = cp.asarray([2.0, 0.0])
        mf.mo_energy = cp.asarray([-0.5, 0.2])
        mf.e_tot = -1.0
        mf.converged = True
        mf.cycles = 4
        td = NTTDA(mf)
        td.nstates = 3
        td.conv_tol = 1e-8
        td.e = np.asarray([0.1, 0.2, 0.3])
        td.xy = [(cp.ones((1, 1)) * value, 0) for value in (1, 2, 3)]
        td.converged = np.ones(3, dtype=bool)
        td._nttda_solver_stats = {'warm_start': True}

        with tempfile.TemporaryDirectory() as tmpdir:
            trajectory = Path(tmpdir) / 'trajectory.h5'
            driver = FSSH_NTTDA(td, states=[1, 2])
            driver.cur_state = 2
            driver.filename = str(trajectory)
            driver.save_force = True
            driver._frame_cache._mol = self.mol.copy()
            driver._frame_cache._signature = ('test-signature',)
            driver._frame_cache._entries = {
                ('grad', 2): (np.eye(2), np.eye(2) * 2),
            }
            position = self.mol.atom_coords()
            velocity = np.arange(6, dtype=float).reshape(2, 3) * 1e-4
            pes = PES(
                energy=np.asarray([-0.9, -0.8]),
                force=np.ones((2, 3)) * 0.01,
                nacv=np.zeros((2, 2, 2, 3)),
            )
            coefficient = np.asarray([0.0, 1.0j])
            np.random.seed(19)
            driver.write_trajectory(
                0, position, velocity, pes, coefficient, 2,
            )
            expected_random = np.random.random()
            driver._finalize()

            with h5py.File(trajectory, 'a') as handle:
                handle.create_group('1')
                handle['velocity'][:] = 0.0

            mismatch = FSSH_NTTDA(
                td, states=[1, 2], state_ordering='overlap',
            )
            with self.assertRaisesRegex(ValueError, 'does not match'):
                mismatch.restore(trajectory)

            restored = FSSH_NTTDA(td, states=[1, 2])
            restored.restore(trajectory)
            self.assertEqual(restored.cur_step, 0)
            self.assertEqual(restored.cur_state, 2)
            self.assertTrue(np.array_equal(restored.position, position))
            self.assertTrue(np.array_equal(restored.velocity, velocity))
            self.assertTrue(np.array_equal(
                restored.coefficient, coefficient,
            ))
            self.assertEqual(np.random.random(), expected_random)
            self.assertEqual(
                set(restored._frame_cache._entries), {('grad', 2)}
            )
            self.assertTrue(np.array_equal(
                restored._prev[1], cp.asnumpy(mf.mo_coeff),
            ))
            restored_pes = restored.evaluate_pes(
                position, 2, with_nacv=False,
            )
            self.assertTrue(np.array_equal(restored_pes.energy, pes.energy))
            self.assertIsNone(restored._restored_pes)
            with h5py.File(trajectory) as handle:
                self.assertEqual(
                    sorted(int(key) for key in handle if key.isdigit()),
                    [0],
                )
                self.assertTrue(np.array_equal(
                    handle['0/velocity'], velocity,
                ))
                self.assertTrue(np.array_equal(
                    handle['velocity'], velocity,
                ))

    def test_resumed_next_step_matches_continuous_trajectory(self):
        from gpu4pyscf.dft import roks
        from gpu4pyscf.sftda import NTTDA

        mol = gto.M(
            atom='N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20',
            basis='sto-3g',
            spin=2,
            unit='Bohr',
            verbose=0,
        )
        mf = roks.ROKS(mol, xc='B3LYP').density_fit()
        mf.conv_tol = 1e-11
        mf.grids.level = 1
        mf.verbose = 0
        mf.kernel()
        self.assertTrue(mf.converged)
        td = NTTDA(mf)
        td.nstates = 3
        td.conv_tol = 1e-9
        td.verbose = 0
        td.kernel()
        self.assertTrue(np.all(
            cp.asnumpy(cp.asarray(td.converged)).astype(bool)
        ))

        velocity = np.asarray([
            [0.0, 0.0, 5e-5],
            [0.0, 0.0, -5e-5],
            [0.0, 5e-5, 0.0],
        ])

        def new_driver(path, nsteps):
            driver = FSSH_NTTDA(
                td, states=[1, 2], cphf_conv_tol=1e-9,
            )
            driver.filename = str(path)
            driver.nsteps = nsteps
            driver.timestep_fs = 0.02
            driver.seed = 23
            driver.save_force = True
            driver.verbose = 0
            return driver

        with tempfile.TemporaryDirectory() as tmpdir:
            continuous_path = Path(tmpdir) / 'continuous.h5'
            resumed_path = Path(tmpdir) / 'resumed.h5'
            continuous = new_driver(continuous_path, 2)

            def capture_first_step(env):
                if env['step'] == 1:
                    shutil.copyfile(continuous_path, resumed_path)
                    shutil.copyfile(
                        continuous._checkpoint_path(),
                        Path(str(resumed_path) + '.checkpoint.pkl'),
                    )

            continuous.callback = capture_first_step
            continuous.kernel(velocity=velocity.copy())
            continuous_rng = np.random.random()

            restored = new_driver(resumed_path, 2)
            restored.restore(resumed_path)
            restored.kernel()
            resumed_rng = np.random.random()

            with h5py.File(continuous_path) as left, \
                    h5py.File(resumed_path) as right:
                self.assertEqual(
                    sorted(int(key) for key in right if key.isdigit()),
                    [0, 1, 2],
                )
                for key, atol in (
                        ('position', 1e-12), ('velocity', 1e-12),
                        ('energy', 1e-11), ('coeffs', 1e-12),
                        ('force', 5e-9), ('nacv', 5e-9)):
                    np.testing.assert_allclose(
                        left[f'2/{key}'], right[f'2/{key}'],
                        atol=atol, rtol=0,
                    )
                self.assertEqual(
                    int(left['2/cur_state'][()]),
                    int(right['2/cur_state'][()]),
                )
            self.assertEqual(continuous_rng, resumed_rng)
            self.assertEqual(
                restored._last_td._nttda_frame_stats[
                    'zvector_cache_hits'
                ],
                2,
            )

    def test_loose_initial_solutions_are_not_reused(self):
        td = FakeNTTDA(self.mol)
        td._scf.converged = True
        td._scf.conv_tol = 1e-6
        td._scf.mo_coeff = np.eye(2)
        td._scf.mo_occ = np.array([1.0, 0.0])
        td.conv_tol = 1e-5
        td.e = np.array([0.1, 0.2, 0.3])
        td.xy = [
            (np.ones((1, 1)), 0),
            (np.ones((1, 1)), 0),
            (np.ones((1, 1)), 0),
        ]
        td.converged = np.ones(3, dtype=bool)

        driver = FSSH_NTTDA(
            td,
            states=[1, 2],
            scf_conv_tol=1e-10,
            td_conv_tol=1e-8,
        )

        self.assertIsNone(driver._last_mf)
        self.assertIsNone(driver._last_td)
        self.assertFalse(driver._initial_frame_available)

    def test_rebuilt_scf_preserves_tuned_range_separation(self):
        from gpu4pyscf.dft import roks

        td = FakeNTTDA(self.mol)
        td._scf = roks.ROKS(self.mol, xc='CAM-B3LYP')
        td._scf.omega = 0.37
        driver = FSSH_NTTDA(td, states=[1, 2])

        rebuilt = driver._new_scf(self.mol.copy())

        self.assertAlmostEqual(rebuilt.omega, 0.37)

    def test_rebuilt_scf_does_not_assign_unset_range_separation(self):
        from gpu4pyscf.dft import roks

        td = FakeNTTDA(self.mol)
        td._scf = roks.ROKS(self.mol, xc='B3LYP')
        self.assertIsNone(td._scf.omega)
        driver = FSSH_NTTDA(td, states=[1, 2])

        rebuilt = driver._new_scf(self.mol.copy())

        self.assertIsNone(rebuilt.omega)


if __name__ == '__main__':
    unittest.main()
