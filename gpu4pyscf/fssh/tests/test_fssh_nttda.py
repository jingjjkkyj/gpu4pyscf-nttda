# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pickle
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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


class ScriptedSCF:
    def __init__(self, mol, mo_coeff, mo_occ, converged=True):
        self.mol = mol
        self._scripted_coeff = cp.asarray(mo_coeff)
        self._scripted_occ = cp.asarray(mo_occ)
        self._scripted_converged = converged
        self.mo_coeff = None
        self.mo_occ = None
        self.mo_energy = None
        self.e_tot = None
        self.converged = False
        self.cycles = 0
        self.kernel_calls = 0

    def get_occ(self, mo_energy=None, mo_coeff=None):
        return self._scripted_occ.copy()

    def kernel(self, dm0=None):
        self.kernel_calls += 1
        self.mo_coeff = self._scripted_coeff.copy()
        self.mo_energy = cp.arange(self.mo_coeff.shape[1], dtype=cp.float64)
        self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
        self.e_tot = -1.0
        self.converged = self._scripted_converged
        self.cycles = 3
        return self.e_tot


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

    def test_energy_ordering_keeps_low_overlap_as_diagnostic(self):
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
                np.array([[0.1, np.sqrt(0.99)]]),
                np.array([[np.sqrt(0.99), 0.1]]),
            ],
            's_occ': np.eye(1),
            's_vir': np.eye(2),
        }

        driver._track_roots(self.mol, td, tracking)

        self.assertEqual(driver.root_assignment, [0, 1])
        np.testing.assert_allclose(driver.root_overlaps, [0.1, 0.1])
        np.testing.assert_allclose(td.e, [0.1, 0.2])

    def test_overlap_ordering_rejects_unmatched_requested_state(self):
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
                np.array([[0.1, 0.0]]),
                np.array([[0.0, 0.1]]),
            ],
            's_occ': np.eye(1),
            's_vir': np.eye(2),
        }

        with self.assertRaisesRegex(RuntimeError, 'root tracking overlap'):
            driver._track_roots(self.mol, td, tracking)

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

    def test_invalid_scf_mom_overlap_tolerance_is_rejected(self):
        td = FakeNTTDA(self.mol)
        for value in (0, -0.1, 1.1, np.inf, '0.5', True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ValueError, 'scf_mom_overlap_tol'):
                    FSSH_NTTDA(
                        td, states=[1, 2],
                        scf_mom_overlap_tol=value,
                    )

    def test_mom_occupation_preserves_closed_open_virtual_counts(self):
        overlap = cp.eye(4)[:, [0, 3, 2, 1]]
        reference_occ = cp.asarray([2.0, 2.0, 1.0, 0.0])

        occupation = FSSH_NTTDA._mom_occupation(
            overlap, reference_occ,
        )

        np.testing.assert_array_equal(
            cp.asnumpy(occupation), [2.0, 0.0, 1.0, 2.0],
        )
        self.assertEqual(int(cp.count_nonzero(occupation == 2)), 2)
        self.assertEqual(int(cp.count_nonzero(occupation == 1)), 1)
        self.assertEqual(int(cp.count_nonzero(occupation == 0)), 1)

    def _orthonormal_mos(self):
        overlap = self.mol.intor_symmetric('int1e_ovlp')
        values, vectors = np.linalg.eigh(overlap)
        return (vectors / np.sqrt(values)) @ vectors.T

    def _mom_recovery_driver(
            self, normal, recovered=None, state_ordering='overlap'):
        driver = FSSH_NTTDA(
            FakeNTTDA(self.mol),
            states=[1, 2],
            scf_mom_overlap_tol=0.5,
            state_ordering=state_ordering,
        )
        previous_coeff = self._orthonormal_mos()
        driver._last_mf = types.SimpleNamespace(
            mol=self.mol,
            mo_coeff=cp.asarray(previous_coeff),
            mo_occ=cp.asarray([2.0, 0.0]),
        )
        scripted = iter(
            [normal] if recovered is None else [normal, recovered]
        )
        driver._new_scf = types.MethodType(
            lambda self, mol: next(scripted), driver,
        )
        return driver

    def test_continuous_scf_does_not_retry_with_mom(self):
        coeff = self._orthonormal_mos()
        normal = ScriptedSCF(self.mol, coeff, [2.0, 0.0])
        driver = self._mom_recovery_driver(normal)

        result = driver._run_scf_with_mom_recovery(
            self.mol, dm0=cp.eye(2),
        )

        self.assertIs(result, normal)
        self.assertFalse(driver.scf_mom_retry)
        self.assertFalse(
            driver.scf_mom_diagnostics['retry_triggered']
        )
        self.assertGreater(
            driver.scf_mom_diagnostics[
                'minimum_occupied_singular_value'
            ],
            0.999999,
        )

    def test_occupied_virtual_exchange_triggers_mom_retry(self):
        coeff = self._orthonormal_mos()[:, [1, 0]]
        normal = ScriptedSCF(self.mol, coeff, [2.0, 0.0])
        recovered = ScriptedSCF(self.mol, coeff, [2.0, 0.0])
        driver = self._mom_recovery_driver(normal, recovered)

        result = driver._run_scf_with_mom_recovery(
            self.mol, dm0=cp.eye(2),
        )

        self.assertIs(result, recovered)
        self.assertTrue(driver.scf_mom_retry)
        np.testing.assert_array_equal(
            cp.asnumpy(recovered.mo_occ), [0.0, 2.0],
        )
        diagnostics = driver.scf_mom_diagnostics
        self.assertTrue(diagnostics['retry_triggered'])
        self.assertLess(
            diagnostics['minimum_occupied_singular_value'], 1e-12,
        )
        self.assertGreater(
            diagnostics[
                'recovered_minimum_occupied_singular_value'
            ],
            0.999999,
        )
        self.assertEqual(diagnostics['recovered_changed_orbitals'], [])

    def test_failed_mom_retry_is_fail_closed(self):
        coeff = self._orthonormal_mos()[:, [1, 0]]
        normal = ScriptedSCF(self.mol, coeff, [2.0, 0.0])
        failed = ScriptedSCF(
            self.mol, coeff, [2.0, 0.0], converged=False,
        )
        driver = self._mom_recovery_driver(normal, failed)

        with self.assertRaisesRegex(RuntimeError, 'MOM retry'):
            driver._run_scf_with_mom_recovery(
                self.mol, dm0=cp.eye(2),
            )
        self.assertTrue(driver.scf_mom_retry)
        self.assertFalse(
            driver.scf_mom_diagnostics['mom_scf_converged']
        )

    def test_failed_ensemble_scf_uses_second_order_recovery(self):
        coeff = self._orthonormal_mos()
        normal = ScriptedSCF(self.mol, coeff, [2.0, 0.0], converged=False)
        normal.is_ensemble_rks = True
        recovered = ScriptedSCF(self.mol, coeff, [2.0, 0.0])
        recovered.kernel()
        driver = self._mom_recovery_driver(normal)
        with mock.patch.object(
                driver, '_ensemble_newton_recovery', return_value=recovered):
            result = driver._run_scf_with_mom_recovery(
                self.mol, dm0=cp.eye(2),
            )
        self.assertIs(result, recovered)
        self.assertFalse(driver.scf_mom_diagnostics['normal_scf_converged'])
        self.assertTrue(driver.scf_mom_diagnostics['newton_scf_converged'])
        self.assertFalse(driver.scf_mom_retry)

    def test_second_order_recovery_keeps_occupation_continuity_guard(self):
        coeff = self._orthonormal_mos()[:, [1, 0]]
        normal = ScriptedSCF(self.mol, coeff, [2.0, 0.0], converged=False)
        normal.is_ensemble_rks = True
        other_branch = ScriptedSCF(self.mol, coeff, [2.0, 0.0])
        other_branch.kernel()
        failed_mom = ScriptedSCF(self.mol, coeff, [2.0, 0.0], converged=False)
        driver = self._mom_recovery_driver(normal, failed_mom)
        with (
                mock.patch.object(
                    driver, '_ensemble_newton_recovery',
                    return_value=other_branch,
                ),
                mock.patch.object(
                    driver, '_ensemble_newton_continuation',
                    return_value=(None, [{'accepted': False}]),
                )):
            with self.assertRaisesRegex(RuntimeError, 'MOM retry'):
                driver._run_scf_with_mom_recovery(self.mol, dm0=cp.eye(2))
        self.assertTrue(driver.scf_mom_retry)
        self.assertTrue(driver.scf_mom_diagnostics['newton_scf_converged'])
        self.assertFalse(
            driver.scf_mom_diagnostics['newton_continuation_converged']
        )

    def test_second_order_continuation_can_restore_local_continuity(self):
        coeff = self._orthonormal_mos()
        exchanged = coeff[:, [1, 0]]
        normal = ScriptedSCF(self.mol, coeff, [2.0, 0.0], converged=False)
        normal.is_ensemble_rks = True
        direct = ScriptedSCF(self.mol, exchanged, [2.0, 0.0])
        direct.kernel()
        continued = ScriptedSCF(self.mol, coeff, [2.0, 0.0])
        continued.kernel()
        driver = self._mom_recovery_driver(normal)
        path = [{'fraction': 1.0, 'accepted': True}]
        with (
                mock.patch.object(
                    driver, '_ensemble_newton_recovery',
                    return_value=direct,
                ),
                mock.patch.object(
                    driver, '_ensemble_newton_continuation',
                    return_value=(continued, path),
                )):
            result = driver._run_scf_with_mom_recovery(
                self.mol, dm0=cp.eye(2),
            )
        self.assertIs(result, continued)
        self.assertFalse(driver.scf_mom_retry)
        self.assertTrue(
            driver.scf_mom_diagnostics['continuity_via_substeps']
        )
        self.assertEqual(
            driver.scf_mom_diagnostics['newton_continuation_path'], path,
        )

    def test_converged_ensemble_scf_can_use_continuation_before_mom(self):
        coeff = self._orthonormal_mos()
        exchanged = coeff[:, [1, 0]]
        normal = ScriptedSCF(self.mol, exchanged, [2.0, 0.0])
        normal.is_ensemble_rks = True
        continued = ScriptedSCF(self.mol, coeff, [2.0, 0.0])
        continued.kernel()
        driver = self._mom_recovery_driver(normal)
        with mock.patch.object(
                driver, '_ensemble_newton_continuation',
                return_value=(continued, [{'fraction': 1.0, 'accepted': True}])):
            result = driver._run_scf_with_mom_recovery(
                self.mol, dm0=cp.eye(2),
            )
        self.assertIs(result, continued)
        self.assertTrue(
            driver.scf_mom_diagnostics['continuity_via_substeps']
        )
        self.assertFalse(driver.scf_mom_retry)

    def test_energy_ordering_accepts_converged_branch_switch(self):
        coeff = self._orthonormal_mos()[:, [1, 0]]
        candidate = ScriptedSCF(self.mol, coeff, [2.0, 0.0])
        candidate.is_ensemble_rks = True
        driver = self._mom_recovery_driver(
            candidate, state_ordering='energy',
        )
        result = driver._run_scf_with_mom_recovery(
            self.mol, dm0=cp.eye(2),
        )
        self.assertIs(result, candidate)
        self.assertFalse(driver.scf_mom_retry)
        self.assertTrue(
            driver.scf_mom_diagnostics[
                'adiabatic_branch_switch_accepted'
            ]
        )

    def test_failed_direct_newton_can_use_continuation(self):
        coeff = self._orthonormal_mos()
        normal = ScriptedSCF(self.mol, coeff, [2.0, 0.0], converged=False)
        normal.is_ensemble_rks = True
        failed = ScriptedSCF(self.mol, coeff, [2.0, 0.0], converged=False)
        failed.kernel()
        continued = ScriptedSCF(self.mol, coeff, [2.0, 0.0])
        continued.kernel()
        driver = self._mom_recovery_driver(normal)
        with (
                mock.patch.object(
                    driver, '_ensemble_newton_recovery',
                    return_value=failed,
                ),
                mock.patch.object(
                    driver, '_ensemble_newton_continuation',
                    return_value=(continued, [{'fraction': 1.0, 'accepted': True}]),
                )):
            result = driver._run_scf_with_mom_recovery(
                self.mol, dm0=cp.eye(2),
            )
        self.assertIs(result, continued)
        self.assertFalse(
            driver.scf_mom_diagnostics['newton_scf_converged']
        )
        self.assertTrue(
            driver.scf_mom_diagnostics['newton_continuation_converged']
        )

    def test_second_order_recovery_accepts_rotation_without_exchange(self):
        coeff = self._orthonormal_mos()
        normal = ScriptedSCF(self.mol, coeff, [2.0, 0.0], converged=False)
        normal.is_ensemble_rks = True
        cosine = .77
        sine = np.sqrt(1 - cosine**2)
        rotated = coeff @ np.array([[cosine, -sine], [sine, cosine]])
        candidate = ScriptedSCF(self.mol, rotated, [2.0, 0.0])
        candidate.kernel()
        driver = self._mom_recovery_driver(normal)
        with mock.patch.object(
                driver, '_ensemble_newton_recovery', return_value=candidate):
            result = driver._run_scf_with_mom_recovery(
                self.mol, dm0=cp.eye(2),
            )
        self.assertIs(result, candidate)
        diagnostics = driver.scf_mom_diagnostics
        self.assertEqual(diagnostics['changed_orbitals'], [])
        self.assertAlmostEqual(diagnostics['minimum_class_singular_value'], .77)
        self.assertFalse(diagnostics['retry_triggered'])

    def test_only_dfj_surface_is_rejected(self):
        td = FakeNTTDA(self.mol)
        td._scf.only_dfj = True
        with self.assertRaisesRegex(NotImplementedError, 'only_dfj'):
            FSSH_NTTDA(td, states=[1, 2])

    def test_checkpoint_restores_electronic_gauge_cache_and_rng(self):
        from gpu4pyscf.dft import roks
        from gpu4pyscf.sftda import (
            EnsembleROKS,
            NTTDA,
            NTTDA_EnsembleROKS,
        )

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

            ensemble_td = NTTDA_EnsembleROKS(
                EnsembleROKS(self.mol, xc='B3LYP'),
            )
            method_mismatch = FSSH_NTTDA(
                ensemble_td, states=[1, 2],
            )
            source_signature = driver._checkpoint_signature()
            target_signature = method_mismatch._checkpoint_signature()
            self.assertEqual(source_signature['method_id'], 'roks')
            self.assertEqual(
                target_signature['method_id'], 'ensemble_roks',
            )
            self.assertEqual(
                {
                    key: value
                    for key, value in source_signature.items()
                    if key != 'method_id'
                },
                {
                    key: value
                    for key, value in target_signature.items()
                    if key != 'method_id'
                },
            )
            with self.assertRaisesRegex(ValueError, 'does not match'):
                method_mismatch.restore(trajectory)

            checkpoint = driver._checkpoint_path()
            with checkpoint.open('rb') as handle:
                legacy_payload = pickle.load(handle)
            legacy_payload['signature'].pop('scf_mom_overlap_tol')
            legacy_payload['electronic'].pop('scf_mom_retry')
            legacy_payload['electronic'].pop('scf_mom_diagnostics')
            with checkpoint.open('wb') as handle:
                pickle.dump(
                    legacy_payload, handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

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

    def test_rebuilt_scf_preserves_ensemble_roks_method(self):
        from pyscf.sftda import nttda_methods
        from gpu4pyscf.sftda import EnsembleROKS, NTTDA_EnsembleROKS

        mf = EnsembleROKS(self.mol, xc='PBE').density_fit()
        td = NTTDA_EnsembleROKS(mf)
        driver = FSSH_NTTDA(td, states=[1, 2])

        rebuilt_mf = driver._new_scf(self.mol.copy())
        rebuilt_td = driver._new_td(rebuilt_mf)

        self.assertIsInstance(rebuilt_mf, EnsembleROKS)
        self.assertEqual(rebuilt_mf.nopen, mf.nopen)
        self.assertIsNotNone(rebuilt_mf.with_df)
        self.assertEqual(
            nttda_methods.resolve_method(rebuilt_td).id,
            'ensemble_roks',
        )
        self.assertEqual(
            rebuilt_td._nttda_explicit_method,
            'ensemble_roks',
        )

    def test_selected_reference_energy_is_used_for_fssh_surfaces(self):
        from gpu4pyscf.sftda import EnsembleROKS, NTTDA_EnsembleROKS

        mf = EnsembleROKS(self.mol, xc='PBE')
        mf.conv_tol = 1e-10
        mf.converged = True
        mf.mo_coeff = cp.eye(self.mol.nao_nr())
        mf.mo_occ = cp.asarray([2.0, 0.0])
        mf.mo_energy = cp.asarray([-0.5, 0.2])
        mf.e_tot = -1.0
        mf.reference_energy = lambda: -2.0

        td = NTTDA_EnsembleROKS(mf)
        td.conv_tol = 1e-8
        td.e = np.asarray([0.1, 0.2, 0.3])
        td.xy = [
            (cp.ones((1, 1)) * value, 0)
            for value in (1.0, 2.0, 3.0)
        ]
        td.converged = np.ones(3, dtype=bool)

        driver = FSSH_NTTDA(td, states=[1, 2])
        energy, _force, _nacv = driver.calc_electronic(
            self.mol.atom_coords(unit='Bohr'),
            cur_state=2,
            with_nacv=False,
            with_frame=False,
        )

        np.testing.assert_allclose(energy, [-1.9, -1.8], atol=1e-14)

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
