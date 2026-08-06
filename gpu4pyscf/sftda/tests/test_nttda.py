# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from unittest import mock

import cupy as cp
import numpy as np
import os

import gpu4pyscf.sftda.nttda as nttda
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh


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

    def test_kernel_records_solver_stats_with_callback(self):
        """``_nttda_solver_stats`` is populated when ``lr_eigh`` calls back."""
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

        callback_info = {
            'cycle': 0,
            'aop_input_width': 6,
            'subspace_size': 6,
            'residuals': np.array([1e-10, 1e-10, 1e-10, 1e-10]),
            'converged': np.array([True, True, True, True]),
            'energies': np.array([-0.1, 0.0, 5e-8, 0.2]),
        }

        def fake_eigh(aop, x0_arg, precond, **kwargs):
            # Exercise the counting wrapper once.
            _ = aop(x0_arg)
            cb = kwargs.get('callback')
            if cb is not None:
                cb(dict(callback_info))
            return (
                np.ones(4, dtype=bool),
                np.array([-0.1, 0.0, 5e-8, 0.2]),
                cp.asarray(vectors),
            )

        with mock.patch.object(nttda, "lr_eigh", side_effect=fake_eigh):
            with mock.patch.dict(os.environ, {'NTTDA_PROFILE': '1'}):
                td.kernel()

        stats = td._nttda_solver_stats
        self.assertEqual(stats['vind_calls'], 1)
        self.assertEqual(stats['vind_widths'], [6])
        self.assertEqual(stats['total_vector_applications'], 6)
        self.assertEqual(stats['davidson_iterations'], 1)
        self.assertTrue(stats['warm_start'] is False)
        self.assertEqual(stats['initial_subspace_width'], 6)
        self.assertEqual(stats['nroots'], 4)
        self.assertEqual(stats['final_residuals'], [1e-10] * 4)

    def test_kernel_warm_start_flag_and_width(self):
        """Warm-start records the flag and concatenates diagonal guesses."""
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
        warm_x0 = cp.asarray(np.eye(9)[:2])

        def fake_eigh(aop, x0_arg, precond, **kwargs):
            _ = aop(x0_arg)
            cb = kwargs.get('callback')
            if cb is not None:
                cb({
                    'cycle': 0,
                    'aop_input_width': int(x0_arg.shape[0]),
                    'subspace_size': int(x0_arg.shape[0]),
                    'residuals': np.array([1e-10, 1e-10, 1e-10, 1e-10]),
                    'converged': np.array([True, True, True, True]),
                    'energies': np.array([-0.1, 0.0, 5e-8, 0.2]),
                })
            return (
                np.ones(4, dtype=bool),
                np.array([-0.1, 0.0, 5e-8, 0.2]),
                cp.asarray(vectors),
            )

        with mock.patch.object(nttda, "lr_eigh", side_effect=fake_eigh):
            with mock.patch.dict(os.environ, {'NTTDA_PROFILE': '1'}):
                td.kernel(x0=warm_x0)

        stats = td._nttda_solver_stats
        self.assertTrue(stats['warm_start'] is True)
        # warm 2 + diagonal 6 = 8
        self.assertEqual(stats['initial_subspace_width'], 8)

    def test_eigh_callback_records_per_iteration_info(self):
        """``_lr_eig.eigh`` invokes callback with per-iteration metadata."""
        n = 6
        mat = cp.diag(cp.arange(1, n + 1, dtype=float))
        mat[0, 1] = mat[1, 0] = 0.1

        def aop(x):
            return x @ mat

        def precond(dx, e):
            diag = cp.arange(1, n + 1, dtype=float)
            e = cp.atleast_1d(cp.asarray(e))
            return dx / (diag - e[:, None] + 1e-3)

        def pick(w, v, nroots, envs):
            return w, v, np.arange(len(w))

        x0 = cp.eye(n)[:4]
        iterations = []

        def callback(info):
            iterations.append(info)

        conv, e, _ = lr_eigh(
            aop, x0, precond, nroots=4, pick=pick,
            max_cycle=10, tol_residual=1e-8, callback=callback,
        )
        self.assertTrue(len(iterations) > 0)
        required_keys = {
            'cycle', 'aop_input_width', 'subspace_size',
            'residuals', 'converged', 'energies',
        }
        for idx, entry in enumerate(iterations):
            self.assertTrue(required_keys <= set(entry))
            self.assertIsInstance(entry['aop_input_width'], int)
            self.assertEqual(entry['cycle'], idx)
        self.assertTrue(all(conv))

    def test_eigh_callback_none_is_zero_overhead(self):
        """``callback=None`` must not raise and must still converge."""
        n = 6
        mat = cp.diag(cp.arange(1, n + 1, dtype=float))

        def aop(x):
            return x @ mat

        def precond(dx, e):
            diag = cp.arange(1, n + 1, dtype=float)
            e = cp.atleast_1d(cp.asarray(e))
            return dx / (diag - e[:, None] + 1e-3)

        def pick(w, v, nroots, envs):
            return w, v, np.arange(len(w))

        x0 = cp.eye(n)[:3]
        conv, e, _ = lr_eigh(
            aop, x0, precond, nroots=3, pick=pick,
            max_cycle=10, tol_residual=1e-8,
        )
        self.assertTrue(all(conv))


if __name__ == "__main__":
    unittest.main()
