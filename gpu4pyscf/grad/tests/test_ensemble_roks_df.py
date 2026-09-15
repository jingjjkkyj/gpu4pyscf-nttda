# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

'''Acceptance tests for the DF EnsembleROKS reference energy and response (T08).

The high-spin evaluator and the charge-response object must share the same
density-fitting auxiliary basis as the differentiated DF energy, the DF
Hessian action must be reproduced by a fixed-orbital finite difference, the
high-spin directional derivative must be reproduced by a DF energy difference
to <=1e-5, and the Z-vector must satisfy the independently recomputed true
residual.  The DF/non-DF numerical difference is never used to justify a sign
change.
'''

import unittest

import cupy as cp
import numpy as np
from scipy.linalg import expm

from pyscf import gto

from gpu4pyscf.grad.ensemble_roks import ReferenceGradients
from gpu4pyscf.sftda import EnsembleROKS

AUXBASIS = 'def2-universal-jkfit'


def noh_radical():
    return gto.M(
        atom='N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20',
        basis='sto-3g',
        spin=2,
        unit='Bohr',
        verbose=0,
    )


class EnsembleROKSDFGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mf = EnsembleROKS(noh_radical()).set(
            xc='B3LYP',
            conv_tol=1e-12,
            conv_tol_grad=1e-9,
            max_cycle=200,
            verbose=0,
        )
        cls.mf.grids.level = 1
        cls.mf = cls.mf.density_fit(auxbasis=AUXBASIS)
        cls.mf.kernel()
        if not cls.mf.converged:
            raise RuntimeError('DF EnsembleROKS reference did not converge')

        cls.gpu_driver = ReferenceGradients(cls.mf)
        cls.gpu_driver.conv_tol = 1e-10
        cls.gpu_driver.max_cycle = 100
        cls.gpu_driver.restart = 40
        # Reuse geometry-fixed intermediates for the response checks.
        cls.gpu_driver._build_intermediates()

    def test_high_spin_and_response_share_df_auxbasis(self):
        with_df = getattr(self.mf, 'with_df', None)
        self.assertIsNotNone(with_df)
        reference_bas = np.asarray(with_df.auxmol._bas)

        for mf in (self.gpu_driver._charge_mf, self.gpu_driver._hs_mf):
            self.assertIsNotNone(getattr(mf, 'with_df', None))
            np.testing.assert_array_equal(mf.with_df.auxmol._bas, reference_bas)
            self.assertFalse(getattr(mf, 'only_dfj', False))

    def test_only_dfj_reference_is_rejected(self):
        mf_only_dfj = self.mf.copy()
        mf_only_dfj.only_dfj = True
        driver = ReferenceGradients(mf_only_dfj)
        with self.assertRaises(NotImplementedError):
            driver._validate()

    def test_df_gradient_entrypoint(self):
        driver = ReferenceGradients(self.mf)
        result = driver.kernel()
        self.assertEqual(result.shape, (self.mf.mol.natm, 3))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(driver.z_solver_diagnostics.converged)

    def test_df_hessian_action_finite_difference_converges(self):
        driver = self.gpu_driver
        space = driver._space
        vector = np.random.default_rng(20260910).normal(size=space.size)
        action = cp.asnumpy(driver.hessian_vector_product(cp.asarray(vector)))
        rotation = space.unpack(vector)

        mol = self.mf.mol
        c0 = driver._c0
        f_occ = cp.asarray(space.f)
        gap = cp.asarray(space.occupation_gap)
        hcore = driver._hcore
        charge_mf = driver._charge_mf

        def orbital_gradient(step):
            unitary = expm(step * rotation)
            mo = c0 @ cp.asarray(unitary)
            dm = (mo * f_occ) @ mo.conj().T
            fock_ao = hcore + charge_mf.get_veff(mol, dm)
            return cp.asnumpy(
                2.0 * gap * space.pack(mo.conj().T @ fock_ao @ mo)
            )

        steps = (1e-3, 5e-4, 2.5e-4)
        finite_difference = {}
        errors = {}
        for step in steps:
            finite_difference[step] = (
                orbital_gradient(step) - orbital_gradient(-step)
            ) / (2.0 * step)
            errors[step] = float(
                np.max(np.abs(finite_difference[step] - action))
            )

        self.assertLess(errors[2.5e-4], errors[5e-4])
        self.assertLess(errors[5e-4], errors[1e-3])

        richardson = (
            4.0 * finite_difference[2.5e-4] - finite_difference[5e-4]
        ) / 3.0
        self.assertLessEqual(
            float(np.max(np.abs(richardson - action))), 1e-5,
        )

    def test_df_high_spin_directional_derivative(self):
        driver = self.gpu_driver
        space = driver._space
        vector = np.random.default_rng(20260911).normal(size=space.size)
        rotation = space.unpack(vector)
        analytic = float(np.dot(cp.asnumpy(driver.g_hs), vector))

        mol = self.mf.mol
        c0 = driver._c0
        f_occ = cp.asarray(space.f)
        hcore = driver._hcore
        hs_mf = driver._hs_mf

        def high_spin_energy(step):
            unitary = expm(step * rotation)
            mo = c0 @ cp.asarray(unitary)
            dm = hs_mf.make_rdm1(mo, f_occ)
            return float(
                hs_mf.energy_tot(
                    dm=dm, h1e=hcore, vhf=hs_mf.get_veff(mol, dm),
                )
            )

        steps = (1e-3, 5e-4, 2.5e-4)
        finite_difference = {}
        errors = {}
        for step in steps:
            finite_difference[step] = (
                high_spin_energy(step) - high_spin_energy(-step)
            ) / (2.0 * step)
            errors[step] = abs(finite_difference[step] - analytic)

        self.assertLessEqual(errors[2.5e-4], 1e-5)
        self.assertLess(errors[2.5e-4], errors[5e-4])
        self.assertLess(errors[5e-4], errors[1e-3])

        richardson = (
            4.0 * finite_difference[2.5e-4] - finite_difference[5e-4]
        ) / 3.0
        self.assertLessEqual(abs(richardson - analytic), 1e-5)

    def test_df_z_solution_satisfies_true_residual(self):
        z = self.gpu_driver._solve_z()
        self.assertIsInstance(z, cp.ndarray)

        diagnostics = self.gpu_driver.z_solver_diagnostics
        self.assertEqual(diagnostics.info, 0)
        self.assertTrue(diagnostics.converged)
        self.assertLessEqual(diagnostics.residual_l2, diagnostics.threshold)

        residual = (
            cp.asarray(self.gpu_driver.g_hs)
            - self.gpu_driver.hessian_vector_product(z)
        )
        self.assertLessEqual(
            float(cp.linalg.norm(residual)), diagnostics.threshold,
        )

    def test_df_response_contraction_stays_on_device(self):
        density = cp.asarray(self.gpu_driver._f0ao) * 1e-3
        result = self.gpu_driver._charge_response(density)
        self.assertIsInstance(result, cp.ndarray)

    def test_df_direct_zb_matches_materialized_b(self):
        driver = self.gpu_driver
        z = cp.asarray(
            np.random.default_rng(20260914).normal(size=driver._space.size),
        )
        expected = cp.einsum(
            'i,iax->ax', z, driver._build_b(), optimize=True,
        )
        result = driver._contract_z_b(z)
        np.testing.assert_allclose(
            cp.asnumpy(result), cp.asnumpy(expected), atol=2e-8, rtol=0,
        )

    def test_df_direct_skeleton_components(self):
        from gpu4pyscf.df.hessian import rhf as df_rhf_hess
        from gpu4pyscf.df.hessian import rks as df_rks_hess
        from gpu4pyscf.grad import rhf as gpu_rhf_grad
        from gpu4pyscf.hessian import rks as gpu_rks_hess

        driver = self.gpu_driver
        space = driver._space
        z = cp.asarray(
            np.random.default_rng(20260914).normal(size=space.size),
        )
        occupied = np.where(space.f > 0.0)[0]
        positions = np.searchsorted(occupied, space.q)
        weight = cp.zeros((space.f.size, len(occupied)))
        weight[space.p, positions] = (
            2.0 * cp.asarray(space.occupation_gap) * z
        )
        driver._contract_df_fock_skeleton(
            weight, driver._c0[:, occupied], cp.asarray(space.f),
        )
        components = driver._z_b_skeleton_components

        charge = driver._charge_mf
        hessobj = df_rks_hess.Hessian(charge)
        vj, vk = df_rhf_hess._get_jk_ip(
            hessobj, driver._c0, cp.asarray(space.f),
        )
        _omega, _alpha, hybrid = charge._numint.rsh_and_hybrid_coeff(
            charge.xc, spin=charge.mol.spin,
        )
        expected_jk = cp.einsum(
            'pq,axpq->ax', weight, vj - 0.5 * hybrid * vk,
        )
        expected_core = cp.einsum(
            'pq,axpq->ax', weight,
            gpu_rhf_grad.get_grad_hcore(
                charge.nuc_grad_method(), driver._c0, cp.asarray(space.f),
            ),
        )
        expected_xc = cp.einsum(
            'pq,axpq->ax', weight,
            gpu_rks_hess._get_vxc_deriv1(
                hessobj, driver._c0, cp.asarray(space.f), 2000,
            ),
        )
        for name, expected in (
                ('core', expected_core), ('df_jk', expected_jk),
                ('xc', expected_xc)):
            error = np.max(np.abs(
                components[name] - cp.asnumpy(expected)
            ))
            self.assertLess(error, 1e-9)


class EnsembleROKSDFDefaultAuxbasisGPU(unittest.TestCase):
    '''R2: ``density_fit()`` without an explicit auxbasis must stay DF.

    ``with_df.auxbasis is None`` means "auto-select", not "no density
    fitting".  The response and high-spin objects must therefore keep their DF
    objects, and the reference-energy evaluator must use the same DF setup.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mf = EnsembleROKS(noh_radical()).set(
            xc='B3LYP',
            conv_tol=1e-12,
            conv_tol_grad=1e-9,
            max_cycle=200,
            verbose=0,
        )
        cls.mf.grids.level = 1
        cls.mf = cls.mf.density_fit()
        cls.mf.kernel()
        if not cls.mf.converged:
            raise RuntimeError('DF EnsembleROKS reference did not converge')

        cls.gpu_driver = ReferenceGradients(cls.mf)
        cls.gpu_driver.conv_tol = 1e-10
        cls.gpu_driver._build_intermediates()

    def test_default_auxbasis_scf_is_df(self):
        self.assertIsNotNone(getattr(self.mf, 'with_df', None))
        self.assertIsNone(getattr(self.mf.with_df, 'auxbasis', None))

    def test_default_auxbasis_response_objects_stay_df(self):
        for mf in (self.gpu_driver._charge_mf, self.gpu_driver._hs_mf):
            self.assertIsNotNone(getattr(mf, 'with_df', None))
            self.assertFalse(getattr(mf, 'only_dfj', False))

    def test_default_auxbasis_reference_energy_evaluator_is_df(self):
        # The reference-energy gradient path must see a DF charge object; the
        # DF auxmol is what the response and skeleton integrals are built from.
        self.assertIsNotNone(self.gpu_driver._charge_mf.with_df.auxmol)

    def test_default_auxbasis_entrypoint(self):
        driver = ReferenceGradients(self.mf)
        result = driver.kernel()
        self.assertEqual(result.shape, (self.mf.mol.natm, 3))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(driver.z_solver_diagnostics.converged)


if __name__ == '__main__':
    unittest.main()
