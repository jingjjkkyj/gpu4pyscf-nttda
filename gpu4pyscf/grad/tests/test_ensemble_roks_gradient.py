# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

'''Acceptance tests for the GPU selected-ROKS reference gradient (T06).

The explicit high-spin derivative, the overlap/density-response Fock
derivative ``B`` and the Z-vector correction must match the T03 CPU oracle and
the read-only scientific reference value, the total gradient must reach 1e-7,
the reference energy must be reproduced by a fixed-grid finite difference to
1e-5, and the device path must not fall back to the CPU response builders.
'''

import unittest
from unittest import mock
from types import SimpleNamespace

import cupy as cp
import numpy as np

from pyscf import gto

from gpu4pyscf.grad.ensemble_roks import ReferenceGradients
from gpu4pyscf.sftda import EnsembleROKS

# Read-only scientific reference
# ``pyscf-forge-ensemble-rks-roks-ref-nttda-grad`` @ ``c25eb35`` for
# NOH/STO-3G/B3LYP/grid-level-1 (spin=2, nopen=2, deltaS=-1).
_R_REFERENCE_GRADIENT = np.array([
    [-1.9102818023616535e-15, 3.6345387668032605e+00, 1.0803852977590358e+01],
    [1.7285668590875121e-15, 4.7772097385290507e-01, -1.2354442742060989e+01],
    [4.4743401606794789e-16, -4.1117858373320937e+00, 1.5503280871393283e+00],
])


def noh_radical():
    return gto.M(
        atom='N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20',
        basis='sto-3g',
        spin=2,
        unit='Bohr',
        verbose=0,
    )


class EnsembleROKSGradientGPU(unittest.TestCase):
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
        cls.mf.kernel()
        if not cls.mf.converged:
            raise RuntimeError('EnsembleROKS reference did not converge')

        cls.cpu_driver = cls.mf.to_cpu().nuc_grad_method()
        cls.cpu_driver.conv_tol = 1e-10
        cls.cpu_driver.max_cycle = 100
        cls.cpu_driver.restart = 40

        cls.gpu_driver = ReferenceGradients(cls.mf)
        cls.gpu_driver.conv_tol = 1e-10
        cls.gpu_driver.max_cycle = 100
        cls.gpu_driver.restart = 40

    def test_reference_gradient_matches_oracle_and_read_only_value(self):
        expected = self.cpu_driver.kernel()
        result = self.gpu_driver.kernel()
        np.testing.assert_allclose(
            result, expected, atol=1e-7, rtol=0,
        )
        np.testing.assert_allclose(
            result, _R_REFERENCE_GRADIENT, atol=1e-7, rtol=0,
        )
        diagnostics = self.gpu_driver.z_solver_diagnostics
        self.assertEqual(diagnostics.info, 0)
        self.assertTrue(diagnostics.converged)
        self.assertLessEqual(diagnostics.residual_l2, diagnostics.threshold)

    def test_gradient_components_match_cpu_oracle(self):
        self.cpu_driver.kernel()
        gpu_driver = ReferenceGradients(self.mf)
        gpu_driver.conv_tol = 1e-10
        gpu_driver.max_cycle = 100
        gpu_driver.restart = 40
        gpu_driver.zb_backend = 'legacy'
        gpu_driver.kernel()

        np.testing.assert_allclose(
            cp.asnumpy(gpu_driver.e_hs_unrelaxed),
            np.asarray(self.cpu_driver.e_hs_unrelaxed),
            atol=1e-8, rtol=0,
        )
        np.testing.assert_allclose(
            cp.asnumpy(gpu_driver.b),
            np.asarray(self.cpu_driver.b),
            atol=1e-8, rtol=0,
        )
        np.testing.assert_allclose(
            cp.asnumpy(gpu_driver.z),
            np.asarray(self.cpu_driver.z),
            atol=1e-8, rtol=0,
        )

    def test_direct_zb_matches_materialized_b(self):
        driver = ReferenceGradients(self.mf)
        driver._build_intermediates()
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

    def test_nttda_fusion_uses_canonical_pairs_and_shared_hessian(self):
        from pyscf.grad.nttda import ensemble

        pairs = ensemble.canonical_pairs(SimpleNamespace(
            _scf=SimpleNamespace(mo_occ=cp.asnumpy(self.mf.mo_occ)),
        ))
        legacy = ReferenceGradients(self.mf)
        legacy._build_intermediates()
        native_pairs = tuple(zip(
            legacy._space.p.tolist(),
            legacy._space.q.tolist(),
        ))
        native_lookup = {
            pair: index for index, pair in enumerate(native_pairs)
        }
        expected_rhs = cp.asnumpy(legacy.g_hs)[
            [native_lookup[(p, q)] for p, q, _name in pairs]
        ]

        shared_hessian = {}

        driver = ReferenceGradients(self.mf)
        cache = SimpleNamespace(
            use_selected_reference_hessian=lambda fock, response: (
                shared_hessian.update(fock=fock, response=response)
            ),
        )
        prepared = driver.prepare_nttda_fusion(
            pairs, response_cache=cache,
        )

        self.assertEqual(prepared.pairs, pairs)
        self.assertIs(shared_hessian['response'], driver._charge_response)
        np.testing.assert_allclose(
            shared_hessian['fock'],
            cp.asnumpy(legacy._f0mo),
            atol=1e-10,
            rtol=0,
        )
        np.testing.assert_allclose(
            prepared.orbital_rhs_shift, expected_rhs, atol=1e-12, rtol=0,
        )
        np.testing.assert_allclose(
            prepared.unrelaxed_gradient,
            cp.asnumpy(legacy._high_spin_unrelaxed_gradient()),
            atol=1e-10,
            rtol=0,
        )

    def test_reference_energy_finite_difference_converges(self):
        gradient = ReferenceGradients(self.mf)
        gradient.conv_tol = 1e-10
        gradient.max_cycle = 100
        gradient.restart = 40
        analytic = gradient.kernel()

        coords0 = self.mf.mol.atom_coords()
        grid_coords = np.array(cp.asnumpy(self.mf.grids.coords), copy=True)
        grid_weights = np.array(cp.asnumpy(self.mf.grids.weights), copy=True)

        def reference_energy_at(coords):
            mol = self.mf.mol.copy()
            mol.set_geom_(coords, unit='Bohr')
            displaced = EnsembleROKS(mol).set(
                xc='B3LYP',
                conv_tol=1e-12,
                conv_tol_grad=1e-9,
                max_cycle=200,
                verbose=0,
            )
            displaced.grids.level = 1
            displaced.grids.coords = np.array(grid_coords, copy=True)
            displaced.grids.weights = np.array(grid_weights, copy=True)
            displaced.grids.non0tab = None
            displaced.kernel()
            if not displaced.converged:
                raise RuntimeError('displaced EnsembleROKS did not converge')
            return displaced.reference_energy()

        errors = {}
        for step in (1e-3, 5e-4, 2.5e-4):
            finite_difference = np.zeros((3, 3))
            for atom in range(3):
                for xyz in range(3):
                    plus = coords0.copy()
                    minus = coords0.copy()
                    plus[atom, xyz] += step
                    minus[atom, xyz] -= step
                    finite_difference[atom, xyz] = (
                        reference_energy_at(plus) - reference_energy_at(minus)
                    ) / (2.0 * step)
            errors[step] = float(
                np.max(np.abs(finite_difference - analytic))
            )

        self.assertLessEqual(errors[5e-4], 1e-5)
        self.assertLessEqual(errors[2.5e-4], 1e-5)
        self.assertLessEqual(abs(errors[5e-4] - errors[2.5e-4]), 1e-5)
        self.assertLess(errors[2.5e-4], errors[5e-4])

    def test_gradient_does_not_recompute_cpu_response(self):
        from pyscf.hessian import rhf as cpu_rhf_hess
        from pyscf.hessian import rks as cpu_rks_hess

        def _forbidden(*args, **kwargs):
            raise AssertionError(
                'GPU reference gradient must not call the CPU response builder'
            )

        with mock.patch.object(
            cpu_rks_hess, '_get_vxc_deriv1', _forbidden,
        ), mock.patch.object(
            cpu_rhf_hess, '_get_jk', _forbidden,
        ):
            result = ReferenceGradients(self.mf).kernel()
        np.testing.assert_allclose(
            result, _R_REFERENCE_GRADIENT, atol=1e-7, rtol=0,
        )

    def test_gradient_does_not_use_cpu_scipy_solver(self):
        from scipy.sparse import linalg as cpu_spla

        def _forbidden(*args, **kwargs):
            raise AssertionError(
                'GPU reference Z-vector must not use the CPU SciPy solver'
            )

        with mock.patch.object(cpu_spla, 'gmres', _forbidden), \
                mock.patch.object(cpu_spla, 'spsolve', _forbidden), \
                mock.patch.object(cpu_spla, 'bicgstab', _forbidden), \
                mock.patch.object(cpu_spla, 'bicg', _forbidden), \
                mock.patch.object(cpu_spla, 'cg', _forbidden):
            result = ReferenceGradients(self.mf).kernel()
        np.testing.assert_allclose(
            result, _R_REFERENCE_GRADIENT, atol=1e-7, rtol=0,
        )


if __name__ == '__main__':
    unittest.main()
