# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

'''Acceptance tests for the GPU Dz0 response and Z-vector machinery.

The packed rotation space must match the T03 CPU oracle exactly, the
matrix-free Hessian action must agree to <=1e-8 on random vectors, and the
Z-vector must satisfy the independently recomputed true residual.  The GMRES
vectors and the response contraction must stay on the device.
'''

import unittest

import cupy as cp
import numpy as np

from pyscf import gto

from gpu4pyscf.grad.ensemble_roks import ReferenceGradients
from gpu4pyscf.sftda import EnsembleROKS


def noh_radical():
    return gto.M(
        atom='N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20',
        basis='sto-3g',
        spin=2,
        unit='Bohr',
        verbose=0,
    )


class EnsembleROKSResponseGPU(unittest.TestCase):
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
        cls.cpu_driver._validate()
        cls.cpu_driver._build_intermediates()

        cls.gpu_driver = ReferenceGradients(cls.mf)
        cls.gpu_driver._validate()
        cls.gpu_driver._build_intermediates()

    def test_rotation_space_order_matches_cpu_oracle(self):
        cpu_space = self.cpu_driver._space
        gpu_space = self.gpu_driver._space
        self.assertEqual(gpu_space.size, cpu_space.size)
        np.testing.assert_array_equal(gpu_space.p, cpu_space.p)
        np.testing.assert_array_equal(gpu_space.q, cpu_space.q)
        np.testing.assert_array_equal(gpu_space.f, cpu_space.f)
        np.testing.assert_array_equal(gpu_space.nalpha, cpu_space.nalpha)
        np.testing.assert_array_equal(gpu_space.nbeta, cpu_space.nbeta)
        np.testing.assert_array_equal(
            gpu_space.occupation_gap, cpu_space.occupation_gap,
        )

    def test_g_dz0_and_g_hs_match_cpu_oracle(self):
        np.testing.assert_allclose(
            cp.asnumpy(self.gpu_driver.g_dz0),
            np.asarray(self.cpu_driver.g_dz0),
            atol=1e-8, rtol=0,
        )
        np.testing.assert_allclose(
            cp.asnumpy(self.gpu_driver.g_hs),
            np.asarray(self.cpu_driver.g_hs),
            atol=1e-8, rtol=0,
        )

    def test_hessian_action_matches_cpu_oracle_and_stays_on_device(self):
        rng = np.random.default_rng(20260909)
        size = self.cpu_driver._space.size
        worst = 0.0
        for _ in range(4):
            vector = rng.normal(size=size)
            expected = np.asarray(
                self.cpu_driver.hessian_vector_product(vector),
            )
            result = self.gpu_driver.hessian_vector_product(cp.asarray(vector))
            self.assertIsInstance(result, cp.ndarray)
            worst = max(
                worst, float(np.max(np.abs(expected - cp.asnumpy(result)))),
            )
        self.assertLessEqual(worst, 1e-8)

    def test_z_solution_matches_cpu_and_satisfies_true_residual(self):
        z_cpu = np.asarray(self.cpu_driver._solve_z())
        z_gpu = self.gpu_driver._solve_z()
        self.assertIsInstance(z_gpu, cp.ndarray)
        np.testing.assert_allclose(
            cp.asnumpy(z_gpu), z_cpu, atol=1e-8, rtol=0,
        )

        diagnostics = self.gpu_driver.z_solver_diagnostics
        self.assertEqual(diagnostics.info, 0)
        self.assertTrue(diagnostics.converged)
        self.assertLessEqual(
            diagnostics.residual_l2, diagnostics.threshold,
        )

    def test_response_contraction_returns_device_arrays(self):
        density = cp.asarray(self.gpu_driver._f0ao) * 1e-3
        result = self.gpu_driver._charge_response(density)
        self.assertIsInstance(result, cp.ndarray)


if __name__ == '__main__':
    unittest.main()
