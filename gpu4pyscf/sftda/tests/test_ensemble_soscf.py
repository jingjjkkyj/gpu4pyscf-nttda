# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

import unittest

import cupy as cp
import numpy as np
from scipy.linalg import expm, expm_frechet
from pyscf import gto

from gpu4pyscf.sftda import EnsembleRKS


class EnsembleSOSCF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.M(
            atom='C 0 0 0; H 0 0 2; H 0 1.7 -.5',
            unit='Bohr', basis='sto-3g', spin=2, verbose=0,
        )
        cls.mf = EnsembleRKS(mol, xc='PBE0').density_fit()
        cls.mf.grids.level = 0
        cls.mf.conv_tol = 1e-12
        cls.mf.conv_tol_grad = 1e-9
        cls.mf.kernel()
        if not cls.mf.converged:
            raise RuntimeError('SCF test reference did not converge')

    def test_hessian_matches_energy_gradient_difference(self):
        mf = self.mf
        solver = mf.newton()
        occ = cp.asnumpy(mf.mo_occ)
        mask = occ[None, :] > occ[:, None]
        rng = np.random.default_rng(62)
        k = np.zeros((len(occ), len(occ)))
        k[mask] = rng.normal(size=mask.sum()) * .03
        k -= k.T
        c = mf.mo_coeff @ cp.asarray(expm(k))
        g, hop, _ = solver.gen_g_hop(c, mf.mo_occ)
        self.assertEqual(g.size, mask.sum())
        np.testing.assert_allclose(
            cp.asnumpy(g), cp.asnumpy(mf.get_grad(c, mf.mo_occ)),
            atol=1e-12, rtol=0,
        )
        # Check each of closed/open, closed/virtual and open/virtual.
        hcore = mf.get_hcore()
        for low, high in ((1, 2), (0, 2), (0, 1)):
            with self.subTest(low=low, high=high):
                direction = np.zeros_like(k)
                subspace = (occ[:, None] == low) & (occ[None, :] == high)
                direction[subspace] = rng.normal(size=subspace.sum())
                direction -= direction.T
                direction /= np.linalg.norm(direction)

                def gradient_at(t):
                    generator = t * direction
                    u = expm(generator)
                    ct = c @ cp.asarray(u)
                    dm = mf.make_rdm1(ct, mf.mo_occ)
                    f = hcore + mf.get_veff(mf.mol, dm)
                    f0 = cp.asnumpy(c.T @ f @ c)
                    du = 2 * (f0 @ u) * occ
                    dk = expm_frechet(
                        generator.T, du, compute_expm=False,
                    )
                    return .5 * (dk - dk.T)[mask]

                eps = 1e-4
                fd = (gradient_at(eps) - gradient_at(-eps)) / (2 * eps)
                np.testing.assert_allclose(
                    cp.asnumpy(hop(cp.asarray(direction[mask]))),
                    fd, atol=2e-8, rtol=2e-7,
                )

        x = cp.asarray(rng.normal(size=mask.sum()))
        y = cp.asarray(rng.normal(size=mask.sum()))
        self.assertAlmostEqual(float(x @ hop(y)), float(y @ hop(x)), places=9)

    def test_newton_preserves_ensemble_and_recovers_reference(self):
        mf = self.mf
        solver = mf.newton()
        solver.conv_tol_grad = 1e-8
        solver.max_cycle = 30
        # Rotate an open orbital into a virtual orbital: the old RHF-only
        # Hessian could neither represent nor remove this perturbation.
        k = np.zeros((mf.mo_occ.size, mf.mo_occ.size))
        p = int(cp.where(mf.mo_occ == 0)[0][0])
        q = int(cp.where(mf.mo_occ == 1)[0][0])
        k[p, q], k[q, p] = .02, -.02
        solver.kernel(mf.mo_coeff @ cp.asarray(expm(k)), mf.mo_occ.copy())
        self.assertTrue(solver.converged)
        self.assertLess(float(cp.linalg.norm(solver.get_grad(
            solver.mo_coeff, solver.mo_occ))), 1e-8)
        self.assertAlmostEqual(solver.e_tot, mf.e_tot, places=10)
        np.testing.assert_array_equal(cp.asnumpy(solver.mo_occ), cp.asnumpy(mf.mo_occ))
        np.testing.assert_allclose(
            cp.asnumpy(solver.mo_coeff.T @ mf.get_ovlp() @ solver.mo_coeff),
            np.eye(mf.mo_occ.size), atol=1e-11, rtol=0,
        )


if __name__ == '__main__':
    unittest.main()
