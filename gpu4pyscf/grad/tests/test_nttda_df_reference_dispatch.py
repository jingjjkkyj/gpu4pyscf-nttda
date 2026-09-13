# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

'''R3: the NTTDA DF total gradient must dispatch the selected-reference driver.

``EnsembleROKS`` inherits ``is_ensemble_rks=True``, so the ordinary DF branch of
``gpu4pyscf.grad.nttda.Gradients.grad_nuc`` would silently select
``gpu4pyscf.df.grad.rks.Gradients`` and bypass the reference-energy response.
These tests pin the selected-reference dispatch: the ordinary DF-RKS driver
must not be constructed, and the selected-reference DF driver must include
the nonstationary reference response.
'''

import unittest
from unittest import mock

import cupy as cp
import numpy as np

from pyscf import gto

from gpu4pyscf.grad import nttda
from gpu4pyscf.sftda import EnsembleROKS, NTTDA


def noh_radical():
    return gto.M(
        atom='N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20',
        basis='sto-3g',
        spin=2,
        unit='Bohr',
        verbose=0,
    )


class NTTDADFReferenceDispatchGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nttda._import_forge()
        cls.mf = EnsembleROKS(noh_radical()).set(
            xc='B3LYP',
            conv_tol=1e-12,
            conv_tol_grad=1e-9,
            max_cycle=200,
            verbose=0,
        )
        cls.mf.grids.level = 1
        cls.mf = cls.mf.density_fit(auxbasis='def2-universal-jkfit')
        cls.mf.kernel()
        if not cls.mf.converged:
            raise RuntimeError('DF EnsembleROKS reference did not converge')
        cls.td = NTTDA(cls.mf).set(
            deltaS=-1, nstates=1, conv_tol=1e-9, max_cycle=200, verbose=0,
        ).run()
        if not np.all(np.asarray(cls.td.converged, dtype=bool)):
            raise RuntimeError('DF NTTDA did not converge')

    def test_reference_is_selected_df_object(self):
        driver = self.td.Gradients()
        self.assertTrue(driver._with_df)
        self.assertIs(
            getattr(driver._gmf, 'reference_energy_stationary', None), False,
        )
        self.assertTrue(getattr(driver._gmf, 'is_ensemble_rks', False))

    def test_grad_nuc_uses_selected_reference_not_plain_df_rks(self):
        driver = self.td.Gradients()
        # The selected-reference driver does construct the ordinary DF-RKS
        # gradient internally, but only as a core-Hamiltonian helper through
        # ``get_grad_hcore``.  The plain driver must never run its own
        # ``kernel`` (that is what would bypass the nonstationary reference
        # response), so spy on ``kernel`` rather than on the class.
        with mock.patch(
            'gpu4pyscf.df.grad.rks.Gradients.kernel',
        ) as plain_df_rks_kernel:
            result = driver.grad_nuc()
        plain_df_rks_kernel.assert_not_called()
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(driver.reference_z_solver_diagnostics['converged'])

    def test_selected_driver_is_the_reference_gradient(self):
        driver = self.td.Gradients()
        reference_driver = driver._gmf.nuc_grad_method()
        self.assertEqual(type(reference_driver).__name__, 'ReferenceGradients')
        expected = reference_driver.kernel()
        np.testing.assert_allclose(driver.grad_nuc(), expected, atol=1e-9, rtol=0)


if __name__ == '__main__':
    unittest.main()
