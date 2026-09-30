# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

'''Acceptance tests for the GPU EnsembleROKS selected reference energy.

``gpu4pyscf.sftda.EnsembleROKS`` keeps the plain ``EnsembleRKS`` SCF objective
and only changes the reference zero used by NTTDA total energies.  These tests
pin the GPU fixed-orbital high-spin ROKS energy against the T02 CPU oracle
(``-123.76580902817798`` Ha for NOH/STO-3G/B3LYP/grid-level-1), check that the
reference evaluation never runs an SCF kernel, and verify that the DF wrapper,
``to_cpu`` preserves the selected reference semantics.
'''

import unittest

import cupy as cp
import numpy as np

from pyscf import gto
from pyscf.sftda import EnsembleROKS as CPUEnsembleROKS

from gpu4pyscf.dft import roks as gpu_roks

from gpu4pyscf.sftda import EnsembleROKS, NTTDA


def noh_radical(basis='sto-3g'):
    return gto.M(
        atom='N 0 0 0; O 0 0 1.20; H 0 0.90 -0.20',
        basis=basis,
        spin=2,
        unit='Bohr',
        verbose=0,
    )


def converged_ensemble_roks(basis='sto-3g', density_fit=False,
                            auxbasis=None):
    mf = EnsembleROKS(noh_radical(basis)).set(
        xc='B3LYP',
        conv_tol=1e-12,
        conv_tol_grad=1e-9,
        max_cycle=200,
        verbose=0,
    )
    mf.grids.level = 1
    if density_fit:
        mf = mf.density_fit(auxbasis=auxbasis)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('EnsembleROKS reference did not converge')
    return mf


class EnsembleROKSGPU(unittest.TestCase):
    def test_reference_energy_matches_t02_cpu_oracle(self):
        mf = converged_ensemble_roks()
        self.assertEqual(
            mf.reference_energy_semantics,
            'roks_energy_on_ensemble_rks_orbitals',
        )
        self.assertTrue(mf.is_ensemble_rks)
        self.assertFalse(mf.reference_energy_stationary)
        self.assertAlmostEqual(
            mf.reference_energy(), -123.76580902817798, places=8,
        )
        # The selected high-spin reference is not the ensemble SCF objective.
        self.assertGreater(abs(mf.reference_energy() - mf.e_tot), 1e-8)

    def test_reference_energy_runs_no_second_scf_and_leaves_state_untouched(self):
        mf = converged_ensemble_roks()
        mo_coeff = cp.asnumpy(cp.asarray(mf.mo_coeff)).copy()
        mo_occ = cp.asnumpy(cp.asarray(mf.mo_occ)).copy()
        mo_energy = cp.asnumpy(cp.asarray(mf.mo_energy)).copy()
        e_tot = float(mf.e_tot)

        original_kernel = gpu_roks.ROKS.kernel
        calls = []

        def counting_kernel(self, *args, **kwargs):
            calls.append(1)
            return original_kernel(self, *args, **kwargs)

        gpu_roks.ROKS.kernel = counting_kernel
        try:
            first = mf.reference_energy()
            second = mf.reference_energy()
        finally:
            gpu_roks.ROKS.kernel = original_kernel

        self.assertEqual(calls, [])
        self.assertEqual(first, second)
        np.testing.assert_array_equal(
            cp.asnumpy(cp.asarray(mf.mo_coeff)), mo_coeff,
        )
        np.testing.assert_array_equal(
            cp.asnumpy(cp.asarray(mf.mo_occ)), mo_occ,
        )
        np.testing.assert_array_equal(
            cp.asnumpy(cp.asarray(mf.mo_energy)), mo_energy,
        )
        self.assertEqual(float(mf.e_tot), e_tot)

    def test_nttda_reference_and_total_energy_dispatch(self):
        mf = converged_ensemble_roks()
        td = NTTDA(mf).set(
            deltaS=-1, nstates=1, conv_tol=1e-8, max_cycle=200, verbose=0,
        )

        with self.assertRaises(RuntimeError):
            _ = td.e_tot

        td.run()
        self.assertTrue(np.all(np.asarray(td.converged, dtype=bool)))
        self.assertAlmostEqual(td.reference_energy(), mf.reference_energy())
        np.testing.assert_allclose(
            np.asarray(td.e_tot),
            mf.reference_energy() + np.asarray(td.e),
            atol=1e-13, rtol=0,
        )

    def test_to_cpu_preserves_ensemble_roks_semantics(self):
        mf = converged_ensemble_roks()
        cpu = mf.to_cpu()
        self.assertIsInstance(cpu, CPUEnsembleROKS)
        self.assertEqual(
            cpu.reference_energy_semantics,
            'roks_energy_on_ensemble_rks_orbitals',
        )
        self.assertFalse(cpu.reference_energy_stationary)
        self.assertAlmostEqual(
            cpu.reference_energy(), mf.reference_energy(), places=8,
        )

    def test_density_fit_reference_energy_uses_same_auxbasis(self):
        auxbasis = 'def2-universal-jkfit'
        mf = converged_ensemble_roks(
            basis='def2-svp', density_fit=True, auxbasis=auxbasis,
        )
        self.assertEqual(type(mf).__name__, 'DFEnsembleROKS')
        self.assertEqual(mf.with_df.auxbasis, auxbasis)
        self.assertEqual(
            mf.reference_energy_semantics,
            'roks_energy_on_ensemble_rks_orbitals',
        )

        captured = []
        original_density_fit = gpu_roks.ROKS.density_fit

        def recording_density_fit(self, *args, **kwargs):
            captured.append(kwargs.get('auxbasis'))
            return original_density_fit(self, *args, **kwargs)

        gpu_roks.ROKS.density_fit = recording_density_fit
        try:
            value = mf.reference_energy()
        finally:
            gpu_roks.ROKS.density_fit = original_density_fit

        self.assertEqual(captured, [auxbasis])
        evaluator = gpu_roks.ROKS(mf.mol, xc=mf.xc).density_fit(
            auxbasis=auxbasis,
        )
        evaluator.verbose = 0
        evaluator.grids = mf.grids
        dm = evaluator.make_rdm1(mf.mo_coeff, mf.mo_occ)
        expected = evaluator.energy_tot(
            dm=dm,
            h1e=evaluator.get_hcore(mf.mol),
            vhf=evaluator.get_veff(mf.mol, dm),
        )
        self.assertAlmostEqual(value, float(expected), places=10)

    def test_density_fit_to_cpu_and_reset_preserve_semantics(self):
        mf = converged_ensemble_roks(
            basis='def2-svp', density_fit=True,
            auxbasis='def2-universal-jkfit',
        )
        cpu = mf.to_cpu()
        self.assertIsInstance(cpu, CPUEnsembleROKS)
        self.assertEqual(type(cpu).__name__, 'DFEnsembleROKS')
        self.assertIsNotNone(cpu.with_df)
        self.assertEqual(
            cpu.reference_energy_semantics,
            'roks_energy_on_ensemble_rks_orbitals',
        )

        mf.reset(mf.mol)
        self.assertEqual(type(mf).__name__, 'DFEnsembleROKS')
        self.assertEqual(
            mf.reference_energy_semantics,
            'roks_energy_on_ensemble_rks_orbitals',
        )



if __name__ == '__main__':
    unittest.main()
