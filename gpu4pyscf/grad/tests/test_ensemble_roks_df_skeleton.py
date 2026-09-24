'''DF Fock derivatives against nuclear differences of the same fitted J/K.

AO density and both output MO transforms stay fixed during displacement.
Every displaced calculation rebuilds its auxiliary basis and Coulomb metric.
No exact-integral result is used as an oracle for a fitted derivative.
'''

import unittest

import cupy as cp
import numpy as np

from pyscf import gto, scf
from gpu4pyscf.dft import rks
from gpu4pyscf.df.hessian import rhf as df_rhf_hess
from gpu4pyscf.df.grad.ensemble_roks import fractional_rks_fock_skeleton


class EnsembleROKSDFSkeletonGPU(unittest.TestCase):
    @staticmethod
    def reference(closed_shell=False):
        mol = gto.M(
            atom='C 0.1 0.2 0; H 0 0 2.0; H 0 1.7 -0.5',
            basis='sto-3g', spin=2, unit='Bohr', verbose=0,
        )
        mf = rks.RKS(mol, xc='HF').density_fit(
            auxbasis='def2-universal-jkfit',
        )
        eig, vec = np.linalg.eigh(mol.intor_symmetric('int1e_ovlp'))
        orthogonal = (vec / np.sqrt(eig)) @ vec.T
        rotation, _ = np.linalg.qr(
            np.random.default_rng(20260912).normal(size=orthogonal.shape),
        )
        mf.mo_coeff = cp.asarray(orthogonal @ rotation)
        occ = np.zeros(mol.nao)
        occ[:3] = 2
        occ[3:5] = (2, 0) if closed_shell else (1, 1)
        mf.mo_occ = cp.asarray(occ)
        return mf

    @staticmethod
    def numerical_jk(mf, omega=None, step=1e-4):
        coeff, occ = mf.mo_coeff, mf.mo_occ
        occupied = coeff[:, occ > 0]
        density = (coeff * occ) @ coeff.T
        coords = mf.mol.atom_coords()
        result = np.empty((2, mf.mol.natm, 3, coeff.shape[1], occupied.shape[1]))
        for atom in range(mf.mol.natm):
            for xyz in range(3):
                values = []
                for sign in (1, -1):
                    displaced = coords.copy()
                    displaced[atom, xyz] += sign * step
                    mol = mf.mol.copy().set_geom_(displaced, unit='Bohr')
                    probe = rks.RKS(mol, xc='HF').density_fit(
                        auxbasis=mf.with_df.auxbasis,
                    )
                    jk = probe.get_jk(mol, density, hermi=1, omega=omega)
                    values.append(np.asarray([
                        cp.asnumpy(coeff.T @ matrix @ occupied) for matrix in jk
                    ]))
                result[:, atom, xyz] = (values[0] - values[1]) / (2 * step)
        return result

    def compare_jk(self, closed_shell=False, omega=None):
        mf = self.reference(closed_shell)
        step = 2e-4 if omega is not None else 1e-4
        numerical = self.numerical_jk(mf, omega, step=step)
        analytic = df_rhf_hess._get_jk_ip(
            df_rhf_hess.Hessian(mf), mf.mo_coeff, mf.mo_occ, omega=omega,
        )
        for name, actual, expected in zip(('J', 'K'), analytic, numerical):
            with self.subTest(operator=name):
                np.testing.assert_allclose(
                    cp.asnumpy(actual), expected, atol=2e-6, rtol=0,
                )

    def test_fractional_jk_nuclear_derivative(self):
        self.compare_jk()

    def test_closed_shell_jk_nuclear_derivative(self):
        self.compare_jk(closed_shell=True)

    def test_fractional_long_range_jk_nuclear_derivative(self):
        self.compare_jk(omega=0.3)

    def compare_ledger_jk(self, omega=None):
        from pyscf.grad.nttda.derivative_jk import _JKDerivativeLedger
        from gpu4pyscf.grad.nttda_ledger import DFLedgerBackend

        mf = self.reference()
        coeff, occ = mf.mo_coeff, mf.mo_occ
        cocc = coeff[:, occ > 0]
        nocc = cocc.shape[1]
        weight = cp.asarray(np.random.default_rng(20260915).normal(
            size=(coeff.shape[1], nocc),
        ))
        probe = coeff @ weight @ cocc.T
        probe = 0.5 * (probe + probe.T)
        probe_host = cp.asnumpy(probe)
        spin_density = 0.5 * cp.asnumpy((coeff * occ) @ coeff.T)
        atoms = tuple(range(mf.mol.natm))

        vj, vk = df_rhf_hess._get_jk_ip(
            df_rhf_hess.Hessian(mf), coeff, occ, omega=omega,
        )
        ledger = _JKDerivativeLedger()
        if omega is None:
            ledger.add('j', 'jk', (
                (probe_host, spin_density, 1.0, None),
                (probe_host, spin_density, 1.0, None),
            ))
        ledger.add('k', 'jk', (
            (0.5 * probe_host, spin_density, -1.0, omega),
            (0.5 * probe_host, spin_density, -1.0, omega),
        ))
        backend = DFLedgerBackend(mf)
        backend.output_backend = 'slot_aware'
        contracted = backend(
            ledger._terms, mf.mol, atoms, slots=('jk',),
        )['jk']
        if omega is None:
            expected = cp.einsum('pq,axpq->ax', weight, vj)
            expected -= 0.5 * cp.einsum('pq,axpq->ax', weight, vk)
        else:
            expected = -0.5 * cp.einsum('pq,axpq->ax', weight, vk)
        np.testing.assert_allclose(
            contracted, cp.asnumpy(expected), atol=1e-8, rtol=0,
        )
        self.assertEqual(backend.stats['output_kernel_tasks'], 1)

    def test_fractional_jk_ledger_contraction(self):
        self.compare_ledger_jk()

    def test_fractional_long_range_jk_ledger_contraction(self):
        self.compare_ledger_jk(omega=0.3)

    def test_fractional_fock_skeleton(self):
        mf = self.reference()
        coeff, occupied = mf.mo_coeff, mf.mo_coeff[:, mf.mo_occ > 0]
        numerical_j, numerical_k = self.numerical_jk(mf)
        # One-electron derivative is independent of density fitting.
        hcore = scf.RHF(mf.mol).nuc_grad_method().hcore_generator(mf.mol)
        h1 = np.asarray([
            cp.asnumpy(cp.einsum(
                'up,xuv,vq->xpq', coeff, cp.asarray(hcore(atom)), occupied,
            ))
            for atom in range(mf.mol.natm)
        ])
        analytic = fractional_rks_fock_skeleton(mf, coeff, mf.mo_occ)
        np.testing.assert_allclose(
            cp.asnumpy(analytic), h1 + numerical_j - 0.5 * numerical_k,
            atol=2e-6, rtol=0,
        )


if __name__ == '__main__':
    unittest.main()
