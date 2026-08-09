# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

'''GPU average-occupation restricted ensemble Kohn--Sham references.

The reference has one spatial-orbital set with fixed ``2/1/0`` occupations.
Its spin densities obey ``D_alpha = D_beta = D/2`` even when ``mol.spin`` is
nonzero; ``mol.spin`` labels the target spin manifold and fixes the number of
singly occupied orbitals.  The orbital residual is therefore the common-Fock
condition ``g_pq = (n_q - n_p) F0_pq``.  CuPy arrays remain device-resident
through occupation selection, density construction, and residual evaluation.
'''

import cupy as cp

from gpu4pyscf.dft import rks
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.scf import hf


class EnsembleRKS(rks.RKS):
    '''RKS with fixed ``2/1/0`` occupations and ``D_alpha=D_beta=D/2``.

    This is the fixed average-occupation reference required by NTTDA, not a
    general state-weighted/GOK ensemble optimizer.
    '''

    is_ensemble_rks = True
    _keys = rks.RKS._keys | {'nopen'}

    def __init__(self, mol, xc='LDA,VWN', nopen=None):
        super().__init__(mol, xc=xc)
        if nopen is None:
            nopen = mol.spin
        if isinstance(nopen, bool) or int(nopen) != nopen:
            raise ValueError('nopen must be a non-negative integer')
        self.nopen = int(nopen)
        self._validate_ensemble()

    def _validate_ensemble(self):
        if self.nopen < 0:
            raise ValueError('nopen must be a non-negative integer')
        if self.nopen != self.mol.spin:
            raise ValueError('nopen must match mol.spin for NTTDA')
        if self.nopen > self.mol.nelectron:
            raise ValueError('nopen cannot exceed the electron count')
        if (self.mol.nelectron - self.nopen) % 2:
            raise ValueError(
                'electron count and nopen have inconsistent parity'
            )

    @property
    def nclosed(self):
        return (self.mol.nelectron - self.nopen) // 2

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'EnsembleRKS nclosed = %d', self.nclosed)
        logger.info(self, 'EnsembleRKS nopen = %d', self.nopen)
        logger.info(self, 'EnsembleRKS spin density constraint: Dz = 0')
        return self

    def check_sanity(self):
        # Avoid RHF's odd/open-shell warning: fractional occupations are the
        # defining ensemble constraint rather than an invalid RHF state.
        return hf.SCF.check_sanity(self)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        self._validate_ensemble()
        if mo_energy is None:
            mo_energy = self.mo_energy
        mo_energy = cp.asarray(mo_energy)
        nmo = mo_energy.size
        if self.nclosed + self.nopen > nmo:
            raise RuntimeError(
                'not enough orbitals for %d closed and %d open orbitals'
                % (self.nclosed, self.nopen)
            )
        # Match the CPU reference for exactly degenerate orbitals.  A stable
        # order keeps the selected open subspace deterministic across hosts.
        order = cp.argsort(mo_energy, kind='stable')
        mo_occ = cp.zeros_like(mo_energy)
        mo_occ[order[:self.nclosed]] = 2
        open_stop = self.nclosed + self.nopen
        mo_occ[order[self.nclosed:open_stop]] = 1
        if self.verbose >= logger.INFO:
            logger.info(self, 'EnsembleRKS occupations = %s', mo_occ)
        return mo_occ

    def make_rdm1s(self, mo_coeff=None, mo_occ=None):
        dm = self.make_rdm1(mo_coeff, mo_occ)
        dm_spin = cp.asarray(dm) * 0.5
        return dm_spin, dm_spin.copy()

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        '''Return independent elements of ``(n_q-n_p) F0_pq``.

        The two spin masks retain closed-open, closed-virtual, and
        open-virtual rotations while excluding rotations inside an
        equal-occupation subspace.  Constructing the masks with CuPy avoids a
        host round-trip in the SCF convergence loop.
        '''
        mo_coeff = cp.asarray(mo_coeff)
        mo_occ = cp.asarray(mo_occ)
        if fock is None:
            dm = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore(self.mol) + self.get_veff(self.mol, dm)
        fock_mo = mo_coeff.conj().T @ cp.asarray(fock) @ mo_coeff
        occupied_alpha = mo_occ > 0
        occupied_beta = mo_occ == 2
        virtual_alpha = ~occupied_alpha
        virtual_beta = ~occupied_beta
        unique = (
            (virtual_alpha[:, None] & occupied_alpha)
            | (virtual_beta[:, None] & occupied_beta)
        )
        occupation_difference = mo_occ[None, :] - mo_occ[:, None]
        return (fock_mo * occupation_difference)[unique]

    def to_cpu(self):
        from pyscf.sftda import EnsembleRKS as CPUEnsembleRKS

        mf = CPUEnsembleRKS(self.mol, xc=self.xc, nopen=self.nopen)
        utils.to_cpu(self, out=mf)
        return mf


__all__ = ['EnsembleRKS']
