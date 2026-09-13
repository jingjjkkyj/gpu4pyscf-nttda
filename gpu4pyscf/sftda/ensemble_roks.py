# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

'''Ensemble-optimized orbitals with a fixed-orbital high-spin ROKS reference.

``EnsembleROKS`` keeps the ``2/1/0`` equal-spin-density ensemble optimization of
:class:`~gpu4pyscf.sftda.ensemble_rks.EnsembleRKS`, but selects a different
reference zero for NTTDA total energies: the high-spin ROKS energy evaluated on
the converged ensemble orbitals.  That reference energy is *not* stationary with
respect to the ensemble orbitals, so its analytic gradient must include the
non-stationary orbital response.
'''

from gpu4pyscf.dft import roks as gpu_roks
from gpu4pyscf.lib import utils

from gpu4pyscf.sftda.ensemble_rks import EnsembleRKS


class EnsembleROKS(EnsembleRKS):
    '''EnsembleRKS with a fixed-orbital high-spin ROKS reference energy.

    The SCF objective is the ensemble energy of :class:`EnsembleRKS`; only the
    reference zero used by NTTDA total energies is redefined by
    :meth:`reference_energy`.
    '''

    reference_energy_semantics = 'roks_energy_on_ensemble_rks_orbitals'
    reference_energy_stationary = False

    def reference_energy(self):
        '''Return the high-spin ROKS energy on the converged ensemble MOs.

        The orbitals remain those optimized by :class:`EnsembleRKS`; this is a
        fixed-orbital energy evaluation and does not run a second ROKS SCF.
        The density is the high-spin density built from the same ``2/1/0``
        occupations, not the equal-spin-density ensemble ``make_rdm1s``.  The
        value is recomputed on every call so a geometry or occupation change
        can never return a stale result.
        '''
        if self.mo_coeff is None or self.mo_occ is None:
            raise RuntimeError(
                'run EnsembleROKS.kernel() before evaluating the reference '
                'energy'
            )

        evaluator = gpu_roks.ROKS(self.mol, xc=self.xc)
        evaluator.verbose = 0
        evaluator.max_memory = self.max_memory
        with_df = getattr(self, 'with_df', None)
        if with_df is not None:
            evaluator = evaluator.density_fit(
                auxbasis=getattr(with_df, 'auxbasis', None),
            )
            evaluator.verbose = 0
        evaluator.grids = self.grids
        evaluator.nlcgrids = self.nlcgrids
        dm = evaluator.make_rdm1(self.mo_coeff, self.mo_occ)
        hcore = evaluator.get_hcore(self.mol)
        veff = evaluator.get_veff(self.mol, dm)
        return float(evaluator.energy_tot(dm=dm, h1e=hcore, vhf=veff))

    def to_cpu(self):
        from pyscf.sftda import EnsembleROKS as CPUEnsembleROKS

        mf = CPUEnsembleROKS(self.mol, xc=self.xc, nopen=self.nopen)
        utils.to_cpu(self, out=mf)
        return mf

    def nuc_grad_method(self):
        '''Return the analytic gradient of the selected reference energy.'''
        from gpu4pyscf.grad.ensemble_roks import ReferenceGradients

        return ReferenceGradients(self)


__all__ = ['EnsembleROKS']
