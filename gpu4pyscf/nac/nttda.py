# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Hybrid GPU driver for analytic NTTDA ``deltaS = -1`` NACs.

Same architecture as :mod:`gpu4pyscf.grad.nttda`: the validated CPU forge
NAC (bilinear cross numerator + analytic interstate RDM + CSF connection)
orchestrates, while all response J/K builds, Z-vector iterations, and J/K
derivative-ledger contractions run on GPU.

The scientific result is preserved as
``d_IJ = N_IJ^HF / (omega_J-omega_I) + d_IJ^CSF``.  GPU execution changes
where the contractions are evaluated, not the gap convention, state phase,
ETF switch, or moving-CSF contribution.
'''

from gpu4pyscf.grad.nttda_bridge import (
    _import_forge, _is_gpu_object, build_cpu_twin, rebuild_reference,
)
from gpu4pyscf.grad.nttda_context import EvaluationContext


def _make_nac_class():
    import pyscf.nac.nttda as forge_nac

    class NAC(forge_nac.NonAdiabaticCouplings):
        '''GPU-accelerated NTTDA NACs (CPU formulas, GPU integrals).'''

        def __init__(self, td, context=None):
            self._context = EvaluationContext(td) if context is None else context
            self._context.validate()
            super().__init__(self._context.cpu_td)
            self._gmf = self._context.gmf
            self.nttda_jk_ledger_backend = self._context.ledger

        def kernel(self, *args, **kwargs):
            self._context.validate()
            return super().kernel(*args, **kwargs)

        def _make_response_cache(self):
            self._context.validate()
            shared = getattr(self, 'shared_response_cache', None)
            if shared is not None:
                return shared
            return self._context.fresh_response_cache()

        def _gradient_driver(self, verbose=None):
            driver = super()._gradient_driver(verbose=verbose)
            driver.nttda_jk_ledger_backend = self.nttda_jk_ledger_backend
            return driver

        def _displaced_td(self, coordinates):
            '''Rebuild the displaced NTTDA with the source's integral model.

            The inherited CPU helper rebuilds a non-density-fitted CPU
            reference, so its finite-difference NAC would not be the same
            integral model as the analytic DF derivative.  For a DF reference
            rebuild on the GPU with the matching auxiliary basis, then hand
            the CPU twin to the wavefunction-overlap post-processing.
            '''
            from gpu4pyscf.df.df_jk import _DFHF
            from gpu4pyscf.sftda import NTTDA as gpu_nttda

            if not isinstance(self._gmf, _DFHF):
                return super()._displaced_td(coordinates)
            mol = self.mol.copy()
            mol.set_geom_(coordinates, unit='Bohr')
            reference = rebuild_reference(self._gmf, mol, self.fixed_grid)
            reference.kernel(dm0=self._gmf.make_rdm1())
            if not reference.converged:
                raise RuntimeError(
                    'displaced NTTDA reference did not converge'
                )
            displaced = gpu_nttda(reference)
            for name in (
                    'deltaS', 'nobeta', 'nstates', 'conv_tol', 'lindep',
                    'max_cycle', 'max_memory'):
                if hasattr(self.base, name):
                    setattr(displaced, name, getattr(self.base, name))
            from pyscf.sftda import nttda_methods as methods
            methods.copy_method(self.base, displaced)
            displaced.verbose = 0
            displaced.kernel()
            return build_cpu_twin(displaced)

    return NAC


_NAC_CLASS = None


def NAC(td):
    '''Build the hybrid GPU NTTDA NAC driver for ``td``.

    ``td`` must be a converged ``gpu4pyscf.sftda.NTTDA``.  CPU forge inputs
    are intentionally rejected to avoid mixing independently configured
    energy and derivative models.
    '''
    if not _is_gpu_object(td):
        raise TypeError(
            'The hybrid derivative driver accepts only a GPU NTTDA object'
        )
    global _NAC_CLASS
    if _NAC_CLASS is None:
        _import_forge()
        _NAC_CLASS = _make_nac_class()
    return _NAC_CLASS(td)


NonAdiabaticCouplings = NAC


def _nac_from_context(td, context):
    global _NAC_CLASS
    if _NAC_CLASS is None:
        _import_forge()
        _NAC_CLASS = _make_nac_class()
    return _NAC_CLASS(td, context=context)
