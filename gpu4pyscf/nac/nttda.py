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
'''

from gpu4pyscf.grad.nttda import (
    DFLedgerBackend,
    LedgerBackend,
    _import_forge,
    _resolve_input,
    make_gpu_response_cache,
)


def _make_nac_class():
    import pyscf.nac.nttda as forge_nac

    class NAC(forge_nac.NonAdiabaticCouplings):
        '''GPU-accelerated NTTDA NACs (CPU formulas, GPU integrals).'''

        def __init__(self, td):
            from gpu4pyscf.df.df_jk import _DFHF

            cpu_td, gmf = _resolve_input(td)
            super().__init__(cpu_td)
            self._gmf = gmf
            if isinstance(gmf, _DFHF):
                self.nttda_jk_ledger_backend = DFLedgerBackend(gmf)
            else:
                self.nttda_jk_ledger_backend = LedgerBackend(gmf)

        def _make_response_cache(self):
            shared = getattr(self, 'shared_response_cache', None)
            if shared is not None:
                return shared
            return make_gpu_response_cache(self.base, self._gmf)

        def _gradient_driver(self, verbose=None):
            driver = super()._gradient_driver(verbose=verbose)
            driver.nttda_jk_ledger_backend = self.nttda_jk_ledger_backend
            return driver

    return NAC


_NAC_CLASS = None


def NAC(td):
    '''Build the hybrid GPU NTTDA NAC driver for ``td``.

    ``td`` may be a converged ``gpu4pyscf.sftda.NTTDA`` or a CPU forge
    NTTDA object.
    '''
    global _NAC_CLASS
    if _NAC_CLASS is None:
        _import_forge()
        _NAC_CLASS = _make_nac_class()
    return _NAC_CLASS(td)


NonAdiabaticCouplings = NAC
