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

"""Per-evaluation GPU response ownership and cross-frame guess factory."""
import os
import numpy as np
import cupy as cp
from .nttda_bridge import _import_forge, _resolve_input, route_jk_to_gpu
from .nttda_ledger import DFLedgerBackend, LedgerBackend

_UNSET = object()

def make_gpu_xc_backend(cpu_td, gmf):
    '''Build the geometry-fixed GPU XC backend where it is supported.'''
    from pyscf.sftda import nttda_methods as methods
    xctype = gmf._numint._xc_type(gmf.xc)
    if xctype not in methods.get_method(cpu_td).gpu_xc_types:
        return None
    from gpu4pyscf.grad.nttda_xc import GPUXCFrameBackend

    return GPUXCFrameBackend(gmf, cpu_td)

def make_gpu_response_cache(cpu_td, gmf, xc_backend=_UNSET):
    '''Forge ResponseCache with GPU-native reference response and GPU J/K.

    EnsembleRKS uses the restricted response of its common Fock; ROKS uses the
    spin-resolved response exposed by ``GPUXCFrameBackend``.  The closures are
    NumPy-in/NumPy-out only at the CPU orchestration seam: every fxc and J/K
    contraction inside a call is performed on the GPU.
    '''
    _forge_grad, forge_roks, _forge_solver = _import_forge()
    if xc_backend is _UNSET:
        xc_backend = make_gpu_xc_backend(cpu_td, gmf)

    from pyscf.grad.nttda.response import ResponseCache

    class _GPUResponseCache(ResponseCache):
        def __init__(self, tdobj):
            super().__init__(tdobj)
            self.frame_xc_backend = xc_backend
            cached_focks = getattr(
                tdobj, '_nttda_gpu_fock0_fockz', None,
            )
            if cached_focks is not None:
                self.extra['fock0_fockz'] = tuple(
                    cp.asnumpy(cp.asarray(value)) for value in cached_focks
                )
            elif xc_backend is not None:
                self.extra['fock0_fockz'] = (
                    xc_backend.spin_lowering_fock0_fockz()
                )
            if (self.method.reference_kind in ('ensemble_rks', 'ensemble_roks')
                    and 'fock0_fockz' in self.extra
                    and os.environ.get('NTTDA_ENSEMBLE_FOCK_CACHE', '1') != '0'):
                # F0 is the ensemble common Fock, not the selected HS reference.
                orbitals = np.asarray(tdobj._scf.mo_coeff)
                self.extra['ensemble_fock_mo'] = (
                    orbitals.conj().T @ self.extra['fock0_fockz'][0] @ orbitals
                )
                self.stats['ensemble_fock_cache_hits'] = 1

        def reference(self):
            reference = super().reference()
            if reference is not self._tdobj._scf:
                route_jk_to_gpu(reference, gmf)
            return reference

        def response(self, hermi):
            if self.method.reference_kind in ('ensemble_rks', 'ensemble_roks'):
                if hermi not in self._responses:
                    from gpu4pyscf.scf import _response_functions

                    gpu_response = _response_functions._gen_rhf_response(
                        gmf, hermi=hermi,
                    )
                    self.stats['response_builds'] += 1

                    def counted_ensemble_response(density):
                        width = (
                            1 if density.ndim == 2
                            else int(np.prod(density.shape[:-2]))
                        )
                        self.stats['response_calls'] += 1
                        self.stats['response_rhs'] += width
                        self.stats['response_batch_widths'].append(width)
                        # cp.asarray strips CPArrayWithTag factor metadata.
                        if not isinstance(density, cp.ndarray):
                            density = cp.asarray(density)
                        return cp.asnumpy(gpu_response(density))

                    self._responses[hermi] = counted_ensemble_response
                return self._responses[hermi]
            if xc_backend is None:
                return super().response(hermi)
            if hermi not in self._responses:
                response = xc_backend.response(hermi)
                self.stats['response_builds'] += 1

                def counted_response(*args, **kwargs):
                    density = np.asarray(args[0])
                    if density.ndim <= 3:
                        width = 1
                    elif density.shape[0] == 2:
                        width = int(np.prod(density.shape[1:-2]))
                    else:
                        width = int(np.prod(density.shape[:-2]))
                    self.stats['response_calls'] += 1
                    self.stats['response_rhs'] += width
                    self.stats['response_batch_widths'].append(width)
                    return response(*args, **kwargs)

                self._responses[hermi] = counted_response
            return self._responses[hermi]

        def orbital_response(self, orbitals, occupation):
            """Make a solve-local exact density builder for the ensemble Z."""
            from ._response_density import OrbitalRotationDensity
            density = OrbitalRotationDensity(orbitals, occupation)
            response = self.response(1)

            def apply(rotation):
                return response(density(rotation))

            def clear():
                self.stats['z_df_cache_hits'] = density.cache.hits
                self.stats['z_df_cache_misses'] = density.cache.misses
                self.stats['z_df_cache_peak_bytes'] = density.cache.peak_bytes
                density.clear()

            apply.clear = clear
            return apply

        def fxc_ref(self):
            if xc_backend is not None:
                raise RuntimeError(
                    'GPU-native GGA frame unexpectedly requested the CPU '
                    'spin-flip fxc reference'
                )
            return super().fxc_ref()

    return _GPUResponseCache(cpu_td)

def make_frame_cache():
    """Persistent AO Z-vector guesses for consecutive dynamics frames."""
    _forge_grad, forge_roks, _forge_solver = _import_forge()
    from pyscf.grad.nttda.frame import ZVectorFrameCache
    return ZVectorFrameCache()


class EvaluationContext:
    """One fixed electronic solution and the backends shared by its tasks."""

    def __init__(self, td):
        from gpu4pyscf.df.df_jk import _DFHF
        self.source = td
        self.cpu_td, self.gmf = _resolve_input(td)
        from pyscf.sftda import nttda_methods as methods
        self.method = methods.get_method(self.cpu_td)
        self.with_df = isinstance(self.gmf, _DFHF)
        self.ledger = DFLedgerBackend(self.gmf) if self.with_df else LedgerBackend(self.gmf)
        self._source_signature = self._signature()
        self._coords = np.array(self.gmf.mol.atom_coords(), copy=True)
        self._xc_backend = _UNSET
        self._response_cache = None

    def _signature(self):
        from pyscf.sftda import nttda_methods as methods
        mf = self.source._scf
        return (
            id(mf), id(mf.mo_coeff), id(mf.mo_occ), id(self.source.xy),
            id(mf.grids.coords), id(mf.grids.weights),
            str(mf.xc), getattr(mf, 'omega', None),
            id(mf._numint), id(getattr(mf, 'with_df', None)),
            repr(getattr(getattr(mf, 'with_df', None), 'auxbasis', None)),
            methods.resolve_method(self.source).id, int(self.source.deltaS),
        )

    def validate(self):
        from pyscf.sftda import nttda_methods as methods
        methods.bind_method(self.source, derivative=True)
        if (self._signature() != self._source_signature
                or not np.array_equal(self.gmf.mol.atom_coords(), self._coords)):
            raise ValueError('electronic solution changed; create a new derivative driver')

    @property
    def xc_backend(self):
        if self._xc_backend is _UNSET:
            self._xc_backend = make_gpu_xc_backend(self.cpu_td, self.gmf)
        return self._xc_backend

    @property
    def response_cache(self):
        if self._response_cache is None:
            self._response_cache = make_gpu_response_cache(
                self.cpu_td, self.gmf, self.xc_backend,
            )
        return self._response_cache

    def fresh_response_cache(self):
        self.validate()
        return make_gpu_response_cache(self.cpu_td, self.gmf)
