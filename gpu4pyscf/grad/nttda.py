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

'''GPU driver for NTTDA excited-state gradients.

Orchestration and scientific formulas reuse the validated CPU forge
implementation (pyscf-forge NTTDA, importable side by side with
gpu4pyscf); expensive integral and GGA grid work run on GPU:

- response J/K builds (M matrix, Fock builds, Z-vector iterations) are
  routed from the CPU twin objects to GPU ROKS/UKS or EnsembleRKS/RKS
  backends;
- the J/K derivative ledger is evaluated with the batched per-atom
  ``_jk_energies_per_atom`` kernels through the ledger backend seam
  (empirically calibrated mapping, machine-precision on J/K and
  range-separated variants: ``cpu_J = 2*gpu(L, R)``,
  ``cpu_K = -2*[gpu(L, R^T) + gpu(L^T, R)]``);
- the ground-state ROKS or EnsembleRKS gradient uses the matching native
  GPU implementation.

For GGA functionals, response, Fock-Z, post-Z derivative contractions, and
the iterative reference fxc action are evaluated by one geometry-fixed GPU XC
backend.  The CPU forge remains the formula and orchestration reference.
This separation is deliberate: the CPU layer owns the spin-adapted
Lagrangian, ``H.T Z = M``, Pulay, and NAC formulas, while the GPU layer owns
only algebraically equivalent integral, grid, and response contractions.
'''

import importlib
import time

import numpy as np
import cupy as cp

from gpu4pyscf.grad.nttda_params import PARAMS as _NTTDA_PARAMS


def _make_gradients_class():
    forge_grad, _forge_roks, _forge_solver = _import_forge()

    class Gradients(forge_grad.Gradients):
        '''GPU-accelerated NTTDA gradients.

        The inherited CPU class assembles ``M``, solves the adjoint equation,
        and combines relaxed electronic, Pulay, and nuclear terms.  This
        subclass replaces the expensive response and derivative-integral
        backends without duplicating those scientific formulas.
        '''

        def __init__(self, td, context=None):
            self._context = EvaluationContext(td) if context is None else context
            self._context.validate()
            super().__init__(self._context.cpu_td)
            self._gmf = self._context.gmf
            self._with_df = self._context.with_df
            self.nttda_jk_ledger_backend = self._context.ledger

        def kernel(self, state=None, atmlst=None, method=None, step=None):
            self._context.validate()
            return super().kernel(state=state, atmlst=atmlst, method=method, step=step)

        def _analytic_components(self, xy, atmlst, response_cache=None):
            if response_cache is None:
                response_cache = getattr(
                    self, 'shared_response_cache', None,
                )
            self._context.validate()
            if response_cache is None:
                response_cache = self._context.fresh_response_cache()
            return super()._analytic_components(
                xy, atmlst, response_cache=response_cache,
            )

        def grad_nuc(self, atmlst=None):
            return self.reference_gradient(atmlst)

        def reference_gradient(self, atmlst=None):
            self._context.validate()
            if self._gmf.grids.coords is None:
                self._gmf.grids.build(sort_grids=True)
            if self._with_df:
                # Selected-reference objects (non-stationary reference energy)
                # must use their own gradient driver.  EnsembleROKS inherits
                # is_ensemble_rks=True, so that flag alone would silently
                # select the ordinary DF-RKS gradient and bypass the reference
                # response.
                selected_reference = (
                    self._context.method.reference_kind == 'ensemble_roks'
                )
                if selected_reference:
                    driver = self._gmf.nuc_grad_method()
                    driver.nttda_xc_backend = self._context.xc_backend
                elif self._context.method.reference_kind == 'ensemble_rks':
                    from gpu4pyscf.df.grad.rks import Gradients as DFGrad
                    driver = DFGrad(self._gmf)
                else:
                    from gpu4pyscf.df.grad.roks import Gradients as DFGrad
                    driver = DFGrad(self._gmf)
            else:
                # The selected-ROKS reference binds nuc_grad_method() to the
                # T06 gpu4pyscf.grad.ensemble_roks.ReferenceGradients driver;
                # legacy ROKS/EnsembleRKS references keep their own gradient.
                driver = self._gmf.nuc_grad_method()
            driver.verbose = 0
            value = np.asarray(driver.kernel())
            diagnostics = getattr(driver, 'z_solver_diagnostics', None)
            self.reference_z_solver_diagnostics = (
                diagnostics.as_dict() if diagnostics is not None else None
            )
            self.reference_z_df_cache_stats = getattr(driver, 'z_df_cache_stats', None)
            self.reference_zb_backend = getattr(driver, 'zb_backend', None)
            self.reference_zb_skeleton_stats = getattr(
                driver, 'z_b_skeleton_stats', None,
            )
            self.reference_gradient_calls = getattr(
                self, 'reference_gradient_calls', 0,
            ) + 1
            if atmlst is not None:
                value = value[list(atmlst)]
            return value

        def _energy_at(self, coords, reference_amplitude):
            '''Displaced total energy for the public finite-difference path.

            The inherited CPU implementation rebuilds a non-density-fitted
            CPU reference, so its displaced energy would not be the same
            integral model as the DF analytic derivative.  For a DF reference
            rebuild the displaced reference on the GPU with the matching
            auxiliary basis instead.
            '''
            if not self._with_df:
                return super()._energy_at(coords, reference_amplitude)
            from gpu4pyscf.sftda import NTTDA as gpu_nttda

            mol = self.mol.copy()
            mol.set_geom_(coords, unit='Bohr')
            reference = rebuild_reference(self._gmf, mol, self.fixed_grid)
            reference.kernel(dm0=self._gmf.make_rdm1())
            if not reference.converged:
                raise RuntimeError(
                    'displaced NTTDA reference did not converge'
                )
            tdobj = gpu_nttda(reference)
            for name in (
                    'deltaS', 'nobeta', 'nstates', 'conv_tol', 'lindep',
                    'max_cycle', 'max_memory'):
                if hasattr(self.base, name):
                    setattr(tdobj, name, getattr(self.base, name))
            from pyscf.sftda import nttda_methods as methods
            methods.copy_method(self.base, tdobj)
            tdobj.verbose = 0
            tdobj.kernel()
            overlaps = np.asarray([
                abs(np.vdot(
                    reference_amplitude, _host_normalized_amplitude(xy),
                ))
                for xy in tdobj.xy
            ])
            root = int(np.argmax(overlaps))
            if overlaps[root] < self.root_overlap_tol:
                raise RuntimeError(
                    'NTTDA state tracking overlap %.6f is below %.6f' %
                    (overlaps[root], self.root_overlap_tol)
                )
            return tdobj.reference_energy() + tdobj.e[root]

    return Gradients


_GRADIENTS_CLASS = None


def Gradients(td):
    '''Build the hybrid GPU NTTDA gradient driver for ``td``.

    ``td`` must be a converged ``gpu4pyscf.sftda.NTTDA``.  A private CPU
    formula twin is created internally without mutating a user-owned object.
    '''
    if not _is_gpu_object(td):
        raise TypeError(
            'The hybrid derivative driver accepts only a GPU NTTDA object'
        )
    global _GRADIENTS_CLASS
    if _GRADIENTS_CLASS is None:
        _GRADIENTS_CLASS = _make_gradients_class()
    return _GRADIENTS_CLASS(td)


Grad = Gradients


def compute_frame(td, active_state, nac_pairs=(), cphf_conv_tol=1e-10,
                  cphf_max_cycle=None, use_etfs=True, frame_cache=None):
    '''One dynamics frame: gradient of the active state plus NAC pairs.

    All geometry-fixed intermediates -- the spin-flip reference kernel,
    reference response closures, spin Fock pair, F0/Fz, and the J/K derivative
    engines (VHFOpt / Int3c2eOpt) -- are built once and shared across the
    gradient and every NAC pair.

    Each property has its own right-hand side ``M``, but all use the same
    reference Hessian in ``H.T Z = M``.  Sharing the response cache therefore
    changes construction cost and initial guesses, not the derivative model.

    ``frame_cache`` may be a forge ``ZVectorFrameCache`` owned by a dynamics
    driver.  It is updated only after every requested property succeeds.

    Returns ``{'grad': (natm, 3), 'nac': {(i, j): (natm, 3)}}``.  NAC
    entries are derivative couplings (the energy-scaled numerator divided
    by the state gap); ``use_etfs=False`` includes the moving-CSF term,
    while ``use_etfs=True`` retains the ETF/Hellmann--Feynman term.
    '''
    if td.deltaS != -1:
        raise NotImplementedError(
            'compute_frame batching currently supports only deltaS=-1; '
            'use td.Gradients().kernel() for a deltaS=0 gradient'
        )

    from gpu4pyscf.nac.nttda import _nac_from_context

    frame_started = time.perf_counter()
    gpu_mem_start = cp.get_default_memory_pool().used_bytes()
    nstates = len(td.e)
    if (
            isinstance(active_state, (bool, np.bool_))
            or not isinstance(active_state, (int, np.integer))
            or not 1 <= active_state <= nstates):
        raise ValueError(
            'active_state must be in [1, %d]' % nstates,
        )
    active_state = int(active_state)
    parsed_pairs = []
    for pair in nac_pairs:
        if len(pair) != 2:
            raise ValueError('each NAC pair must contain two state indices')
        state_i, state_j = pair
        if any(
                isinstance(state, (bool, np.bool_))
                or not isinstance(state, (int, np.integer))
                for state in (state_i, state_j)):
            raise ValueError('NAC states must be integer root indices')
        parsed_pairs.append((int(state_i), int(state_j)))
    nac_pairs = tuple(parsed_pairs)
    for state_i, state_j in nac_pairs:
        if not 1 <= state_i <= nstates or not 1 <= state_j <= nstates:
            raise ValueError(
                'NAC states must be in [1, %d]' % nstates,
            )
        if state_i == state_j:
            raise ValueError('NAC pairs require two distinct states')

    driver_started = time.perf_counter()
    context = EvaluationContext(td)
    grad = _gradients_from_context(td, context)
    grad.verbose = 0
    grad.cphf_conv_tol = cphf_conv_tol
    grad.cphf_max_cycle = cphf_max_cycle
    xctype = grad.base._scf._numint._xc_type(grad.base._scf.xc)
    gpu_xc_backend = context.xc_backend
    cache = context.response_cache
    grad.shared_response_cache = cache
    backend = grad.nttda_jk_ledger_backend

    delta = importlib.import_module(
        'pyscf.grad.nttda.delta_s_minus_one',
    )
    forge_response = importlib.import_module('pyscf.grad.nttda.response')
    atmlst = tuple(range(td.mol.natm))
    task_keys = [('grad', active_state)]
    nac = None
    nac_gradient = None
    if nac_pairs:
        nac = _nac_from_context(td, context)
        nac.verbose = 0
        nac.cphf_conv_tol = cphf_conv_tol
        nac.cphf_max_cycle = cphf_max_cycle
        nac.use_etfs = use_etfs
        nac.shared_response_cache = cache
        nac.nttda_jk_ledger_backend = backend
        nac_gradient = nac._gradient_driver(verbose=0)
    driver_seconds = time.perf_counter() - driver_started

    xc_started = time.perf_counter()
    grad_xc_terms = None
    nac_xc_terms = [None] * len(nac_pairs)
    if xctype in ('GGA', 'MGGA') and gpu_xc_backend is not None:
        grad_channel, spaces, grad_pz = delta.gradient_xc_request(
            grad.base, grad.base.xy[active_state - 1],
        )
        channels = [grad_channel]
        pz_batch = [grad_pz]
        for state_i, state_j in nac_pairs:
            request = delta.cross_xc_request(
                nac.base,
                nac.base.xy[state_i - 1],
                nac.base.xy[state_j - 1],
            )
            channels.extend(request[0])
            pz_batch.append(request[2])
        response_builder = getattr(
            gpu_xc_backend, xctype.lower() + '_response_terms_batch',
        )
        fockz_builder = getattr(
            gpu_xc_backend, xctype.lower() + '_fockz_terms_batch',
        )
        response_xc = response_builder(
            grad, grad.base, channels, atmlst=atmlst,
        )
        fockz_xc = fockz_builder(
            grad, grad.base, spaces, pz_batch, atmlst=atmlst,
        )
        grad_xc_terms = (response_xc[0], fockz_xc[0])
        for index in range(len(nac_pairs)):
            offset = 1 + 2 * index
            nac_xc_terms[index] = (
                response_xc[offset],
                response_xc[offset + 1],
                fockz_xc[index + 1],
            )
    xc_seconds = time.perf_counter() - xc_started

    prepare_started = time.perf_counter()
    prepared = [context.method.prepare_gradient(
        grad,
        grad.base,
        grad.base.xy[active_state - 1],
        atmlst=atmlst,
        tolerance=cphf_conv_tol,
        max_cycle=cphf_max_cycle,
        cache=cache,
        xc_terms=grad_xc_terms,
    )]
    for index, (state_i, state_j) in enumerate(nac_pairs):
        task_keys.append(('nac', state_i, state_j, bool(use_etfs)))
        prepared.append(context.method.prepare_cross(
            nac_gradient,
            nac.base,
            nac.base.xy[state_i - 1],
            nac.base.xy[state_j - 1],
            atmlst=atmlst,
            tolerance=cphf_conv_tol,
            max_cycle=cphf_max_cycle,
            cache=cache,
            xc_terms=nac_xc_terms[index],
        ))
    prepare_seconds = time.perf_counter() - prepare_started

    pairs = context.method.orbital_backend().canonical_pairs(grad.base, compact=True)
    initial = None
    cache_hits = 0
    if frame_cache is not None:
        initial, cache_hits = frame_cache.project(
            grad.base, pairs, task_keys,
        )
    finish_started = time.perf_counter()
    components = forge_response.finish_prepared_gradients(
        prepared, initial=initial,
    )
    finish_seconds = time.perf_counter() - finish_started
    grad.nttda_details = components[0]
    nuclear_started = time.perf_counter()
    result = {
        'grad': np.asarray(grad.grad_nuc()) + components[0].total,
        'nac': {},
    }
    nuclear_seconds = time.perf_counter() - nuclear_started

    def record_stats(nac_postprocess_seconds=0.0):
        '''Collect per-frame stats from all backends into ``td._nttda_frame_stats``.

        Called once at the end of ``compute_frame`` (after all gradient/NAC
        work succeeds).  The stats dict is read by the FSSH profile script
        and by ``frame_cache.last_stats``.  Fields:

        - ``timings`` — phase-level wall seconds (drivers, xc_batch, prepare,
          zvector_and_derivatives, nuclear, nac_postprocess, total).
        - ``nttda_solver`` — Davidson vind/Davidson stats from
          ``NTTDA._nttda_solver_stats``.
        - ``scf`` — SCF cycle count and convergence.
        - ``gpu_memory`` — cupy memory pool usage at frame start/end.
        - ``runtime_params`` — snapshot of ``nttda_params.PARAMS``.
        - ``jk_backend`` — DFLedgerBackend stats (pair counts, compression,
          layer timing, lower-bound gate counters).
        - ``xc_backend`` / ``response_cache`` / ``response_jk`` — backend stats.
        '''
        scf_obj = getattr(td, '_scf', None)
        scf_stats = {
            'cycles': int(getattr(scf_obj, 'cycles', 0)),
            'converged': bool(getattr(scf_obj, 'converged', False)),
        } if scf_obj is not None else {}
        stats = {
            'method_id': context.method.id,
            'active_state': active_state,
            'nac_pairs': len(nac_pairs),
            'zvector_batch_width': len(prepared),
            'zvector_cache_hits': int(np.count_nonzero(cache_hits)),
            'xc_type': xctype,
            'xc_response_channels': (
                1 + 2 * len(nac_pairs)
                if gpu_xc_backend is not None else 0
            ),
            'xc_fockz_tasks': (
                1 + len(nac_pairs)
                if gpu_xc_backend is not None else 0
            ),
            'timings': {
                'drivers': driver_seconds,
                'xc_batch': xc_seconds,
                'prepare': prepare_seconds,
                'zvector_and_derivatives': finish_seconds,
                'nuclear': nuclear_seconds,
                'nac_postprocess': nac_postprocess_seconds,
                'total': time.perf_counter() - frame_started,
            },
            'nttda_solver': dict(getattr(td, '_nttda_solver_stats', {})),
            'scf': scf_stats,
            'gpu_memory': {
                'start_bytes': int(gpu_mem_start),
                'end_bytes': int(cp.get_default_memory_pool().used_bytes()),
            },
            'runtime_params': dict(_NTTDA_PARAMS),
            'jk_backend': dict(getattr(backend, 'stats', {})),
            'xc_backend': dict(getattr(gpu_xc_backend, 'stats', {})),
            'response_cache': dict(getattr(cache, 'stats', {})),
            'response_jk': dict(getattr(
                cache.reference(), '_nttda_jk_route_stats', {},
            )),
            'reference_gradient': {
                'calls': int(getattr(grad, 'reference_gradient_calls', 0)),
                'semantics': getattr(
                    grad._gmf, 'reference_energy_semantics', None,
                ),
                'z_b_backend': getattr(
                    grad, 'reference_zb_backend', None,
                ),
                'z_solver': dict(
                    getattr(grad, 'reference_z_solver_diagnostics', None) or {},
                ),
                'z_df_cache': dict(
                    getattr(grad, 'reference_z_df_cache_stats', None) or {},
                ),
                'z_b_skeleton': dict(
                    getattr(grad, 'reference_zb_skeleton_stats', None) or {},
                ),
            },
        }
        td._nttda_frame_stats = stats
        if frame_cache is not None:
            frame_cache.last_stats = stats

    if not nac_pairs:
        if frame_cache is not None:
            frame_cache.commit(
                grad.base,
                pairs,
                task_keys,
                np.asarray([components[0].zvector]),
            )
        record_stats()
        return result

    forge_nac = importlib.import_module('pyscf.nac.nttda')
    csf_started = time.perf_counter()
    for (state_i, state_j), item in zip(nac_pairs, components[1:]):
        gap = float(nac.base.e[state_j - 1] - nac.base.e[state_i - 1])
        if abs(gap) < nac.gap_tol:
            raise ZeroDivisionError(
                'NTTDA state gap %.6e is below gap_tol %.6e'
                % (gap, nac.gap_tol)
            )
        numerator = np.asarray(item.total)
        if not use_etfs:
            csf = forge_nac.nac_csf_components(
                nac,
                nac.base.xy[state_i - 1],
                nac.base.xy[state_j - 1],
                cache=cache,
            )
            numerator = numerator + gap * np.asarray(csf.total)
        result['nac'][(state_i, state_j)] = np.real_if_close(
            numerator / gap,
        )
    if frame_cache is not None:
        frame_cache.commit(
            grad.base,
            pairs,
            task_keys,
            np.asarray([item.zvector for item in components]),
        )
    record_stats(time.perf_counter() - csf_started)
    return result

# Compatibility exports. Internal consumers import the owning module.
from .nttda_bridge import (
    _assert_module_under_root,
    _prepend_path,
    _validate_forge_capabilities,
    _import_forge,
    _is_gpu_object,
    _validate_supported_reference,
    rebuild_reference,
    _host_normalized_amplitude,
    _resolve_input,
)
from .nttda_ledger import (
    _density_view_key,
    _DensityCombination,
    _density_cache_key,
    _linear_combination,
    _compress_bilinear_pairs,
    _expanded_slot_pair_groups,
    DFLedgerBackend,
    LedgerBackend,
)
from .nttda_context import (
    EvaluationContext, make_gpu_xc_backend, make_gpu_response_cache, make_frame_cache,
)


def build_cpu_twin(gpu_td):
    from . import nttda_bridge as bridge
    return bridge.build_cpu_twin(gpu_td, forge=_import_forge())


def _gradients_from_context(td, context):
    global _GRADIENTS_CLASS
    if _GRADIENTS_CLASS is None:
        _GRADIENTS_CLASS = _make_gradients_class()
    return _GRADIENTS_CLASS(td, context=context)


def route_jk_to_gpu(cpu_mf, gmf):
    """Compatibility entry retaining the host/device adapter test seam."""
    from . import nttda_bridge as bridge
    return bridge.route_jk_to_gpu(cpu_mf, gmf, array_backend=cp)
