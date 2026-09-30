"""Cross-frame AO initial guesses; no geometry-fixed response data.

The historical spin encoding is retained during structural migration; it is
only an initial guess, and each method solves its own current-frame equations.
"""

import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.sftda import nttda_methods as methods

_PAIR_ALPHA_WEIGHT = {
    'cc': 0.5,
    'oo': 0.5,
    'vv': 0.5,
    'co': 0.0,
    'cv': 1.0,
    'ov': 1.0,
}
_PAIR_BETA_WEIGHT = {
    'cc': 0.5,
    'oo': 0.5,
    'vv': 0.5,
    'co': 1.0,
    'cv': 1.0,
    'ov': 0.0,
}


def _pair_index_arrays(pairs):
    """Row/column index arrays and per-spin unpack weights for ``pairs``."""
    try:
        alpha = cp.asarray(
            [_PAIR_ALPHA_WEIGHT[name] for _p, _q, name in pairs],
        )
        beta = cp.asarray(
            [_PAIR_BETA_WEIGHT[name] for _p, _q, name in pairs],
        )
    except KeyError as error:
        raise ValueError('unknown ROKS pair type %s' % error) from error
    rows = cp.asarray([p for p, _q, _name in pairs], dtype=cp.intp)
    columns = cp.asarray([q for _p, q, _name in pairs], dtype=cp.intp)
    return rows, columns, alpha, beta


def _frame_cache_signature(tdobj, pairs):
    """Electronic-model identity; nuclear coordinates are intentionally absent."""
    mf = tdobj._scf
    mol = mf.mol
    return (
        int(mol.nao_nr()),
        bool(getattr(mol, 'cart', False)),
        int(mol.charge),
        int(mol.spin),
        tuple(mol.atom_symbol(index) for index in range(mol.natm)),
        tuple(cp.asarray(mf.mo_occ).tolist()),
        str(getattr(mf, 'xc', 'HF')),
        getattr(mf, 'omega', None),
        methods.resolve_method(tdobj).id,
        int(getattr(tdobj, 'deltaS', 0)),
        tuple(pairs),
    )


class ZVectorFrameCache:
    """AO-projected adjoint guesses shared by consecutive geometries.

    Both spin source matrices are retained because a packed ROKS rotation can
    belong to different alpha/beta response sectors.  Entries are committed as
    one geometry snapshot so guesses from different frames are never mixed.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self._mol = None
        self._signature = None
        self._entries = {}
        self.last_stats = None
        return self

    def commit(self, tdobj, pairs, task_keys, zvectors):
        task_keys = tuple(task_keys)
        zvectors = cp.asarray(zvectors)
        if zvectors.ndim == 1:
            zvectors = zvectors.reshape(1, -1)
        if zvectors.shape != (len(task_keys), len(pairs)):
            raise ValueError(
                'cached Z-vectors have shape %s; expected (%d, %d)' % (zvectors.shape, len(task_keys), len(pairs))
            )
        mo = cp.asarray(tdobj._scf.mo_coeff)
        rows, columns, alpha, beta = _pair_index_arrays(pairs)
        source_alpha, source_beta = _unpack_pair_sources(
            zvectors,
            mo.shape[1],
            rows,
            columns,
            alpha,
            beta,
        )
        ao_alpha = mo @ source_alpha @ mo.conj().T
        ao_beta = mo @ source_beta @ mo.conj().T
        self._mol = tdobj.mol.copy()
        self._signature = _frame_cache_signature(tdobj, pairs)
        self._entries = {
            key: (cp.array(item_a, copy=True), cp.array(item_b, copy=True))
            for key, item_a, item_b in zip(task_keys, ao_alpha, ao_beta)
        }
        return self

    def project(self, tdobj, pairs, task_keys):
        task_keys = tuple(task_keys)
        misses = cp.zeros(len(task_keys), dtype=bool)
        if self._mol is None or self._signature != _frame_cache_signature(tdobj, pairs):
            return None, misses
        hits = cp.asarray([key in self._entries for key in task_keys])
        if not cp.any(hits):
            return None, hits

        mo = cp.asarray(tdobj._scf.mo_coeff)
        overlap = cp.asarray(
            gto.intor_cross(
                'int1e_ovlp',
                self._mol,
                tdobj.mol,
            )
        )
        overlap_mo = overlap @ mo
        rows, columns, alpha, beta = _pair_index_arrays(pairs)
        norm = alpha * alpha + beta * beta
        guesses = cp.zeros((len(task_keys), len(pairs)))
        for index, key in enumerate(task_keys):
            if not hits[index]:
                continue
            source_alpha, source_beta = self._entries[key]
            current_alpha = overlap_mo.conj().T @ source_alpha @ overlap_mo
            current_beta = overlap_mo.conj().T @ source_beta @ overlap_mo
            guesses[index] = (alpha * current_alpha[rows, columns] + beta * current_beta[rows, columns]) / norm
        return guesses, hits


def _unpack_pair_sources(vectors, nmo, rows, columns, alpha, beta):
    """Batched kappa sources from packed vectors of shape ``(nvec, npair)``."""
    nvec = vectors.shape[0]
    source_alpha = cp.zeros((nvec, nmo, nmo))
    source_beta = cp.zeros((nvec, nmo, nmo))
    source_alpha[:, rows, columns] = vectors * alpha
    source_beta[:, rows, columns] = vectors * beta
    return source_alpha, source_beta


def make_frame_cache():
    return ZVectorFrameCache()


import time
from dataclasses import replace
from gpu4pyscf.grad.nttda_params import PARAMS as _NTTDA_PARAMS


def compute_frame(
    td,
    active_state,
    nac_pairs=(),
    cphf_conv_tol=1e-10,
    cphf_max_cycle=None,
    use_etfs=True,
    frame_cache=None,
    gradient_states=None,
):
    """One dynamics frame: one or more state gradients plus NAC pairs.

    All geometry-fixed intermediates -- the spin-flip reference kernel,
    reference response closures, spin Fock pair, F0/Fz, and the J/K derivative
    engines (VHFOpt / Int3c2eOpt) -- are built once and shared across the
    gradient and every NAC pair.

    Each property has its own right-hand side ``M``, but all use the same
    reference Hessian in ``H.T Z = M``.  Sharing the response cache therefore
    changes construction cost and initial guesses, not the derivative model.

    ``frame_cache`` may be a ``ZVectorFrameCache`` owned by a dynamics
    driver.  It is updated only after every requested property succeeds.

    ``gradient_states`` defaults to ``(active_state,)``.  When multiple
    states are requested, ``active_state`` must be included and the return
    value additionally contains ``'gradients': {state: (natm, 3)}``;
    ``'grad'`` remains the active-state gradient for compatibility.  NAC
    entries are derivative couplings (the energy-scaled numerator divided by
    the state gap); ``use_etfs=False`` includes the moving-CSF term, while
    ``use_etfs=True`` retains the ETF/Hellmann--Feynman term.
    """
    if td.deltaS != -1:
        raise NotImplementedError(
            'compute_frame batching currently supports only deltaS=-1; '
            'use td.Gradients().kernel() for a deltaS=0 gradient'
        )

    from gpu4pyscf.nac.nttda import _nac_from_context
    from gpu4pyscf.grad.nttda import _gradients_from_context
    from .response import ResponseCache

    frame_started = time.perf_counter()
    gpu_mem_start = cp.get_default_memory_pool().used_bytes()
    nstates = len(td.e)
    if (
        isinstance(active_state, (bool, np.bool_))
        or not isinstance(active_state, (int, np.integer))
        or not 1 <= active_state <= nstates
    ):
        raise ValueError(
            'active_state must be in [1, %d]' % nstates,
        )
    active_state = int(active_state)
    if gradient_states is None:
        gradient_states = (active_state,)
    else:
        parsed_states = []
        for state in gradient_states:
            if isinstance(state, (bool, np.bool_)) or not isinstance(state, (int, np.integer)):
                raise ValueError('gradient states must be integer root indices')
            parsed_states.append(int(state))
        gradient_states = tuple(parsed_states)
        if len(set(gradient_states)) != len(gradient_states):
            raise ValueError('gradient states must be unique')
        if active_state not in gradient_states:
            raise ValueError('gradient_states must include active_state')
        if any(not 1 <= state <= nstates for state in gradient_states):
            raise ValueError(
                'gradient states must be in [1, %d]' % nstates,
            )
    active_gradient_index = gradient_states.index(active_state)
    parsed_pairs = []
    for pair in nac_pairs:
        if len(pair) != 2:
            raise ValueError('each NAC pair must contain two state indices')
        state_i, state_j = pair
        if any(
            isinstance(state, (bool, np.bool_)) or not isinstance(state, (int, np.integer))
            for state in (state_i, state_j)
        ):
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
    context = ResponseCache(td)
    grad = _gradients_from_context(td, context)
    grad.verbose = 0
    grad.cphf_conv_tol = cphf_conv_tol
    grad.cphf_max_cycle = cphf_max_cycle
    xctype = grad.base._scf._numint._xc_type(grad.base._scf.xc)
    gpu_xc_backend = context.xc_backend
    cache = context
    backend = context.ledger

    from . import delta_s_minus_one as delta, response

    atmlst = tuple(range(td.mol.natm))
    task_keys = [('grad', state) for state in gradient_states]
    nac = None
    nac_gradient = None
    if nac_pairs:
        nac = _nac_from_context(td, context)
        nac.verbose = 0
        nac.cphf_conv_tol = cphf_conv_tol
        nac.cphf_max_cycle = cphf_max_cycle
        nac.use_etfs = use_etfs
        nac_gradient = nac._gradient_driver(verbose=0)
    driver_seconds = time.perf_counter() - driver_started

    xc_started = time.perf_counter()
    grad_xc_terms = [None] * len(gradient_states)
    nac_xc_terms = [None] * len(nac_pairs)
    if xctype in ('GGA', 'MGGA') and gpu_xc_backend is not None:
        gradient_requests = [
            delta.gradient_xc_request(
                grad.base,
                grad.base.xy[state - 1],
            )
            for state in gradient_states
        ]
        channels = [request[0] for request in gradient_requests]
        spaces = gradient_requests[0][1]
        pz_batch = [request[2] for request in gradient_requests]
        for state_i, state_j in nac_pairs:
            request = delta.cross_xc_request(
                nac.base,
                nac.base.xy[state_i - 1],
                nac.base.xy[state_j - 1],
            )
            channels.extend(request[0])
            pz_batch.append(request[2])
        response_builder = getattr(
            gpu_xc_backend,
            xctype.lower() + '_response_terms_batch',
        )
        fockz_builder = getattr(
            gpu_xc_backend,
            xctype.lower() + '_fockz_terms_batch',
        )
        response_xc = response_builder(
            grad,
            grad.base,
            channels,
            atmlst=atmlst,
        )
        fockz_xc = fockz_builder(
            grad,
            grad.base,
            spaces,
            pz_batch,
            atmlst=atmlst,
        )
        for index in range(len(gradient_states)):
            grad_xc_terms[index] = (response_xc[index], fockz_xc[index])
        for index in range(len(nac_pairs)):
            offset = len(gradient_states) + 2 * index
            nac_xc_terms[index] = (
                response_xc[offset],
                response_xc[offset + 1],
                fockz_xc[len(gradient_states) + index],
            )
    xc_seconds = time.perf_counter() - xc_started

    prepare_started = time.perf_counter()
    prepared = [
        methods.prepare_gradient(
            context.method,
            grad,
            grad.base,
            grad.base.xy[state - 1],
            atmlst=atmlst,
            tolerance=cphf_conv_tol,
            max_cycle=cphf_max_cycle,
            cache=cache,
            xc_terms=grad_xc_terms[index],
        )
        for index, state in enumerate(gradient_states)
    ]
    for index, (state_i, state_j) in enumerate(nac_pairs):
        task_keys.append(('nac', state_i, state_j, bool(use_etfs)))
        prepared.append(
            methods.prepare_cross(
                context.method,
                nac_gradient,
                nac.base,
                nac.base.xy[state_i - 1],
                nac.base.xy[state_j - 1],
                atmlst=atmlst,
                tolerance=cphf_conv_tol,
                max_cycle=cphf_max_cycle,
                cache=cache,
                xc_terms=nac_xc_terms[index],
            )
        )
    prepare_seconds = time.perf_counter() - prepare_started

    pairs = methods.orbital_backend(context.method).canonical_pairs(grad.base, compact=True)
    reference_prepared = None
    reference_started = time.perf_counter()
    if context.method.reference_kind == 'ensemble_roks':
        reference_driver = grad._gmf.nuc_grad_method()
        reference_driver.verbose = 0
        reference_prepared = reference_driver.prepare_nttda_fusion(
            pairs,
            atmlst=atmlst,
            response_cache=cache,
        )
        if reference_prepared.pairs != pairs:
            raise RuntimeError('selected-reference preparation changed canonical pair order')
        for index in range(len(gradient_states)):
            prepared[index] = replace(
                prepared[index],
                orbital_rhs_shift=reference_prepared.orbital_rhs_shift,
                tolerance=min(float(cphf_conv_tol), 1e-12),
            )
    reference_prepare_seconds = time.perf_counter() - reference_started

    initial = None
    cache_hits = 0
    if frame_cache is not None:
        initial, cache_hits = frame_cache.project(
            grad.base,
            pairs,
            task_keys,
        )
    finish_profile = _NTTDA_PARAMS['finish_profile']
    finish_profile_timings = {} if finish_profile else None
    finish_started = time.perf_counter()
    components = response.finish_prepared_gradients(
        prepared,
        initial=initial,
        timings=finish_profile_timings,
        synchronize=(cp.cuda.get_current_stream().synchronize if finish_profile else None),
    )
    finish_seconds = time.perf_counter() - finish_started
    gradient_components = components[: len(gradient_states)]
    grad.nttda_details = gradient_components[active_gradient_index]
    nuclear_started = time.perf_counter()
    if reference_prepared is None:
        reference_gradient = np.asarray(grad.grad_nuc())
    else:
        reference_gradient = reference_prepared.unrelaxed_gradient
        grad.reference_gradient_calls = 1
        grad.reference_z_solver_diagnostics = {
            'converged': True,
            'fused': True,
            'residual_max_abs': max(float(item.residual) for item in gradient_components),
            'solve_tolerance': min(float(cphf_conv_tol), 1e-12),
        }
        grad.reference_zb_backend = 'fused_nttda'
        grad.reference_z_df_cache_stats = dict(
            getattr(reference_driver, 'z_df_cache_stats', {}),
        )
        grad.reference_zb_skeleton_stats = {}
    gradients = {
        state: cp.asnumpy(cp.asarray(reference_gradient) + item.total)
        for state, item in zip(gradient_states, gradient_components)
    }
    result = {
        'grad': gradients[active_state],
        'gradients': gradients,
        'nac': {},
    }
    nuclear_seconds = reference_prepare_seconds + time.perf_counter() - nuclear_started

    def record_stats(nac_postprocess_seconds=0.0):
        """Collect per-frame stats from all backends into ``td._nttda_frame_stats``.

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
        """
        scf_obj = getattr(td, '_scf', None)
        scf_stats = (
            {
                'cycles': int(getattr(scf_obj, 'cycles', 0)),
                'converged': bool(getattr(scf_obj, 'converged', False)),
            }
            if scf_obj is not None
            else {}
        )
        stats = {
            'method_id': context.method.id,
            'active_state': active_state,
            'gradient_states': gradient_states,
            'gradient_count': len(gradient_states),
            'nac_pairs': len(nac_pairs),
            'zvector_batch_width': len(prepared),
            'zvector_cache_hits': int(np.count_nonzero(cache_hits)),
            'xc_type': xctype,
            'xc_response_channels': (len(gradient_states) + 2 * len(nac_pairs) if gpu_xc_backend is not None else 0),
            'xc_fockz_tasks': (len(gradient_states) + len(nac_pairs) if gpu_xc_backend is not None else 0),
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
            'response_jk': {},
            'finish_prepared_timings': dict(
                finish_profile_timings or {},
            ),
            'reference_gradient': {
                'calls': int(getattr(grad, 'reference_gradient_calls', 0)),
                'fused': reference_prepared is not None,
                'semantics': getattr(
                    grad._gmf,
                    'reference_energy_semantics',
                    None,
                ),
                'z_b_backend': getattr(
                    grad,
                    'reference_zb_backend',
                    None,
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
                cp.asarray([item.zvector for item in components]),
            )
        record_stats()
        return result

    from gpu4pyscf.nac import nttda as nac_module

    csf_started = time.perf_counter()
    for (state_i, state_j), item in zip(nac_pairs, components[len(gradient_states) :]):
        gap = float(nac.base.e[state_j - 1] - nac.base.e[state_i - 1])
        if abs(gap) < nac.gap_tol:
            raise ZeroDivisionError('NTTDA state gap %.6e is below gap_tol %.6e' % (gap, nac.gap_tol))
        numerator = cp.asnumpy(cp.asarray(item.total))
        if not use_etfs:
            csf = nac_module.nac_csf_components(
                nac,
                nac.base.xy[state_i - 1],
                nac.base.xy[state_j - 1],
                cache=cache,
            )
            numerator = numerator + gap * cp.asnumpy(cp.asarray(csf.total))
        result['nac'][(state_i, state_j)] = np.real_if_close(
            numerator / gap,
        )
    if frame_cache is not None:
        frame_cache.commit(
            grad.base,
            pairs,
            task_keys,
            cp.asarray([item.zvector for item in components]),
        )
    record_stats(time.perf_counter() - csf_started)
    return result
