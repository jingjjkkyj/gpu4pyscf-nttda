"""Shared response lifetime, derivative task scheduling and final contraction.

Orbital Hessians remain in their own backends. This module receives their
adjoints/probes and never infers a scientific method from user flags.
"""

from dataclasses import dataclass
from contextlib import contextmanager
from inspect import signature
import json
import os
import sys
import time
import numpy as np
import cupy as cp
from scipy.sparse.linalg import LinearOperator, gmres
from pyscf.lib import logger
from gpu4pyscf.sftda import nttda_methods as methods
from .frame import _pair_index_arrays
from .reference import _validate_supported_reference
from gpu4pyscf.grad.nttda_ledger import DFLedgerBackend, LedgerBackend
from gpu4pyscf.grad.ensemble_roks import _hcore_derivative_generator

_GMRES_RTOL_KEY = 'rtol' if 'rtol' in signature(gmres).parameters else 'tol'
_FINISH_TIMING_SECTIONS = (
    'setup',
    'zvector_solve',
    'zvector_data',
    'post_z_prepare',
    'df_derivative_ledger',
    'post_z_hcore',
    'post_z_xc',
    'post_z_accumulate',
    'final_assembly',
)


class _FinishTimings:
    """Optional synchronized, exclusive timings for one finish call."""

    def __init__(self, output, synchronize):
        self.output = output
        self.synchronize = synchronize or (lambda: None)
        self.accounted = 0.0
        self.synchronize()
        self.started = time.perf_counter()

    @contextmanager
    def section(self, name):
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - started
            self.output[name] = self.output.get(name, 0.0) + elapsed
            self.accounted += elapsed

    def finish(self):
        self.synchronize()
        total = time.perf_counter() - self.started
        for name in _FINISH_TIMING_SECTIONS:
            self.output.setdefault(name, 0.0)
        self.output['total'] = total
        self.output['accounted'] = self.accounted
        self.output['unattributed'] = max(0.0, total - self.accounted)


@contextmanager
def _untimed_section(_name):
    yield


@dataclass(frozen=True)
class OrbitalSpaces:
    """Closed, open, and virtual spatial-orbital partitions."""

    closed: cp.ndarray
    open: cp.ndarray
    virtual: cp.ndarray
    c_closed: cp.ndarray
    c_open: cp.ndarray
    c_virtual: cp.ndarray

    @property
    def spin(self):
        return 0.5 * len(self.open)


def orbital_spaces(tdobj):
    """Return the spatial-reference ``C/O/V`` partition used by NTTDA."""
    mf = tdobj._scf
    occ = cp.asarray(mf.mo_occ)
    if occ.ndim != 1:
        raise ValueError('NTTDA gradients require spatial reference orbitals')
    closed = cp.flatnonzero(occ == 2)
    open_ = cp.flatnonzero(occ == 1)
    virtual = cp.flatnonzero(occ == 0)
    coeff = cp.asarray(mf.mo_coeff)
    return OrbitalSpaces(
        closed=closed,
        open=open_,
        virtual=virtual,
        c_closed=coeff[:, closed],
        c_open=coeff[:, open_],
        c_virtual=coeff[:, virtual],
    )


def pair_density(c_left, coefficient, c_right):
    """Build ``C_left coefficient C_right^T`` without symmetrizing it."""
    return c_left @ cp.asarray(coefficient) @ c_right.conj().T


@dataclass(frozen=True)
class GradientComponents:
    """Excitation-gradient pieces; ``residual`` is the adjoint inf-norm."""

    m_matrix: cp.ndarray
    direct: cp.ndarray
    orbital: cp.ndarray
    total: cp.ndarray
    zvector: cp.ndarray
    residual: float


@dataclass(frozen=True)
class PreparedGradient:
    """Unrelaxed derivative task waiting for its orbital adjoint."""

    gradient_driver: object
    tdobj: object
    m_matrix: cp.ndarray
    direct: cp.ndarray
    atmlst: tuple
    tolerance: float
    max_cycle: object
    jk_ledger: object
    nobeta_p0: object
    direct_fock_probes: object
    cache: object
    orbital_rhs_shift: object = None


class ResponseCache:
    """Share integrals, XC and response within one fixed electronic solution."""

    def __init__(self, td):
        from gpu4pyscf.df.df_jk import _DFHF
        from gpu4pyscf.sftda.nttda import NTTDA

        if not isinstance(td, NTTDA):
            raise TypeError('NTTDA GPU derivatives require a GPU NTTDA object')
        self._tdobj = td
        self.method = methods.bind_method(td, derivative=True)
        self._orbitals = td._scf.mo_coeff
        self._occupations = td._scf.mo_occ
        self._reference = None
        self._responses = {}
        self._fxc_ref = False
        self._focks_mo = None
        self.stats = {
            'response_builds': 0,
            'response_calls': 0,
            'response_rhs': 0,
            'response_batch_widths': [],
        }
        # Free-form memo for channel-level geometry-fixed intermediates
        # (e.g. the F0/Fz pair); keyed by short strings.
        self.extra = {}
        self.source = td
        self.gmf = td._scf
        _validate_supported_reference(self.gmf)
        self.with_df = isinstance(self.gmf, _DFHF)
        self.ledger = DFLedgerBackend(self.gmf) if self.with_df else LedgerBackend(self.gmf)
        self._coords = self.gmf.mol.atom_coords().copy()
        self._source_signature = self._signature()
        self._xc_backend = None
        self._xc_initialized = False
        cached = getattr(td, '_nttda_gpu_fock0_fockz', None)
        if cached is not None:
            self.extra['fock0_fockz'] = tuple(cp.asarray(value) for value in cached)
        if self.method.reference_kind != 'roks' and cached is not None:
            mo = cp.asarray(self.gmf.mo_coeff)
            self.extra['ensemble_fock_mo'] = mo.conj().T @ cp.asarray(cached[0]) @ mo
            self.stats['ensemble_fock_cache_hits'] = 1

    def _signature(self):
        mf = self.gmf
        return (
            id(mf),
            id(mf.mo_coeff),
            id(mf.mo_occ),
            id(self.source.xy),
            id(mf.grids.coords),
            id(mf.grids.weights),
            str(mf.xc),
            getattr(mf, 'omega', None),
            id(mf._numint),
            id(getattr(mf, 'with_df', None)),
            repr(getattr(getattr(mf, 'with_df', None), 'auxbasis', None)),
            methods.resolve_method(self.source).id,
            int(self.source.deltaS),
        )

    def validate(self):
        methods.bind_method(self.source, derivative=True)
        if self._signature() != self._source_signature or not np.array_equal(self.gmf.mol.atom_coords(), self._coords):
            raise ValueError('electronic solution changed; create a new derivative driver')

    @property
    def xc_backend(self):
        if not self._xc_initialized:
            if self.gmf._numint._xc_type(self.gmf.xc) in self.method.gpu_xc_types:
                from gpu4pyscf.grad.nttda_xc import GPUXCFrameBackend

                self._xc_backend = GPUXCFrameBackend(self.gmf, self.source)
            self._xc_initialized = True
        return self._xc_backend

    def spin_focks_mo(self):
        if self._focks_mo is None:
            self._focks_mo = methods.spin_focks_mo(self.method, self.gmf)
        return self._focks_mo

    def response(self, hermi):
        if hermi not in self._responses:
            if self.method.reference_kind != 'roks' and self.gmf._numint._xc_type(self.gmf.xc) == 'HF':

                def response(density):
                    coulomb, exchange = self.gmf.get_jk(self.gmf.mol, density, hermi=hermi)
                    return coulomb - 0.5 * exchange
            elif self.method.reference_kind != 'roks':
                from gpu4pyscf.scf._response_functions import _gen_rhf_response

                response = _gen_rhf_response(self.gmf, hermi=hermi)
            elif self.xc_backend is not None:
                response = self.xc_backend.response(hermi)
            else:
                from gpu4pyscf.scf._response_functions import _gen_uhf_response

                response = _gen_uhf_response(
                    self.gmf,
                    mo_coeff=cp.stack((self.gmf.mo_coeff, self.gmf.mo_coeff)),
                    mo_occ=cp.stack(methods.spin_occupations(self.method, self.gmf)),
                    hermi=hermi,
                )
            self.stats['response_builds'] += 1

            def counted(density):
                # Preserve exact DF factor tags on device density matrices.
                if not isinstance(density, cp.ndarray):
                    density = cp.asarray(density)
                width = int(np.prod(density.shape[:-2]))
                if self.method.reference_kind == 'roks':
                    width //= 2
                self.stats['response_calls'] += 1
                self.stats['response_rhs'] += width
                self.stats['response_batch_widths'].append(width)
                return response(density)

            self._responses[hermi] = counted
        return self._responses[hermi]

    def orbital_response(self, orbitals, occupation):
        from gpu4pyscf.grad._response_density import OrbitalRotationDensity

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

    def use_selected_reference_hessian(self, fock_mo, response):
        self.extra['ensemble_fock_mo'] = cp.asarray(fock_mo)
        self._responses[1] = response
        self.stats['response_builds'] += 1

    def fxc_ref(self):
        if self._fxc_ref is False:
            from gpu4pyscf.sftda.nttda import spin_flip_reference_fxc

            self._fxc_ref = (
                None if self.gmf._numint._xc_type(self.gmf.xc) == 'HF' else spin_flip_reference_fxc(self.gmf)
            )
        return self._fxc_ref

    def assert_compatible(self, tdobj):
        self.validate()
        if (
            tdobj is not self._tdobj
            or methods.resolve_method(tdobj).id != self.method.id
            or tdobj._scf.mo_coeff is not self._orbitals
            or tdobj._scf.mo_occ is not self._occupations
        ):
            raise ValueError('response cache belongs to a different NTTDA evaluation')


def prepare_gradient(
    gradient_driver,
    tdobj,
    m_matrix,
    direct,
    atmlst,
    tolerance,
    max_cycle,
    *,
    jk_ledger,
    nobeta_p0,
    direct_fock_probes,
    cache=None,
    orbital_rhs_shift=None,
):
    """Package one derivative after its unrelaxed terms are assembled."""
    if cache is None:
        cache = ResponseCache(tdobj)
    return PreparedGradient(
        gradient_driver=gradient_driver,
        tdobj=tdobj,
        m_matrix=cp.asarray(m_matrix),
        direct=cp.asarray(direct),
        atmlst=tuple(atmlst),
        tolerance=float(tolerance),
        max_cycle=max_cycle,
        jk_ledger=jk_ledger,
        nobeta_p0=nobeta_p0,
        direct_fock_probes=direct_fock_probes,
        cache=cache,
        orbital_rhs_shift=orbital_rhs_shift,
    )


def _prepared_orbital_rhs(prepared, pairs):
    """Pack one adjoint RHS and apply any objective-specific additive shift."""
    rhs = pack_m_matrix(prepared.m_matrix, pairs)
    shift = prepared.orbital_rhs_shift
    if shift is None:
        return rhs
    shift = cp.asarray(shift)
    if shift.shape != rhs.shape or not cp.all(cp.isfinite(shift)):
        raise ValueError('orbital RHS shift must be finite and match the orbital pairs')
    return rhs + shift


def _prepared_zvector_data(prepared, pairs, zvector):
    backend = methods.orbital_backend(methods.get_method(prepared.tdobj))
    tdobj = prepared.tdobj
    adjoint = backend.zvector_adjoint_matrix(tdobj, pairs, zvector, cache=prepared.cache)
    rhs = _prepared_orbital_rhs(prepared, pairs)
    residual = float(cp.max(cp.abs(pack_m_matrix(adjoint, pairs) - rhs)))
    probes = backend.zvector_probe_densities(tdobj, pairs, zvector)
    return adjoint, residual, probes


def stage_fock_derivatives(prepared, probe_alpha, probe_beta, *, defer_xc=False, defer_hcore=False):
    """Append Fock terms to the task's ledger and return local derivatives.

    The prepared task holds arrays and its J/K ledger explicitly. No closure
    hides its probe, reference, or pending contractions.
    """
    from .derivative_jk import spin_fock_direct_hf, spin_fock_direct_dft

    driver, td = prepared.gradient_driver, prepared.tdobj
    options = dict(
        atmlst=prepared.atmlst,
        jk_ledger=prepared.jk_ledger,
        output_slots=('direct', 'zvector'),
        with_hcore=not defer_hcore,
    )
    if td._scf._numint._xc_type(td._scf.xc) == 'HF':
        local = spin_fock_direct_hf(driver, td, probe_alpha, probe_beta, **options)
    else:
        local = spin_fock_direct_dft(
            driver, td, probe_alpha, probe_beta, nobeta_p0=prepared.nobeta_p0, with_xc=not defer_xc, **options
        )
    return local, prepared.jk_ledger


def _finish_prepared_gradient(prepared, pairs, zvector, zvector_data=None, staged_fock_contractions=None):
    """Assemble one prepared task from a solved adjoint vector."""
    gradient_driver = prepared.gradient_driver
    tdobj = prepared.tdobj
    m_matrix = prepared.m_matrix
    direct = prepared.direct
    atmlst = prepared.atmlst
    direct_fock_probes = prepared.direct_fock_probes
    cache = prepared.cache
    if zvector_data is None:
        zvector_data = _prepared_zvector_data(prepared, pairs, zvector)
    adjoint, residual, (probe_alpha, probe_beta) = zvector_data
    if staged_fock_contractions is not None:
        direct_total = direct + staged_fock_contractions[0]
        fock_contraction = staged_fock_contractions[1]
    else:
        direct_alpha, direct_beta = direct_fock_probes
        fock_contractions, ledger = stage_fock_derivatives(
            prepared,
            cp.stack((direct_alpha, probe_alpha)),
            cp.stack((direct_beta, probe_beta)),
        )
        contracted = ledger.contract(gradient_driver, tdobj.mol, atmlst, slots=('direct', 'zvector'))
        direct_total = direct + fock_contractions[0] + contracted['direct']
        fock_contraction = fock_contractions[1] + contracted['zvector']
    overlap_derivative = cache.extra.get('overlap_derivative')
    if overlap_derivative is None:
        overlap_derivative = cp.asarray(-tdobj.mol.intor('int1e_ipovlp', comp=3))
        cache.extra['overlap_derivative'] = overlap_derivative
    orbital = _orbital_gradient(
        tdobj,
        m_matrix,
        adjoint,
        fock_contraction,
        atmlst=atmlst,
        overlap_derivative=overlap_derivative,
    )
    return GradientComponents(
        m_matrix=m_matrix,
        direct=direct_total,
        orbital=orbital,
        total=direct_total + orbital,
        zvector=zvector,
        residual=residual,
    )


def _stage_frame_jk(prepared, pairs, zvectors, zvector_data=None, section=_untimed_section):
    """Merge all post-Z ledgers when one accelerator backend is shared."""
    backends = [item.cache.ledger for item in prepared]
    if (
        backends[0] is None
        or any(backend is not backends[0] for backend in backends[1:])
        or any(item.jk_ledger is None for item in prepared)
        or any(item.direct_fock_probes is None for item in prepared)
    ):
        return None

    if zvector_data is None:
        backend = methods.orbital_backend(methods.get_method(prepared[0].tdobj))
        zvector_data = backend._prepared_zvector_batch_data(prepared, pairs, zvectors)
    with section('post_z_prepare'):
        local_contractions = []
        probe_alpha_batch = []
        probe_beta_batch = []
        merged_terms = {'j': [], 'k': []}
        slots = []
        defer_xc = all(
            methods.get_method(item.tdobj).batch_xc
            and item.tdobj._scf._numint._xc_type(
                item.tdobj._scf.xc,
            )
            in ('GGA', 'MGGA')
            for item in prepared
        )
        for index, (item, data) in enumerate(zip(prepared, zvector_data)):
            _adjoint, _residual, (probe_alpha, probe_beta) = data
            direct_alpha, direct_beta = item.direct_fock_probes
            probes_alpha = cp.stack((direct_alpha, probe_alpha))
            probes_beta = cp.stack((direct_beta, probe_beta))
            probe_alpha_batch.append(probes_alpha)
            probe_beta_batch.append(probes_beta)
            local, ledger = stage_fock_derivatives(
                item,
                probes_alpha,
                probes_beta,
                defer_xc=defer_xc,
                defer_hcore=True,
            )
            local_contractions.append(cp.asarray(local))
            for name in ('direct', 'zvector'):
                slots.append((index, name))
            for operator in ('j', 'k'):
                for term in ledger._terms[operator]:
                    merged_terms[operator].append(
                        type(term)(
                            term.left,
                            term.right,
                            term.scale,
                            term.omega,
                            (index, term.slot),
                        )
                    )

    with section('df_derivative_ledger'):
        contracted = backends[0](
            merged_terms,
            prepared[0].tdobj.mol,
            prepared[0].atmlst,
            slots=tuple(slots),
        )
    with section('post_z_hcore'):
        hcore_derivative = _hcore_derivative_generator(prepared[0].tdobj.mol)
        p_total_batch = [alpha + beta for alpha, beta in zip(probe_alpha_batch, probe_beta_batch)]
        for atom_index, atom in enumerate(prepared[0].atmlst):
            derivative = cp.asarray(hcore_derivative(atom))
            for local, p_total in zip(local_contractions, p_total_batch):
                local[:, atom_index] += cp.einsum(
                    'npq,xpq->nx',
                    p_total,
                    derivative,
                )
    with section('post_z_xc'):
        if defer_xc:
            mf = prepared[0].tdobj._scf
            xctype = mf._numint._xc_type(mf.xc)
            from .xc import _reference_spin_densities

            density_alpha, density_beta = _reference_spin_densities(
                prepared[0].tdobj,
            )
            probe_counts = [len(value) for value in probe_alpha_batch]
            frame_xc_backend = prepared[0].cache.xc_backend
            if frame_xc_backend is None:
                from . import xc as xc_backend

                xc_contractions = getattr(
                    xc_backend,
                    'contract_' + xctype.lower() + '_vxc_derivative',
                )(
                    mf,
                    density_alpha,
                    density_beta,
                    cp.concatenate(probe_alpha_batch),
                    cp.concatenate(probe_beta_batch),
                    atmlst=prepared[0].atmlst,
                    max_memory=min(item.gradient_driver.max_memory for item in prepared),
                )
            else:
                xc_contractions = getattr(
                    frame_xc_backend,
                    'contract_' + xctype.lower() + '_vxc_derivative',
                )(
                    density_alpha,
                    density_beta,
                    cp.concatenate(probe_alpha_batch),
                    cp.concatenate(probe_beta_batch),
                    atmlst=prepared[0].atmlst,
                )
            offset = 0
            for local, count in zip(local_contractions, probe_counts):
                local += xc_contractions[offset : offset + count]
                offset += count
    with section('post_z_accumulate'):
        for index, local in enumerate(local_contractions):
            local[0] += contracted[(index, 'direct')]
            local[1] += contracted[(index, 'zvector')]
    return tuple(zvector_data), tuple(local_contractions)


def pack_m_matrix(matrix, pairs):
    antisymmetric = matrix - matrix.T
    rows, columns, _alpha, _beta = _pair_index_arrays(pairs)
    return antisymmetric[rows, columns]


def _orbital_gradient(tdobj, m_matrix, adjoint, fock_contraction, atmlst=None, overlap_derivative=None):
    mol = tdobj.mol
    mf = tdobj._scf
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    mo = cp.asarray(mf.mo_coeff)
    if overlap_derivative is None:
        overlap_derivative = cp.asarray(-mol.intor('int1e_ipovlp', comp=3))
    offsets = mol.offset_nr_by_atom()
    # Folding the MO-space traces over kappa_sym = -1/2 C^dag (dS/dR) C into
    # one AO-basis weight matrix replaces the per-(atom, xyz) O(nao^3)
    # projections by a single transform plus per-atom slice contractions.
    weight = mo @ (adjoint.T - m_matrix) @ mo.conj().T
    weight = weight + weight.T
    result = cp.zeros((len(atmlst), 3))
    for k, atom in enumerate(atmlst):
        p0, p1 = offsets[atom][2:]
        result[k] = -fock_contraction[k] + 0.5 * cp.einsum(
            'xpq,pq->x',
            overlap_derivative[:, p0:p1],
            weight[p0:p1],
        )
    return result


def _solve_preconditioned_krylov(action, rhs, diagonal, tolerance, max_cycle, max_memory, initial=None):
    """Historical entry, now restarted GMRES with true-residual acceptance.

    A matrix RHS is flattened as a block-diagonal system with one shared
    Hessian. Matrix actions still receive all rows together. The absolute
    global 2-norm target guarantees the same target for every individual RHS.
    ``max_cycle`` counts global inner iterations; restart is min(20, npair).
    ``max_memory`` is retained for compatibility; restart bounds basis storage.
    """
    rhs = cp.asnumpy(cp.asarray(rhs))
    diagonal = cp.asnumpy(cp.asarray(diagonal))
    dtype = np.result_type(rhs, diagonal, np.float64)
    shape = rhs.shape
    size = rhs.size
    npair = rhs.shape[-1]
    if size == 0 or not np.any(rhs):
        return np.zeros(shape, dtype=dtype)
    last_vector = last_product = None
    iterations = 0
    tracing = os.environ.get('NTTDA_ZVECTOR_TRACE', '0') == '1'
    trace_started = time.perf_counter() if tracing else 0.0
    solve_id = '%d-%d' % (os.getpid(), time.monotonic_ns()) if tracing else None
    hvp_calls = 0
    hvp_seconds = 0.0

    def trace_event(event, **fields):
        if tracing:
            record = dict(
                event=event,
                solve_id=solve_id,
                elapsed_seconds=time.perf_counter() - trace_started,
                hvp_calls=hvp_calls,
                hvp_seconds=hvp_seconds,
                **fields,
            )
            # Failed solves may have infinite residuals; keep valid JSON.
            record = {
                key: str(value) if isinstance(value, float) and not np.isfinite(value) else value
                for key, value in record.items()
            }
            print('NTTDA_ZVECTOR ' + json.dumps(record), file=sys.stderr, flush=True)

    if tracing:
        trace_event(
            'start',
            source=os.path.realpath(__file__),
            solver='scipy.gmres',
            npair=npair,
            nrhs=size // npair,
            tolerance=float(tolerance),
            restart=min(20, npair),
            max_inner_iterations=int(max_cycle),
            rhs_l2=float(np.linalg.norm(rhs)),
            thread_env={
                name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
            },
        )

    def matvec(vector):
        nonlocal last_vector, last_product, hvp_calls, hvp_seconds
        if tracing:
            hvp_calls += 1
            trace_event('hvp_start')
            action_started = time.perf_counter()
        try:
            last_vector = np.array(vector, copy=True)
            product = cp.asnumpy(action(cp.asarray(vector.reshape(shape))))
            if product.shape != shape or not np.all(np.isfinite(product)):
                raise RuntimeError('NTTDA Z-vector Hessian action must have finite, matching output')
            last_product = product.reshape(-1)
        except Exception as error:
            if tracing:
                seconds = time.perf_counter() - action_started
                hvp_seconds += seconds
                trace_event('hvp_error', seconds=seconds, error_type=type(error).__name__)
            raise
        if tracing:
            seconds = time.perf_counter() - action_started
            hvp_seconds += seconds
            trace_event('hvp_end', seconds=seconds)
        return last_product

    def count_iteration(preconditioned_residual):
        nonlocal iterations
        iterations += 1
        if tracing:
            trace_event(
                'iteration',
                iteration=iterations,
                preconditioned_relative_residual=float(preconditioned_residual),
            )

    operator = LinearOperator((size, size), matvec=matvec, dtype=dtype)
    preconditioner = LinearOperator(
        (size, size),
        matvec=lambda vector: (vector.reshape(shape) / diagonal).ravel(),
        dtype=dtype,
    )
    solution, info = gmres(
        operator,
        rhs.ravel(),
        M=preconditioner,
        x0=None if initial is None else cp.asnumpy(cp.asarray(initial)).ravel(),
        atol=tolerance,
        **{_GMRES_RTOL_KEY: 0.0},
        restart=min(20, npair),
        maxiter=max_cycle,
        callback=count_iteration,
        callback_type='legacy',
    )
    solution = np.asarray(solution).reshape(-1)
    if solution.size == size and np.all(np.isfinite(solution)):
        if last_vector is None or not np.array_equal(solution, last_vector):
            matvec(solution)
        residual = last_product - rhs.ravel()
        residual_norm = float(np.linalg.norm(residual))
        residual_inf = float(np.max(np.abs(residual)))
    else:
        residual_norm = residual_inf = float('inf')
    if tracing:
        rejected = info != 0 or not np.isfinite(residual_norm) or residual_norm > tolerance
        trace_event(
            'rejected' if rejected else 'accepted',
            iterations=iterations,
            info=int(info),
            true_residual_l2=residual_norm,
            residual_inf=residual_inf,
            tolerance=float(tolerance),
        )
    if info != 0 or not np.isfinite(residual_norm) or residual_norm > tolerance:
        raise RuntimeError(
            'NTTDA Z-vector GMRES did not converge: '
            'true residual 2-norm=%.6e, tolerance=%.6e, '
            'residual inf-norm=%.6e, global iterations=%d/%d, info=%d'
            % (residual_norm, tolerance, residual_inf, iterations, max_cycle, info)
        )
    return solution.reshape(shape)


def solve_zvector_equations(
    action,
    pairs,
    tdobj,
    rhs,
    tolerance=1e-12,
    max_cycle=None,
    cache=None,
    initial=None,
    *,
    preconditioner,
    solver=_solve_preconditioned_krylov,
):
    """Validate, solve and accept a batch using the original equation's norm.

    Orbital backends supply their own diagonal preconditioner. Already
    converged warm guesses are removed per RHS, without a tolerance floor.
    Acceptance reuses the last action if it belongs to the returned solution;
    it also protects callers using a replacement solver at the legacy seam.
    """
    rhs = cp.asarray(rhs)
    single = rhs.ndim == 1
    if single:
        rhs = rhs.reshape(1, -1)
    elif rhs.ndim != 2:
        raise ValueError('Z-vector RHS must have shape (npair,) or (nrhs,npair)')
    if rhs.shape[1] != len(pairs):
        raise ValueError('Z-vector rhs must have one entry per orbital pair')
    if not cp.all(cp.isfinite(rhs)):
        raise ValueError('Z-vector rhs must be finite')
    if not cp.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Z-vector tolerance must be finite and positive')
    if max_cycle is not None and (not isinstance(max_cycle, (int, np.integer)) or max_cycle <= 0):
        raise ValueError('Z-vector max_cycle must be a positive integer or None')
    if initial is not None:
        initial = cp.asarray(initial)
        if single and initial.ndim == 1:
            initial = initial.reshape(1, -1)
        if initial.shape != rhs.shape or not cp.all(cp.isfinite(initial)):
            raise ValueError('Z-vector initial guess must be finite and match the RHS shape')
    solution = cp.zeros_like(rhs, dtype=cp.result_type(rhs, cp.float64))
    pending = cp.any(rhs != 0, axis=1)
    if initial is not None and cp.any(pending):
        indices = cp.flatnonzero(pending)
        product = cp.asarray(action(initial[pending]))
        if product.shape != rhs[pending].shape or not cp.all(cp.isfinite(product)):
            raise RuntimeError('NTTDA Z-vector Hessian action must have finite, matching output')
        residuals = product - rhs[pending]
        norms = cp.linalg.norm(residuals, axis=1)
        accepted = cp.isfinite(norms) & (norms <= tolerance)
        solution[indices[accepted]] = initial[indices[accepted]]
        pending[indices[accepted]] = False
    if not cp.any(pending):
        return solution[0] if single else solution

    diagonal = cp.asarray(preconditioner(tdobj, pairs, cache=cache))
    if diagonal.shape != (len(pairs),) or not cp.all(cp.isfinite(diagonal)) or cp.any(diagonal == 0):
        raise ValueError('Z-vector preconditioner must be finite, nonzero and match the orbital pairs')
    if max_cycle is None:
        max_cycle = len(pairs)
    last_vector = last_product = None

    def cached_action(vector):
        nonlocal last_vector, last_product
        last_vector = cp.array(vector, copy=True)
        last_product = cp.array(action(vector), copy=True)
        if last_product.shape != last_vector.shape or not cp.all(cp.isfinite(last_product)):
            raise RuntimeError('NTTDA Z-vector Hessian action must have finite, matching output')
        return last_product

    pending_rhs = rhs[pending]
    solved = cp.asarray(
        solver(
            cached_action,
            pending_rhs,
            diagonal,
            tolerance,
            max_cycle,
            float(getattr(tdobj, 'max_memory', 2000.0)),
            initial=None if initial is None else initial[pending],
        )
    ).reshape(pending_rhs.shape)
    if not cp.all(cp.isfinite(solved)):
        raise RuntimeError('NTTDA Z-vector solution is not finite')
    if last_vector is None or not cp.array_equal(solved, last_vector):
        cached_action(solved)
    residuals = last_product - pending_rhs
    norms = cp.linalg.norm(residuals, axis=1)
    failed = ~cp.isfinite(norms) | (norms > tolerance)
    if cp.any(failed):
        raise RuntimeError(
            'NTTDA Z-vector RHS %s did not converge: true residual 2-norms=%s, tolerance=%.6e'
            % (cp.flatnonzero(pending)[failed].tolist(), norms[failed].tolist(), tolerance)
        )
    solution[pending] = solved
    logger.debug(
        tdobj, 'NTTDA Z-vector maximum true residual 2-norm=%.6e, tolerance=%.6e', float(cp.max(norms)), tolerance
    )
    return solution[0] if single else solution


def finish_prepared_gradients(prepared, initial=None, *, timings=None, synchronize=None):
    """Solve and assemble one method's tasks through its orbital backend.

    ``timings`` enables exclusive wall-time sections. ``synchronize`` may be
    a CUDA stream synchronizer; neither argument changes the scientific path.
    """
    timer = None if timings is None else _FinishTimings(timings, synchronize)
    section = _untimed_section if timer is None else timer.section
    prepared = tuple(prepared)
    if not prepared:
        if timer is not None:
            timer.finish()
        return tuple()
    try:
        with section('setup'):
            reference = prepared[0]
            method = methods.get_method(reference.tdobj)
            for item in prepared:
                if methods.get_method(item.tdobj).id != method.id:
                    raise ValueError('prepared gradients use different NTTDA methods')
                item.cache.assert_compatible(item.tdobj)
            backend = methods.orbital_backend(method)
            pairs = backend.canonical_pairs(reference.tdobj, compact=True)
            for item in prepared[1:]:
                item_pairs = backend.canonical_pairs(item.tdobj, compact=True)
                if item_pairs != pairs:
                    raise ValueError('prepared gradients use different orbital pair spaces')
                if item.atmlst != reference.atmlst:
                    raise ValueError('prepared gradients use different atom lists')
                if not cp.array_equal(item.tdobj._scf.mo_occ, reference.tdobj._scf.mo_occ):
                    raise ValueError('prepared gradients use different occupations')
                if not cp.allclose(item.tdobj._scf.mo_coeff, reference.tdobj._scf.mo_coeff, atol=1e-12, rtol=0.0):
                    raise ValueError('prepared gradients use different orbitals')
            action, _pairs = backend.make_hessian_transpose_action(
                reference.tdobj,
                pairs=pairs,
                cache=reference.cache,
            )
            rhs = cp.asarray([_prepared_orbital_rhs(item, pairs) for item in prepared])
            tolerance = min(item.tolerance for item in prepared)
            finite_cycles = [item.max_cycle for item in prepared if item.max_cycle is not None]
            max_cycle = max(finite_cycles) if finite_cycles else None
        with section('zvector_solve'):
            zvectors = backend.solve_zvectors(
                action,
                pairs,
                reference.tdobj,
                rhs,
                tolerance=tolerance,
                max_cycle=max_cycle,
                cache=reference.cache,
                initial=initial,
            )
        with section('zvector_data'):
            zvector_data = backend._prepared_zvector_batch_data(
                prepared,
                pairs,
                zvectors,
            )
            zvector_data = tuple(
                (
                    adjoint,
                    float(cp.max(cp.abs(pack_m_matrix(adjoint, pairs) - _prepared_orbital_rhs(item, pairs)))),
                    probes,
                )
                for item, (adjoint, _residual, probes) in zip(
                    prepared,
                    zvector_data,
                )
            )
        staged = _stage_frame_jk(
            prepared,
            pairs,
            zvectors,
            zvector_data=zvector_data,
            section=section,
        )
        with section('final_assembly'):
            if staged is not None:
                zvector_data, fock_contractions = staged
                return tuple(
                    _finish_prepared_gradient(
                        item,
                        pairs,
                        zvector,
                        zvector_data=data,
                        staged_fock_contractions=fock,
                    )
                    for item, zvector, data, fock in zip(
                        prepared,
                        zvectors,
                        zvector_data,
                        fock_contractions,
                    )
                )
            return tuple(
                _finish_prepared_gradient(
                    item,
                    pairs,
                    zvector,
                    zvector_data=data,
                )
                for item, zvector, data in zip(
                    prepared,
                    zvectors,
                    zvector_data,
                )
            )
    finally:
        if timer is not None:
            timer.finish()
