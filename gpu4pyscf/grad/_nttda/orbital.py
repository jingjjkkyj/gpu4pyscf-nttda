"""Orbital Hessians and adjoints for ROKS and ensemble occupations.

Public functions dispatch on the bound physical reference. The two sets of
weights are explicit below; they never share spin weights or occupations.
"""

from .response import ResponseCache, pack_m_matrix, _solve_preconditioned_krylov, solve_zvector_equations
from .frame import _pair_index_arrays, _unpack_pair_sources
from gpu4pyscf.sftda import nttda_methods as methods
from pyscf.scf import hf
import cupy as cp


def roks_prepared_zvector_batch_data(prepared, pairs, zvectors):
    """Build all post-solve adjoints and AO probes in one response pass."""
    prepared = tuple(prepared)
    reference = prepared[0]
    tdobj = reference.tdobj
    vectors = cp.asarray(zvectors).reshape(len(prepared), len(pairs))
    mo = cp.asarray(tdobj._scf.mo_coeff)
    rows, columns, alpha, beta = _pair_index_arrays(pairs)
    source_alpha, source_beta = _unpack_pair_sources(
        vectors,
        mo.shape[1],
        rows,
        columns,
        alpha,
        beta,
    )
    adjoints = _hessian_transpose_dense(
        tdobj,
        source_alpha,
        source_beta,
        reference.cache,
    )
    probe_alpha = mo @ source_alpha @ mo.conj().T
    probe_beta = mo @ source_beta @ mo.conj().T
    output = []
    for index, item in enumerate(prepared):
        rhs = pack_m_matrix(item.m_matrix, pairs)
        residual = float(
            cp.max(
                cp.abs(
                    pack_m_matrix(adjoints[index], pairs) - rhs,
                )
            )
        )
        output.append(
            (
                adjoints[index],
                residual,
                (probe_alpha[index], probe_beta[index]),
            )
        )
    return tuple(output)


def roks_canonical_pairs(tdobj, compact=True):
    """Canonical spatial-orbital rotations and their ROKS residual type."""
    occ = cp.asarray(tdobj._scf.mo_occ)
    closed = cp.flatnonzero(occ == 2)
    open_ = cp.flatnonzero(occ == 1)
    virtual = cp.flatnonzero(occ == 0)
    pairs = []
    if not compact:
        for indices, name in ((closed, 'cc'), (open_, 'oo'), (virtual, 'vv')):
            for p_local in range(1, len(indices)):
                for q_local in range(p_local):
                    pairs.append((indices[p_local], indices[q_local], name))
    pairs.extend((o, c, 'co') for o in open_ for c in closed)
    pairs.extend((v, c, 'cv') for v in virtual for c in closed)
    pairs.extend((v, o, 'ov') for v in virtual for o in open_)
    return tuple(pairs)


def _spin_focks_mo(mf):
    return methods.spin_focks_mo(methods.method_for_reference(mf), mf)


def _response_reference(mf):
    return methods.response_reference(methods.method_for_reference(mf), mf)


def _hessian_transpose_dense(tdobj, source_alpha, source_beta, cache):
    """Batched full-matrix action of the transpose ROKS Hessian.

    ``source_alpha``/``source_beta`` carry a leading batch dimension; the
    response closure receives all densities of the batch in one call.
    """
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    fock_alpha, fock_beta = cache.spin_focks_mo()
    occupation_alpha = (occ > 0).astype(float)
    occupation_beta = (occ == 2).astype(float)
    gradient = fock_alpha @ (source_alpha + source_alpha.transpose(0, 2, 1))
    gradient += fock_beta @ (source_beta + source_beta.transpose(0, 2, 1))
    density_alpha = mo @ source_alpha @ mo.conj().T
    density_beta = mo @ source_beta @ mo.conj().T
    density_alpha = 0.5 * (density_alpha + density_alpha.transpose(0, 2, 1))
    density_beta = 0.5 * (density_beta + density_beta.transpose(0, 2, 1))
    potentials = cache.response(1)(
        cp.stack((density_alpha, density_beta)),
    )
    potential_alpha = mo.conj().T @ cp.asarray(potentials[0]) @ mo
    potential_beta = mo.conj().T @ cp.asarray(potentials[1]) @ mo
    gradient += potential_alpha * occupation_alpha[None, :]
    gradient += potential_alpha.transpose(0, 2, 1) * occupation_alpha[None, :]
    gradient += potential_beta * occupation_beta[None, :]
    gradient += potential_beta.transpose(0, 2, 1) * occupation_beta[None, :]
    return gradient


def roks_make_hessian_transpose_action(tdobj, pairs=None, cache=None):
    """Return a matrix-free action for the transpose ROKS Hessian."""
    if pairs is None:
        pairs = roks_canonical_pairs(tdobj, compact=True)
    if cache is None:
        cache = ResponseCache(tdobj)
    nmo = cp.asarray(tdobj._scf.mo_coeff).shape[1]
    rows, columns, alpha, beta = _pair_index_arrays(pairs)

    def apply(vector):
        vector = cp.asarray(vector)
        single = vector.ndim == 1
        vectors = vector.reshape(-1, len(pairs))
        source_alpha, source_beta = _unpack_pair_sources(
            vectors,
            nmo,
            rows,
            columns,
            alpha,
            beta,
        )
        gradient = _hessian_transpose_dense(
            tdobj,
            source_alpha,
            source_beta,
            cache,
        )
        packed = gradient[:, rows, columns] - gradient[:, columns, rows]
        return packed[0] if single else packed

    return apply, pairs


def roks_preconditioner(tdobj, pairs, cache=None):
    if cache is None:
        cache = ResponseCache(tdobj)
    fock_alpha, fock_beta = cache.spin_focks_mo()
    epsilon_alpha = cp.diag(fock_alpha)
    epsilon_beta = cp.diag(fock_beta)
    rows, columns, alpha, beta = _pair_index_arrays(pairs)
    diagonal = alpha * (epsilon_alpha[rows] - epsilon_alpha[columns]) + beta * (
        epsilon_beta[rows] - epsilon_beta[columns]
    )
    small = cp.abs(diagonal) < 1e-8
    diagonal[small] = cp.where(diagonal[small] < 0.0, -1e-8, 1e-8)
    return diagonal


def _unpack_zvector_source(tdobj, pairs, zvector):
    nmo = cp.asarray(tdobj._scf.mo_coeff).shape[1]
    rows, columns, alpha, beta = _pair_index_arrays(pairs)
    source_alpha, source_beta = _unpack_pair_sources(
        cp.asarray(zvector).reshape(1, -1),
        nmo,
        rows,
        columns,
        alpha,
        beta,
    )
    return source_alpha[0], source_beta[0]


def roks_zvector_adjoint_matrix(tdobj, pairs, zvector, cache=None):
    """Full MO adjoint matrix satisfying ``z.H(kappa)=Tr(G.T kappa)``."""
    if cache is None:
        cache = ResponseCache(tdobj)
    source_alpha, source_beta = _unpack_zvector_source(
        tdobj,
        pairs,
        zvector,
    )
    return _hessian_transpose_dense(
        tdobj,
        source_alpha[None],
        source_beta[None],
        cache,
    )[0]


def roks_zvector_probe_densities(tdobj, pairs, zvector):
    mo = cp.asarray(tdobj._scf.mo_coeff)
    source_alpha, source_beta = _unpack_zvector_source(
        tdobj,
        pairs,
        zvector,
    )
    return (
        mo @ source_alpha @ mo.conj().T,
        mo @ source_beta @ mo.conj().T,
    )


def ensemble_canonical_pairs(tdobj, compact=True):
    """Independent rotations ``(lower occupation p, higher occupation q)``.

    Same-occupation rotations leave the ensemble density unchanged and are
    redundant.  The returned names therefore cover only ``CV/CO/OV``.
    """
    occupation = cp.asarray(tdobj._scf.mo_occ)
    labels = {2: 'c', 1: 'o', 0: 'v'}
    rows, columns = cp.where(cp.asarray(hf.uniq_var_indices(cp.asnumpy(occupation))))
    return tuple(
        (int(p), int(q), labels[int(occupation[q])] + labels[int(occupation[p])]) for p, q in zip(rows, columns)
    )


def _pair_indices(pairs):
    rows = cp.asarray([p for p, _q, _name in pairs], dtype=cp.intp)
    columns = cp.asarray([q for _p, q, _name in pairs], dtype=cp.intp)
    return rows, columns


def _rotation_matrices(vectors, pairs, nmo):
    """Unpack independent variables into anti-Hermitian ``kappa`` matrices."""
    vectors = cp.asarray(vectors)
    single = vectors.ndim == 1
    vectors = vectors.reshape(-1, len(pairs))
    rows, columns = _pair_indices(pairs)
    rotation = cp.zeros((len(vectors), nmo, nmo))
    rotation[:, rows, columns] = vectors
    rotation[:, columns, rows] = -vectors
    return rotation[0] if single else rotation


def _weighted_sources(tdobj, pairs, vectors):
    """Build ``Z_pq = z_pq (n_q-n_p)`` in the full MO space."""
    occupation = cp.asarray(tdobj._scf.mo_occ)
    nmo = occupation.size
    vectors = cp.asarray(vectors)
    single = vectors.ndim == 1
    vectors = vectors.reshape(-1, len(pairs))
    rows, columns = _pair_indices(pairs)
    source = cp.zeros((len(vectors), nmo, nmo))
    source[:, rows, columns] = vectors * (occupation[columns] - occupation[rows])
    return source[0] if single else source


def _fock_mo(mf, cache=None):
    if cache is not None and 'ensemble_fock_mo' in cache.extra:
        return cache.extra['ensemble_fock_mo']
    orbitals = cp.asarray(mf.mo_coeff)
    fock = orbitals.conj().T @ cp.asarray(mf.get_fock()) @ orbitals
    if cache is not None:
        cache.extra['ensemble_fock_mo'] = fock
    return fock


def ensemble_make_hessian_transpose_action(tdobj, pairs=None, cache=None):
    """Return the symmetric average-occupation orbital Hessian action.

    For a trial rotation ``kappa``, ``density_mo = [kappa,n]`` and
    ``dF_mo = [F0,kappa] + C^H V_RKS[dD] C``.  Multiplication by the outer
    occupation difference differentiates the stationary residual itself.
    """
    mf = tdobj._scf
    orbitals = cp.asarray(mf.mo_coeff)
    occupation = cp.asarray(mf.mo_occ)
    occupation_difference = occupation[None, :] - occupation[:, None]
    nmo = orbitals.shape[1]
    if pairs is None:
        pairs = ensemble_canonical_pairs(tdobj)
    fock = _fock_mo(mf, cache=cache)
    response = cache.response(1) if cache is not None else mf.gen_response(hermi=1)
    orbital_response_factory = getattr(cache, 'orbital_response', None)
    orbital_response = orbital_response_factory(orbitals, occupation) if orbital_response_factory is not None else None
    rows, columns = _pair_indices(pairs)

    def apply(vector):
        vector = cp.asarray(vector)
        single = vector.ndim == 1
        rotation = _rotation_matrices(vector, pairs, nmo)
        rotations = rotation[None] if single else rotation
        # [kappa,n]_pq = (n_q-n_p) kappa_pq.  Keeping this directed matrix
        # avoids introducing a factor of two before the RKS response call.
        if orbital_response is None:
            density_mo = rotations * occupation_difference
            density_ao = orbitals @ density_mo @ orbitals.conj().T
            potential = response(density_ao)
        else:
            potential = orbital_response(rotations)
        potential_mo = orbitals.conj().T @ potential @ orbitals
        fock_derivative = fock @ rotations - rotations @ fock + potential_mo
        output = occupation_difference[rows, columns][None] * fock_derivative[:, rows, columns]
        return output[0] if single else output

    if orbital_response is not None:
        apply.clear = orbital_response.clear
    return apply, pairs


def ensemble_zvector_adjoint_matrix(tdobj, pairs, zvector, cache=None):
    """Return the full MO coefficient derivative of ``z . g_orbital``.

    The packed residual of this matrix is ``H.T z``.  Retaining the full
    matrix is also required by the final overlap/Pulay contraction.
    """
    mf = tdobj._scf
    orbitals = cp.asarray(mf.mo_coeff)
    occupation = cp.asarray(mf.mo_occ)
    zvector = cp.asarray(zvector)
    single = zvector.ndim == 1
    source = _weighted_sources(tdobj, pairs, zvector)
    sources = source[None] if single else source
    fock = _fock_mo(mf, cache=cache)
    gradient = fock @ (sources + sources.swapaxes(-1, -2))

    density = orbitals @ sources @ orbitals.conj().T
    density = 0.5 * (density + density.conj().swapaxes(-1, -2))
    response = cache.response(1) if cache is not None else mf.gen_response(hermi=1)
    potential = response(density)
    potential = orbitals.conj().T @ potential @ orbitals
    gradient += potential * occupation[None, :]
    gradient += potential.conj().swapaxes(-1, -2) * occupation[None, :]
    return gradient[0] if single else gradient


def ensemble_zvector_probe_densities(tdobj, pairs, zvector):
    """Equal-spin AO probes for the nuclear derivative of the common Fock.

    ``total`` is the directed occupation-weighted Z source.  Both spin
    channels receive exactly half; assigning ROKS alpha/beta weights here
    would silently change the EnsembleRKS Hessian derivative.
    """
    orbitals = cp.asarray(tdobj._scf.mo_coeff)
    source = _weighted_sources(tdobj, pairs, zvector)
    total = orbitals @ source @ orbitals.conj().T
    return 0.5 * total, 0.5 * total


def ensemble_preconditioner(tdobj, pairs, cache=None):
    occupation = cp.asarray(tdobj._scf.mo_occ)
    epsilon = cp.diag(_fock_mo(tdobj._scf, cache=cache))
    diagonal = cp.asarray([(occupation[q] - occupation[p]) * (epsilon[p] - epsilon[q]) for p, q, _name in pairs])
    small = cp.abs(diagonal) < 1e-8
    diagonal[small] = cp.where(diagonal[small] < 0.0, -1e-8, 1e-8)
    return diagonal


def ensemble_prepared_zvector_batch_data(prepared, pairs, zvectors):
    """Build ensemble adjoints and equal-spin AO probes for a solved batch."""
    reference = prepared[0]
    rhs = cp.asarray([pack_m_matrix(item.m_matrix, pairs) for item in prepared])
    adjoints = ensemble_zvector_adjoint_matrix(
        reference.tdobj,
        pairs,
        zvectors,
        cache=reference.cache,
    )
    probe_alpha, probe_beta = ensemble_zvector_probe_densities(
        reference.tdobj,
        pairs,
        zvectors,
    )
    return tuple(
        (
            adjoint,
            float(cp.max(cp.abs(pack_m_matrix(adjoint, pairs) - item_rhs))),
            (alpha, beta),
        )
        for item, item_rhs, adjoint, alpha, beta in zip(
            prepared,
            rhs,
            adjoints,
            probe_alpha,
            probe_beta,
        )
    )


def canonical_pairs(tdobj, compact=True):
    if methods.get_method(tdobj).reference_kind == 'roks':
        return roks_canonical_pairs(tdobj, compact=compact)
    return ensemble_canonical_pairs(tdobj, compact=compact)


def make_hessian_transpose_action(tdobj, pairs=None, cache=None):
    if methods.get_method(tdobj).reference_kind == 'roks':
        return roks_make_hessian_transpose_action(tdobj, pairs=pairs, cache=cache)
    return ensemble_make_hessian_transpose_action(tdobj, pairs=pairs, cache=cache)


def zvector_adjoint_matrix(tdobj, pairs, zvector, cache=None):
    if methods.get_method(tdobj).reference_kind == 'roks':
        return roks_zvector_adjoint_matrix(tdobj, pairs, zvector, cache=cache)
    return ensemble_zvector_adjoint_matrix(tdobj, pairs, zvector, cache=cache)


def zvector_probe_densities(tdobj, pairs, zvector):
    if methods.get_method(tdobj).reference_kind == 'roks':
        return roks_zvector_probe_densities(tdobj, pairs, zvector)
    return ensemble_zvector_probe_densities(tdobj, pairs, zvector)


def _preconditioner(tdobj, pairs, cache=None):
    if methods.get_method(tdobj).reference_kind == 'roks':
        return roks_preconditioner(tdobj, pairs, cache=cache)
    return ensemble_preconditioner(tdobj, pairs, cache=cache)


def _prepared_zvector_batch_data(prepared, pairs, zvectors):
    if methods.get_method(prepared[0].tdobj).reference_kind == 'roks':
        return roks_prepared_zvector_batch_data(prepared, pairs, zvectors)
    return ensemble_prepared_zvector_batch_data(prepared, pairs, zvectors)


def solve_zvectors(action, pairs, tdobj, rhs, tolerance=1e-12, max_cycle=None, cache=None, initial=None):
    """Use the selected Hessian diagonal and require the true residual target."""
    try:
        return solve_zvector_equations(
            action,
            pairs,
            tdobj,
            rhs,
            tolerance=tolerance,
            max_cycle=max_cycle,
            cache=cache,
            initial=initial,
            preconditioner=_preconditioner,
            solver=_solve_preconditioned_krylov,
        )
    finally:
        clear = getattr(action, 'clear', None)
        if clear is not None:
            clear()


def solve_zvector(action, pairs, tdobj, rhs, tolerance=1e-12, max_cycle=None, cache=None, initial=None):
    return solve_zvectors(
        action, pairs, tdobj, rhs, tolerance=tolerance, max_cycle=max_cycle, cache=cache, initial=initial
    )
