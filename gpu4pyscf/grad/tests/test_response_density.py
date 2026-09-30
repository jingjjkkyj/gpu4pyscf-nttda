"""Exact orbital densities and fixed DF contractions against dense oracles."""

from unittest import mock

import cupy as cp
import numpy as np
import pytest
from pyscf import gto

from gpu4pyscf.df.df import DF
from gpu4pyscf.grad._response_density import OrbitalRotationDensity


def case(width=2):
    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='sto-3g', verbose=0)
    rng = np.random.default_rng(97)
    c = np.linalg.qr(rng.normal(size=(mol.nao, mol.nao)))[0]
    n = np.array([2., 2., 1., 1., 0., 0., 0.])
    k = rng.normal(size=(width, mol.nao, mol.nao))
    k -= k.swapaxes(-1, -2)
    if width == 1:
        k = k[0]
    dense = c @ (k * (n[None, :] - n[:, None])) @ c.T
    builder = OrbitalRotationDensity(c, n)
    return mol, builder, cp.asarray(k), cp.asarray(dense)


@pytest.mark.parametrize('width', [1, 2, 3])
@pytest.mark.parametrize('with_j,with_k', [(True, True), (True, False), (False, True)])
def test_exact_density_and_jk(width, with_j, with_k):
    mol, builder, k, dense = case(width)
    df = DF(mol, auxbasis='weigend')
    dm = builder(k)
    np.testing.assert_allclose(cp.asnumpy(dm), cp.asnumpy(dense), atol=2e-14, rtol=0)
    expected = df.get_jk(dense, hermi=1, with_j=with_j, with_k=with_k)
    for _ in range(2):
        actual = df.get_jk(dm, hermi=1, with_j=with_j, with_k=with_k)
        for a, b in zip(actual, expected):
            if b is not None:
                np.testing.assert_allclose(cp.asnumpy(a), cp.asnumpy(b), atol=2e-12, rtol=0)
    if with_k:
        assert builder.cache.hits > 0
        assert builder.cache.nbytes > 0
    builder.clear()
    assert builder.cache.nbytes == 0


def test_cache_budget_reset_range_separation_and_block_ranges():
    mol, builder, k, dense = case()
    df = DF(mol, auxbasis='weigend')

    def check(omega=None):
        a = df.get_jk(builder(k), hermi=1, omega=omega)
        b = df.get_jk(dense, hermi=1, omega=omega)
        for x, y in zip(a, b):
            np.testing.assert_allclose(cp.asnumpy(x), cp.asnumpy(y), atol=2e-12, rtol=0)

    check()
    check(.3)
    check(.3)
    assert builder.cache.hits > 0
    with mock.patch.object(df, 'get_blksize', return_value=16):
        check()
        check()
    # Different geometry, same AO count: old contractions must never be used.
    moved = mol.copy()
    moved.set_geom_('O 0 0 0; H 0 0 1.1; H 0 1 0')
    df.reset(moved)
    check()
    builder.clear()
    builder.cache.max_bytes = 1
    check()
    assert builder.cache.nbytes == 0


def test_factor_ownership_and_complex_dense_fallback():
    mol, builder, k, dense = case()
    # The owner snapshots orbitals rather than aliasing mutable SCF state.
    orbitals = builder.orbitals.copy()
    other = OrbitalRotationDensity(orbitals, builder.occupation)
    orbitals[:] = 0
    np.testing.assert_allclose(cp.asnumpy(other(k)), cp.asnumpy(dense), atol=2e-14, rtol=0)
    complex_builder = OrbitalRotationDensity(builder.orbitals.astype(complex),
                                             builder.occupation)
    result = complex_builder(k)
    assert not hasattr(result, 'factor_l')
    np.testing.assert_allclose(cp.asnumpy(result), cp.asnumpy(dense), atol=2e-14, rtol=0)


def test_nttda_hessian_seam_matches_dense_response():
    from types import SimpleNamespace
    from pyscf.grad.nttda.orbital import (
        ensemble_make_hessian_transpose_action as make_hessian_transpose_action,
        ensemble_canonical_pairs as canonical_pairs)

    mol, builder, _, _ = case()
    c = cp.asnumpy(builder.orbitals)
    n = cp.asnumpy(builder.occupation)
    rng = np.random.default_rng(73)
    fock = rng.normal(size=(mol.nao, mol.nao))
    fock += fock.T
    df = DF(mol, auxbasis='weigend')

    def response(dm):
        j, k = df.get_jk(cp.asarray(dm) if isinstance(dm, np.ndarray) else dm, hermi=1)
        return cp.asnumpy(j - .1 * k)

    dense_cache = SimpleNamespace(extra={'ensemble_fock_mo': fock},
                                  response=lambda hermi: response)

    def factory(orbitals, occupation):
        def action(rotation):
            return response(builder(rotation))
        action.clear = builder.clear
        return action

    fast_cache = SimpleNamespace(extra=dense_cache.extra, response=dense_cache.response,
                                 orbital_response=factory)
    td = SimpleNamespace(_scf=SimpleNamespace(mo_coeff=c, mo_occ=n))
    pairs = canonical_pairs(td)
    dense_action, _ = make_hessian_transpose_action(td, cache=dense_cache)
    fast_action, _ = make_hessian_transpose_action(td, cache=fast_cache)
    for shape in [(len(pairs),), (2, len(pairs))]:
        v = rng.normal(size=shape)
        np.testing.assert_allclose(fast_action(v), dense_action(v), atol=2e-12, rtol=0)
    assert builder.cache.hits > 0
    fast_action.clear()
    assert builder.cache.nbytes == 0


def test_nondefault_stream_and_failed_solve_cleanup():
    from gpu4pyscf.grad.ensemble_roks import ReferenceGradients
    from pyscf.grad.nttda import orbital as ensemble, response

    mol, builder, k, dense = case()
    df = DF(mol, auxbasis='weigend')
    expected = df.get_jk(dense, hermi=1)
    cp.cuda.get_current_stream().synchronize()
    for _ in range(2):
        with cp.cuda.Stream(non_blocking=True):
            actual = df.get_jk(builder(k), hermi=1)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(cp.asnumpy(a), cp.asnumpy(b), atol=2e-12, rtol=0)
    assert builder.cache.hits > 0

    def action(vector):
        return vector
    action.clear = builder.clear
    with mock.patch.object(ensemble, 'solve_zvector_equations', side_effect=RuntimeError('failed')):
        with pytest.raises(RuntimeError, match='failed'):
            ensemble.solve_zvectors(action, (), None, None)
    assert builder.cache.nbytes == 0

    df.get_jk(builder(k), hermi=1)
    driver = ReferenceGradients.__new__(ReferenceGradients)
    driver._rotation_density = builder
    with mock.patch.object(driver, '_solve_z_impl', side_effect=RuntimeError('failed')):
        with pytest.raises(RuntimeError, match='failed'):
            driver._solve_z()
    assert builder.cache.nbytes == 0
