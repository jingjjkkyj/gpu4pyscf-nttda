"""Exact DF exchange factors and same-evaluation ensemble Fock reuse."""

import os
from unittest import mock

import cupy as cp
import numpy as np
import pytest
from pyscf import gto

from gpu4pyscf.dft.roks import ROKS
from gpu4pyscf.sftda import EnsembleRKS, EnsembleROKS, NTTDA
from gpu4pyscf.sftda.nttda import _transition_density
from gpu4pyscf.grad.nttda_context import (
    EvaluationContext, make_gpu_response_cache,
)


def make_reference(kind, xc='B3LYP', df=True):
    mol = gto.M(
        atom='C 0.02 -0.03 0.01; H -0.02 0.8 0.62; H 0.03 -0.91 0.5',
        basis='sto-3g', spin=2, verbose=0,
    )
    mf = kind(mol, xc=xc)
    if df:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level = 0
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf


@pytest.mark.parametrize('nleft,nright', [(2, 5), (5, 2), (2, 2)])
def test_transition_density_keeps_exact_directed_factors(nleft, nright):
    rng = np.random.default_rng(18)
    left = cp.asarray(rng.normal(size=(8, nleft)))
    right = cp.asarray(rng.normal(size=(8, nright)))
    amplitude = cp.asarray(rng.normal(size=(3, nleft, nright)))
    dense = _transition_density(amplitude, left, right)
    tagged = _transition_density(amplitude, left, right, True)
    reconstructed = tagged.factor_l @ tagged.factor_r.swapaxes(-1, -2)
    np.testing.assert_allclose(cp.asnumpy(tagged), cp.asnumpy(dense), atol=0, rtol=0)
    np.testing.assert_allclose(
        cp.asnumpy(reconstructed), cp.asnumpy(dense), atol=2e-13, rtol=1e-14,
    )
    assert tagged.factor_l.shape[-1] == min(nleft, nright)
    assert tagged.symmetrize == 0


CASES = (
    (ROKS, 'B3LYP', True),
    (EnsembleRKS, 'B3LYP', True),
    (EnsembleROKS, 'B3LYP', True),
    (ROKS, 'CAM-B3LYP', True),
    (ROKS, 'B3LYP', False),
    (ROKS, 'PBE', True),
)


@pytest.fixture(scope='module', params=CASES)
def operator_reference(request):
    return make_reference(*request.param), request.param


def test_dense_and_factorized_full_operator(operator_reference):
    mf, (_kind, xc, df) = operator_reference
    td = NTTDA(mf)
    rng = np.random.default_rng(72)
    for delta_s in (-1, 0):
        td.deltaS = delta_s
        gen = td.gen_vind_sfd if delta_s == -1 else td.gen_vind_sc
        with mock.patch.dict(os.environ, {'NTTDA_DF_EXCHANGE_BACKEND': 'dense'}):
            dense, diagonal = gen()
        with mock.patch.dict(os.environ, {'NTTDA_DF_EXCHANGE_BACKEND': 'factorized'}):
            factors, other_diagonal = gen()
        expected_backend = 'factorized' if df and xc != 'PBE' and delta_s == -1 else 'dense'
        assert td._nttda_df_exchange_backend == expected_backend
        np.testing.assert_allclose(
            cp.asnumpy(diagonal), cp.asnumpy(other_diagonal), atol=1e-12, rtol=0,
        )
        for width in (1, 3, 6):
            vectors = cp.asarray(rng.normal(size=(width, diagonal.size)))
            expected = dense(vectors)
            with mock.patch.object(mf, 'get_k', wraps=mf.get_k) as exchange:
                actual = factors(vectors)
            np.testing.assert_allclose(
                cp.asnumpy(actual), cp.asnumpy(expected), atol=2e-10, rtol=1e-11,
            )
            if expected_backend == 'factorized':
                assert exchange.call_count >= 4
                assert all(hasattr(call.args[1], 'factor_l')
                           for call in exchange.call_args_list)
        if df and delta_s == -1:
            with mock.patch.object(mf, 'only_dfj', True):
                with mock.patch.dict(os.environ, {'NTTDA_DF_EXCHANGE_BACKEND': 'factorized'}):
                    gen()
                assert td._nttda_df_exchange_backend == 'dense'


@pytest.mark.parametrize('kind', [EnsembleRKS, EnsembleROKS])
@pytest.mark.parametrize('df', [False, True])
def test_ensemble_fock_cache_matches_cpu_and_hessian(kind, df):
    from pyscf.grad.nttda import ensemble

    td = NTTDA(make_reference(kind, df=df))
    td.set(nstates=3, conv_tol=1e-10, max_cycle=200).run()
    assert np.all(td.converged)
    with mock.patch.dict(os.environ, {'NTTDA_ENSEMBLE_FOCK_CACHE': '1'}):
        context = EvaluationContext(td)
        cache = context.response_cache
    with mock.patch.dict(os.environ, {'NTTDA_ENSEMBLE_FOCK_CACHE': '0'}):
        legacy = context.fresh_response_cache()
    assert 'ensemble_fock_mo' not in legacy.extra
    expected = ensemble._fock_mo(context.cpu_td._scf, legacy)
    assert cache.stats['ensemble_fock_cache_hits'] == 1
    np.testing.assert_allclose(cache.extra['ensemble_fock_mo'], expected, atol=2e-10, rtol=0)
    old_action, pairs = ensemble.make_hessian_transpose_action(context.cpu_td, cache=legacy)
    with mock.patch.object(context.cpu_td._scf, 'get_fock',
                           side_effect=AssertionError('unexpected Fock rebuild')):
        action, new_pairs = ensemble.make_hessian_transpose_action(context.cpu_td, cache=cache)
        assert new_pairs == pairs
        rng = np.random.default_rng(46)
        for shape in ((len(pairs),), (3, len(pairs))):
            vectors = rng.normal(size=shape)
            np.testing.assert_allclose(action(vectors), old_action(vectors), atol=2e-9, rtol=0)
    other = EvaluationContext(td).response_cache
    assert not np.shares_memory(cache.extra['ensemble_fock_mo'], other.extra['ensemble_fock_mo'])
    td.xy = list(td.xy)
    with pytest.raises(ValueError, match='create a new derivative driver'):
        context.validate()


def test_missing_gpu_fock_keeps_cpu_fallback_and_roks_is_unchanged():
    from gpu4pyscf.grad.nttda_bridge import build_cpu_twin

    for kind in (ROKS, EnsembleRKS):
        td = NTTDA(make_reference(kind))
        td.set(nstates=3, conv_tol=1e-10, max_cycle=200).run()
        cpu_td = build_cpu_twin(td)
        if kind is ROKS:
            cache = make_gpu_response_cache(cpu_td, td._scf, xc_backend=None)
            assert 'ensemble_fock_mo' not in cache.extra
        else:
            del cpu_td._nttda_gpu_fock0_fockz
            cache = make_gpu_response_cache(cpu_td, td._scf, xc_backend=None)
            assert 'ensemble_fock_mo' not in cache.extra


def test_unknown_exchange_backend_is_rejected():
    mf = make_reference(EnsembleROKS)
    with mock.patch.dict(os.environ, {'NTTDA_DF_EXCHANGE_BACKEND': 'typo'}):
        with pytest.raises(ValueError, match='NTTDA_DF_EXCHANGE_BACKEND'):
            NTTDA(mf).gen_vind_sfd()
