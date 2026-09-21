"""Four GPU method entries, shared frames and cache ownership on real CUDA."""

import numpy as np
import pytest

from pyscf import gto
from gpu4pyscf.dft.roks import ROKS
from gpu4pyscf.sftda import (
    EnsembleRKS, EnsembleROKS, NTTDA, NTTDA_ROKS, NTTDA_ROKS_NoBeta,
    NTTDA_EnsembleRKS, NTTDA_EnsembleROKS,
)
from gpu4pyscf.grad.nttda import compute_frame, make_frame_cache
from gpu4pyscf.grad.nttda_context import EvaluationContext
from gpu4pyscf.grad.nttda_bridge import build_cpu_twin


CASES = (
    (ROKS, NTTDA_ROKS, 'roks', False),
    (ROKS, NTTDA_ROKS_NoBeta, 'roks_nobeta', True),
    (EnsembleRKS, NTTDA_EnsembleRKS, 'ensemble_rks', False),
    (EnsembleROKS, NTTDA_EnsembleROKS, 'ensemble_roks', False),
)


def reference(kind, df=False):
    mol = gto.M(
        atom='C 0.02 -0.03 0.01; H -0.02 0.8 0.62; H 0.03 -0.91 0.5',
        basis='sto-3g', spin=2, verbose=0,
    )
    mf = kind(mol, xc='PBE')
    if df:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level = 0
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    assert mf.converged
    return mf


def solve(td, delta_s=-1):
    td.set(deltaS=delta_s, nstates=3, conv_tol=1e-10, max_cycle=200).run()
    assert np.all(td.converged)
    return td


@pytest.mark.parametrize('case', CASES, ids=[case[2] for case in CASES])
@pytest.mark.parametrize('delta_s', [-1, 0])
@pytest.mark.parametrize('df', [False, True])
def test_explicit_legacy_and_cpu_twin_agree(case, delta_s, df):
    kind, factory, ident, nobeta = case
    mf = reference(kind, df)
    legacy = solve(NTTDA(mf).set(nobeta=nobeta), delta_s)
    explicit = solve(factory(mf), delta_s)
    assert explicit.method_id == ident
    assert build_cpu_twin(explicit).method_id == ident
    np.testing.assert_allclose(explicit.e, legacy.e, atol=1e-11, rtol=0)
    expected = legacy.Gradients().kernel(state=1, atmlst=[1])
    actual = explicit.Gradients().kernel(state=1, atmlst=[1])
    np.testing.assert_allclose(actual, expected, atol=5e-9, rtol=0)


@pytest.mark.parametrize('case', CASES, ids=[case[2] for case in CASES])
def test_joint_frame_and_warm_guess_match_independent_properties(case):
    from unittest import mock
    from gpu4pyscf.grad import nttda_bridge

    kind, factory, ident, _ = case
    td = solve(factory(reference(kind, df=True)))
    grad = td.Gradients().set(cphf_conv_tol=1e-10).kernel(state=1)
    nac = td.NAC().set(cphf_conv_tol=1e-10).kernel(state_I=1, state_J=2, ediff=True, use_etfs=True)
    cache = make_frame_cache()
    for _ in range(2):
        with mock.patch.object(nttda_bridge, 'build_cpu_twin', wraps=build_cpu_twin) as twin:
            result = compute_frame(td, 1, [(1, 2)], frame_cache=cache)
            twin.assert_called_once()
        np.testing.assert_allclose(result['grad'], grad, atol=5e-9, rtol=0)
        np.testing.assert_allclose(result['nac'][(1, 2)], nac, atol=5e-9, rtol=0)
        assert td._nttda_frame_stats['method_id'] == ident
    assert td._nttda_frame_stats['zvector_cache_hits'] == 2
    snapshot = cache._entries
    stats = cache.last_stats
    energies = td.e.copy()
    try:
        td.e[1] = td.e[0]
        with pytest.raises(ZeroDivisionError):
            compute_frame(td, 1, [(1, 2)], frame_cache=cache)
        assert cache._entries is snapshot
        assert cache.last_stats is stats
    finally:
        td.e[:] = energies
    # These task keys have no stored guesses, so one GMRES iteration cannot
    # borrow the already converged vectors from the successful frame above.
    with pytest.raises(RuntimeError, match='GMRES did not converge'):
        compute_frame(td, 2, [(1, 3)], frame_cache=cache, cphf_max_cycle=1)
    assert cache._entries is snapshot
    assert cache.last_stats is stats


@pytest.mark.parametrize('kind', [EnsembleRKS, EnsembleROKS])
def test_ensemble_nobeta_does_not_change_execution(kind):
    td = solve(NTTDA(reference(kind, df=True)))
    before = compute_frame(td, 1)
    before_stats = td._nttda_frame_stats
    td.nobeta = True
    after = compute_frame(td, 1)
    np.testing.assert_allclose(before['grad'], after['grad'], atol=5e-9, rtol=0)
    for key in ('method_id', 'xc_response_channels', 'xc_fockz_tasks'):
        assert td._nttda_frame_stats[key] == before_stats[key]
    assert td._nttda_frame_stats['xc_backend']['cpu_fallbacks'] == before_stats['xc_backend']['cpu_fallbacks']


def test_contexts_do_not_share_mutable_response_state():
    td = solve(NTTDA_ROKS(reference(ROKS)))
    first, second = EvaluationContext(td), EvaluationContext(td)
    assert first.response_cache is not second.response_cache
    assert first.ledger is not second.ledger
    with pytest.raises(ValueError, match='different NTTDA evaluation'):
        first.response_cache.assert_compatible(second.cpu_td)
    td.nobeta = True
    with pytest.raises(ValueError, match='explicit NTTDA method'):
        first.validate()


@pytest.mark.parametrize('case', CASES, ids=[case[2] for case in CASES])
@pytest.mark.parametrize('df', [False, True])
def test_spin_fock_cache_uses_gpu_and_preserves_method(case, df):
    from unittest import mock
    from pyscf.dft.numint import NumInt

    kind, factory, _, _ = case
    td = solve(factory(reference(kind, df=df)))
    context = EvaluationContext(td)
    cpu_mf = context.cpu_td._scf
    expected = context.method.spin_focks_mo(cpu_mf)
    with (
        mock.patch.object(cpu_mf, 'get_fock', side_effect=AssertionError('CPU Fock')),
        mock.patch.object(NumInt, 'nr_uks', side_effect=AssertionError('CPU UKS XC')),
        mock.patch.object(NumInt, 'nr_rks', side_effect=AssertionError('CPU RKS XC')),
    ):
        cache = context.response_cache
        assert cache._focks_mo is None
        with mock.patch.object(td._scf, 'get_fock', wraps=td._scf.get_fock) as build:
            actual = cache.spin_focks_mo()
            assert cache.spin_focks_mo() is actual
            build.assert_called_once()
    for spin, reference_fock in zip(actual, expected):
        assert isinstance(spin, np.ndarray)
        np.testing.assert_allclose(spin, reference_fock, atol=1e-9, rtol=0)


def test_existing_drivers_reject_replaced_solution_for_all_paths():
    td = solve(NTTDA_ROKS(reference(ROKS)))
    grad, nac = td.Gradients(), td.NAC()
    td.xy = list(td.xy)
    for mode in ('analytic', 'finite_diff'):
        with pytest.raises(ValueError, match='create a new derivative driver'):
            grad.kernel(state=1, method=mode)
    with pytest.raises(ValueError, match='create a new derivative driver'):
        nac.kernel(state_I=1, state_J=2)


@pytest.mark.parametrize('case', CASES, ids=[case[2] for case in CASES])
def test_joint_frame_uses_selected_method_preparation(case):
    from dataclasses import replace
    from importlib import import_module
    from unittest import mock

    kind, factory, ident, _ = case
    td = solve(factory(reference(kind, df=True)))
    module = import_module('pyscf.sftda.nttda_methods.' + ident)
    for entry, pairs in (('prepare_gradient', ()), ('prepare_cross', ((1, 2),))):
        replacement = mock.Mock(side_effect=RuntimeError('selected method preparation'))
        method = replace(module.METHOD, **{entry: replacement})
        with mock.patch.object(module, 'METHOD', method):
            with pytest.raises(RuntimeError, match='selected method preparation'):
                compute_frame(td, 1, pairs)
            replacement.assert_called_once()
