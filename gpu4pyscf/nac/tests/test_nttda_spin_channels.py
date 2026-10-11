"""Independent spin-adapted density and derivative checks for deltaS=0/+1."""

from types import SimpleNamespace

import cupy as cp
import numpy as np
import pytest
from pyscf import gto

from gpu4pyscf.dft.roks import ROKS
from gpu4pyscf.nac.nttda import (
    _apply_one_body, _state_combine, awf_overlap, build_spin_adapted_awf,
    get_hf_interstate_numerator, interstate_rdm1,
)
from gpu4pyscf.nac.tests.test_nttda_etf import overlap_derivatives
from gpu4pyscf.sftda import (
    EnsembleRKS, EnsembleROKS, NTTDA_ROKS, NTTDA_ROKS_NoBeta,
    NTTDA_EnsembleRKS, NTTDA_EnsembleROKS,
)


DERIVATIVE_CASES = [
    pytest.param((channel, xc, kind, factory, df, symbol, spin), id=f'{name}-{channel:+d}')
    for name, xc, kind, factory, df, symbol, spin, channels in (
        ('roks-hf', 'HF', ROKS, NTTDA_ROKS, False, 'C', 2, (0, 1)),
        ('roks-pbe', 'PBE', ROKS, NTTDA_ROKS, False, 'C', 2, (0, 1)),
        ('roks-rsh-df', 'CAM-B3LYP', ROKS, NTTDA_ROKS, True, 'C', 2, (0, 1)),
        ('nobeta-mgga', 'TPSS', ROKS, NTTDA_ROKS_NoBeta, False, 'C', 2, (0, 1)),
        ('ensemble-hf', 'HF', EnsembleRKS, NTTDA_EnsembleRKS, False, 'C', 2, (0, 1)),
        ('ensemble-roks-df', 'PBE', EnsembleROKS, NTTDA_EnsembleROKS, True, 'C', 2, (0, 1)),
        ('closed-hf', 'HF', ROKS, NTTDA_ROKS, False, 'O', 0, (1,)),
        ('closed-pbe', 'PBE', ROKS, NTTDA_ROKS, False, 'O', 0, (1,)),
        ('doublet-hf', 'HF', ROKS, NTTDA_ROKS, False, 'N', 1, (0,)),
    )
    for channel in channels
]


@pytest.mark.parametrize('channel,nopen', [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)])
def test_transition_density_matches_determinants(channel, nopen):
    nc, nv = 2, 2
    size = nc * nv if channel == 1 else nc * nopen + 2 * nc * nv + 1 + nopen * nv
    rng = np.random.default_rng(42)
    roots = np.linalg.qr(rng.normal(size=(size, 2)) + 1j * rng.normal(size=(size, 2)))[0].T
    # Exercise actual MO indices, rather than an assumed C/O/V ordering.
    occ = rng.permutation([2] * nc + [1] * nopen + [0] * nv)
    td = SimpleNamespace(deltaS=channel, _scf=SimpleNamespace(mo_occ=occ))
    for root in roots:
        awf = build_spin_adapted_awf(td, (root, 0))
        assert abs(awf.norm - 1) < 1e-13
        assert awf.nalpha - awf.nbeta == nopen + 2 * channel
        raised = _state_combine([
            (1, _apply_one_body(awf.coefficients, p, awf.nmo + p)) for p in range(awf.nmo)
        ])
        assert sum(abs(c) ** 2 for c in raised.values()) < 1e-26
    analytic = interstate_rdm1(td, (roots[0], 0), (roots[1], 0))
    determinant = interstate_rdm1(td, (roots[0], 0), (roots[1], 0), method='determinant')
    np.testing.assert_allclose(analytic, determinant, atol=1e-13, rtol=0)
    reverse = interstate_rdm1(td, (roots[1], 0), (roots[0], 0))
    np.testing.assert_allclose(reverse, analytic.conj().T, atol=1e-13, rtol=0)

    # Permuting the occupation blocks preserves the physical state.
    original_occ = np.array([2] * nc + [1] * nopen + [0] * nv)
    permutation = np.concatenate([np.flatnonzero(original_occ == value) for value in (0, 2, 1)])
    left = SimpleNamespace(deltaS=channel, _scf=SimpleNamespace(
        mo_occ=original_occ, mo_coeff=np.eye(len(occ)),
    ))
    right = SimpleNamespace(deltaS=channel, _scf=SimpleNamespace(
        mo_occ=original_occ[permutation], mo_coeff=np.eye(len(occ))[:, permutation],
    ))
    overlap = awf_overlap(left, (roots[0], 0), right, (roots[0], 0), ao_overlap=np.eye(len(occ)))
    np.testing.assert_allclose(overlap, 1, atol=1e-13, rtol=0)


@pytest.fixture(scope='module', params=DERIVATIVE_CASES)
def td_solution(request):
    channel, xc, kind, factory, df, symbol, spin = request.param
    mol = gto.M(atom=f'{symbol} .02 -.03 .01; H -.02 .8 .62; H .03 -.91 .5',
                basis='sto-3g', spin=spin, verbose=0)
    mf = kind(mol, xc=xc).set(conv_tol=1e-12, conv_tol_grad=1e-9, max_cycle=200)
    if df:
        mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.grids.level = 0
    mf.small_rho_cutoff = 0
    mf.run()
    assert mf.converged
    td = factory(mf).set(deltaS=channel, nstates=3, conv_tol=1e-10, max_cycle=200).run()
    assert np.all(td.converged)
    return td


def test_gradient_matches_energy_difference(td_solution):
    grad = td_solution.Gradients().set(fixed_grid=True)
    analytic = grad.kernel(state=1, atmlst=[1])
    numerical = grad.kernel(state=1, atmlst=[1], method='finite_diff', step=1e-4)
    np.testing.assert_allclose(analytic, numerical, atol=3e-5, rtol=0)


def test_full_and_etf_match_wavefunction_difference(td_solution):
    nac = td_solution.NAC().set(fixed_grid=True)
    full = nac.kernel(1, 2, ediff=True, use_etfs=False)
    orbital = nac.nttda_nac_details['csf_orbital']
    etf = nac.kernel(1, 2, ediff=True, use_etfs=True)
    # The raising roots in this fixture have large couplings; the O(step^2)
    # error at 1e-4 exceeds 3e-5. A halved step resolves that truncation error.
    numerical_full, numerical_etf = overlap_derivatives(nac, step=5e-5)
    np.testing.assert_allclose(full[0], numerical_full, atol=3e-5, rtol=0)
    np.testing.assert_allclose(etf[0], numerical_etf, atol=3e-5, rtol=0)
    if td_solution.deltaS == 1:
        np.testing.assert_allclose(orbital, 0, atol=1e-12, rtol=0)
    elif td_solution._scf.xc == 'PBE' and td_solution.method_id == 'roks':
        assert np.max(np.abs(orbital)) > 1e-4


def test_cross_preparation_matches_full_polarization(td_solution):
    td = td_solution
    nac = td.NAC()
    direct = get_hf_interstate_numerator(nac, *td.xy[:2], method='direct', atmlst=[2, 0])
    polarized = get_hf_interstate_numerator(nac, *td.xy[:2], method='polarization', atmlst=[2, 0])
    # Use the existing joint-frame DF tolerance for independently contracted
    # functionals; the RSH/DF fixture differs by ~1e-9 with either ledger path.
    tolerance = 5e-9 if getattr(td._scf, 'with_df', None) is not None else 1e-10
    np.testing.assert_allclose(cp.asnumpy(direct), polarized, atol=tolerance, rtol=0)
    forward = nac.kernel(1, 2, ediff=True, use_etfs=True, atmlst=[2, 0])
    backward = td.NAC().kernel(2, 1, ediff=True, use_etfs=True, atmlst=[2, 0])
    np.testing.assert_allclose(forward, -backward, atol=1e-10, rtol=0)
