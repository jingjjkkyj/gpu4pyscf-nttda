"""NTTDA ETFs against independently displaced spin-adapted wavefunctions."""

import numpy as np
import pytest
from pyscf import gto

from gpu4pyscf.dft.roks import ROKS
from gpu4pyscf.nac.nttda import _align_displaced_roots, awf_overlap
from gpu4pyscf.sftda import NTTDA_ROKS


@pytest.fixture(scope='module', params=['HF', 'PBE'])
def td_solution(request):
    mol = gto.M(
        atom='C 0.02 -0.03 0.01; H -0.02 0.8 0.62; H 0.03 -0.91 0.5',
        basis='sto-3g', spin=2, verbose=0,
    )
    mf = ROKS(mol, xc=request.param).set(
        conv_tol=1e-12, conv_tol_grad=1e-9, max_cycle=200,
    )
    mf.grids.level = 0
    mf.small_rho_cutoff = 0
    mf.kernel()
    assert mf.converged
    td = NTTDA_ROKS(mf).set(
        deltaS=-1, nstates=3, conv_tol=1e-10, max_cycle=200,
    ).run()
    assert np.all(td.converged)
    return td


def overlap_derivatives(nac, atom=0, step=1e-4):
    """Differentiate full and AO-ETF overlaps without analytic connections.

    The symmetric cross-geometry AO overlap has no first-order antisymmetric
    AO derivative. Relaxed MO coefficients and amplitudes are kept on both
    sides, so this removes only the AO part of the moving-CSF connection.
    """
    reference = nac.base
    full, etf = np.zeros(3), np.zeros(3)
    for axis in range(3):
        displaced = []
        for sign in (-1, 1):
            coords = reference.mol.atom_coords()
            coords[atom, axis] += sign * step
            point = nac._displaced_td(coords)
            assert np.all(point.converged)
            _align_displaced_roots(
                reference, point, nac.root_overlap_tol,
                required=(0, 1), method='awf',
            )
            displaced.append(point)
        left, right = displaced
        overlap = gto.intor_cross('int1e_ovlp', left.mol, right.mol)
        for result, ao in ((full, overlap), (etf, (overlap + overlap.T) / 2)):
            ij = awf_overlap(left, left.xy[0], right, right.xy[1], ao_overlap=ao)
            ji = awf_overlap(left, left.xy[1], right, right.xy[0], ao_overlap=ao)
            result[axis] = np.real(ij - ji) / (4 * step)
    return full, etf


def test_etf_matches_symmetrized_ao_overlap(td_solution):
    nac = td_solution.NAC()
    full = nac.kernel(1, 2, ediff=True, use_etfs=False)
    details = nac.nttda_nac_details
    etf = nac.kernel(1, 2, ediff=True, use_etfs=True)
    numerical_full, numerical_etf = overlap_derivatives(nac)
    np.testing.assert_allclose(full[0], numerical_full, atol=3e-5, rtol=0)
    np.testing.assert_allclose(etf[0], numerical_etf, atol=3e-5, rtol=0)
    # This fixture must distinguish an AO-only ETF from dropping all CSFs.
    assert np.max(np.abs(details['csf_orbital'][0])) > 1e-4


def test_etf_gap_scaling_atom_order_and_state_exchange(td_solution):
    nac = td_solution.NAC()
    full = nac.kernel(1, 2, ediff=True, use_etfs=False)
    ao = nac.nttda_nac_details['csf_ao']
    etf = nac.kernel(1, 2, ediff=True, use_etfs=True)
    np.testing.assert_allclose(etf, full - ao, atol=1e-9, rtol=0)

    atoms = [2, 0]
    gap = td_solution.e[1] - td_solution.e[0]
    weighted = nac.kernel(1, 2, atmlst=atoms, ediff=False, use_etfs=True)
    np.testing.assert_allclose(weighted, gap * etf[atoms], atol=1e-9, rtol=0)
    reverse = td_solution.NAC().kernel(2, 1, atmlst=atoms, ediff=True, use_etfs=True)
    np.testing.assert_allclose(reverse, -etf[atoms], atol=1e-9, rtol=0)
