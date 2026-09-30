"""Independent installation, and derivatives on the same electronic solution."""
import subprocess
import sys

import cupy as cp
import numpy as np
import pytest
from pyscf import gto
from gpu4pyscf.dft.roks import ROKS
from gpu4pyscf.sftda import (
    EnsembleRKS, EnsembleROKS, NTTDA_ROKS, NTTDA_ROKS_NoBeta,
    NTTDA_EnsembleRKS, NTTDA_EnsembleROKS,
)

CASES = (
    (ROKS, NTTDA_ROKS), (ROKS, NTTDA_ROKS_NoBeta),
    (EnsembleRKS, NTTDA_EnsembleRKS), (EnsembleROKS, NTTDA_EnsembleROKS),
)


def solution(kind, factory, xc, channel):
    mol = gto.M(atom='C .02 -.03 .01; H -.02 .8 .62; H .03 -.91 .5',
                basis='sto-3g', spin=2, verbose=0)
    mf = kind(mol, xc=xc).set(conv_tol=1e-12)
    mf.grids.level = 0
    mf.small_rho_cutoff = 0
    mf.run()
    assert mf.converged
    td = factory(mf).set(deltaS=channel, nstates=3, conv_tol=1e-10, max_cycle=200).run()
    assert np.all(td.converged)
    return td


def cpu_solution(td):
    """Test oracle with identical orbitals, grid, root order and phases."""
    from pyscf import dft
    from pyscf import sftda
    from pyscf.sftda import nttda_methods as methods
    kind = {'roks': dft.ROKS, 'roks_nobeta': dft.ROKS,
            'ensemble_rks': sftda.EnsembleRKS,
            'ensemble_roks': sftda.EnsembleROKS}[td.method_id]
    mf = kind(td.mol)
    mf.xc = td._scf.xc
    if td._scf.omega is not None:
        mf.omega = td._scf.omega
    mf.e_tot = td._scf.e_tot
    mf.verbose = 0
    mf.mo_coeff = cp.asnumpy(td._scf.mo_coeff)
    mf.mo_occ = cp.asnumpy(td._scf.mo_occ)
    mf.mo_energy = cp.asnumpy(td._scf.mo_energy)
    mf.grids.coords = cp.asnumpy(td._scf.grids.coords)
    mf.grids.weights = cp.asnumpy(td._scf.grids.weights)
    mf.converged = True
    result = sftda.NTTDA(mf).set(deltaS=td.deltaS, nobeta=td.nobeta)
    result.e = td.e.copy()
    result.xy = [(cp.asnumpy(cp.asarray(x)), 0) for x, _ in td.xy]
    result.converged = td.converged
    methods.bind_method(result)
    methods.record_solution(result)
    return result


@pytest.mark.parametrize('case', CASES, ids=[item[1].__name__ for item in CASES])
@pytest.mark.parametrize('xc', ['HF', 'LDA,VWN', 'PBE', 'TPSS', 'CAM-B3LYP'])
@pytest.mark.parametrize('channel', [-1, 0])
def test_derivatives_match_cpu_on_identical_solution(case, xc, channel):
    td = solution(*case, xc, channel)
    reference = cpu_solution(td)
    driver = td.Gradients()
    assert driver.base is td
    np.testing.assert_allclose(driver.kernel(state=1),
                               reference.Gradients().kernel(state=1), atol=1e-7, rtol=0)
    if channel == -1:
        actual = td.NAC().kernel(state_I=1, state_J=2, ediff=True, use_etfs=False)
        expected = reference.NAC().kernel(state_I=1, state_J=2, ediff=True, use_etfs=False)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)


def test_solver_gradient_and_nac_do_not_import_forge():
    script = '''
import importlib.abc
import sys
class NoForge(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('pyscf.sftda', 'pyscf.grad.nttda', 'pyscf.nac')):
            raise AssertionError('unexpected forge import: ' + fullname)
sys.meta_path.insert(0, NoForge())
from pyscf import gto
from gpu4pyscf.sftda import EnsembleROKS, NTTDA_EnsembleROKS
from gpu4pyscf.grad.nttda import compute_frame
mol = gto.M(atom='C 0 0 0; H 0 .8 .6; H 0 -.9 .5', basis='sto-3g', spin=2, verbose=0)
mf = EnsembleROKS(mol, xc='PBE').set(conv_tol=1e-12)
mf.grids.level = 0
mf.run()
td = NTTDA_EnsembleROKS(mf).set(nstates=3, conv_tol=1e-9).run()
td.Gradients().kernel(state=1)
td.NAC().kernel(state_I=1, state_J=2, use_etfs=False)
compute_frame(td, 1, [(1, 2)], use_etfs=False)
assert not any(n.startswith(('pyscf.sftda', 'pyscf.grad.nttda', 'pyscf.nac')) for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('setting,value', [('only_dfj', True), ('disp', 'd3bj'),
                                          ('nlc', 'VV10'), ('with_solvent', object()),
                                          ('mm_mol', object())])
def test_unsupported_reference_models_fail_before_derivatives(setting, value):
    mol = gto.M(atom='C 0 0 0; H 0 .8 .6; H 0 -.9 .5', basis='sto-3g', spin=2, verbose=0)
    mf = ROKS(mol, xc='PBE')
    setattr(mf, setting, value)
    td = NTTDA_ROKS(mf)
    with pytest.raises(NotImplementedError):
        td.Gradients()
    with pytest.raises(NotImplementedError):
        td.NAC()


@pytest.mark.parametrize('density_fit', [False, True])
def test_displaced_roks_retains_functional_and_integral_model(density_fit):
    from gpu4pyscf.grad._nttda.reference import rebuild_reference
    mol = gto.M(atom='C 0 0 0; H 0 .8 .6; H 0 -.9 .5', basis='sto-3g', spin=2, verbose=0)
    mf = ROKS(mol, xc='CAM-B3LYP')
    if density_fit:
        mf = mf.density_fit(auxbasis='weigend')
    mf.omega = .2
    mf.grids.level = 0
    mf.grids.build()
    displaced = rebuild_reference(mf, mol.copy(), fixed_grid=True)
    assert displaced.xc == mf.xc
    assert displaced.omega == mf.omega
    assert bool(getattr(displaced, 'with_df', None)) == density_fit
    if density_fit:
        assert displaced.with_df.auxbasis == 'weigend'
    np.testing.assert_array_equal(cp.asnumpy(displaced.grids.coords), cp.asnumpy(mf.grids.coords))
    assert not cp.shares_memory(displaced.grids.coords, mf.grids.coords)


def test_nac_scanner_uses_the_new_electronic_solution():
    td = solution(ROKS, NTTDA_ROKS, 'PBE', -1)
    scanner = td.NAC().set(state_I=1, state_J=2, use_etfs=False).as_scanner()
    previous_cache = scanner._context
    coordinates = td.mol.atom_coords()
    coordinates[1, 2] += 1e-3
    mol = td.mol.copy().set_geom_(coordinates, unit='Bohr')
    actual = scanner(mol)
    expected = td.NAC().kernel(state_I=1, state_J=2, use_etfs=False)
    assert scanner._context is not previous_cache
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
