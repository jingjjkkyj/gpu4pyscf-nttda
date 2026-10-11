"""The spin-raising closed-shell limit is ordinary triplet TDA."""

import cupy as cp
import numpy as np
import pytest
from pyscf import dft, gto, tdscf

from gpu4pyscf.dft.roks import ROKS
from gpu4pyscf.sftda import NTTDA_ROKS
from gpu4pyscf.tdscf._lr_eig import eigh


@pytest.mark.parametrize('complex_vectors', [False, True])
def test_tight_davidson_retains_orthonormal_metric(complex_vectors):
    rng = np.random.default_rng(15)
    vectors = rng.normal(size=(14, 14))
    guesses = rng.normal(size=(6, 14))
    if complex_vectors:
        vectors = vectors + 1j * rng.normal(size=vectors.shape)
        guesses = guesses + 1j * rng.normal(size=guesses.shape)
    rotation = np.linalg.qr(vectors)[0]
    spectrum = np.r_[[-0.005, 0.074, 0.374], np.linspace(1, 20, 11)]
    matrix = cp.asarray((rotation * spectrum) @ rotation.conj().T)
    # Near-dependent guesses expose the metric drift behind strict residual
    # stagnation; the requested residual must still hold for the actual A*X.
    guesses[1] = guesses[0] + 1e-8 * guesses[1]
    diagonal = matrix.diagonal().real

    def precond(residual, energy):
        return residual / (diagonal[None] - energy[:, None])

    converged, energies, roots = eigh(
        lambda x: x @ matrix.T, cp.asarray(guesses), precond,
        nroots=3, tol_residual=1e-10, lindep=1e-26, max_cycle=100,
        pick=lambda w, v, n, env: (w, v, cp.arange(w.size)),
        reorthogonalize=True, verbose=0,
    )
    assert bool(cp.all(converged))
    np.testing.assert_allclose(energies, spectrum[:3], atol=1e-10, rtol=0)
    residual = roots @ matrix.T - cp.asarray(energies)[:, None] * roots
    assert float(cp.max(cp.linalg.norm(residual, axis=1))) < 1e-10


@pytest.mark.parametrize('xc', ['HF', 'PBE', 'CAM-B3LYP'])
def test_closed_shell_matches_triplet_tda(xc):
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', basis='sto-3g', verbose=0)
    mf = ROKS(mol, xc=xc).set(conv_tol=1e-12, conv_tol_grad=1e-9, max_cycle=200)
    mf.grids.level = 0
    mf.small_rho_cutoff = 0
    mf.kernel()
    assert mf.converged
    td = NTTDA_ROKS(mf).set(deltaS=1, nstates=3, conv_tol=1e-10).run()
    assert np.all(td.converged)

    reference = dft.RKS(mol).set(xc=xc)
    for name in ('mo_coeff', 'mo_occ', 'mo_energy'):
        setattr(reference, name, cp.asnumpy(getattr(mf, name)))
    reference.grids.coords = cp.asnumpy(mf.grids.coords)
    reference.grids.weights = cp.asnumpy(mf.grids.weights)
    reference.converged = True
    # Compare the same final Fock operator: SCF's stored energies can precede
    # its last density update, whereas NTTDA rebuilds F0 from that density.
    fock = reference.get_fock()
    reference.mo_energy = np.diag(reference.mo_coeff.T @ fock @ reference.mo_coeff)
    triplet = tdscf.TDA(reference).set(singlet=False, nstates=3, conv_tol=1e-10).run()
    assert np.all(triplet.converged)
    np.testing.assert_allclose(td.e, triplet.e, atol=1e-8, rtol=0)
