# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

'''Second-order optimization of fixed-occupation restricted ensembles.'''

import cupy as cp
from pyscf import gto, lib
from cupyx.scipy.linalg import expm
from cupyx.scipy.sparse.linalg import LinearOperator, gmres

from gpu4pyscf.scf import soscf, _response_functions


def project_orbitals(mf, previous):
    '''Transport orbitals while retaining each occupation subspace.'''
    overlap = cp.asarray(mf.get_ovlp())
    cross = cp.asarray(gto.intor_cross('int1e_ovlp', mf.mol, previous.mol))
    projected = cp.linalg.solve(overlap, cross @ cp.asarray(previous.mo_coeff))
    occ = cp.asarray(previous.mo_occ)
    c = cp.zeros_like(projected)
    completed = []
    for occupation in (2, 1, 0):
        indices = cp.where(occ == occupation)[0]
        if not indices.size:
            continue
        block = projected[:, indices]
        if completed:
            filled = c[:, cp.concatenate(completed)]
            block -= filled @ (filled.conj().T @ overlap @ block)
        values, vectors = cp.linalg.eigh(block.conj().T @ overlap @ block)
        if float(values.min()) <= 1e-12:
            raise RuntimeError('projected ensemble orbital block is linearly dependent')
        c[:, indices] = block @ (
            (vectors / cp.sqrt(values)) @ vectors.conj().T
        )
        completed.append(indices)
    return c


def gen_g_hop(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None):
    '''Gradient and exponential-coordinate Hessian for 2/1/0 occupations.

    The normalization matches ``EnsembleRKS.get_grad`` (half the energy
    derivative). Rotations within an equal-occupation subspace are redundant.
    The connection term makes the Hessian symmetric away from stationarity,
    including when all three occupation classes are present.
    '''
    c = cp.asarray(mo_coeff)
    occ = cp.asarray(mo_occ)
    if fock_ao is None:
        dm = mf.make_rdm1(c, occ)
        fock_ao = mf.get_fock(h1e=h1e, dm=dm)
    fock = c.conj().T @ fock_ao @ c
    difference = occ[None, :] - occ[:, None]
    mask = difference > 0
    gradient = difference * fock
    response = _response_functions._gen_rhf_response(
        mf, c, occ, singlet=None, hermi=1,
    )

    def h_op(x):
        rotation = cp.zeros_like(fock)
        rotation[mask] = x
        rotation -= rotation.conj().T
        dm1 = c @ (difference * rotation) @ c.conj().T
        df = (
            fock @ rotation - rotation @ fock
            + c.conj().T @ response(dm1) @ c
        )
        connection = .5 * (
            rotation @ gradient - gradient @ rotation
        )
        return (difference * df + connection)[mask]

    diagonal = difference * (
        fock.diagonal().real[:, None] - fock.diagonal().real[None, :]
    )
    return gradient[mask], h_op, diagonal[mask]


class _SecondOrderEnsembleRKS(soscf._CIAH_SOSCF):
    gen_g_hop = gen_g_hop

    def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
        requested = self.conv_tol_grad
        tolerance = self.conv_tol ** .5 if requested is None else requested
        # CIAH supplies the global step control. Close to the stationary
        # point its tiny augmented eigenvalue becomes poorly resolved;
        # finish with the linear Newton equation and the requested residual.
        self.conv_tol_grad = max(tolerance, 1e-5)
        try:
            super().kernel(mo_coeff, mo_occ, dm0)
        finally:
            self.conv_tol_grad = requested
        self.converged = False
        base = self._scf
        hcore = base.get_hcore()
        occ = self.mo_occ

        def evaluate(c):
            dm = base.make_rdm1(c, occ)
            v = base.get_veff(base.mol, dm)
            fock = hcore + v
            g = base.get_grad(c, occ, fock)
            return float(base.energy_tot(dm, hcore, v)), fock, g

        c = self.mo_coeff
        energy, fock, g = evaluate(c)
        delta_energy = energy - self.e_tot
        for _ in range(min(self.max_cycle, 10)):
            residual = float(cp.linalg.norm(g))
            if residual < tolerance and abs(delta_energy) < self.conv_tol:
                self.converged = True
                break
            if residual > 1e-3:
                break
            _, hop, diagonal = self.gen_g_hop(c, occ, fock)
            safe = cp.where(
                abs(diagonal) > 1e-4, diagonal,
                cp.where(diagonal < 0, -1e-4, 1e-4),
            )
            operator = LinearOperator((g.size, g.size), matvec=hop, dtype=g.dtype)
            preconditioner = LinearOperator(
                operator.shape, matvec=lambda x: x / safe, dtype=g.dtype,
            )
            step, info = gmres(
                operator, -g, M=preconditioner, tol=1e-5,
                atol=tolerance * .01, restart=40, maxiter=160,
            )
            if not bool(cp.all(cp.isfinite(step))):
                break
            step *= min(1., .1 / max(float(cp.max(abs(step))), 1e-30))
            for scale in (1., .5, .25, .125):
                rotation = cp.zeros((occ.size, occ.size), dtype=c.dtype)
                rotation[occ[None, :] > occ[:, None]] = scale * step
                rotation -= rotation.conj().T
                trial = c @ expm(rotation)
                next_energy, next_fock, next_g = evaluate(trial)
                next_residual = float(cp.linalg.norm(next_g))
                if (next_residual < residual
                        or (next_residual < tolerance
                            and abs(next_energy - energy) < self.conv_tol)):
                    delta_energy = next_energy - energy
                    c, energy, fock, g = trial, next_energy, next_fock, next_g
                    break
            else:
                break
        self.e_tot = energy
        self.mo_energy, self.mo_coeff = base.canonicalize(c, occ, fock)
        # Recompute the residual after canonicalization as well.
        self.converged = bool(
            float(cp.linalg.norm(base.get_grad(
                self.mo_coeff, occ, fock))) < tolerance
            and abs(delta_energy) < self.conv_tol
        )
        return self.e_tot


def newton(mf):
    '''Return CIAH with every unequal-occupation orbital rotation included.'''
    if isinstance(mf, soscf._CIAH_SOSCF):
        return mf
    cls = _SecondOrderEnsembleRKS
    return lib.set_class(cls(mf), (cls, mf.__class__))
