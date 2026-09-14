# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

'''GPU response and Z-vector machinery for the fixed-orbital ROKS reference.

The differentiated energy is the high-spin ROKS energy evaluated on the
converged EnsembleRKS (Dz0) orbitals, i.e.
``gpu4pyscf.sftda.EnsembleROKS.reference_energy()``.  The orbitals are
stationary for the ensemble energy, not for this selected energy, so the
gradient needs the orbital-relaxation (Z-vector) contribution: Lagrange
multipliers for the ensemble stationarity conditions are solved with a
matrix-free Hessian-vector product and contracted with the explicit derivative
of those conditions.

This module ports the reference CPU implementation
(``pyscf-forge-ensemble-rks-roks-ref-nttda-grad`` @ ``c25eb35``) to the GPU:
the packed rotation space and its ordering are identical, while the response
contraction, the Hessian action and the GMRES vectors stay on the device.
'''

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import LinearOperator, gmres

from pyscf import lib
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger

from gpu4pyscf.dft import rks as gpu_rks
from gpu4pyscf.dft import roks as gpu_roks
from gpu4pyscf.grad import rhf as gpu_rhf_grad
from gpu4pyscf.hessian import rhf as gpu_rhf_hess
from gpu4pyscf.hessian import rks as gpu_rks_hess
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.scf import _response_functions


_OCC_TOL = 1e-8
_ZB_BACKENDS = ('contracted', 'legacy')


@dataclass(frozen=True)
class RotationSpace:
    '''Packed nonredundant rotations and their orbital occupations.'''

    p: np.ndarray
    q: np.ndarray
    f: np.ndarray
    nalpha: np.ndarray
    nbeta: np.ndarray

    @property
    def size(self) -> int:
        return int(self.p.size)

    @property
    def occupation_gap(self) -> np.ndarray:
        return self.f[self.q] - self.f[self.p]

    def unpack(self, vector):
        '''Map a packed vector to the full real anti-symmetric MO matrix.'''
        xp = cp if isinstance(vector, cp.ndarray) else np
        vector = xp.asarray(vector, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError(
                'Expected a packed rotation of shape (%d,), got %s.'
                % (self.size, vector.shape)
            )
        matrix = xp.zeros((self.f.size, self.f.size))
        matrix[self.p, self.q] = vector
        matrix[self.q, self.p] = -vector
        return matrix

    def pack(self, matrix):
        '''Extract entries matching the packed (p,q) rotation order.'''
        return matrix[..., self.p, self.q]


@dataclass(frozen=True)
class GMRESDiagnostics:
    '''Auditable convergence evidence for the Dz0 Z-vector solve.'''

    info: int
    iterations: int
    last_preconditioned_residual: float | None
    residual_l2: float
    rhs_l2: float
    relative_residual_l2: float
    residual_max_abs: float
    threshold: float
    converged: bool

    def as_dict(self) -> dict:
        return {
            'info': self.info,
            'iterations': self.iterations,
            'last_preconditioned_residual': self.last_preconditioned_residual,
            'residual_l2': self.residual_l2,
            'rhs_l2': self.rhs_l2,
            'relative_residual_l2': self.relative_residual_l2,
            'residual_max_abs': self.residual_max_abs,
            'threshold': self.threshold,
            'converged': self.converged,
        }


def _as_float(value) -> float:
    if hasattr(value, 'get'):
        value = value.get()
    return float(value)


def validate_gmres_solution(
        operator, rhs, solution, info, tolerance,
        preconditioned_residuals=()) -> GMRESDiagnostics:
    '''Validate GMRES with the recomputed, unpreconditioned device residual.'''
    rhs = cp.asarray(rhs, dtype=float)
    solution = cp.asarray(solution, dtype=float)
    if not bool(cp.all(cp.isfinite(solution))):
        raise RuntimeError('Dz0 Z-vector GMRES returned a non-finite solution.')

    residual = rhs - cp.asarray(operator.matvec(solution), dtype=float)
    if not bool(cp.all(cp.isfinite(residual))):
        raise RuntimeError(
            'Dz0 Z-vector GMRES produced a non-finite true residual.'
        )

    residual_l2 = _as_float(cp.linalg.norm(residual))
    rhs_l2 = _as_float(cp.linalg.norm(rhs))
    threshold = float(tolerance) * max(rhs_l2, 1.0)
    relative = residual_l2 / rhs_l2 if rhs_l2 > 0.0 else residual_l2
    residual_max = _as_float(cp.max(cp.abs(residual))) if residual.size else 0.0
    history = [_as_float(value) for value in preconditioned_residuals]
    diagnostics = GMRESDiagnostics(
        info=int(info),
        iterations=len(history),
        last_preconditioned_residual=(history[-1] if history else None),
        residual_l2=residual_l2,
        rhs_l2=rhs_l2,
        relative_residual_l2=relative,
        residual_max_abs=residual_max,
        threshold=threshold,
        converged=bool(residual_l2 <= threshold),
    )
    if not diagnostics.converged:
        last = diagnostics.last_preconditioned_residual
        last_text = 'none' if last is None else '%.3e' % last
        raise RuntimeError(
            'Dz0 Z-vector GMRES true residual did not converge: '
            'info=%d, ||b-Az||_2=%.3e, threshold=%.3e, '
            'relative_residual=%.3e, max_abs=%.3e, '
            'last preconditioned residual=%s.'
            % (
                diagnostics.info,
                diagnostics.residual_l2,
                diagnostics.threshold,
                diagnostics.relative_residual_l2,
                diagnostics.residual_max_abs,
                last_text,
            )
        )
    if diagnostics.info != 0:
        warnings.warn(
            'Dz0 Z-vector GMRES reported info=%d, but the independently '
            'recomputed true residual %.3e satisfies the %.3e threshold.'
            % (diagnostics.info, diagnostics.residual_l2, diagnostics.threshold),
            RuntimeWarning,
        )
    return diagnostics


def _block_pairs(rows: np.ndarray, cols: np.ndarray):
    '''Pairs matching ``matrix[np.ix_(rows, cols)].ravel()`` order.'''
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if rows.size == 0 or cols.size == 0:
        empty = np.empty(0, dtype=int)
        return empty, empty
    return np.repeat(rows, cols.size), np.tile(cols, rows.size)


def _rotation_space(mo_occ) -> RotationSpace:
    f = np.asarray(mo_occ, dtype=float)
    if f.ndim != 1:
        raise ValueError(
            'Dz0 occupations must be a one-dimensional array; got %s.'
            % (f.shape,)
        )

    is_c = np.isclose(f, 2.0, atol=_OCC_TOL, rtol=0.0)
    is_o = np.isclose(f, 1.0, atol=_OCC_TOL, rtol=0.0)
    is_v = np.isclose(f, 0.0, atol=_OCC_TOL, rtol=0.0)
    if not np.all(is_c | is_o | is_v):
        bad = np.where(~(is_c | is_o | is_v))[0]
        raise NotImplementedError(
            'This implementation requires Dz0 occupations 0, 1, or 2. '
            'Nonstandard occupations were found at MO indices %s.'
            % bad.tolist()
        )

    c = np.where(is_c)[0]
    o = np.where(is_o)[0]
    v = np.where(is_v)[0]
    blocks = (_block_pairs(o, c), _block_pairs(v, c), _block_pairs(v, o))
    p = np.concatenate([block[0] for block in blocks])
    q = np.concatenate([block[1] for block in blocks])

    nalpha = (f > 0.0).astype(float)
    nbeta = np.isclose(f, 2.0, atol=_OCC_TOL, rtol=0.0).astype(float)
    return RotationSpace(p=p, q=q, f=f, nalpha=nalpha, nbeta=nbeta)


def _copy_ks_settings(source, target) -> None:
    '''Copy numerical-integration settings without copying wrappers.'''
    for name in (
        'xc',
        'nlc',
        'grids',
        'nlcgrids',
        '_numint',
        'max_memory',
        'verbose',
        'stdout',
        'direct_scf_tol',
        'small_rho_cutoff',
    ):
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))


def _ao_density(mo_coeff, occupation):
    '''Return ``D_AO = C occupation C^dagger``.'''
    return (mo_coeff * cp.asarray(occupation)) @ mo_coeff.conj().T


def _transform_ao_to_mo(mo_coeff, matrices):
    '''Transform one AO matrix or a leading batch to the MO basis.'''
    return cp.einsum(
        'up,...uv,vq->...pq', mo_coeff.conj(), matrices, mo_coeff,
        optimize=True,
    )


def _hcore_derivative_generator(mol):
    '''Core-Hamiltonian derivative from the CPU one-electron integral entry.

    J/K and the XC derivative run on the GPU; the core derivative is a pure
    one-electron integral and is intentionally taken from the CPU path.
    '''
    h1 = rhf_grad.get_hcore(mol)
    aoslices = mol.aoslice_by_atom()

    def hcore_deriv(atm_id):
        _shl0, _shl1, p0, p1 = aoslices[atm_id]
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3)
            vrinv *= -mol.atom_charge(atm_id)
        vrinv[:, p0:p1] += h1[:, p0:p1]
        return vrinv + vrinv.transpose(0, 2, 1)

    return hcore_deriv


def _full_overlap_derivative(one_sided_s1, atom, aoslices):
    '''Build the symmetrized ``S^A`` from the one-sided derivative.'''
    p0, p1 = aoslices[atom][2:]
    s1 = np.zeros_like(one_sided_s1)
    s1[:, p0:p1, :] += one_sided_s1[:, p0:p1, :]
    s1[:, :, p0:p1] += one_sided_s1[:, p0:p1, :].transpose(0, 2, 1)
    return s1


def _fractional_rks_fock_skeleton(charge_mf, mo_coeff, mo_occ):
    '''Return ``C^T F_AO^(0,[A]) C_occ`` on the device for all atoms.

    This is the GPU RKS ``make_h1`` construction with the density corrected
    from the closed-shell ``2 C_occ C_occ^T`` to the fractional-occupation
    density ``C mo_occ C^T`` required by the Dz0 reference.  Range-separated
    exchange is included through the ``(omega, alpha, hyb)`` decomposition.
    J/K and the XC derivative stay on the GPU; the core derivative is the
    allowed CPU one-electron integral entry.
    '''
    mol = charge_mf.mol
    mo_coeff = cp.asarray(mo_coeff)
    f_occ = cp.asarray(mo_occ)
    mocc = mo_coeff[:, cp.asnumpy(f_occ) > 0]
    dm0 = _ao_density(mo_coeff, f_occ)

    hessobj = gpu_rks_hess.Hessian(charge_mf)
    ni = charge_mf._numint
    ni.libxc.test_deriv_order(charge_mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(charge_mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(charge_mf.xc)

    max_memory = max(
        2000, charge_mf.max_memory * 0.9 - lib.current_memory()[0],
    )
    explicit = gpu_rks_hess._get_vxc_deriv1(
        hessobj, mo_coeff, f_occ, max_memory,
    )

    vj, vk = gpu_rhf_hess._get_jk_ip1(mol, dm0, with_k=bool(hybrid))
    veff = vj
    if hybrid:
        veff = veff - 0.5 * hyb * vk
    if abs(omega) > 1e-10 and abs(alpha - hyb) > 1e-10:
        with mol.with_range_coulomb(omega):
            vk_lr = gpu_rhf_hess._get_jk_ip1(
                mol, dm0, with_j=False,
            )[1]
        veff = veff - 0.5 * (alpha - hyb) * vk_lr
    explicit = explicit + cp.einsum('up,xauv,vq->xapq', mo_coeff, veff, mocc)

    hcore_gen = _hcore_derivative_generator(mol)
    hcore = cp.stack([cp.asarray(hcore_gen(atom)) for atom in range(mol.natm)])
    explicit = explicit + cp.einsum(
        'up,xauv,vq->xapq', mo_coeff, hcore, mocc,
    )
    return explicit


def _occupied_positions(space):
    '''Map global packed-column indices to positions in the occupied block.'''
    occ = np.where(space.f > 0.0)[0]
    return np.searchsorted(occ, space.q)


class ReferenceGradients(lib.StreamObject):
    '''GPU response and Z-vector driver of the selected ROKS reference.

    The driver differentiates ``EnsembleROKS.reference_energy()`` (the
    high-spin ROKS energy on the fixed Dz0 ensemble orbitals).  The response
    contraction, Hessian action and GMRES vectors stay on the device; the
    packed rotation space and its ordering match the CPU reference exactly.
    '''

    _keys = {
        'base',
        'mol',
        'max_memory',
        'conv_tol',
        'max_cycle',
        'restart',
        'grid_response',
        'atmlst',
        'de',
        'z',
        'g_hs',
        'g_dz0',
        'b',
        'e_hs_unrelaxed',
        'z_solver_diagnostics',
        'zb_backend',
        'z_b_correction',
    }

    def __init__(self, mf):
        self.base = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory

        self.conv_tol = float(getattr(self.base, 'conv_tol_grad', 0.0) or 1e-9)
        self.max_cycle = 80
        self.restart = 40
        self.grid_response = False
        self.atmlst = None

        self.de = None
        self.z = None
        self.g_hs = None
        self.g_dz0 = None
        self.b = None
        self.e_hs_unrelaxed = None
        self.z_solver_diagnostics = None
        self.zb_backend = os.environ.get(
            'NTTDA_REFERENCE_ZB_BACKEND', 'contracted',
        ).strip().lower()
        self.z_b_correction = None

        self._space = None
        self._charge_mf = None
        self._hs_mf = None
        self._charge_response = None
        self._c0 = None
        self._f0ao = None
        self._f0mo = None
        self._f_hs_ao = None
        self._f_hs_mo = None
        self._dm_hs = None
        self._w_hs_mo = None
        self._hcore = None
        self._veff_hs = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** GPU Dz0SCF high-spin-reference response ********')
        log.info('Z-vector tolerance = %.3g', self.conv_tol)
        log.info('Z-vector max cycles = %d', self.max_cycle)
        log.info('GMRES restart = %d', self.restart)
        log.info('grid response = %s', self.grid_response)
        log.info('Z.B contraction backend = %s', self.zb_backend)
        return self

    def _validate(self) -> None:
        mf = self.base
        mol = self.mol
        if (
                getattr(mf, 'mo_coeff', None) is None
                or getattr(mf, 'mo_occ', None) is None):
            raise RuntimeError(
                'Run EnsembleROKS before requesting its analytic gradient.'
            )
        if hasattr(mf, 'converged') and not mf.converged:
            warnings.warn(
                'EnsembleROKS is not converged; its analytic gradient is '
                'not stationary.'
            )
        if np.iscomplexobj(mf.mo_coeff) and np.max(
                np.abs(np.asarray(mf.mo_coeff).imag)) > 1e-12:
            raise NotImplementedError(
                'Complex-orbital EnsembleROKS gradients are not implemented.'
            )
        if self.grid_response:
            raise NotImplementedError(
                'Moving-grid response is not implemented consistently in '
                'B^(0,A); use grid_response=False.'
            )
        if getattr(mf, 'only_dfj', False):
            raise NotImplementedError(
                'only_dfj EnsembleROKS gradients require a mixed DF-J/exact-K '
                'Fock derivative, which is not implemented.'
            )
        if getattr(mf, 'with_x2c', None) is not None:
            raise NotImplementedError(
                'X2C EnsembleROKS gradients are not implemented.'
            )
        if getattr(mf, 'with_solvent', None) is not None:
            raise NotImplementedError(
                'Solvent-response EnsembleROKS gradients are not implemented.'
            )
        if hasattr(mf, 'do_nlc') and mf.do_nlc():
            raise NotImplementedError(
                'Nonlocal-correlation (NLC/VV10) EnsembleROKS gradients are '
                'not implemented.'
            )
        if hasattr(mf, 'do_disp') and mf.do_disp():
            raise NotImplementedError(
                'Dispersion-corrected EnsembleROKS gradients are not '
                'implemented.'
            )
        if getattr(mol, 'dimension', 3) != 3:
            raise NotImplementedError(
                'Only molecular (three-dimensional) calculations are '
                'supported.'
            )
        if self.zb_backend not in _ZB_BACKENDS:
            raise ValueError(
                'Unknown EnsembleROKS Z.B backend %r; expected one of %s.'
                % (self.zb_backend, _ZB_BACKENDS)
            )

    def _build_intermediates(self) -> None:
        mf = self.base
        mol = self.mol
        c0 = cp.asarray(mf.mo_coeff).real
        space = _rotation_space(cp.asnumpy(cp.asarray(mf.mo_occ)))
        f_occ = cp.asarray(space.f)

        # Do not call the gpu4pyscf.dft.RKS factory here: for mol.spin != 0 it
        # dispatches to ROKS, whereas the ensemble (charge-only) kernel is the
        # spin-unpolarized RKS one.
        with_df = getattr(mf, 'with_df', None)
        auxbasis = None if with_df is None else getattr(
            with_df, 'auxbasis', None,
        )
        charge_mf = gpu_rks.RKS(mol)
        _copy_ks_settings(mf, charge_mf)
        if with_df is not None:
            # ``auxbasis=None`` means "auto-select", not "no DF"; keep the DF
            # object so the response and reference-energy evaluators match the
            # differentiated SCF.
            charge_mf = charge_mf.density_fit(auxbasis=auxbasis)
            _copy_ks_settings(mf, charge_mf)
        charge_mf.mo_coeff = c0
        charge_mf.mo_occ = f_occ

        dm0 = _ao_density(c0, f_occ)
        hcore = charge_mf.get_hcore(mol)
        f0ao = hcore + charge_mf.get_veff(mol, dm0)
        f0mo = _transform_ao_to_mo(c0, f0ao)

        charge_response = _response_functions._gen_rhf_response(
            charge_mf,
            mo_coeff=c0,
            mo_occ=f_occ,
            singlet=None,
            hermi=1,
            max_memory=self.max_memory,
            with_nlc=False,
        )

        hs_mf = gpu_roks.ROKS(mol)
        _copy_ks_settings(mf, hs_mf)
        if with_df is not None:
            hs_mf = hs_mf.density_fit(auxbasis=auxbasis)
            _copy_ks_settings(mf, hs_mf)
        hs_mf.mo_coeff = c0
        hs_mf.mo_occ = f_occ
        dm_hs = hs_mf.make_rdm1(c0, f_occ)
        veff_hs = hs_mf.get_veff(mol, dm_hs)
        f_hs_ao = cp.stack((hcore + veff_hs[0], hcore + veff_hs[1]))
        f_hs_mo = _transform_ao_to_mo(c0, f_hs_ao)

        occ_spin = cp.asarray((space.nalpha, space.nbeta))
        w_hs_mo = 0.5 * cp.sum(
            occ_spin[:, :, None] * f_hs_mo + f_hs_mo * occ_spin[:, None, :],
            axis=0,
        )

        self._space = space
        self._charge_mf = charge_mf
        self._hs_mf = hs_mf
        self._charge_response = charge_response
        self._c0 = c0
        self._f0ao = f0ao
        self._f0mo = f0mo
        self._f_hs_ao = f_hs_ao
        self._f_hs_mo = f_hs_mo
        self._dm_hs = dm_hs
        self._w_hs_mo = w_hs_mo
        self._hcore = hcore
        self._veff_hs = veff_hs

        gap = cp.asarray(space.occupation_gap)
        self.g_dz0 = 2.0 * gap * space.pack(f0mo)
        nalpha = cp.asarray(space.nalpha)
        nbeta = cp.asarray(space.nbeta)
        self.g_hs = 2.0 * (
            (nalpha[space.q] - nalpha[space.p]) * space.pack(f_hs_mo[0])
            + (nbeta[space.q] - nbeta[space.p]) * space.pack(f_hs_mo[1])
        )

    def hessian_vector_product(self, vector):
        '''Evaluate the ensemble orbital-Hessian action ``A^(0) v``.'''
        if self._space is None:
            self._validate()
            self._build_intermediates()

        space = self._space
        kappa = space.unpack(cp.asarray(vector, dtype=float))
        f_occ = cp.asarray(space.f)
        delta_dm_mo = kappa * f_occ[None, :] - f_occ[:, None] * kappa
        delta_dm_ao = self._c0 @ delta_dm_mo @ self._c0.T
        delta_f_ao = self._charge_response(delta_dm_ao)
        delta_f_mo = _transform_ao_to_mo(self._c0, delta_f_ao)
        moving_mo = self._f0mo @ kappa - kappa @ self._f0mo
        result = 2.0 * cp.asarray(space.occupation_gap) * space.pack(
            moving_mo + delta_f_mo,
        )
        return cp.asarray(result).real

    def _solve_z(self):
        space = self._space
        if space.size == 0:
            self.z_solver_diagnostics = GMRESDiagnostics(
                info=0,
                iterations=0,
                last_preconditioned_residual=None,
                residual_l2=0.0,
                rhs_l2=0.0,
                relative_residual_l2=0.0,
                residual_max_abs=0.0,
                threshold=float(self.conv_tol),
                converged=True,
            )
            return cp.empty(0)

        operator = LinearOperator(
            (space.size, space.size),
            matvec=self.hessian_vector_product,
            rmatvec=self.hessian_vector_product,
            dtype=float,
        )

        # The exact real-orbital Dz0 Hessian is symmetric, so A^T z = g_HS
        # is solved with the same matrix-free action.  This diagonal contains
        # the one-electron commutator part and is used only as a
        # preconditioner.
        diagonal = 2.0 * cp.asarray(space.occupation_gap) * (
            self._f0mo.diagonal()[space.p] - self._f0mo.diagonal()[space.q]
        )
        floor = max(1e-8, 1e-6 * _as_float(cp.max(cp.abs(diagonal))))
        safe_diagonal = cp.where(
            cp.abs(diagonal) > floor,
            diagonal,
            cp.where(diagonal < 0.0, -floor, floor),
        )
        preconditioner = LinearOperator(
            operator.shape,
            matvec=lambda x: cp.asarray(x) / safe_diagonal,
            dtype=float,
        )

        residuals = []
        rhs = cp.asarray(self.g_hs).real
        z, info = gmres(
            operator,
            rhs,
            M=preconditioner,
            restart=self.restart,
            maxiter=self.max_cycle,
            tol=self.conv_tol,
            atol=0.0,
            callback=residuals.append,
            callback_type='pr_norm',
        )

        self.z_solver_diagnostics = validate_gmres_solution(
            operator,
            rhs,
            z,
            info,
            self.conv_tol,
            residuals,
        )
        return z

    def _high_spin_unrelaxed_gradient(self):
        '''Evaluate ``E_HS^(A)``, including the nonstationary overlap term.'''
        mol = self.mol
        c0 = self._c0
        dm_hs = self._dm_hs
        space = self._space
        one_sided_s1 = cp.asarray(rhf_grad.get_ovlp(mol))
        dm_total = dm_hs[0] + dm_hs[1]
        w_hs_ao = c0 @ self._w_hs_mo @ c0.T
        aoslices = mol.aoslice_by_atom()

        if getattr(self._hs_mf, 'with_df', None) is not None:
            from gpu4pyscf.df.grad import roks as df_roks_grad

            hs_grad = df_roks_grad.Gradients(self._hs_mf)
        else:
            hs_grad = self._hs_mf.nuc_grad_method()
        f_occ = cp.asarray(space.f)
        dma, dmb = self._hs_mf.make_rdm1(c0, f_occ)
        dm_ee = tag_array(
            cp.stack((cp.asarray(dma), cp.asarray(dmb))),
            mo_coeff=cp.repeat(c0[None], 2, axis=0),
            mo_occ=cp.asarray(
                [cp.asnumpy(f_occ) > 0, cp.asnumpy(f_occ) == 2], dtype=float,
            ),
        )
        e2 = cp.asarray(hs_grad.energy_ee(mol, dm_ee))

        de = cp.zeros((mol.natm, 3))
        hcore_gen = _hcore_derivative_generator(mol)
        for atom, (_, _, p0, p1) in enumerate(aoslices):
            hcore = cp.asarray(hcore_gen(atom))
            de[atom] += cp.einsum('xij,ij->x', hcore, dm_total)
            # -Tr[W_HS^MO S_MO^A], written with PySCF's one-sided S derivative.
            de[atom] -= 2.0 * cp.einsum(
                'xij,ij->x', one_sided_s1[:, p0:p1], w_hs_ao[p0:p1],
            )
        de += e2

        grad_nuc = cp.asarray(hs_grad.grad_nuc(mol))
        return de + grad_nuc

    def _build_b(self):
        '''Build ``B_i^(0,A)`` for all atoms and Cartesian components.'''
        mol = self.mol
        c0 = self._c0
        space = self._space
        f_occ = cp.asarray(space.f)
        occ = np.where(space.f > 0.0)[0]
        q_pos = _occupied_positions(space)
        cocc = c0[:, occ]
        aoslices = mol.aoslice_by_atom()
        one_sided_s1 = rhf_grad.get_ovlp(mol)
        if getattr(self._charge_mf, 'with_df', None) is not None:
            from gpu4pyscf.df.grad.ensemble_roks import (
                fractional_rks_fock_skeleton,
            )

            skeleton = fractional_rks_fock_skeleton(
                self._charge_mf, c0, f_occ,
            )
        else:
            skeleton = _fractional_rks_fock_skeleton(
                self._charge_mf, c0, f_occ,
            )
        gap = cp.asarray(space.occupation_gap)[:, None]

        b = cp.empty((space.size, mol.natm, 3))
        for atom in range(mol.natm):
            s1ao = _full_overlap_derivative(one_sided_s1, atom, aoslices)
            s1mo = _transform_ao_to_mo(c0, cp.asarray(s1ao))

            anticommutator_sf = s1mo * f_occ[None, None, :]
            anticommutator_sf += f_occ[None, :, None] * s1mo
            delta_dm_sym_mo = -0.5 * anticommutator_sf
            delta_dm_sym_ao = cp.einsum(
                'up,xpq,vq->xuv', c0, delta_dm_sym_mo, c0.conj(),
                optimize=True,
            )
            response_ao = self._charge_response(delta_dm_sym_ao)

            explicit_mo_occ = skeleton[atom]
            response_mo_occ = cp.einsum(
                'up,xuv,vq->xpq', c0, response_ao, cocc, optimize=True,
            )
            overlap_mo = -0.5 * (
                cp.einsum('xpq,qr->xpr', s1mo, self._f0mo, optimize=True)
                + cp.einsum('pq,xqr->xpr', self._f0mo, s1mo, optimize=True)
            )
            overlap_mo_occ = overlap_mo[:, :, occ]
            fock_fixed_k_occ = (
                explicit_mo_occ + response_mo_occ + overlap_mo_occ
            )
            packed = fock_fixed_k_occ[:, space.p, q_pos]
            b[:, atom, :] = (2.0 * gap * packed.T)
        return b

    def _contract_z_b(self, z):
        '''Contract ``z_i B_i^(0,A)`` without materializing the full B tensor.

        The charge response is self-adjoint on real Hermitian densities.  Move
        it from every nuclear overlap density onto the single Z-derived probe,
        then reduce the remaining overlap derivatives by AO centre.  Explicit
        DF/XC/core skeleton terms are contracted before they are discarded.
        '''
        mol = self.mol
        c0 = self._c0
        space = self._space
        z = cp.asarray(z, dtype=float)
        if z.shape != (space.size,):
            raise ValueError(
                'Expected a Z vector of shape (%d,), got %s.'
                % (space.size, z.shape)
            )

        f_occ = cp.asarray(space.f)
        occ = np.where(space.f > 0.0)[0]
        q_pos = _occupied_positions(space)
        cocc = c0[:, occ]
        weighted_z = 2.0 * cp.asarray(space.occupation_gap) * z

        # W[p,q_occ] is the only part of B selected by the packed Z vector.
        weight_occ = cp.zeros((space.f.size, occ.size), dtype=c0.dtype)
        weight_occ[space.p, q_pos] = weighted_z

        if getattr(self._charge_mf, 'with_df', None) is not None:
            from gpu4pyscf.df.grad.ensemble_roks import (
                fractional_rks_fock_skeleton,
            )

            skeleton = fractional_rks_fock_skeleton(
                self._charge_mf, c0, f_occ,
            )
        else:
            skeleton = _fractional_rks_fock_skeleton(
                self._charge_mf, c0, f_occ,
            )
        correction = cp.einsum(
            'pq,axpq->ax', weight_occ, skeleton, optimize=True,
        )
        del skeleton

        # Contract W with the common-Fock response by adjointness:
        # <Pz, R[dD(S^A)]> = <R[Pz], dD(S^A)>.
        probe_ao = c0 @ weight_occ @ cocc.T
        probe_ao = 0.5 * (probe_ao + probe_ao.T)
        response_probe_mo = _transform_ao_to_mo(
            c0, self._charge_response(probe_ao),
        )
        response_weight_mo = -0.5 * response_probe_mo * (
            f_occ[:, None] + f_occ[None, :]
        )

        # The explicit overlap term is also linear in S^A.  Fold both MO
        # contractions into one AO weight and use the vectorized AO-centre
        # reduction instead of one full MO transform per atom.
        weight_mo = cp.zeros_like(self._f0mo)
        weight_mo[:, occ] = weight_occ
        overlap_weight_mo = -0.5 * (
            weight_mo @ self._f0mo.T + self._f0mo.T @ weight_mo
        )
        overlap_weight_ao = c0 @ (
            response_weight_mo + overlap_weight_mo
        ) @ c0.T
        one_sided_s1 = cp.asarray(rhf_grad.get_ovlp(mol))
        correction += cp.asarray(gpu_rhf_grad.contract_h1e_dm(
            mol, one_sided_s1, overlap_weight_ao.T, hermi=0,
        ))
        return correction

    def get_ovlp(self, mol=None):
        '''Overlap-derivative helper used by the NTTDA gradient and NAC.'''
        return rhf_grad.get_ovlp(self.mol if mol is None else mol)

    def get_hcore(self, mol=None):
        '''Core Hamiltonian (geometry only, X2C/ECP aware).'''
        if mol is None:
            mol = self.mol
        return rhf_grad.get_hcore(mol)

    def hcore_generator(self, mol=None):
        '''Core-Hamiltonian derivative generator (geometry only).'''
        if mol is None:
            mol = self.mol
        return _hcore_derivative_generator(mol)

    def kernel(self, atmlst=None, verbose=None):
        '''Compute and return the selected reference-energy gradient.'''
        log = logger.new_logger(self, verbose)
        self._validate()
        self.dump_flags(verbose)
        self._build_intermediates()

        max_g0 = (
            _as_float(cp.max(cp.abs(self.g_dz0)))
            if self.g_dz0.size else 0.0
        )
        log.info('max |g_Dz0| = %.6g', max_g0)
        scf_grad_tol = getattr(self.base, 'conv_tol_grad', 0.0) or 0.0
        if max_g0 > max(1e-6, 100.0 * scf_grad_tol):
            warnings.warn(
                'The packed Dz0 orbital gradient is not small (max=%.3e); '
                'the analytic-gradient stationarity equation may be '
                'inaccurate.' % max_g0
            )

        self.z = self._solve_z()
        self.e_hs_unrelaxed = self._high_spin_unrelaxed_gradient()
        if self.zb_backend == 'legacy':
            self.b = self._build_b()
            z_correction = cp.einsum(
                'i,iax->ax', self.z, self.b, optimize=True,
            )
        else:
            self.b = None
            z_correction = self._contract_z_b(self.z)
        self.z_b_correction = z_correction
        de = self.e_hs_unrelaxed - z_correction

        self.atmlst = atmlst
        result = cp.asnumpy(de) if atmlst is None else cp.asnumpy(de)[
            np.asarray(atmlst, dtype=int)
        ]
        self.de = result

        if log.verbose >= logger.NOTE:
            logger.note(
                self,
                '--------------- GPU Dz0SCF reference gradients ---------------',
            )
            rhf_grad._write(log, self.mol, result, atmlst)
            logger.note(
                self,
                '--------------------------------------------------------------',
            )
        return result

    grad = kernel


Gradients = ReferenceGradients
Grad = ReferenceGradients

__all__ = [
    'GMRESDiagnostics',
    'ReferenceGradients',
    'Gradients',
    'Grad',
    'RotationSpace',
    'validate_gmres_solution',
]
