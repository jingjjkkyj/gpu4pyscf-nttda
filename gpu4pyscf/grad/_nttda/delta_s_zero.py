"""Analytic gradient for current NTTDA ``deltaS=0``.

The ``prepare_grad_elec`` function exposes the channel's scientific data flow.
All same-spin amplitudes, Fock projections, response coefficients, and direct
J/K contractions live in this module.  Only XC quadrature and the
reference-specific ROKS/EnsembleRKS adjoint are delegated to sibling modules.
The scalar ledger and differentiation convention are documented in
``docs/derivations/ensemble_rks_nttda_gradient_nac.md``.
"""

from dataclasses import dataclass

import numpy as np
import cupy as cp

from gpu4pyscf.sftda import nttda as nttda_mod
from gpu4pyscf.sftda.nttda import gen_rohf_response_sc

from . import xc as xc_backend
from gpu4pyscf.sftda import nttda_methods as methods
from .derivative_jk import (
    _JKDerivativeLedger,
    response_direct_hfx,
)
from .response import orbital_spaces, pair_density, ResponseCache, prepare_gradient, finish_prepared_gradients


# Orbital spaces and native amplitudes


@dataclass(frozen=True)
class SameSpinAmplitudes:
    """Five amplitude blocks used by ``NTTDA(deltaS=0)``."""

    co: cp.ndarray
    cv: cp.ndarray
    oo: float
    ov: cp.ndarray
    cv0: cp.ndarray


def same_spin_slices(spaces):
    """Return canonical slices for ``CO/CV/OO/OV/CV0`` amplitudes."""
    nc = len(spaces.closed)
    no = len(spaces.open)
    nv = len(spaces.virtual)
    nco = nc * no
    ncv = nc * nv
    nov = no * nv
    i1 = nco
    i2 = i1 + ncv
    i3 = i2 + 1
    i4 = i3 + nov
    return {
        'CO': slice(0, i1),
        'CV': slice(i1, i2),
        'OO': slice(i2, i3),
        'OV': slice(i3, i4),
        'CV0': slice(i4, i4 + ncv),
    }


def split_same_spin(tdobj, xy):
    """Split one packed ``deltaS=0`` vector into its five native blocks."""
    spaces = orbital_spaces(tdobj)
    if spaces.spin < 0.5:
        raise ValueError('NTTDA deltaS=0 requires at least one open orbital')
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = cp.asarray(vector).reshape(-1)
    slices = same_spin_slices(spaces)
    expected = slices['CV0'].stop
    if vector.size != expected:
        raise ValueError('deltaS=0 amplitude has size %d; expected %d' % (vector.size, expected))
    nc = len(spaces.closed)
    no = len(spaces.open)
    nv = len(spaces.virtual)
    return spaces, SameSpinAmplitudes(
        co=vector[slices['CO']].reshape(nc, no),
        cv=vector[slices['CV']].reshape(nc, nv),
        oo=float(vector[slices['OO']][0]),
        ov=vector[slices['OV']].reshape(no, nv),
        cv0=vector[slices['CV0']].reshape(nc, nv),
    )


def same_spin_transition_densities(tdobj, xy):
    """Return directed AO transition densities for the four response blocks."""
    spaces, amp = split_same_spin(tdobj, xy)
    return (
        spaces,
        amp,
        {
            'CO': pair_density(spaces.c_open, amp.co.T, spaces.c_closed),
            'CV': pair_density(spaces.c_virtual, amp.cv.T, spaces.c_closed),
            'OV': pair_density(spaces.c_virtual, amp.ov.T, spaces.c_open),
            'CV0': pair_density(spaces.c_virtual, amp.cv0.T, spaces.c_closed),
        },
    )


# Explicit F0/Fz ledger


@dataclass(frozen=True)
class FockProjection:
    """One scalar term ``Tr[P (weight_f0 F0 + weight_fz Fz)]``."""

    name: str
    left_indices: cp.ndarray
    left_orbitals: cp.ndarray
    coefficient: cp.ndarray
    right_indices: cp.ndarray
    right_orbitals: cp.ndarray
    weight_f0: float
    weight_fz: float

    def density(self):
        return pair_density(
            self.left_orbitals,
            self.coefficient,
            self.right_orbitals,
        )


def fock0_fockz(tdobj, max_memory=None, cache=None):
    """Build exactly the ``F0`` and ``Fz`` matrices used by ``gen_vind_sc``."""
    if cache is not None and 'fock0_fockz' in cache.extra:
        return cache.extra['fock0_fockz']
    mf = tdobj._scf
    if max_memory is None:
        max_memory = tdobj.max_memory
    _response, fockz = gen_rohf_response_sc(
        mf,
        hermi=0,
        fxc_ref=None if cache is None else cache.fxc_ref(),
    )
    fock0 = methods.fock0(methods.get_method(tdobj), mf)
    result = fock0, fockz
    if cache is not None:
        cache.extra['fock0_fockz'] = result
    return result


def same_spin_fock_projections(tdobj, xy):
    """Return the complete five-block explicit-Fock ledger."""
    spaces, x = split_same_spin(tdobj, xy)
    spin = spaces.spin
    c = spaces.c_closed
    o = spaces.c_open
    v = spaces.c_virtual
    a = np.sqrt((spin + 1.0) / (2.0 * spin))
    b = np.sqrt(2.0 * (spin + 1.0) / spin)
    d = np.sqrt((spin + 1.0) / spin)
    h = np.sqrt(0.5)

    terms = []

    def indices(orbitals):
        if orbitals is c:
            return spaces.closed
        if orbitals is o:
            return spaces.open
        if orbitals is v:
            return spaces.virtual
        raise ValueError('Fock projection uses an unknown orbital space')

    def add(name, left, coefficient, right, f0, fz):
        coefficient = cp.asarray(coefficient)
        if coefficient.size:
            terms.append(
                FockProjection(
                    name,
                    indices(left),
                    left,
                    coefficient,
                    indices(right),
                    right,
                    float(f0),
                    float(fz),
                )
            )

    # CO row/column and its couplings.
    add('co-oo', o, x.co.T @ x.co, o, 1.0, -1.0)
    add('co-cc', c, -x.co @ x.co.T, c, 1.0, -1.0)
    add('co-cv', o, 2.0 * a * (x.co.T @ x.cv), v, 1.0, -1.0)
    add('co-oo1', o, -2.0 * x.oo * x.co.T, c, 1.0, -1.0)
    add('co-cv0', o, 2.0 * h * (x.co.T @ x.cv0), v, 1.0, -1.0)

    # CV block and its OO/OV/CV0 couplings.
    add('cv-vv', v, x.cv.T @ x.cv, v, 1.0, -1.0 / spin)
    add('cv-cc', c, -x.cv @ x.cv.T, c, 1.0, 1.0 / spin)
    add('cv-oo1', v, 2.0 * b * x.oo * x.cv.T, c, 0.0, 1.0)
    add('cv-ov', o, -2.0 * a * (x.ov @ x.cv.T), c, 1.0, 1.0)
    add(
        'cv-cv0-vv',
        v,
        -d * (x.cv.T @ x.cv0 + x.cv0.T @ x.cv),
        v,
        0.0,
        1.0,
    )
    add(
        'cv-cv0-cc',
        c,
        d * (x.cv0 @ x.cv.T + x.cv @ x.cv0.T),
        c,
        0.0,
        1.0,
    )

    # OV and CV0 diagonal/coupling terms.
    add('ov-vv', v, x.ov.T @ x.ov, v, 1.0, 1.0)
    add('ov-oo', o, -x.ov @ x.ov.T, o, 1.0, 1.0)
    add('ov-oo1', v, 2.0 * x.oo * x.ov.T, o, 1.0, 1.0)
    add('ov-cv0', c, 2.0 * h * (x.cv0 @ x.ov.T), o, 1.0, 1.0)
    add('cv0-vv', v, x.cv0.T @ x.cv0, v, 1.0, 0.0)
    add('cv0-cc', c, -x.cv0 @ x.cv0.T, c, 1.0, 0.0)
    add('cv0-oo1', v, -2.0 * np.sqrt(2.0) * x.oo * x.cv0.T, c, 1.0, 0.0)
    return tuple(terms)


def same_spin_fock_probes(tdobj, xy):
    """Return AO probes ``(P0, Pz)`` generated from the Fock ledger."""
    nao = tdobj.mol.nao_nr()
    p0 = cp.zeros((nao, nao))
    pz = cp.zeros_like(p0)
    for term in same_spin_fock_projections(tdobj, xy):
        density = term.density()
        p0 += term.weight_f0 * density
        pz += term.weight_fz * density
    return p0, pz


def _fock_response_q(tdobj, p_alpha, p_beta, cache=None):
    return methods.fock_response_q(methods.get_method(tdobj), tdobj, p_alpha, p_beta, cache)


def same_spin_fock_q(tdobj, xy, max_memory=None, cache=None):
    """Return the explicit-Fock contribution to ``(Q_alpha,Q_beta)``.

    For HF the complete ``Fz`` response is exactly represented by the
    spin-resolved probes.  DFT callers add the independent ``Fz`` and
    ``nobeta`` response ledgers after this common ``F0`` contribution.
    """
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    nmo = mo.shape[1]
    fock0, fockz = fock0_fockz(tdobj, max_memory=max_memory, cache=cache)
    fock0_mo = mo.conj().T @ fock0 @ mo
    fockz_mo = mo.conj().T @ fockz @ mo
    q_alpha = cp.zeros((nmo, nmo))
    q_beta = cp.zeros_like(q_alpha)

    for term in same_spin_fock_projections(tdobj, xy):
        left = term.left_indices
        right = term.right_indices
        coeff = term.coefficient

        def project(target, operator, scale):
            if scale == 0.0:
                return
            target[:, left] += scale * operator[:, right] @ coeff.T
            target[:, right] += scale * operator[:, left] @ coeff

        project(q_alpha, fock0_mo, 0.5 * term.weight_f0)
        project(q_beta, fock0_mo, 0.5 * term.weight_f0)
        if mf._numint._xc_type(mf.xc) == 'HF':
            project(q_alpha, fockz_mo, 0.5 * term.weight_fz)
            project(q_beta, fockz_mo, 0.5 * term.weight_fz)
        else:
            project(q_alpha, fockz_mo, term.weight_fz)

    p0, pz = same_spin_fock_probes(tdobj, xy)
    p_alpha = 0.5 * p0
    p_beta = 0.5 * p0
    if mf._numint._xc_type(mf.xc) == 'HF':
        p_alpha = p_alpha + 0.5 * pz
        p_beta = p_beta - 0.5 * pz
    response_alpha, response_beta = _fock_response_q(
        tdobj,
        p_alpha,
        p_beta,
        cache=cache,
    )
    q_alpha += response_alpha
    q_beta += response_beta
    return q_alpha, q_beta


# vref0/vref1 response ledger


@dataclass(frozen=True)
class ResponseTerm:
    """One ``T_target : (c0*V0+c1*V1)[T_source]`` term."""

    target: str
    source: str
    vref0: float
    vref1: float


def same_spin_response_terms(spin):
    """Exact response coefficients in ``(CO,CV,OV,CV0)`` block order.

    ``V0`` and ``V1`` are distinct GGA/MGGA index placements for directed
    transition densities; for LDA the two maps are identical.
    """
    a = np.sqrt((spin + 1.0) / (2.0 * spin))
    h = np.sqrt(0.5)
    r2 = np.sqrt(2.0)
    return (
        ResponseTerm('CO', 'CO', 1.0, -1.0),
        ResponseTerm('CO', 'CV', a, 0.0),
        ResponseTerm('CO', 'OV', 0.0, 1.0),
        ResponseTerm('CO', 'CV0', h, -r2),
        ResponseTerm('CV', 'CO', a, 0.0),
        ResponseTerm('CV', 'CV', 1.0, 0.0),
        ResponseTerm('CV', 'OV', a, 0.0),
        ResponseTerm('OV', 'CO', 0.0, 1.0),
        ResponseTerm('OV', 'CV', a, 0.0),
        ResponseTerm('OV', 'OV', 1.0, -1.0),
        ResponseTerm('OV', 'CV0', -h, r2),
        ResponseTerm('CV0', 'CO', h, -r2),
        ResponseTerm('CV0', 'OV', -h, r2),
        ResponseTerm('CV0', 'CV0', 1.0, -2.0),
    )


def _apply_hfx_responses(tdobj, densities):
    """Return only the hybrid/RSH J/K portions of ``vref0/vref1``."""
    mf = tdobj._scf
    labels = tuple(densities)
    dms = cp.asarray([densities[label] for label in labels])
    vref0 = cp.zeros_like(dms)
    vref1 = cp.zeros_like(dms)
    ni = mf._numint
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
    if ni.libxc.is_hybrid_xc(mf.xc):
        vref0 -= hybrid * mf.get_k(mf.mol, dms, hermi=0)
        vref1 -= hybrid * mf.get_j(mf.mol, dms, hermi=0)
        if omega != 0:
            scale = alpha - hybrid
            vref0 -= scale * mf.get_k(
                mf.mol,
                dms,
                hermi=0,
                omega=omega,
            )
            vref1 -= scale * nttda_mod._get_j_range_separated(mf, dms, 0, omega)
    return (
        {label: value for label, value in zip(labels, vref0)},
        {label: value for label, value in zip(labels, vref1)},
    )


def _derivative_potentials(spaces, densities, vref0, vref1):
    potentials = {label: cp.zeros_like(dm) for label, dm in densities.items()}
    for term in same_spin_response_terms(spaces.spin):
        if term.vref0:
            potentials[term.target] += term.vref0 * vref0[term.source]
            potentials[term.source] += term.vref0 * vref0[term.target]
        if term.vref1:
            potentials[term.target] += term.vref1 * vref1[term.source]
            potentials[term.source] += term.vref1 * vref1[term.target]
    return potentials


def same_spin_hfx_projection_q(tdobj, xy):
    """Transition-factor derivative of only the hybrid/RSH response scalar."""
    spaces, amp, densities = same_spin_transition_densities(tdobj, xy)
    vref0, vref1 = _apply_hfx_responses(tdobj, densities)
    potentials = _derivative_potentials(
        spaces,
        densities,
        vref0,
        vref1,
    )
    mo = cp.asarray(tdobj._scf.mo_coeff)
    q_alpha = cp.zeros((mo.shape[1], mo.shape[1]))
    q_beta = cp.zeros_like(q_alpha)
    block_data = {
        'CO': (spaces.open, spaces.closed, amp.co.T),
        'CV': (spaces.virtual, spaces.closed, amp.cv.T),
        'OV': (spaces.virtual, spaces.open, amp.ov.T),
        'CV0': (spaces.virtual, spaces.closed, amp.cv0.T),
    }
    for label, (target, source, coefficient) in block_data.items():
        potential_mo = mo.conj().T @ potentials[label] @ mo
        q_beta[:, target] += potential_mo[:, source] @ coefficient.T
        q_alpha[:, source] += potential_mo[target, :].T @ coefficient
    return q_alpha, q_beta


# AO J/K nuclear derivatives


# Hybrid/RSH Fz correction


def same_spin_fockz_hfx_terms(gradient_driver, tdobj, pz, atmlst=None, with_direct=True, jk_ledger=None, output_slot=0):
    """Differentiate ``-1/2 Pz:K(D_OO)`` excluding the Pz projection."""
    mf = tdobj._scf
    mol = mf.mol
    ni = mf._numint
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    mo = cp.asarray(mf.mo_coeff)
    q_alpha = cp.zeros((mo.shape[1], mo.shape[1]))
    q_beta = cp.zeros_like(q_alpha)
    direct = cp.zeros((len(atmlst), 3))
    if not ni.libxc.is_hybrid_xc(mf.xc):
        return xc_backend.XCGradientTerms(q_alpha, q_beta, direct)

    spaces = orbital_spaces(tdobj)
    density_open = spaces.c_open @ spaces.c_open.T
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    scales = [(hybrid, None)]
    if omega != 0:
        scales.append((alpha - hybrid, omega))
    k_terms = []
    for coefficient, range_omega in scales:
        if coefficient == 0.0:
            continue
        if range_omega is None:
            potential = mf.get_k(mol, pz, hermi=0)
        else:
            potential = mf.get_k(
                mol,
                pz,
                hermi=0,
                omega=range_omega,
            )
        q_alpha[:, spaces.open] -= 0.5 * coefficient * (mo.conj().T @ (potential + potential.T) @ spaces.c_open)
        if with_direct:
            k_terms.append(
                (
                    pz,
                    density_open,
                    -0.5 * coefficient,
                    range_omega,
                )
            )
    if with_direct:
        local_ledger = _JKDerivativeLedger()
        ledger = jk_ledger if jk_ledger is not None else local_ledger
        ledger.add('k', output_slot, k_terms)
        if jk_ledger is None:
            direct += local_ledger.contract(
                gradient_driver,
                mol,
                atmlst,
                slots=(output_slot,),
            )[output_slot]
    return xc_backend.XCGradientTerms(q_alpha, q_beta, direct)


# Scalar closure diagnostics (private to this channel)


# Channel assembly


def prepare_grad_elec(gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12, max_cycle=None, cache=None):
    """Build the complete analytic excitation gradient for deltaS=0.

    The function follows the physical order of the Lagrangian: native
    amplitudes and AO probes, XC/J/K contributions to the M matrix and direct
    derivative, the ROKS adjoint, and the final overlap contraction.
    """
    if tdobj.deltaS != 0:
        raise ValueError('deltaS=0 gradient received a different spin channel')
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    if cache is None:
        cache = ResponseCache(tdobj)
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)

    # 1. Native amplitudes, transition densities, and explicit Fock probes.
    spaces, amplitudes, densities = same_spin_transition_densities(tdobj, xy)
    blocks = {
        'CO': (spaces.open, spaces.closed, amplitudes.co.T),
        'CV': (spaces.virtual, spaces.closed, amplitudes.cv.T),
        'OV': (spaces.virtual, spaces.open, amplitudes.ov.T),
        'CV0': (spaces.virtual, spaces.closed, amplitudes.cv0.T),
    }
    response_terms = same_spin_response_terms(spaces.spin)
    channel_data = (spaces, amplitudes, densities, blocks, response_terms)
    p0, pz = same_spin_fock_probes(tdobj, xy)
    jk_ledger = _JKDerivativeLedger()
    direct_slot = 'direct'
    zvector_slot = 'zvector'

    # 2. Explicit Fock contribution to the orbital-rotation M matrix.
    fock_alpha, fock_beta = same_spin_fock_q(tdobj, xy, cache=cache)

    if xctype == 'HF':
        # 3a. HF response and fixed-orbital AO derivative.
        response_alpha, response_beta = same_spin_hfx_projection_q(tdobj, xy)
        m_matrix = fock_alpha + fock_beta + response_alpha + response_beta
        direct = response_direct_hfx(
            gradient_driver,
            tdobj,
            densities,
            response_terms,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
        )
        if methods.get_method(tdobj).separate_hf_fockz:
            fockz_hfx = same_spin_fockz_hfx_terms(
                gradient_driver,
                tdobj,
                pz,
                atmlst=atmlst,
                jk_ledger=jk_ledger,
                output_slot=direct_slot,
            )
            m_matrix += fockz_hfx.q_alpha + fockz_hfx.q_beta
            direct += fockz_hfx.direct
        direct_fock_probes = methods.hf_probes(methods.get_method(tdobj), p0, pz)

    else:
        # 3b. Semilocal XC, hybrid/RSH, Fz, and nobeta contributions.
        try:
            response_builder, fockz_builder = {
                'LDA': (
                    xc_backend.lda_response_terms,
                    xc_backend.lda_fockz_terms,
                ),
                'GGA': (
                    xc_backend.gga_response_terms,
                    xc_backend.gga_fockz_terms,
                ),
                'MGGA': (
                    xc_backend.mgga_response_terms,
                    xc_backend.mgga_fockz_terms,
                ),
            }[xctype]
        except KeyError as error:
            raise NotImplementedError('NTTDA deltaS=0 gradient does not support XC type %s' % xctype) from error

        hfx_alpha, hfx_beta = same_spin_hfx_projection_q(tdobj, xy)
        response_xc = response_builder(
            gradient_driver,
            tdobj,
            channel_data,
            atmlst=atmlst,
        )
        fockz_xc = fockz_builder(
            gradient_driver,
            tdobj,
            spaces,
            pz,
            atmlst=atmlst,
        )
        fockz_hfx = same_spin_fockz_hfx_terms(
            gradient_driver,
            tdobj,
            pz,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
        )
        common_alpha, common_beta = methods.common_fock_q(methods.get_method(tdobj), tdobj, p0)
        m_matrix = (
            fock_alpha
            + fock_beta
            + hfx_alpha
            + hfx_beta
            + response_xc.q_alpha
            + response_xc.q_beta
            + fockz_xc.q_alpha
            + fockz_xc.q_beta
            + fockz_hfx.q_alpha
            + fockz_hfx.q_beta
            + common_alpha
            + common_beta
        )

        direct_fock_probes = (0.5 * p0, 0.5 * p0)
        direct = response_direct_hfx(
            gradient_driver,
            tdobj,
            densities,
            response_terms,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
        )
        direct += response_xc.direct
        direct += fockz_xc.direct
        direct += fockz_hfx.direct

    # 4-5. Dispatch to the reference-specific transpose Hessian (ROKS or
    # EnsembleRKS), then add the Fock derivative and overlap/Pulay terms.
    return prepare_gradient(
        gradient_driver,
        tdobj,
        m_matrix,
        direct,
        atmlst,
        tolerance,
        max_cycle,
        jk_ledger=jk_ledger,
        nobeta_p0=p0,
        direct_fock_probes=direct_fock_probes,
        cache=cache,
    )


def grad_elec(gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12, max_cycle=None, cache=None):
    """Compatibility channel entry using the selected method's preparation."""
    if tdobj.deltaS != 0:
        raise ValueError('deltaS=0 gradient received a different spin channel')
    prepared = methods.prepare_gradient(
        methods.get_method(tdobj),
        gradient_driver,
        tdobj,
        xy,
        atmlst=atmlst,
        tolerance=tolerance,
        max_cycle=max_cycle,
        cache=cache,
    )
    return finish_prepared_gradients((prepared,))[0]
