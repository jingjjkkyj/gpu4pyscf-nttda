"""Analytic gradient and interstate numerator for NTTDA ``deltaS=-1``.

This module owns the complete spin-lowering amplitude, Fock, response, and
AO derivative formulas.  The scalar action is recorded as
``P0:F0 + Pz:Fz + sum(T_target:V0/1[T_source])``; differentiating that
single ledger generates both the orbital ``M`` matrix and the fixed-orbital
nuclear derivative.  EnsembleRKS uses the same channel algebra but supplies
an equal-spin common-Fock response and its own occupation Hessian.

The bilinear NAC numerator is evaluated by polarizing this quadratic
functional, while avoiding two complete gradient/Z-vector calculations.
See ``docs/derivations/ensemble_rks_nttda_gradient_nac.md`` for the equations.
"""

from dataclasses import dataclass

import numpy as np
import cupy as cp

from gpu4pyscf.sftda import nttda as nttda_mod
from gpu4pyscf.sftda.nttda import gen_rohf_response_sfd

from . import xc as xc_backend
from gpu4pyscf.sftda import nttda_methods as methods
from .derivative_jk import (
    _JKDerivativeLedger,
    response_direct_hfx,
)
from .response import (
    orbital_spaces,
    pair_density,
    ResponseCache,
    finish_prepared_gradients,
    prepare_gradient,
)


# Orbital spaces and native amplitudes


@dataclass(frozen=True)
class SpinLoweringAmplitudes:
    """Four native blocks used by ``NTTDA(deltaS=-1)``."""

    co: cp.ndarray
    cv: cp.ndarray
    oo: cp.ndarray
    ov: cp.ndarray


def split_spin_lowering(tdobj, xy):
    """Split a lowering-channel amplitude into ``CO/CV/OO/OV`` blocks."""
    spaces = orbital_spaces(tdobj)
    if spaces.spin < 1.0:
        raise ValueError('NTTDA deltaS=-1 requires reference spin Si >= 1')
    vector = xy[0] if isinstance(xy, (tuple, list)) else xy
    vector = cp.asarray(vector)
    nc = len(spaces.closed)
    no = len(spaces.open)
    nv = len(spaces.virtual)
    expected = (nc + no, no + nv)
    if vector.size != expected[0] * expected[1]:
        raise ValueError('deltaS=-1 amplitude has size %d; expected %d' % (vector.size, expected[0] * expected[1]))
    vector = vector.reshape(expected)
    return spaces, SpinLoweringAmplitudes(
        co=vector[:nc, :no],
        cv=vector[:nc, no:],
        oo=vector[nc:, :no],
        ov=vector[nc:, no:],
    )


def spin_lowering_transition_densities(tdobj, xy):
    """Directed alpha-occupied to beta-target transition densities."""
    spaces, amp = split_spin_lowering(tdobj, xy)
    return (
        spaces,
        amp,
        {
            'CO': pair_density(spaces.c_open, amp.co.T, spaces.c_closed),
            'CV': pair_density(spaces.c_virtual, amp.cv.T, spaces.c_closed),
            'OO': pair_density(spaces.c_open, amp.oo.T, spaces.c_open),
            'OV': pair_density(spaces.c_virtual, amp.ov.T, spaces.c_open),
        },
    )


def spin_lowering_block_data(spaces, amplitudes):
    """MO index/factor map for variations of lowering transition densities."""
    return {
        'CO': (spaces.open, spaces.closed, amplitudes.co.T),
        'CV': (spaces.virtual, spaces.closed, amplitudes.cv.T),
        'OO': (spaces.open, spaces.open, amplitudes.oo.T),
        'OV': (spaces.virtual, spaces.open, amplitudes.ov.T),
    }


# Channel-local immutable records


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


@dataclass(frozen=True)
class ResponseTerm:
    """One ``T_target : (c0*V0+c1*V1)[T_source]`` term."""

    target: str
    source: str
    vref0: float
    vref1: float


# Shared-response evaluators copied into the lowering channel


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


# Reference Fock response helper


def _fock_response_q(tdobj, p_alpha, p_beta, cache=None):
    return methods.fock_response_q(methods.get_method(tdobj), tdobj, p_alpha, p_beta, cache)


# Complete lowering scalar and M-matrix ledger


def spin_lowering_response_terms(spin):
    """Exact ``V0/V1`` coefficient matrices in ``gen_rohf_response_sfd``.

    The returned order is row-major in ``(CO,CV,OO,OV)``.  Keeping the
    coefficients as data makes the energy, M matrix, nuclear derivative,
    and GPU batch path share one source of truth.
    """
    denominator = 2.0 * spin - 1.0
    a = np.sqrt((2.0 * spin + 1.0) / (2.0 * spin))
    b = np.sqrt(2.0 * spin / denominator)
    c = np.sqrt((2.0 * spin + 1.0) / denominator)
    return (
        ResponseTerm('CO', 'CO', 1.0, 1.0 / denominator),
        ResponseTerm('CO', 'CV', a, 0.0),
        ResponseTerm('CO', 'OO', b, 0.0),
        ResponseTerm(
            'CO',
            'OV',
            2.0 * spin / denominator,
            -1.0 / denominator,
        ),
        ResponseTerm('CV', 'CO', a, 0.0),
        ResponseTerm('CV', 'CV', 1.0, 0.0),
        ResponseTerm('CV', 'OO', c, 0.0),
        ResponseTerm('CV', 'OV', a, 0.0),
        ResponseTerm('OO', 'CO', b, 0.0),
        ResponseTerm('OO', 'CV', c, 0.0),
        ResponseTerm('OO', 'OO', 1.0, 0.0),
        ResponseTerm('OO', 'OV', b, 0.0),
        ResponseTerm(
            'OV',
            'CO',
            2.0 * spin / denominator,
            -1.0 / denominator,
        ),
        ResponseTerm('OV', 'CV', a, 0.0),
        ResponseTerm('OV', 'OO', b, 0.0),
        ResponseTerm('OV', 'OV', 1.0, 1.0 / denominator),
    )


def spin_lowering_fock0_fockz(tdobj, max_memory=None, cache=None):
    """Operators used by the current lowering-channel action."""
    if cache is not None and 'fock0_fockz' in cache.extra:
        return cache.extra['fock0_fockz']
    mf = tdobj._scf
    if max_memory is None:
        max_memory = tdobj.max_memory
    _response, fockz = gen_rohf_response_sfd(
        mf,
        hermi=0,
        fxc_ref=None if cache is None else cache.fxc_ref(),
    )
    result = methods.fock0(methods.get_method(tdobj), mf), fockz
    if cache is not None:
        cache.extra['fock0_fockz'] = result
    return result


def spin_lowering_fock_projections(tdobj, xy):
    """Complete ``P0:F0 + Pz:Fz`` ledger of ``X.T A_sfd X``.

    The first double loop is the ordinary alpha-to-beta particle/hole Fock
    difference.  The following seven terms are the tensor spin-adaptation
    correction; they must remain in the same F0/Fz basis used by the energy
    action so that analytic differentiation is closed.
    """
    spaces, amplitudes = split_spin_lowering(tdobj, xy)
    c = spaces.c_closed
    o = spaces.c_open
    v = spaces.c_virtual
    block_data = (
        ('C', 'O', amplitudes.co),
        ('C', 'V', amplitudes.cv),
        ('O', 'O', amplitudes.oo),
        ('O', 'V', amplitudes.ov),
    )
    orbital_data = {
        'C': (spaces.closed, c),
        'O': (spaces.open, o),
        'V': (spaces.virtual, v),
    }
    terms = []

    def add(name, left_label, coefficient, right_label, f0, fz):
        left_indices, left_orbitals = orbital_data[left_label]
        right_indices, right_orbitals = orbital_data[right_label]
        coefficient = cp.asarray(coefficient)
        if coefficient.size:
            terms.append(
                FockProjection(
                    name=name,
                    left_indices=left_indices,
                    left_orbitals=left_orbitals,
                    coefficient=coefficient,
                    right_indices=right_indices,
                    right_orbitals=right_orbitals,
                    weight_f0=float(f0),
                    weight_fz=float(fz),
                )
            )

    # Ordinary alpha-to-beta spin-flip Fock difference.
    for row_left, column_left, x_left in block_data:
        for row_right, column_right, x_right in block_data:
            if row_left == row_right:
                add(
                    'base-beta-%s%s-%s%s'
                    % (
                        row_left,
                        column_left,
                        row_right,
                        column_right,
                    ),
                    column_left,
                    x_left.T @ x_right,
                    column_right,
                    1.0,
                    -1.0,
                )
            if column_left == column_right:
                add(
                    'base-alpha-%s%s-%s%s'
                    % (
                        row_left,
                        column_left,
                        row_right,
                        column_right,
                    ),
                    row_right,
                    -(x_right @ x_left.T),
                    row_left,
                    1.0,
                    1.0,
                )

    # Tensor spin-adaptation correction, expressed in the same F0/Fz basis.
    spin = spaces.spin
    trace_oo = float(cp.trace(amplitudes.oo))
    eta = np.sqrt((2.0 * spin + 1.0) / (2.0 * spin)) - 1.0
    gamma = np.sqrt((2.0 * spin + 1.0) / (2.0 * spin - 1.0))
    zeta = np.sqrt(2.0 * spin / (2.0 * spin - 1.0)) - 1.0
    chi = 1.0 / np.sqrt(2.0 * spin * (2.0 * spin - 1.0))
    t_cc = amplitudes.cv @ amplitudes.cv.T / spin + amplitudes.co @ amplitudes.co.T * 2.0 / (2.0 * spin - 1.0)
    t_vv = amplitudes.cv.T @ amplitudes.cv / spin + amplitudes.ov.T @ amplitudes.ov * 2.0 / (2.0 * spin - 1.0)
    # OO->CV and CV->OO each contribute gamma / spin to the scalar.
    t_cv = 2.0 * gamma / spin * trace_oo * amplitudes.cv
    t_beta_vo = 2.0 * eta * amplitudes.cv.T @ amplitudes.co + 2.0 * zeta * amplitudes.ov.T @ amplitudes.oo
    t_beta_co = 2.0 * chi * trace_oo * amplitudes.co
    t_alpha_oc = (-2.0 * eta * amplitudes.cv @ amplitudes.ov.T - 2.0 * zeta * amplitudes.co @ amplitudes.oo.T).T
    t_alpha_vo = -2.0 * chi * trace_oo * amplitudes.ov.T

    add('adapt-spin-cc', 'C', t_cc, 'C', 0.0, -1.0)
    add('adapt-spin-vv', 'V', t_vv, 'V', 0.0, -1.0)
    add('adapt-spin-cv', 'C', t_cv, 'V', 0.0, -1.0)
    add('adapt-beta-vo', 'V', t_beta_vo, 'O', 1.0, -1.0)
    add('adapt-beta-co', 'C', t_beta_co, 'O', 1.0, -1.0)
    add('adapt-alpha-oc', 'O', t_alpha_oc, 'C', 1.0, 1.0)
    add('adapt-alpha-vo', 'V', t_alpha_vo, 'O', 1.0, 1.0)
    return tuple(terms)


def spin_lowering_fock_probes(tdobj, xy):
    """Return AO probes ``P0,Pz`` for the lowering Fock ledger."""
    nao = tdobj.mol.nao_nr()
    p0 = cp.zeros((nao, nao))
    pz = cp.zeros_like(p0)
    for term in spin_lowering_fock_projections(tdobj, xy):
        density = term.density()
        p0 += term.weight_f0 * density
        pz += term.weight_fz * density
    return p0, pz


def _response_potentials(densities, vref0, vref1, terms):
    potentials = {label: cp.zeros_like(dm) for label, dm in densities.items()}
    for term in terms:
        if term.vref0:
            potentials[term.target] += term.vref0 * vref0[term.source]
            potentials[term.source] += term.vref0 * vref0[term.target]
        if term.vref1:
            potentials[term.target] += term.vref1 * vref1[term.source]
            potentials[term.source] += term.vref1 * vref1[term.target]
    return potentials


def _project_transition_potentials(tdobj, blocks, potentials):
    mo = cp.asarray(tdobj._scf.mo_coeff)
    q_alpha = cp.zeros((mo.shape[1], mo.shape[1]))
    q_beta = cp.zeros_like(q_alpha)
    for label, (target, source, coefficient) in blocks.items():
        potential = mo.conj().T @ potentials[label] @ mo
        q_beta[:, target] += potential[:, source] @ coefficient.T
        q_alpha[:, source] += potential[target, :].T @ coefficient
    return q_alpha, q_beta


def spin_lowering_hfx_projection_q(tdobj, xy):
    """Transition-factor derivative of the lowering response scalar."""
    spaces, amplitudes, densities = spin_lowering_transition_densities(
        tdobj,
        xy,
    )
    blocks = spin_lowering_block_data(spaces, amplitudes)
    vref0, vref1 = _apply_hfx_responses(tdobj, densities)
    potentials = _response_potentials(
        densities,
        vref0,
        vref1,
        spin_lowering_response_terms(spaces.spin),
    )
    return _project_transition_potentials(tdobj, blocks, potentials)


def spin_lowering_fock_q(tdobj, xy, max_memory=None, cache=None, include_probe_response=True):
    """Explicit-Fock projection and reference-density response M matrices.

    ``include_probe_response=False`` returns only the frozen-operator
    projection part; the bilinear cross assembly adds the probe response
    once, evaluated on cross probes, instead of once per polarization.
    """
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    nmo = mo.shape[1]
    fock0, fockz = spin_lowering_fock0_fockz(
        tdobj,
        max_memory=max_memory,
        cache=cache,
    )
    fock0_mo = mo.conj().T @ fock0 @ mo
    fockz_mo = mo.conj().T @ fockz @ mo
    q_alpha = cp.zeros((nmo, nmo))
    q_beta = cp.zeros_like(q_alpha)
    is_hf = mf._numint._xc_type(mf.xc) == 'HF'

    for term in spin_lowering_fock_projections(tdobj, xy):
        left = term.left_indices
        right = term.right_indices
        coefficient = term.coefficient

        def project(target, operator, scale):
            if scale:
                target[:, left] += scale * operator[:, right] @ coefficient.T
                target[:, right] += scale * operator[:, left] @ coefficient

        project(q_alpha, fock0_mo, 0.5 * term.weight_f0)
        project(q_beta, fock0_mo, 0.5 * term.weight_f0)
        if is_hf:
            project(q_alpha, fockz_mo, 0.5 * term.weight_fz)
            project(q_beta, fockz_mo, 0.5 * term.weight_fz)
        else:
            project(q_alpha, fockz_mo, term.weight_fz)

    if not include_probe_response:
        return q_alpha, q_beta

    p0, pz = spin_lowering_fock_probes(tdobj, xy)
    p_alpha = 0.5 * p0
    p_beta = 0.5 * p0
    if is_hf:
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


# AO J/K nuclear derivatives


# Hybrid/RSH Fz correction


def spin_lowering_fockz_hfx_terms(
    gradient_driver, tdobj, pz, atmlst=None, with_direct=True, jk_ledger=None, output_slot=0
):
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


# Channel assembly


def gradient_xc_request(tdobj, xy):
    """Return the GGA/MGGA channel and Fz probe for frame batching."""
    spaces, amplitudes, densities = spin_lowering_transition_densities(
        tdobj,
        xy,
    )
    channel_data = (
        spaces,
        amplitudes,
        densities,
        spin_lowering_block_data(spaces, amplitudes),
        spin_lowering_response_terms(spaces.spin),
    )
    _p0, pz = spin_lowering_fock_probes(tdobj, xy)
    return channel_data, spaces, pz


def cross_xc_request(tdobj, xy_i, xy_j):
    """Return the two polarized XC channels and cross Fz probe."""
    quarter = 0.25
    x_plus = _combine_amplitudes(xy_i, xy_j, 1.0)
    x_minus = _combine_amplitudes(xy_i, xy_j, -1.0)
    spaces, amp_plus, dens_plus = spin_lowering_transition_densities(
        tdobj,
        x_plus,
    )
    _spaces_m, amp_minus, dens_minus = spin_lowering_transition_densities(
        tdobj,
        x_minus,
    )
    terms = spin_lowering_response_terms(spaces.spin)
    channel_plus = (
        spaces,
        amp_plus,
        dens_plus,
        spin_lowering_block_data(spaces, amp_plus),
        terms,
    )
    channel_minus = (
        spaces,
        amp_minus,
        dens_minus,
        spin_lowering_block_data(spaces, amp_minus),
        terms,
    )
    _p0_plus, pz_plus = spin_lowering_fock_probes(tdobj, x_plus)
    _p0_minus, pz_minus = spin_lowering_fock_probes(tdobj, x_minus)
    return (channel_plus, channel_minus), spaces, quarter * (pz_plus - pz_minus)


def prepare_grad_elec(
    gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12, max_cycle=None, cache=None, xc_terms=None
):
    """Prepare the analytic excitation gradient through its Z-vector RHS."""
    if tdobj.deltaS != -1:
        raise ValueError('deltaS=-1 gradient received a different spin channel')
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    if cache is None:
        cache = ResponseCache(tdobj)
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)

    # 1. Native amplitudes, transition densities, and explicit Fock probes.
    spaces, amplitudes, densities = spin_lowering_transition_densities(
        tdobj,
        xy,
    )
    blocks = spin_lowering_block_data(spaces, amplitudes)
    response_terms = spin_lowering_response_terms(spaces.spin)
    channel_data = (spaces, amplitudes, densities, blocks, response_terms)
    p0, pz = spin_lowering_fock_probes(tdobj, xy)
    jk_ledger = _JKDerivativeLedger()
    direct_slot = 'direct'
    zvector_slot = 'zvector'

    # 2. Explicit Fock contribution to the orbital-rotation M matrix.
    fock_alpha, fock_beta = spin_lowering_fock_q(tdobj, xy, cache=cache)
    hfx_alpha, hfx_beta = spin_lowering_hfx_projection_q(tdobj, xy)

    if xctype == 'HF':
        # 3a. HF response and fixed-orbital AO derivative.
        m_matrix = fock_alpha + fock_beta + hfx_alpha + hfx_beta
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
            fockz_hfx = spin_lowering_fockz_hfx_terms(
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
            raise NotImplementedError('NTTDA deltaS=-1 gradient does not support XC type %s' % xctype) from error

        frame_xc_backend = cache.xc_backend
        if xc_terms is None and xctype in ('GGA', 'MGGA') and frame_xc_backend:
            response_xc = getattr(
                frame_xc_backend,
                xctype.lower() + '_response_terms_batch',
            )(
                gradient_driver,
                tdobj,
                (channel_data,),
                atmlst=atmlst,
            )[0]
            fockz_xc = getattr(
                frame_xc_backend,
                xctype.lower() + '_fockz_terms_batch',
            )(
                gradient_driver,
                tdobj,
                spaces,
                (pz,),
                atmlst=atmlst,
            )[0]
        elif xc_terms is None:
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
        else:
            response_xc, fockz_xc = xc_terms
        fockz_hfx = spin_lowering_fockz_hfx_terms(
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
    """Build the complete analytic excitation gradient for deltaS=-1."""
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


# Bilinear interstate (cross) assembly


def _combine_amplitudes(xy_i, xy_j, sign):
    x_i = xy_i[0] if isinstance(xy_i, (tuple, list)) else xy_i
    x_j = xy_j[0] if isinstance(xy_j, (tuple, list)) else xy_j
    return (cp.asarray(x_i) + sign * cp.asarray(x_j), 0)


def _paired_hfx_responses(tdobj, densities_plus, densities_minus):
    """Hybrid/RSH ``vref0/vref1`` for two density sets in one batched pass."""
    merged = {}
    for label, value in densities_plus.items():
        merged[('+', label)] = value
    for label, value in densities_minus.items():
        merged[('-', label)] = value
    vref0, vref1 = _apply_hfx_responses(tdobj, merged)

    def split(values):
        return (
            {label: values[('+', label)] for label in densities_plus},
            {label: values[('-', label)] for label in densities_minus},
        )

    return split(vref0), split(vref1)


def prepare_grad_elec_cross(
    gradient_driver, tdobj, xy_i, xy_j, atmlst=None, tolerance=1e-12, max_cycle=None, cache=None, xc_terms=None
):
    """Prepare the bilinear interstate numerator through its Z-vector RHS.

    Mathematically identical to
    ``0.25*(grad_elec(X_I+X_J) - grad_elec(X_I-X_J))`` but evaluated with
    one Z-vector solve, one shared J/K derivative-ledger contraction, one
    batched Fock-derivative pass, and single evaluations of every layer
    that is linear in its probe (probe response, Fz, nobeta): the
    polarization trick pays all of these twice.  Mathematically this is the
    symmetric bilinear form ``X_I.T A^[R] X_J`` with relaxed orbitals.
    """
    if tdobj.deltaS != -1:
        raise ValueError('deltaS=-1 cross gradient received a different spin channel')
    if atmlst is None:
        atmlst = range(tdobj.mol.natm)
    atmlst = tuple(atmlst)
    if cache is None:
        cache = ResponseCache(tdobj)
    mf = tdobj._scf
    xctype = mf._numint._xc_type(mf.xc)
    quarter = 0.25
    x_plus = _combine_amplitudes(xy_i, xy_j, 1.0)
    x_minus = _combine_amplitudes(xy_i, xy_j, -1.0)

    # Quadratic probes -> cross probes via the polarization difference.
    p0_plus, pz_plus = spin_lowering_fock_probes(tdobj, x_plus)
    p0_minus, pz_minus = spin_lowering_fock_probes(tdobj, x_minus)
    p0_cross = quarter * (p0_plus - p0_minus)
    pz_cross = quarter * (pz_plus - pz_minus)

    # Frozen-operator Fock projections per polarization (cheap MO algebra);
    # the probe response is linear in the probe and evaluated once.
    fock_plus = spin_lowering_fock_q(
        tdobj,
        x_plus,
        cache=cache,
        include_probe_response=False,
    )
    fock_minus = spin_lowering_fock_q(
        tdobj,
        x_minus,
        cache=cache,
        include_probe_response=False,
    )
    is_hf = xctype == 'HF'
    probe_alpha = 0.5 * p0_cross
    probe_beta = 0.5 * p0_cross
    if is_hf:
        probe_alpha, probe_beta = methods.hf_probes(methods.get_method(tdobj), p0_cross, pz_cross)
    response_alpha, response_beta = _fock_response_q(
        tdobj,
        probe_alpha,
        probe_beta,
        cache=cache,
    )
    fock_alpha = quarter * (fock_plus[0] - fock_minus[0]) + response_alpha
    fock_beta = quarter * (fock_plus[1] - fock_minus[1]) + response_beta

    # Hybrid/RSH response projections with one batched J/K pass over the
    # union of both polarizations' transition densities.
    spaces, amp_plus, dens_plus = spin_lowering_transition_densities(
        tdobj,
        x_plus,
    )
    _spaces_m, amp_minus, dens_minus = spin_lowering_transition_densities(
        tdobj,
        x_minus,
    )
    response_terms = spin_lowering_response_terms(spaces.spin)
    (v0_plus, v0_minus), (v1_plus, v1_minus) = _paired_hfx_responses(
        tdobj,
        dens_plus,
        dens_minus,
    )
    pot_plus = _response_potentials(
        dens_plus,
        v0_plus,
        v1_plus,
        response_terms,
    )
    pot_minus = _response_potentials(
        dens_minus,
        v0_minus,
        v1_minus,
        response_terms,
    )
    hfx_plus = _project_transition_potentials(
        tdobj,
        spin_lowering_block_data(spaces, amp_plus),
        pot_plus,
    )
    hfx_minus = _project_transition_potentials(
        tdobj,
        spin_lowering_block_data(spaces, amp_minus),
        pot_minus,
    )
    hfx_alpha = quarter * (hfx_plus[0] - hfx_minus[0])
    hfx_beta = quarter * (hfx_plus[1] - hfx_minus[1])

    jk_ledger = _JKDerivativeLedger()
    direct_slot = 'direct'
    zvector_slot = 'zvector'

    if xctype == 'HF':
        m_matrix = fock_alpha + fock_beta + hfx_alpha + hfx_beta
        direct = response_direct_hfx(
            gradient_driver,
            tdobj,
            dens_plus,
            response_terms,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
            scale=quarter,
        )
        direct += response_direct_hfx(
            gradient_driver,
            tdobj,
            dens_minus,
            response_terms,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
            scale=-quarter,
        )
        if methods.get_method(tdobj).separate_hf_fockz:
            fockz_hfx = spin_lowering_fockz_hfx_terms(
                gradient_driver,
                tdobj,
                pz_cross,
                atmlst=atmlst,
                jk_ledger=jk_ledger,
                output_slot=direct_slot,
            )
            m_matrix += fockz_hfx.q_alpha + fockz_hfx.q_beta
            direct += fockz_hfx.direct
        direct_fock_probes = methods.hf_probes(methods.get_method(tdobj), p0_cross, pz_cross)

    else:
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
            raise NotImplementedError('NTTDA deltaS=-1 cross gradient does not support XC type %s' % xctype) from error

        blocks_plus = spin_lowering_block_data(spaces, amp_plus)
        blocks_minus = spin_lowering_block_data(spaces, amp_minus)
        channel_plus = (
            spaces,
            amp_plus,
            dens_plus,
            blocks_plus,
            response_terms,
        )
        channel_minus = (
            spaces,
            amp_minus,
            dens_minus,
            blocks_minus,
            response_terms,
        )
        frame_xc_backend = cache.xc_backend
        if xc_terms is None and xctype in ('GGA', 'MGGA') and frame_xc_backend:
            response_xc_plus, response_xc_minus = getattr(
                frame_xc_backend,
                xctype.lower() + '_response_terms_batch',
            )(
                gradient_driver,
                tdobj,
                (channel_plus, channel_minus),
                atmlst=atmlst,
            )
        elif xc_terms is None:
            response_xc_plus = response_builder(
                gradient_driver,
                tdobj,
                channel_plus,
                atmlst=atmlst,
            )
            response_xc_minus = response_builder(
                gradient_driver,
                tdobj,
                channel_minus,
                atmlst=atmlst,
            )
        else:
            response_xc_plus, response_xc_minus, _fockz_xc = xc_terms
        # Fz, hybrid-Fz, and nobeta layers are linear in their probes and
        # are evaluated once on the cross probes.
        if xc_terms is None and xctype in ('GGA', 'MGGA') and frame_xc_backend:
            fockz_xc = getattr(
                frame_xc_backend,
                xctype.lower() + '_fockz_terms_batch',
            )(
                gradient_driver,
                tdobj,
                spaces,
                (pz_cross,),
                atmlst=atmlst,
            )[0]
        elif xc_terms is None:
            fockz_xc = fockz_builder(
                gradient_driver,
                tdobj,
                spaces,
                pz_cross,
                atmlst=atmlst,
            )
        else:
            fockz_xc = _fockz_xc
        fockz_hfx = spin_lowering_fockz_hfx_terms(
            gradient_driver,
            tdobj,
            pz_cross,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
        )
        nobeta_alpha, nobeta_beta = methods.common_fock_q(methods.get_method(tdobj), tdobj, p0_cross)
        m_matrix = (
            fock_alpha
            + fock_beta
            + hfx_alpha
            + hfx_beta
            + quarter
            * (
                response_xc_plus.q_alpha
                + response_xc_plus.q_beta
                - response_xc_minus.q_alpha
                - response_xc_minus.q_beta
            )
            + fockz_xc.q_alpha
            + fockz_xc.q_beta
            + fockz_hfx.q_alpha
            + fockz_hfx.q_beta
            + nobeta_alpha
            + nobeta_beta
        )

        direct_fock_probes = (0.5 * p0_cross, 0.5 * p0_cross)
        direct = response_direct_hfx(
            gradient_driver,
            tdobj,
            dens_plus,
            response_terms,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
            scale=quarter,
        )
        direct += response_direct_hfx(
            gradient_driver,
            tdobj,
            dens_minus,
            response_terms,
            atmlst=atmlst,
            jk_ledger=jk_ledger,
            output_slot=direct_slot,
            scale=-quarter,
        )
        direct += quarter * (response_xc_plus.direct - response_xc_minus.direct)
        direct += fockz_xc.direct
        direct += fockz_hfx.direct

    return prepare_gradient(
        gradient_driver,
        tdobj,
        m_matrix,
        direct,
        atmlst,
        tolerance,
        max_cycle,
        jk_ledger=jk_ledger,
        nobeta_p0=p0_cross,
        direct_fock_probes=direct_fock_probes,
        cache=cache,
    )


def grad_elec_cross(gradient_driver, tdobj, xy_i, xy_j, atmlst=None, tolerance=1e-12, max_cycle=None, cache=None):
    """Relaxed bilinear interstate numerator ``X_I^T A^[R] X_J``."""
    prepared = methods.prepare_cross(
        methods.get_method(tdobj),
        gradient_driver,
        tdobj,
        xy_i,
        xy_j,
        atmlst=atmlst,
        tolerance=tolerance,
        max_cycle=max_cycle,
        cache=cache,
    )
    return finish_prepared_gradients((prepared,))[0]
