"""Shared fixed-AO J/K derivative ledger and Fock contractions."""

from dataclasses import dataclass
import cupy as cp
from gpu4pyscf.grad.ensemble_roks import _hcore_derivative_generator
from gpu4pyscf.sftda import nttda_methods as methods
from . import xc as xc_backend


@dataclass(frozen=True)
class _JKDerivativeTerm:
    """One fixed-AO bilinear derivative with a named output slot."""

    left: cp.ndarray
    right: cp.ndarray
    scale: float
    omega: float
    slot: object


class _JKDerivativeLedger:
    """Spin-lowering scheduler for fixed-AO J/K derivatives."""

    def __init__(self):
        self._terms = {'j': [], 'k': []}

    def add(self, operator, slot, terms):
        self._terms[operator].extend(
            _JKDerivativeTerm(left, right, float(scale), omega, slot)
            for left, right, scale, omega in terms
            if scale != 0.0
        )

    def contract(self, gradient_driver, mol, atoms, slots=()):
        return gradient_driver._context.ledger(self._terms, mol, atoms, slots)


def _reference_spin_densities(tdobj):
    return methods.spin_densities(methods.get_method(tdobj), tdobj._scf)


def _spin_probe_stacks(p_alpha, p_beta):
    p_alpha = cp.asarray(p_alpha)
    p_beta = cp.asarray(p_beta)
    single_probe = p_alpha.ndim == 2
    if single_probe:
        p_alpha = p_alpha[None]
        p_beta = p_beta[None]
    return p_alpha, p_beta, single_probe


def spin_fock_direct_dft(
    gradient_driver,
    tdobj,
    p_alpha,
    p_beta,
    atmlst=None,
    nobeta_p0=None,
    jk_ledger=None,
    output_slots=None,
    with_xc=True,
    with_hcore=True,
):
    """Differentiate one or more ordinary UKS Fock scalar probes.

    The optional ``nobeta_p0`` correction belongs to the first, explicit-direct
    probe in the batch.
    """
    mf = tdobj._scf
    mol = tdobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    p_alpha, p_beta, single_probe = _spin_probe_stacks(
        p_alpha,
        p_beta,
    )
    if output_slots is None:
        output_slots = tuple(range(len(p_alpha)))
    p_total = p_alpha + p_beta
    density_alpha, density_beta = _reference_spin_densities(tdobj)
    gradient = cp.zeros((len(p_alpha), len(atmlst), 3))
    if with_hcore:
        hcore_derivative = _hcore_derivative_generator(mol)
        for k, atom in enumerate(atmlst):
            gradient[:, k] += cp.einsum(
                'npq,xpq->nx',
                p_total,
                cp.asarray(hcore_derivative(atom)),
            )
    ni = mf._numint
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    local_ledger = _JKDerivativeLedger()
    ledger = jk_ledger if jk_ledger is not None else local_ledger
    for probe in range(len(p_alpha)):
        j_terms = [
            (p_total[probe], density_alpha, 1.0, None),
            (p_total[probe], density_beta, 1.0, None),
        ]
        k_terms = []
        if ni.libxc.is_hybrid_xc(mf.xc):
            k_terms.extend(
                (
                    (p_alpha[probe], density_alpha, -hybrid, None),
                    (p_beta[probe], density_beta, -hybrid, None),
                )
            )
            if omega != 0:
                long_range = -(alpha - hybrid)
                k_terms.extend(
                    (
                        (p_alpha[probe], density_alpha, long_range, omega),
                        (p_beta[probe], density_beta, long_range, omega),
                    )
                )
        ledger.add('j', output_slots[probe], j_terms)
        ledger.add('k', output_slots[probe], k_terms)
    xctype = ni._xc_type(mf.xc)
    if xctype == 'LDA':
        derivative_contractor = xc_backend.contract_lda_vxc_derivative
    elif xctype == 'GGA':
        derivative_contractor = xc_backend.contract_gga_vxc_derivative
    elif xctype == 'MGGA':
        derivative_contractor = xc_backend.contract_mgga_vxc_derivative
    else:
        raise NotImplementedError('ordinary Fock direct derivative is not implemented for %s' % xctype)
    if with_xc:
        terms = methods.direct_xc_terms(
            methods.get_method(tdobj),
            density_alpha,
            density_beta,
            p_alpha,
            p_beta,
            nobeta_p0,
        )
        for dma, dmb, pa, pb, row in terms:
            contribution = derivative_contractor(
                mf,
                dma,
                dmb,
                pa,
                pb,
                atmlst=atmlst,
                max_memory=gradient_driver.max_memory,
            )
            if row is None:
                gradient += contribution
            else:
                gradient[row] += contribution
    if jk_ledger is None:
        contractions = local_ledger.contract(
            gradient_driver,
            mol,
            atmlst,
            slots=output_slots,
        )
        for probe, slot in enumerate(output_slots):
            gradient[probe] += contractions[slot]
    return gradient[0] if single_probe else gradient


def spin_fock_direct_hf(
    gradient_driver, tdobj, p_alpha, p_beta, atmlst=None, jk_ledger=None, output_slots=None, with_hcore=True
):
    """Differentiate one or more spin-resolved HF Fock scalar probes."""
    mol = tdobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    p_alpha, p_beta, single_probe = _spin_probe_stacks(
        p_alpha,
        p_beta,
    )
    if output_slots is None:
        output_slots = tuple(range(len(p_alpha)))
    p_total = p_alpha + p_beta
    dm_alpha, dm_beta = _reference_spin_densities(tdobj)
    gradient = cp.zeros((len(p_alpha), len(atmlst), 3))

    if with_hcore:
        hcore_derivative = _hcore_derivative_generator(mol)
        for k, atom in enumerate(atmlst):
            gradient[:, k] += cp.einsum(
                'npq,xpq->nx',
                p_total,
                cp.asarray(hcore_derivative(atom)),
            )
    local_ledger = _JKDerivativeLedger()
    ledger = jk_ledger if jk_ledger is not None else local_ledger
    for probe in range(len(p_alpha)):
        ledger.add(
            'j',
            output_slots[probe],
            (
                (p_total[probe], dm_alpha, 1.0, None),
                (p_total[probe], dm_beta, 1.0, None),
            ),
        )
        ledger.add(
            'k',
            output_slots[probe],
            (
                (p_alpha[probe], dm_alpha, -1.0, None),
                (p_beta[probe], dm_beta, -1.0, None),
            ),
        )
    if jk_ledger is None:
        contractions = local_ledger.contract(
            gradient_driver,
            mol,
            atmlst,
            slots=output_slots,
        )
        for probe, slot in enumerate(output_slots):
            gradient[probe] += contractions[slot]
    return gradient[0] if single_probe else gradient


def response_direct_hfx(
    gradient_driver, tdobj, densities, response_terms, atmlst=None, jk_ledger=None, output_slot=0, scale=1.0
):
    """J/K skeleton derivative for a channel response-term ledger.

    ``scale`` multiplies every term; the bilinear cross assembly pushes the
    two polarizations into one shared ledger with scales +/- 1/4.
    """
    mol = tdobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)
    atmlst = tuple(atmlst)
    gradient = cp.zeros((len(atmlst), 3))
    ni = tdobj._scf._numint
    omega, alpha, hybrid = ni.rsh_and_hybrid_coeff(
        tdobj._scf.xc,
        mol.spin,
    )
    if not ni.libxc.is_hybrid_xc(tdobj._scf.xc):
        return gradient

    scales = [(hybrid, None)]
    if omega != 0:
        scales.append((alpha - hybrid, omega))
    j_terms = []
    k_terms = []
    for term in response_terms:
        target = densities[term.target]
        source = densities[term.source]
        for coefficient, range_omega in scales:
            if term.vref0:
                k_terms.append(
                    (
                        target,
                        source,
                        -scale * coefficient * term.vref0,
                        range_omega,
                    )
                )
            if term.vref1:
                j_terms.append(
                    (
                        target,
                        source,
                        -scale * coefficient * term.vref1,
                        range_omega,
                    )
                )
    local_ledger = _JKDerivativeLedger()
    ledger = jk_ledger if jk_ledger is not None else local_ledger
    ledger.add('j', output_slot, j_terms)
    ledger.add('k', output_slot, k_terms)
    if jk_ledger is None:
        gradient += local_ledger.contract(
            gradient_driver,
            mol,
            atmlst,
            slots=(output_slot,),
        )[output_slot]
    return gradient
