"""NTTDA method identity and explicit choices of reference equations.

Method records contain only physical identity. Ordinary functions select the
reference equations; no callback registry or dynamically imported recipes are
involved. Each backend owns its own implementation of these choices.
"""

from dataclasses import dataclass
import cupy as cp


@dataclass(frozen=True)
class Method:
    id: str
    reference_kind: str

    @property
    def separate_hf_fockz(self):
        return self.reference_kind != 'roks'

    @property
    def batch_xc(self):
        return self.id != 'roks_nobeta'

    @property
    def gpu_xc_types(self):
        return ('GGA',) if self.id == 'roks_nobeta' else ('GGA', 'MGGA')


METHODS = {
    name: Method(name, kind)
    for name, kind in (
        ('roks', 'roks'),
        ('roks_nobeta', 'roks'),
        ('ensemble_rks', 'ensemble_rks'),
        ('ensemble_roks', 'ensemble_roks'),
    )
}


def reference_kind(mf):
    if getattr(mf, 'reference_energy_semantics', None) == 'roks_energy_on_ensemble_rks_orbitals':
        return 'ensemble_roks'
    if getattr(mf, 'is_ensemble_rks', False):
        return 'ensemble_rks'
    return 'roks'


def method_for_reference(mf, nobeta=False):
    kind = reference_kind(mf)
    name = 'roks_nobeta' if kind == 'roks' and nobeta else kind
    return METHODS[name]


def resolve_method(td):
    method = method_for_reference(td._scf, bool(getattr(td, 'nobeta', False)))
    explicit = getattr(td, '_nttda_explicit_method', None)
    if explicit is not None and method.id != explicit:
        raise ValueError('explicit NTTDA method %s cannot switch to %s' % (explicit, method.id))
    return method


def bind_method(td, *, derivative=False):
    method = resolve_method(td)
    signature = (method.id, int(td.deltaS))
    if derivative and hasattr(td, '_nttda_solution_signature'):
        if td._nttda_solution_signature != signature:
            raise ValueError('NTTDA method or channel changed; rerun NTTDA.kernel() before derivatives')
    td._nttda_method = method
    return method


def get_method(td):
    """Return the method identity bound to the current evaluation."""
    method = getattr(td, '_nttda_method', None)
    return resolve_method(td) if method is None else method


def begin_solution(td):
    method = bind_method(td)
    td._nttda_solution_signature = None
    # A failed/new solve must never leave reusable Fock data from the old one.
    for name in ('_nttda_gpu_fxc_ref', '_nttda_gpu_fock0_fockz'):
        td.__dict__.pop(name, None)
    return method


def record_solution(td):
    td._nttda_solution_signature = (get_method(td).id, int(td.deltaS))


def explicit_solver(solver, mf, method_id):
    method = METHODS[method_id]
    if reference_kind(mf) != method.reference_kind:
        raise TypeError('%s requires a %s reference' % (method_id, method.reference_kind))
    if method.reference_kind == 'roks' and not any(
        cls.__name__ in ('ROKS', 'SymAdaptedROKS') for cls in type(mf).__mro__
    ):
        raise TypeError('%s requires a ROKS reference' % method_id)
    td = solver(mf)
    td.nobeta = method_id == 'roks_nobeta'
    td._nttda_explicit_method = method_id
    bind_method(td)
    return td


def copy_method(source, target):
    """Preserve method identity when cloning a solver."""
    method = bind_method(source, derivative=getattr(source, 'xy', None) is not None)
    target._nttda_explicit_method = getattr(source, '_nttda_explicit_method', None)
    target._nttda_method = method
    if hasattr(source, '_nttda_solution_signature'):
        target._nttda_solution_signature = source._nttda_solution_signature


def roks_fock0(mf, *, xp=cp):
    fock = mf.get_fock()
    return 0.5 * (xp.asarray(fock.focka) + xp.asarray(fock.fockb))


def ensemble_fock0(mf, *, xp=cp):
    return xp.asarray(mf.get_fock())


def roks_occupations(mf, *, xp=cp):
    occ = xp.asarray(mf.mo_occ)
    return (occ > 0).astype(float), (occ == 2).astype(float)


def ensemble_occupations(mf, *, xp=cp):
    occ = 0.5 * xp.asarray(mf.mo_occ)
    return occ, occ


def roks_densities(mf, *, xp=cp):
    mo = xp.asarray(mf.mo_coeff)
    occ = xp.asarray(mf.mo_occ)
    return mo[:, occ > 0] @ mo[:, occ > 0].T, mo[:, occ == 2] @ mo[:, occ == 2].T


def ensemble_densities(mf, *, xp=cp):
    return tuple(xp.asarray(dm) for dm in mf.make_rdm1s())


def stationary_energy(mf):
    return float(mf.e_tot)


def selected_energy(mf):
    return float(mf.reference_energy())


def reference_gradient(method, mf):
    """Return a reference driver, including its own DF dispatch."""
    return mf.nuc_grad_method()


def no_common_fock_q(td, p0, max_memory=None):
    nmo = td._scf.mo_coeff.shape[1]
    return cp.zeros((nmo, nmo)), cp.zeros((nmo, nmo))


def ordinary_xc_terms(dma, dmb, pa, pb, p0):
    return ((dma, dmb, pa, pb, None),)


def roks_hf_probes(p0, pz):
    return 0.5 * (p0 + pz), 0.5 * (p0 - pz)


def ensemble_hf_probes(p0, pz):
    return 0.5 * p0, 0.5 * p0


def prepare_gradient(method, driver, td, xy, **options):
    from gpu4pyscf.grad._nttda import delta_s_minus_one, delta_s_zero

    channels = {-1: delta_s_minus_one, 0: delta_s_zero}
    if td.deltaS == 1:
        raise NotImplementedError(
            "Analytic NTTDA gradients are not implemented for deltaS=1; use method='finite_diff'."
        )
    if td.deltaS not in channels:
        raise ValueError('deltaS must be -1, 0, or 1')
    return channels[td.deltaS].prepare_grad_elec(driver, td, xy, **options)


def prepare_cross(method, driver, td, xy_i, xy_j, **options):
    from gpu4pyscf.grad._nttda import delta_s_minus_one

    if td.deltaS != -1:
        raise NotImplementedError('NTTDA NAC currently supports only deltaS=-1')
    return delta_s_minus_one.prepare_grad_elec_cross(driver, td, xy_i, xy_j, **options)


def gradient_components(method, driver, xy, atmlst, response_cache=None):
    from gpu4pyscf.grad._nttda.response import finish_prepared_gradients

    td = driver.base
    prepared = prepare_gradient(
        get_method(td),
        driver,
        td,
        xy,
        atmlst=atmlst,
        tolerance=driver.cphf_conv_tol,
        max_cycle=driver.cphf_max_cycle,
        cache=response_cache,
    )
    return finish_prepared_gradients((prepared,))[0]


def roks_fock_response_q(tdobj, p_alpha, p_beta, cache=None):
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    occ_alpha, occ_beta = roks_occupations(mf)
    if mf._numint._xc_type(mf.xc) != 'HF':
        if cache is not None:
            response = cache.response(0)
        else:
            unrestricted = mf.to_uks()
            unrestricted.verbose = 0
            response = unrestricted.gen_response(hermi=0)
        v_alpha, v_beta = response(cp.stack((p_alpha.T, p_beta.T)))
    else:
        coulomb = mf.get_j(mf.mol, (p_alpha + p_beta).T, hermi=0)
        v_alpha = coulomb - mf.get_k(mf.mol, p_alpha.T, hermi=0)
        v_beta = coulomb - mf.get_k(mf.mol, p_beta.T, hermi=0)
    return (
        (mo.conj().T @ (v_alpha + v_alpha.T) @ mo) * occ_alpha[None, :],
        (mo.conj().T @ (v_beta + v_beta.T) @ mo) * occ_beta[None, :],
    )


def ensemble_fock_response_q(tdobj, p_alpha, p_beta, cache=None):
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    response = cache.response(0) if cache is not None else mf.gen_response(hermi=0)
    potential = response((cp.asarray(p_alpha) + cp.asarray(p_beta)).T)
    q = (mo.conj().T @ (potential + potential.T) @ mo) * cp.asarray(mf.mo_occ)[None, :]
    return 0.5 * q, 0.5 * q


def roks_response_reference(mf):
    return mf


def ensemble_response_reference(mf):
    return mf


def roks_spin_focks_mo(mf):
    fock = mf.get_fock()
    mo = cp.asarray(mf.mo_coeff)
    return mo.conj().T @ fock.focka @ mo, mo.conj().T @ fock.fockb @ mo


def ensemble_spin_focks_mo(mf):
    mo = cp.asarray(mf.mo_coeff)
    fock = mo.conj().T @ cp.asarray(mf.get_fock()) @ mo
    return fock, fock


def nobeta_fock0(mf, *, xp=cp):
    dma, dmb = mf.make_rdm1()
    dm0 = 0.5 * (xp.asarray(dma) + xp.asarray(dmb))
    fock = mf.get_fock(dm=xp.stack((dm0, dm0)))
    return 0.5 * (xp.asarray(fock.focka) + xp.asarray(fock.fockb))


def nobeta_common_fock_q(td, p0, max_memory=None):
    from gpu4pyscf.grad._nttda import xc as correction

    xctype = td._scf._numint._xc_type(td._scf.xc).lower()
    return getattr(correction, xctype + '_nobeta_reference_q')(td, p0, max_memory=max_memory)


def nobeta_direct_xc_terms(dma, dmb, pa, pb, p0):
    if p0 is None:
        return ordinary_xc_terms(dma, dmb, pa, pb, p0)
    actual_a, actual_b = cp.array(pa, copy=True), cp.array(pb, copy=True)
    actual_a[0] -= 0.5 * p0
    actual_b[0] -= 0.5 * p0
    dm0 = 0.5 * (dma + dmb)
    return (
        (dma, dmb, actual_a, actual_b, None),
        (dm0, dm0, 0.5 * p0, 0.5 * p0, 0),
    )


def fock0(method, mf, *, xp=cp):
    if method.id == 'roks_nobeta':
        return nobeta_fock0(mf, xp=xp)
    if method.reference_kind == 'roks':
        return roks_fock0(mf, xp=xp)
    return ensemble_fock0(mf, xp=xp)


def spin_densities(method, mf, *, xp=cp):
    if method.reference_kind == 'roks':
        return roks_densities(mf, xp=xp)
    return ensemble_densities(mf, xp=xp)


def spin_occupations(method, mf, *, xp=cp):
    if method.reference_kind == 'roks':
        return roks_occupations(mf, xp=xp)
    return ensemble_occupations(mf, xp=xp)


def fock_response_q(method, td, pa, pb, cache=None):
    if method.reference_kind == 'roks':
        return roks_fock_response_q(td, pa, pb, cache)
    return ensemble_fock_response_q(td, pa, pb, cache)


def response_reference(method, mf):
    if method.reference_kind == 'roks':
        return roks_response_reference(mf)
    return ensemble_response_reference(mf)


def spin_focks_mo(method, mf):
    if method.reference_kind == 'roks':
        return roks_spin_focks_mo(mf)
    return ensemble_spin_focks_mo(mf)


def reference_energy(method, mf):
    if method.reference_kind == 'ensemble_roks':
        return selected_energy(mf)
    return stationary_energy(mf)


def common_fock_q(method, td, p0, max_memory=None):
    if method.id == 'roks_nobeta':
        return nobeta_common_fock_q(td, p0, max_memory)
    return no_common_fock_q(td, p0, max_memory)


def direct_xc_terms(method, dma, dmb, pa, pb, p0):
    if method.id == 'roks_nobeta':
        return nobeta_direct_xc_terms(dma, dmb, pa, pb, p0)
    return ordinary_xc_terms(dma, dmb, pa, pb, p0)


def hf_probes(method, p0, pz):
    if method.reference_kind == 'roks':
        return roks_hf_probes(p0, pz)
    return ensemble_hf_probes(p0, pz)


def orbital_backend(method):
    from gpu4pyscf.grad._nttda import orbital

    return orbital
