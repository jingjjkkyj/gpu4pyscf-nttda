"""GPU XC projections and explicitly bounded host quadrature fallback."""

from dataclasses import dataclass
import cupy as cp
from gpu4pyscf.sftda import nttda_methods as methods


@dataclass(frozen=True)
class XCGradientTerms:
    q_alpha: cp.ndarray
    q_beta: cp.ndarray
    direct: cp.ndarray


def _project_channel_potentials(tdobj, potentials, blocks):
    """Project transition-factor potentials for any NTTDA spin channel."""
    mo = cp.asarray(tdobj._scf.mo_coeff)
    q_alpha = cp.zeros((mo.shape[1], mo.shape[1]))
    q_beta = cp.zeros_like(q_alpha)
    for label, (target, source, coefficient) in blocks.items():
        potential = mo.conj().T @ potentials[label] @ mo
        q_beta[:, target] += potential[:, source] @ coefficient.T
        q_alpha[:, source] += potential[target, :].T @ coefficient
    return q_alpha, q_beta


def _reference_spin_densities(tdobj):
    return methods.spin_densities(methods.get_method(tdobj), tdobj._scf)


def _reference_spin_occupations(tdobj):
    return methods.spin_occupations(methods.get_method(tdobj), tdobj._scf)


def _add_reference_q(tdobj, q_alpha, q_beta, matrix_alpha, matrix_beta):
    mf = tdobj._scf
    mo = cp.asarray(mf.mo_coeff)
    occupation_alpha, occupation_beta = _reference_spin_occupations(tdobj)
    q_alpha += (mo.conj().T @ (matrix_alpha + matrix_alpha.T) @ mo) * occupation_alpha[None, :]
    q_beta += (mo.conj().T @ (matrix_beta + matrix_beta.T) @ mo) * occupation_beta[None, :]


@dataclass(frozen=True)
class QuadratureReference:
    mol: object
    grids: object
    xc: str
    mo_coeff: object
    mo_occ: object
    _numint: object
    max_memory: float


@dataclass(frozen=True)
class QuadratureInput:
    _scf: QuadratureReference
    method: object
    max_memory: float

    @property
    def mol(self):
        return self._scf.mol


def _host(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    if isinstance(value, dict):
        return {key: _host(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_host(item) for item in value)
    from dataclasses import is_dataclass, fields, replace

    if is_dataclass(value):
        return replace(value, **{field.name: _host(getattr(value, field.name)) for field in fields(value)})
    return value


def _reference(mf):
    from pyscf.dft import numint, gen_grid

    grids = gen_grid.Grids(mf.mol)
    grids.coords = _host(mf.grids.coords)
    grids.weights = _host(mf.grids.weights)
    ni = numint.NumInt()
    ni.omega = mf._numint.omega
    return QuadratureReference(mf.mol, grids, mf.xc, _host(mf.mo_coeff), _host(mf.mo_occ), ni, mf.max_memory)


def _evaluate(name, driver, td, *args, **kwargs):
    from . import xc_host

    inputs = QuadratureInput(_reference(td._scf), methods.get_method(td), td.max_memory)
    result = getattr(xc_host, name)(driver, inputs, *_host(args), **_host(kwargs))
    return XCGradientTerms(cp.asarray(result.q_alpha), cp.asarray(result.q_beta), cp.asarray(result.direct))


def lda_response_terms(driver, td, *args, **kwargs):
    return _evaluate('lda_response_terms', driver, td, *args, **kwargs)


def lda_fockz_terms(driver, td, *args, **kwargs):
    return _evaluate('lda_fockz_terms', driver, td, *args, **kwargs)


def gga_response_terms(driver, td, *args, **kwargs):
    from gpu4pyscf.grad.nttda_xc import GPUXCFrameBackend

    return GPUXCFrameBackend(td._scf, td).gga_response_terms_batch(driver, td, [args[0]], **kwargs)[0]


def gga_fockz_terms(driver, td, *args, **kwargs):
    from gpu4pyscf.grad.nttda_xc import GPUXCFrameBackend

    return GPUXCFrameBackend(td._scf, td).gga_fockz_terms_batch(driver, td, args[0], [args[1]], **kwargs)[0]


def mgga_response_terms(driver, td, *args, **kwargs):
    if methods.get_method(td).id == 'roks_nobeta':
        return _evaluate('mgga_response_terms', driver, td, *args, **kwargs)
    from gpu4pyscf.grad.nttda_xc import GPUXCFrameBackend

    return GPUXCFrameBackend(td._scf, td).mgga_response_terms_batch(driver, td, [args[0]], **kwargs)[0]


def mgga_fockz_terms(driver, td, *args, **kwargs):
    if methods.get_method(td).id == 'roks_nobeta':
        return _evaluate('mgga_fockz_terms', driver, td, *args, **kwargs)
    from gpu4pyscf.grad.nttda_xc import GPUXCFrameBackend

    return GPUXCFrameBackend(td._scf, td).mgga_fockz_terms_batch(driver, td, args[0], [args[1]], **kwargs)[0]


def _contract(kind, mf, *args, **kwargs):
    from . import xc_host

    return cp.asarray(
        getattr(xc_host, 'contract_' + kind + '_vxc_derivative')(_reference(mf), *_host(args), **_host(kwargs))
    )


def contract_lda_vxc_derivative(mf, *args, **kwargs):
    return _contract('lda', mf, *args, **kwargs)


def contract_gga_vxc_derivative(mf, *args, **kwargs):
    return _contract('gga', mf, *args, **kwargs)


def contract_mgga_vxc_derivative(mf, *args, **kwargs):
    return _contract('mgga', mf, *args, **kwargs)


def _nobeta_q(kind, td, p0, max_memory=None):
    from . import xc_host

    inputs = QuadratureInput(_reference(td._scf), methods.get_method(td), td.max_memory)
    return tuple(
        cp.asarray(value)
        for value in getattr(xc_host, kind + '_nobeta_reference_q')(inputs, _host(p0), max_memory=max_memory)
    )


def lda_nobeta_reference_q(td, p0, max_memory=None):
    return _nobeta_q('lda', td, p0, max_memory)


def gga_nobeta_reference_q(td, p0, max_memory=None):
    return _nobeta_q('gga', td, p0, max_memory)


def mgga_nobeta_reference_q(td, p0, max_memory=None):
    return _nobeta_q('mgga', td, p0, max_memory)
