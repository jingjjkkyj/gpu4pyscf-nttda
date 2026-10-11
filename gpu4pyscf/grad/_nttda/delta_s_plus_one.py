# Copyright 2026 The NEST Developers and The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU spin-raising gradient: (Dp-Dh):F0 + (Dp+Dh):Fz + T:K0[T].

Only the CV block occurs. The prepared derivative uses the same adjoint,
J/K ledger and XC contractions as the other spin channels.
"""

import cupy as cp

from gpu4pyscf.sftda import nttda_methods as methods
from gpu4pyscf.sftda.nttda import gen_rohf_response_sfu
from . import xc as xc_backend
from .delta_s_zero import FockProjection, ResponseTerm, fock_projection_q, same_spin_fockz_hfx_terms
from .derivative_jk import _JKDerivativeLedger, response_direct_hfx
from .response import orbital_spaces, pair_density, ResponseCache, prepare_gradient


RESPONSE_TERMS = (ResponseTerm('CV', 'CV', 1.0, 0.0),)


def split_spin_raising(tdobj, xy):
    spaces = orbital_spaces(tdobj)
    vector = cp.asarray(xy[0] if isinstance(xy, (tuple, list)) else xy)
    nc, nv = len(spaces.closed), len(spaces.virtual)
    if vector.size != nc * nv:
        raise ValueError('deltaS=+1 amplitude has size %d; expected %d' % (vector.size, nc * nv))
    return spaces, vector.reshape(nc, nv)


def spin_raising_fock_projections(tdobj, xy):
    spaces, cv = split_spin_raising(tdobj, xy)
    c, v = spaces.closed, spaces.virtual
    return (
        FockProjection('cv-vv', v, spaces.c_virtual, cv.T @ cv, v, spaces.c_virtual, 1.0, 1.0),
        FockProjection('cv-cc', c, spaces.c_closed, cv @ cv.T, c, spaces.c_closed, -1.0, 1.0),
    )


def spin_raising_fock_probes(tdobj, xy):
    particle, hole = (term.density() for term in spin_raising_fock_projections(tdobj, xy))
    return particle - hole, particle + hole


def fock0_fockz(tdobj, cache=None):
    if cache is not None and 'fock0_fockz' in cache.extra:
        return cache.extra['fock0_fockz']
    _, fockz = gen_rohf_response_sfu(tdobj._scf, fxc_ref=None if cache is None else cache.fxc_ref())
    result = methods.fock0(methods.get_method(tdobj), tdobj._scf), fockz
    if cache is not None:
        cache.extra['fock0_fockz'] = result
    return result


def gradient_xc_request(tdobj, xy):
    spaces, cv = split_spin_raising(tdobj, xy)
    densities = {'CV': pair_density(spaces.c_virtual, cv.T, spaces.c_closed)}
    blocks = {'CV': (spaces.virtual, spaces.closed, cv.T)}
    _, pz = spin_raising_fock_probes(tdobj, xy)
    return (spaces, cv, densities, blocks, RESPONSE_TERMS), spaces, pz


def _hfx_projection_q(tdobj, spaces, cv, density):
    """Differentiate the two transition factors of T:K0[T]."""
    mf = tdobj._scf
    omega, alpha, hybrid = mf._numint.rsh_and_hybrid_coeff(mf.xc, mf.mol.spin)
    potential = cp.zeros_like(density)
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        potential -= 2 * hybrid * mf.get_k(mf.mol, density, hermi=0)
        if omega != 0:
            potential -= 2 * (alpha - hybrid) * mf.get_k(mf.mol, density, hermi=0, omega=omega)
    mo = cp.asarray(mf.mo_coeff)
    potential = mo.T @ potential @ mo
    result = cp.zeros_like(potential)
    result[:, spaces.virtual] += potential[:, spaces.closed] @ cv
    result[:, spaces.closed] += potential[spaces.virtual, :].T @ cv.T
    return result


def prepare_grad_elec(
    gradient_driver, tdobj, xy, atmlst=None, tolerance=1e-12, max_cycle=None, cache=None, xc_terms=None,
):
    if tdobj.deltaS != 1:
        raise ValueError('deltaS=+1 gradient received a different spin channel')
    if cache is None:
        cache = ResponseCache(tdobj)
    atmlst = tuple(range(tdobj.mol.natm) if atmlst is None else atmlst)
    channel, spaces, pz = gradient_xc_request(tdobj, xy)
    _, cv, densities, _, _ = channel
    p0, _ = spin_raising_fock_probes(tdobj, xy)
    q_alpha, q_beta = fock_projection_q(
        tdobj, spin_raising_fock_projections(tdobj, xy),
        fock0_fockz(tdobj, cache), (p0, pz), cache=cache,
    )
    m_matrix = q_alpha + q_beta + _hfx_projection_q(tdobj, spaces, cv, densities['CV'])
    ledger = _JKDerivativeLedger()
    direct = response_direct_hfx(
        gradient_driver, tdobj, densities, RESPONSE_TERMS,
        atmlst=atmlst, jk_ledger=ledger, output_slot='direct',
    )
    method = methods.get_method(tdobj)
    xctype = tdobj._scf._numint._xc_type(tdobj._scf.xc)
    if xctype == 'HF':
        probes = methods.hf_probes(method, p0, pz)
    else:
        if xc_terms is None:
            response_xc = getattr(xc_backend, xctype.lower() + '_response_terms')(
                gradient_driver, tdobj, channel, atmlst=atmlst,
            )
            fockz_xc = getattr(xc_backend, xctype.lower() + '_fockz_terms')(
                gradient_driver, tdobj, spaces, pz, atmlst=atmlst,
            )
        else:
            response_xc, fockz_xc = xc_terms
        m_matrix += response_xc.q_alpha + response_xc.q_beta + fockz_xc.q_alpha + fockz_xc.q_beta
        direct += response_xc.direct + fockz_xc.direct
        common_alpha, common_beta = methods.common_fock_q(method, tdobj, p0)
        m_matrix += common_alpha + common_beta
        probes = (0.5 * p0, 0.5 * p0)
    if xctype != 'HF' or method.separate_hf_fockz:
        fockz_hfx = same_spin_fockz_hfx_terms(
            gradient_driver, tdobj, pz, atmlst=atmlst,
            jk_ledger=ledger, output_slot='direct',
        )
        m_matrix += fockz_hfx.q_alpha + fockz_hfx.q_beta
        direct += fockz_hfx.direct
    return prepare_gradient(
        gradient_driver, tdobj, m_matrix, direct, atmlst, tolerance, max_cycle,
        jk_ledger=ledger, nobeta_p0=p0, direct_fock_probes=probes, cache=cache,
    )
