# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

'''Non-relativistic ROHF analytical nuclear gradients'''

from functools import reduce
import numpy as np
import cupy
from pyscf import gto
from gpu4pyscf.df import int3c2e
from gpu4pyscf.gto.ecp import get_ecp_ip
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array, contract, ensure_numpy
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import uhf as uhf_grad


def make_rdm1e(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None):
    '''Energy-weighted density matrix from the spin-resolved ROHF Fock.

    Semicanonical ROHF orbitals do not diagonalize focka/fockb separately,
    so the UHF eps-weighted construction does not apply; project the full
    spin Fock matrices instead (same as pyscf.grad.rohf.make_rdm1e).
    '''
    mf = mf_grad.base
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    # gpu4pyscf.scf.rohf.ROHF.get_fock tags the Roothaan Fock with the plain
    # spin Focks focka/fockb; those are what enter the Pulay term.
    fock = mf.get_fock(dm=dm)
    focka = cupy.asarray(fock.focka)
    fockb = cupy.asarray(fock.fockb)
    mocc_a = mo_coeff[:, mo_occ > 0]
    mocc_b = mo_coeff[:, mo_occ == 2]
    proj_a = mocc_a.dot(mocc_a.conj().T)
    proj_b = mocc_b.dot(mocc_b.conj().T)
    rdm1e_a = reduce(cupy.dot, (proj_a, focka, proj_a))
    rdm1e_b = reduce(cupy.dot, (proj_b, fockb, proj_b))
    return cupy.stack((rdm1e_a, rdm1e_b))


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of ROHF/ROKS gradients.

    ROHF supplies spatial orbitals (nao, nmo) with a 1D occupancy vector;
    downstream UKS-style XC/JK backends expect stacked per-spin quantities,
    so the density matrix is tagged with duplicated orbitals and the
    (occ > 0, occ == 2) split occupancies before it is handed over.
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    t0 = t1 = log.init_timer()

    mo_coeff = cupy.asarray(mo_coeff)
    mo_occ = cupy.asarray(mo_occ)
    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dma, dmb = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = cupy.stack((cupy.asarray(dma), cupy.asarray(dmb)))
    mo_coeff_u = cupy.repeat(mo_coeff[None], 2, axis=0)
    mo_occ_u = cupy.asarray([mo_occ > 0, mo_occ == 2], dtype=np.double)
    dm0 = tag_array(dm0, mo_coeff=mo_coeff_u, mo_occ=mo_occ_u)
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]

    # (\nabla i | hcore | j) - (\nabla i | j)
    h1 = cupy.asarray(mf_grad.get_hcore(mol, exclude_ecp=True))
    s1 = cupy.asarray(mf_grad.get_ovlp(mol))

    # (i | \nabla hcore | j)
    dh1e = int3c2e.get_dh1e(mol, dm0_sf)

    if len(mol._ecpbas) > 0:
        ecp_atoms = sorted(set(mol._ecpbas[:, gto.ATOM_OF]))
        h1_ecp = get_ecp_ip(mol, ecp_atoms=ecp_atoms)
        h1 -= h1_ecp.sum(axis=0)
        dh1e[ecp_atoms] += 2.0 * contract('nxij,ij->nx', h1_ecp, dm0_sf)

    if mol._pseudo:
        raise NotImplementedError('Pseudopotential gradient not supported '
                                  'for molecular system yet')

    t1 = log.timer_debug1('gradients of h1e', *t1)
    e2_grad = mf_grad.energy_ee(mol, dm0)

    extra_force = np.zeros((len(atmlst), 3))
    for k, ia in enumerate(atmlst):
        extra_force[k] += ensure_numpy(mf_grad.extra_force(ia, locals()))
    log.timer_debug1('gradients of 2e part', *t1)

    dh = rhf_grad.contract_h1e_dm(mol, h1, dm0_sf, hermi=1)
    ds = rhf_grad.contract_h1e_dm(mol, s1, dme0_sf, hermi=1)
    de = dh - ds + e2_grad
    de += ensure_numpy(dh1e)
    de += extra_force
    log.timer_debug1('gradients of electronic part', *t0)
    return de


class Gradients(uhf_grad.Gradients):

    grad_elec = grad_elec

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        return make_rdm1e(self, mo_energy, mo_coeff, mo_occ)


Grad = Gradients
