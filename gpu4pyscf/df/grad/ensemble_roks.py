# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.

'''Density-fitted helpers for the fractional-occupation EnsembleROKS reference.

The non-DF ``_fractional_rks_fock_skeleton`` uses the exact GPU J/K derivative
integrals.  For a DF reference the same skeleton ``C^T F_AO^(0,[A]) C_occ``
must be built from the DF three-centre derivative integrals, the two-centre
metric derivative and the auxiliary-centre response.  The existing
:func:`gpu4pyscf.df.hessian.rhf._get_jk_ip` provides that machinery for the
closed-shell ``dm0 = 2 mocc mocc^T`` density.

Occupation weights enter only internal density contractions.  The free
output-orbital index remains unweighted.  Both three-centre derivatives and
the inverse-metric derivative follow this convention.
'''


import cupy as cp

from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.hessian import rks as gpu_rks_hess
from gpu4pyscf.df.hessian import rhf as gpu_df_rhf_hess
from gpu4pyscf.df.hessian import rks as gpu_df_rks_hess


def fractional_rks_fock_skeleton(charge_mf, mo_coeff, mo_occ):
    '''Return ``C^T F_AO^(0,[A]) C_occ`` from DF integrals for all atoms.

    ``mo_occ`` must be the fractional Dz0 occupation vector (0/1/2).  The
    returned tensor has shape ``(natm, 3, nmo, nocc)`` with ``nocc`` the number
    of orbitals with ``mo_occ > 0``.

    '''
    mol = charge_mf.mol
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)

    hessobj = gpu_df_rks_hess.Hessian(charge_mf)
    ni = charge_mf._numint
    ni.libxc.test_deriv_order(charge_mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(charge_mf.xc, spin=mol.spin)
    with_k = ni.libxc.is_hybrid_xc(charge_mf.xc)

    vj1, vk1 = gpu_df_rhf_hess._get_jk_ip(
        hessobj, mo_coeff, mo_occ, None, None, 0,
        with_j=True, with_k=with_k,
    )
    skeleton = vj1
    if with_k:
        skeleton -= 0.5 * hyb * vk1
    if abs(omega) > 1e-10 and abs(alpha - hyb) > 1e-10:
        _, vk1_lr = gpu_df_rhf_hess._get_jk_ip(
            hessobj, mo_coeff, mo_occ, None, None, 0,
            with_j=False, with_k=True, omega=omega,
        )
        skeleton -= 0.5 * (alpha - hyb) * vk1_lr

    skeleton += rhf_grad.get_grad_hcore(
        charge_mf.nuc_grad_method(), mo_coeff, mo_occ,
    )
    max_memory = max(
        2000, charge_mf.max_memory * 0.9 - cp.get_default_memory_pool().used_bytes() / 1e6,
    )
    skeleton += gpu_rks_hess._get_vxc_deriv1(
        hessobj, mo_coeff, mo_occ, max_memory,
    )
    return skeleton


__all__ = ['fractional_rks_fock_skeleton']
