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

'''Noncollinear-Tensor TDA (NTTDA) excitation energies on GPU.

GPU port of pyscf.sftda.nttda for high-spin ROKS and average-occupation
EnsembleRKS references.  This module implements the production
``deltaS = -1`` and ``deltaS = 0`` channels.  The vref0
(spin-flip kernel) action reuses the stock gpu4pyscf ``nr_rks_fxc`` with an
injected reference kernel; the GGA/MGGA vref1 action is evaluated directly
in the MO blocks on the grid (batched GEMMs), which avoids the CPU-only
sparse-AO primitives of the reference implementation.
'''

import numpy as np
import cupy as cp
import os

from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.dft import numint as gpu_numint
from gpu4pyscf.scf import jk as jk_mod
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh


def _get_j_range_separated(mf, dms, hermi, omega):
    '''Long-range (erf-kernel) Coulomb J.

    RSH functionals only range-separate the exchange, so the stock GPU
    J engine silently ignores ``omega``; the NTTDA vref1 term needs the
    genuine long-range J.  For a density-fitted SCF the per-omega cderi
    path computes the erf-kernel J correctly, so it is used directly;
    otherwise the range-separated VHFOpt stored under
    ``mf._opt_gpu[omega]`` (the same object the omega ``get_k`` path
    uses) drives the conventional integrals.
    '''
    from gpu4pyscf.df.df_jk import _DFHF

    if isinstance(mf, _DFHF):
        return mf.get_j(mf.mol, dms, hermi, omega=omega)
    mol = mf.mol
    vhfopt = mf._opt_gpu.get(omega)
    if vhfopt is None:
        with mol.with_range_coulomb(omega):
            vhfopt = mf._opt_gpu[omega] = jk_mod._VHFOpt(
                mol, mf.direct_scf_tol, tile=1).build()
    vj, _vk = jk_mod.get_jk(mol, dms, hermi, vhfopt,
                            with_j=True, with_k=False)
    return vj


def _orbital_indices(mf):
    mo_occ = cp.asarray(mf.mo_occ).get()
    csidx = np.flatnonzero(mo_occ == 2)
    osidx = np.flatnonzero(mo_occ == 1)
    vsidx = np.flatnonzero(mo_occ == 0)
    return csidx, osidx, vsidx


def _sc_vector_slices(nclosed, nopen, nvirtual):
    co = nclosed * nopen
    cv = nclosed * nvirtual
    oo = 1
    ov = nopen * nvirtual
    p_cv = co
    p_oo = p_cv + cv
    p_ov = p_oo + oo
    p_cv0 = p_ov + ov
    return {
        'CO(1)': slice(0, p_cv),
        'CV(1)': slice(p_cv, p_oo),
        'OO(1)': slice(p_oo, p_ov),
        'OV(1)': slice(p_ov, p_cv0),
        'CV(0)': slice(p_cv0, p_cv0 + cv),
    }


def _spin_lowered_reference_vector(nocc, nvir, nopen):
    '''Normalized amplitude of ``S_- |Phi_S,S>`` in the NTTDA layout.

    Closed-shell alpha-to-beta flips vanish by Pauli exclusion.  The
    reference therefore has equal amplitudes only on the diagonal of the
    open-occupied/open-virtual block.
    '''
    if nopen < 1 or nopen > min(nocc, nvir):
        raise ValueError('nopen must fit in both NTTDA orbital dimensions')
    reference = np.zeros((nocc, nvir))
    indices = np.arange(nopen)
    reference[nocc - nopen + indices, indices] = 1.0 / np.sqrt(nopen)
    return reference.ravel()


def _select_physical_root_order(
        energies, reference_overlaps, nstates, overlap_tol, energy_tol):
    '''Identify one spin-lowered reference root and energy-sort the rest.'''
    energies = np.asarray(energies)
    overlaps = np.asarray(reference_overlaps)
    if energies.ndim != 1 or overlaps.shape != energies.shape:
        raise ValueError('root energies and reference overlaps must be 1D peers')
    if len(energies) < nstates + 1:
        raise RuntimeError(
            f'NTTDA returned {len(energies)} roots; {nstates + 1} are required'
        )
    reference = int(np.argmax(overlaps))
    if overlaps[reference] < overlap_tol:
        raise RuntimeError(
            'NTTDA reference-root overlap '
            f'{overlaps[reference]:.6f} is below {overlap_tol:.6f}'
        )
    if abs(energies[reference]) > energy_tol:
        raise RuntimeError(
            'NTTDA spin-lowered reference root has energy '
            f'{energies[reference]:.6e} Ha, exceeding {energy_tol:.3e} Ha'
        )

    physical = np.delete(np.arange(len(energies)), reference)
    physical = physical[
        np.argsort(energies[physical], kind='stable')
    ][:nstates]
    return reference, physical


def spin_flip_reference_fxc(mf):
    '''``1/2 (f_aa - f_ab - f_ba + f_bb)`` on the (sorted) grid.

    Passing the SPATIAL ROKS orbitals with the 0/1/2 occupancy reproduces
    the CPU reference exactly: ``cache_xc_kernel`` takes its restricted
    branch and, for ``spin=1``, evaluates the spin-resolved kernel at the
    spin-averaged density ``(rho/2, rho/2)`` -- this is the NTTDA
    convention, not the spin-polarized ROKS density.
    '''
    ni = mf._numint
    mo = cp.asarray(mf.mo_coeff)
    occ = cp.asarray(mf.mo_occ)
    fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc, mo, occ, 1)[2]
    return 0.5 * (
        fxc[0, :, 0] - fxc[0, :, 1] - fxc[1, :, 0] + fxc[1, :, 1]
    )


def _fxc1_gga_mo_wv(fxc, t, i):
    nvec, ngrids = t.shape[0], t.shape[-1]
    wv = cp.empty((nvec, 4, ngrids))
    t00 = t[:, 0, 0]
    if i == 0:
        wv[:, 0] = cp.einsum('ijg,xijg->xg', fxc[:4, :4], t)
        wv[:, 1:4] = fxc[0, 1:4][None] * t00[:, None]
        wv[:, 1:4] += cp.einsum('ijg,xig->xjg', fxc[1:4, 1:4], t[:, 1:4, 0])
    else:
        wv[:, 0] = fxc[i, 0][None] * t00
        wv[:, 0] += cp.einsum('jg,xjg->xg', fxc[i, 1:4], t[:, 0, 1:4])
        wv[:, 1:4] = fxc[i, 1:4][None] * t00[:, None]
    return wv


def _fxc1_mgga_mo_wv(fxc, t, i):
    nvec, ngrids = t.shape[0], t.shape[-1]
    wv = cp.empty((nvec, 4, ngrids))
    t00 = t[:, 0, 0]
    if i == 0:
        wv[:, 0] = cp.einsum('ijg,xijg->xg', fxc[:4, :4], t)
        wv[:, 1:4] = fxc[0, 1:4][None] * t00[:, None]
        wv[:, 1:4] += cp.einsum('ijg,xig->xjg', fxc[1:4, 1:4], t[:, 1:4, 0])
        wv[:, 1:4] += 0.5 * fxc[0, 4][None, None] * t[:, 0, 1:4]
        wv[:, 1:4] += 0.5 * cp.einsum('ig,xijg->xjg', fxc[1:4, 4],
                                      t[:, 1:4, 1:4])
    else:
        wv[:, 0] = fxc[i, 0][None] * t00
        wv[:, 0] += cp.einsum('jg,xjg->xg', fxc[i, 1:4], t[:, 0, 1:4])
        wv[:, 0] += 0.5 * fxc[4, 0][None] * t[:, i, 0]
        wv[:, 0] += 0.5 * cp.einsum('jg,xjg->xg', fxc[4, 1:4], t[:, i, 1:4])
        wv[:, 1:4] = fxc[i, 1:4][None] * t00[:, None]
        wv[:, 1:4] += 0.5 * fxc[i, 4][None, None] * t[:, 0, 1:4]
        wv[:, 1:4] += 0.5 * fxc[4, 1:4][None] * t[:, i, 0][:, None]
        wv[:, 1:4] += 0.25 * fxc[4, 4][None, None] * t[:, i, 1:4]
    return wv


def nr_rks_fxc1_mo(mf, mo_blocks, in_blocks, out_blocks, terms, fxc_ref):
    '''Contract the fxc1 kernel directly in selected MO spaces (GPU).

    ``in_blocks``: name -> (X (nvec, nleft, nright) cupy, left_key,
    right_key); ``out_blocks``: name -> (left_key, right_key); ``terms``:
    (input_name, output_name, coefficient).  Mirrors the CPU
    ``_nr_rks_fxc1_mo`` with the gpu4pyscf sorted-AO block loop; only
    GGA/MGGA reach this path.
    '''
    ni = mf._numint
    mol = mf.mol
    grids = mf.grids
    xctype = ni._xc_type(mf.xc)
    if xctype == 'GGA':
        fill_wv = _fxc1_gga_mo_wv
    elif xctype == 'MGGA':
        fill_wv = _fxc1_mgga_mo_wv
    else:
        raise ValueError(f'MO-grid fxc1 only supports GGA/MGGA, got {xctype}')

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao

    sorted_blocks = {
        key: opt.sort_orbitals(cp.asarray(coeff), axis=[0])
        for key, coeff in mo_blocks.items()
    }
    needed = set()
    for _x, left_key, right_key in in_blocks.values():
        needed.add(left_key)
        needed.add(right_key)
    for left_key, right_key in out_blocks.values():
        needed.add(left_key)
        needed.add(right_key)

    nvec = next(iter(in_blocks.values()))[0].shape[0]
    out = {
        name: cp.zeros((
            nvec,
            mo_blocks[left_key].shape[1],
            mo_blocks[right_key].shape[1],
        ))
        for name, (left_key, right_key) in out_blocks.items()
    }
    terms_by_input = {}
    for in_name, out_name, coef in terms:
        terms_by_input.setdefault(in_name, []).append((out_name, coef))

    p1 = 0
    for ao_mask, idx, weight, _coords in ni.block_loop(
            _sorted_mol, grids, nao, 1):
        p0, p1 = p1, p1 + weight.size
        wfxc = fxc_ref[:, :, p0:p1] * weight

        mo_cache = {}
        for key in needed:
            coeff_mask = sorted_blocks[key][idx]
            mo_cache[key] = contract('cig,ip->cpg', ao_mask[:4], coeff_mask)

        for in_name, (x, left_key, right_key) in in_blocks.items():
            input_terms = terms_by_input.get(in_name)
            if not input_terms:
                continue
            left_mo = mo_cache[left_key]
            right_mo = mo_cache[right_key]
            # t[n, i, j, g] = sum_lr L[j][l, g] X[n, l, r] R[i][r, g]
            xr = contract('nlr,irg->nilg', x, right_mo)
            t = contract('nilg,jlg->nijg', xr, left_mo)
            wv = cp.empty((nvec, 4, 4, weight.size))
            for i in range(4):
                wv[:, i] = fill_wv(wfxc, t, i)
            for out_name, coef in input_terms:
                out_left, out_right = out_blocks[out_name]
                lmo = mo_cache[out_left]
                rmo = mo_cache[out_right]
                # out[n] += coef * sum_ij L[j] diag(wv[n,i,j]) R[i]^T
                weighted = contract('nijg,jlg->nilg', wv, lmo)
                out[out_name] += coef * contract(
                    'nilg,irg->nlr', weighted, rmo,
                )
    return out


def gen_rohf_response_sfd(mf, fxc_ref=None, hermi=0, use_mo_grid_fxc1=True):
    '''Response function for ``Sf = Si - 1`` (GPU).

    Returns ``(vind, fockz)``; with ``use_mo_grid_fxc1`` the GGA/MGGA
    vref1 action is skipped here and evaluated in MO blocks by the caller.
    '''
    mol = mf.mol
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    xctype = ni._xc_type(mf.xc)
    spin = (mol.nelec[0] - mol.nelec[1]) * 0.5

    if xctype != 'HF' and fxc_ref is None:
        fxc_ref = spin_flip_reference_fxc(mf)
    skip_vref1 = use_mo_grid_fxc1 and xctype in ('GGA', 'MGGA')

    def vind(dms_co, dms_cv, dms_oo, dms_ov):
        n_co, n_cv, n_oo = len(dms_co), len(dms_cv), len(dms_oo)
        idx1 = n_co
        idx2 = n_co + n_cv
        idx3 = n_co + n_cv + n_oo

        dms0 = cp.concatenate((dms_co, dms_cv, dms_oo, dms_ov), axis=0)
        dms1 = cp.concatenate((dms_co, dms_ov), axis=0)

        if xctype != 'HF':
            vref0 = gpu_numint.nr_rks_fxc(
                ni, mol, mf.grids, mf.xc, None, dms0, 0, hermi,
                None, None, fxc_ref,
            )
            vref0 = cp.asarray(vref0)
            if skip_vref1:
                vref1 = cp.zeros_like(dms1)
            elif xctype == 'LDA':
                vref1 = cp.asarray(gpu_numint.nr_rks_fxc(
                    ni, mol, mf.grids, mf.xc, None, dms1, 0, hermi,
                    None, None, fxc_ref,
                ))
            else:
                raise NotImplementedError(
                    'AO-basis vref1 is not ported; use the MO-grid path'
                )
        else:
            vref0 = cp.zeros_like(dms0)
            vref1 = cp.zeros_like(dms1)

        if hybrid:
            vk = mf.get_k(mol, dms0, hermi) * hyb
            vj = mf.get_j(mol, dms1, hermi) * hyb
            if omega != 0:
                vk += mf.get_k(mol, dms0, hermi, omega=omega) * (alpha - hyb)
                vj += _get_j_range_separated(
                    mf, dms1, hermi, omega,
                ) * (alpha - hyb)
            vref0 -= cp.asarray(vk)
            vref1 -= cp.asarray(vj)

        vref0_co = vref0[:idx1]
        vref0_cv = vref0[idx1:idx2]
        vref0_oo = vref0[idx2:idx3]
        vref0_ov = vref0[idx3:]
        vref1_co = vref1[:n_co]
        vref1_ov = vref1[n_co:]

        s = spin
        v1ao_co = (vref0_co + vref1_co / (2 * s - 1)
                   + np.sqrt((2 * s + 1) / 2 / s) * vref0_cv)
        v1ao_co += (np.sqrt(2 * s / (2 * s - 1)) * vref0_oo
                    + 2 * s / (2 * s - 1) * vref0_ov
                    - vref1_ov / (2 * s - 1))
        v1ao_cv = (vref0_co * np.sqrt((2 * s + 1) / 2 / s) + vref0_cv
                   + np.sqrt((2 * s + 1) / (2 * s - 1)) * vref0_oo)
        v1ao_cv += np.sqrt((2 * s + 1) / 2 / s) * vref0_ov
        v1ao_oo = (np.sqrt(2 * s / (2 * s - 1)) * vref0_co
                   + np.sqrt((2 * s + 1) / (2 * s - 1)) * vref0_cv)
        v1ao_oo += vref0_oo + np.sqrt(2 * s / (2 * s - 1)) * vref0_ov
        v1ao_ov = (2 * s / (2 * s - 1) * vref0_co - vref1_co / (2 * s - 1)
                   + np.sqrt((2 * s + 1) / 2 / s) * vref0_cv)
        v1ao_ov += (np.sqrt(2 * s / (2 * s - 1)) * vref0_oo + vref0_ov
                    + vref1_ov / (2 * s - 1))
        return v1ao_co, v1ao_cv, v1ao_oo, v1ao_ov

    orbos = cp.asarray(mf.mo_coeff)[:, cp.asarray(mf.mo_occ) == 1]
    dmoo = orbos @ orbos.T
    if xctype != 'HF':
        delta = cp.asarray(gpu_numint.nr_rks_fxc(
            ni, mol, mf.grids, mf.xc, None, dmoo[None], 0, 1,
            None, None, fxc_ref,
        ))[0]
    else:
        delta = cp.zeros_like(dmoo)
    if hybrid:
        delta -= cp.asarray(mf.get_k(mol, dmoo, 1)) * hyb
        if omega != 0:
            delta -= cp.asarray(
                mf.get_k(mol, dmoo, 1, omega=omega)
            ) * (alpha - hyb)
    return vind, 0.5 * delta


def gen_rohf_response_sc(mf, fxc_ref=None, hermi=0,
                         use_mo_grid_fxc1=True):
    '''Response function for the ``Sf = Si`` NTTDA channel on GPU.'''
    mol = mf.mol
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    xctype = ni._xc_type(mf.xc)
    spin = (mol.nelec[0] - mol.nelec[1]) * 0.5

    if xctype != 'HF' and fxc_ref is None:
        fxc_ref = spin_flip_reference_fxc(mf)
    skip_vref1 = use_mo_grid_fxc1 and xctype in ('GGA', 'MGGA')

    def vind(dms_co, dms_cv, dms_ov, dms_cv0):
        n_co, n_cv, n_ov = len(dms_co), len(dms_cv), len(dms_ov)
        idx1 = n_co
        idx2 = idx1 + n_cv
        idx3 = idx2 + n_ov
        dms0 = cp.concatenate((dms_co, dms_cv, dms_ov, dms_cv0))
        dms1 = cp.concatenate((dms_co, dms_ov, dms_cv0))

        if xctype != 'HF':
            vref0 = cp.asarray(gpu_numint.nr_rks_fxc(
                ni, mol, mf.grids, mf.xc, None, dms0, 0, hermi,
                None, None, fxc_ref,
            ))
            if skip_vref1:
                vref1 = cp.zeros_like(dms1)
            elif xctype == 'LDA':
                vref1 = cp.asarray(gpu_numint.nr_rks_fxc(
                    ni, mol, mf.grids, mf.xc, None, dms1, 0, hermi,
                    None, None, fxc_ref,
                ))
            else:
                raise NotImplementedError(
                    'AO-basis vref1 is not ported; use the MO-grid path'
                )
        else:
            vref0 = cp.zeros_like(dms0)
            vref1 = cp.zeros_like(dms1)

        if hybrid:
            vk = mf.get_k(mol, dms0, hermi) * hyb
            vj = mf.get_j(mol, dms1, hermi) * hyb
            if omega != 0:
                vk += mf.get_k(
                    mol, dms0, hermi, omega=omega,
                ) * (alpha - hyb)
                vj += _get_j_range_separated(
                    mf, dms1, hermi, omega,
                ) * (alpha - hyb)
            vref0 -= cp.asarray(vk)
            vref1 -= cp.asarray(vj)

        vref0_co = vref0[:idx1]
        vref0_cv = vref0[idx1:idx2]
        vref0_ov = vref0[idx2:idx3]
        vref0_cv0 = vref0[idx3:]
        vref1_co = vref1[:idx1]
        vref1_ov = vref1[idx1:idx1 + n_ov]
        vref1_cv0 = vref1[idx1 + n_ov:]
        factor = np.sqrt((spin + 1) / (2 * spin))

        v1ao_co = (
            vref0_co - vref1_co + factor * vref0_cv + vref1_ov
            + np.sqrt(0.5) * vref0_cv0
            - np.sqrt(2.0) * vref1_cv0
        )
        v1ao_cv = (
            factor * vref0_co + vref0_cv + factor * vref0_ov
        )
        v1ao_ov = (
            vref1_co + factor * vref0_cv - vref1_ov + vref0_ov
            - np.sqrt(0.5) * vref0_cv0
            + np.sqrt(2.0) * vref1_cv0
        )
        v1ao_cv0 = (
            np.sqrt(0.5) * vref0_co - np.sqrt(2.0) * vref1_co
            - np.sqrt(0.5) * vref0_ov + np.sqrt(2.0) * vref1_ov
            + vref0_cv0 - 2.0 * vref1_cv0
        )
        return v1ao_co, v1ao_cv, v1ao_ov, v1ao_cv0

    orbos = cp.asarray(mf.mo_coeff)[:, cp.asarray(mf.mo_occ) == 1]
    dmoo = orbos @ orbos.T
    if xctype != 'HF':
        delta = cp.asarray(gpu_numint.nr_rks_fxc(
            ni, mol, mf.grids, mf.xc, None, dmoo[None], 0, 1,
            None, None, fxc_ref,
        ))[0]
    else:
        delta = cp.zeros_like(dmoo)
    if hybrid:
        delta -= cp.asarray(mf.get_k(mol, dmoo, 1)) * hyb
        if omega != 0:
            delta -= cp.asarray(
                mf.get_k(mol, dmoo, 1, omega=omega)
            ) * (alpha - hyb)
    return vind, 0.5 * delta


def gen_vind_sc(td):
    '''GPU matrix-vector action for the ``deltaS = 0`` NTTDA channel.'''
    mf = td._scf
    mo = cp.asarray(mf.mo_coeff)
    csidx, osidx, vsidx = _orbital_indices(mf)
    c = mo[:, csidx]
    o = mo[:, osidx]
    v = mo[:, vsidx]
    mo_blocks = {'c': c, 'o': o, 'v': v}
    nc, no, nv = c.shape[1], o.shape[1], v.shape[1]
    slices = _sc_vector_slices(nc, no, nv)
    spin = 0.5 * no
    if spin < 0.5:
        raise ValueError('deltaS=0 NTTDA requires at least one open orbital')

    xctype = mf._numint._xc_type(mf.xc)
    use_mo_grid_fxc1 = xctype in ('GGA', 'MGGA')
    fxc_ref = None
    if xctype != 'HF':
        fxc_ref = spin_flip_reference_fxc(mf)
    vresp, fockz = gen_rohf_response_sc(
        mf, fxc_ref=fxc_ref, hermi=0,
        use_mo_grid_fxc1=use_mo_grid_fxc1,
    )
    if getattr(mf, 'is_ensemble_rks', False):
        fock0 = cp.asarray(mf.get_fock())
    elif td.nobeta:
        density_alpha, density_beta = mf.make_rdm1()
        density0 = 0.5 * (density_alpha + density_beta)
        fock = mf.get_fock(dm=cp.stack((density0, density0)))
        fock0 = 0.5 * (cp.asarray(fock.focka) + cp.asarray(fock.fockb))
    else:
        fock = mf.get_fock()
        fock0 = 0.5 * (cp.asarray(fock.focka) + cp.asarray(fock.fockb))
    td._nttda_gpu_fxc_ref = fxc_ref
    td._nttda_gpu_fock0_fockz = (fock0, fockz)

    focka = fock0 + fockz
    fockb = fock0 - fockz
    f_coco1 = o.T @ fockb @ o
    f_coco2 = c.T @ fockb @ c
    f_cocv = o.T @ fockb @ v
    f_cvcv1 = v.T @ (fock0 - fockz / spin) @ v
    f_cvcv2 = c.T @ (fock0 + fockz / spin) @ c
    f_cocv0 = o.T @ fockb @ v
    f_cvov = o.T @ focka @ c
    f_cvcv01 = v.T @ fockz @ v
    f_cvcv02 = c.T @ fockz @ c
    f_ovov1 = v.T @ focka @ v
    f_ovov2 = o.T @ focka @ o
    f_ovcv0 = c.T @ focka @ o
    f_cv0cv01 = v.T @ fock0 @ v
    f_cv0cv02 = c.T @ fock0 @ c
    f_cooo = o.T @ fockb @ c
    f_cvoo = v.T @ fockz @ c
    f_ovoo = v.T @ focka @ o
    f_cv0oo = v.T @ fock0 @ c

    hdiag = cp.concatenate((
        (f_coco1.diagonal()[None] - f_coco2.diagonal()[:, None]).ravel(),
        (f_cvcv1.diagonal()[None] - f_cvcv2.diagonal()[:, None]).ravel(),
        cp.zeros(1),
        (f_ovov1.diagonal()[None] - f_ovov2.diagonal()[:, None]).ravel(),
        (f_cv0cv01.diagonal()[None]
         - f_cv0cv02.diagonal()[:, None]).ravel(),
    ))

    def vind(zs):
        zs = cp.asarray(zs).reshape(-1, hdiag.size)
        zco = zs[:, slices['CO(1)']].reshape(-1, nc, no)
        zcv = zs[:, slices['CV(1)']].reshape(-1, nc, nv)
        zoo = zs[:, slices['OO(1)']].reshape(-1, 1)
        zov = zs[:, slices['OV(1)']].reshape(-1, no, nv)
        zcv0 = zs[:, slices['CV(0)']].reshape(-1, nc, nv)

        def density(amplitude, right, left):
            tmp = contract('xov,pv->xpo', amplitude, right)
            return contract('xpo,qo->xpq', tmp, left)

        dco = density(zco, o, c)
        dcv = density(zcv, v, c)
        dov = density(zov, v, o)
        dcv0 = density(zcv0, v, c)
        vao_co, vao_cv, vao_ov, vao_cv0 = vresp(
            dco, dcv, dov, dcv0,
        )

        def project(potential, left, right):
            tmp = contract('xpq,qo->xpo', potential, left)
            return contract('xpo,pv->xov', tmp, right)

        vco = project(vao_co, c, o)
        vcv = project(vao_cv, c, v)
        vov = project(vao_ov, o, v)
        vcv0 = project(vao_cv0, c, v)

        if use_mo_grid_fxc1:
            in_blocks = {
                'co': (zco, 'c', 'o'),
                'ov': (zov, 'o', 'v'),
                'cv0': (zcv0, 'c', 'v'),
            }
            out_blocks = {
                'co': ('c', 'o'),
                'ov': ('o', 'v'),
                'cv0': ('c', 'v'),
            }
            root2 = np.sqrt(2.0)
            terms = (
                ('co', 'co', -1.0), ('ov', 'co', 1.0),
                ('cv0', 'co', -root2), ('co', 'ov', 1.0),
                ('ov', 'ov', -1.0), ('cv0', 'ov', root2),
                ('co', 'cv0', -root2), ('ov', 'cv0', root2),
                ('cv0', 'cv0', -2.0),
            )
            vref1 = nr_rks_fxc1_mo(
                mf, mo_blocks, in_blocks, out_blocks, terms, fxc_ref,
            )
            vco += vref1['co']
            vov += vref1['ov']
            vcv0 += vref1['cv0']

        factor = np.sqrt((spin + 1) / (2 * spin))
        vco += contract('uv,xiv->xiu', f_coco1, zco)
        vco -= contract('ji,xju->xiu', f_coco2, zco)
        vco += factor * contract('ub,xib->xiu', f_cocv, zcv)
        vco -= cp.einsum('ui,xv->xiu', f_cooo, zoo)
        vco += np.sqrt(0.5) * contract('ub,xib->xiu', f_cocv0, zcv0)

        vcv += factor * contract('av,xiv->xia', f_cocv.T, zco)
        vcv += contract('ab,xib->xia', f_cvcv1, zcv)
        vcv -= contract('ji,xja->xia', f_cvcv2, zcv)
        vcv += np.sqrt(2 * (spin + 1) / spin) * cp.einsum(
            'ai,xv->xia', f_cvoo, zoo,
        )
        vcv -= factor * contract('vi,xva->xia', f_cvov, zov)
        vcv -= np.sqrt((spin + 1) / spin) * contract(
            'ab,xib->xia', f_cvcv01, zcv0,
        )
        vcv += np.sqrt((spin + 1) / spin) * contract(
            'ji,xja->xia', f_cvcv02, zcv0,
        )

        vov -= factor * contract('ju,xja->xua', f_cvov.T, zcv)
        vov += cp.einsum('au,xv->xua', f_ovoo, zoo)
        vov += contract('ab,xub->xua', f_ovov1, zov)
        vov -= contract('vu,xva->xua', f_ovov2, zov)
        vov += np.sqrt(0.5) * contract('ju,xja->xua', f_ovcv0, zcv0)

        vcv0 += np.sqrt(0.5) * contract('av,xiv->xia', f_cocv0.T, zco)
        vcv0 -= np.sqrt((spin + 1) / spin) * contract(
            'ab,xib->xia', f_cvcv01, zcv,
        )
        vcv0 += np.sqrt((spin + 1) / spin) * contract(
            'ji,xja->xia', f_cvcv02, zcv,
        )
        vcv0 -= np.sqrt(2.0) * cp.einsum('ai,xv->xia', f_cv0oo, zoo)
        vcv0 += np.sqrt(0.5) * contract('vi,xva->xia', f_ovcv0.T, zov)
        vcv0 += contract('ab,xib->xia', f_cv0cv01, zcv0)
        vcv0 -= contract('ji,xja->xia', f_cv0cv02, zcv0)

        voo = cp.zeros(len(zs))
        voo -= contract('jv,xjv->x', f_cooo.T, zco)
        voo += np.sqrt(2 * (spin + 1) / spin) * contract(
            'jb,xjb->x', f_cvoo.T, zcv,
        )
        voo += contract('vb,xvb->x', f_ovoo.T, zov)
        voo -= np.sqrt(2.0) * contract('jb,xjb->x', f_cv0oo.T, zcv0)

        return cp.concatenate((
            vco.reshape(len(zs), -1),
            vcv.reshape(len(zs), -1),
            voo[:, None],
            vov.reshape(len(zs), -1),
            vcv0.reshape(len(zs), -1),
        ), axis=1)

    return vind, hdiag


def gen_vind_sfd(td):
    mf = td._scf
    mo_coeff = cp.asarray(mf.mo_coeff)

    csidx, osidx, vsidx = _orbital_indices(mf)
    orbcs = mo_coeff[:, csidx]
    orbos = mo_coeff[:, osidx]
    orbvs = mo_coeff[:, vsidx]
    mo_blocks = {'c': orbcs, 'o': orbos, 'v': orbvs}
    ncs, nos, nvs = len(csidx), len(osidx), len(vsidx)
    nocc = ncs + nos
    nvir = nos + nvs
    core_rows = slice(None, ncs)
    open_rows = slice(ncs, None)
    open_cols = slice(None, nos)
    virt_cols = slice(nos, None)

    s = nos * 0.5
    assert s >= 1.0, 'NTTDA for Sf=Si-1 only supports Si>=1.'

    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    use_mo_grid_fxc1 = xctype in ('GGA', 'MGGA')
    fxc_ref = None
    if xctype != 'HF':
        fxc_ref = spin_flip_reference_fxc(mf)
    vresp, fockz = gen_rohf_response_sfd(
        mf, fxc_ref=fxc_ref, hermi=0, use_mo_grid_fxc1=use_mo_grid_fxc1,
    )

    if getattr(mf, 'is_ensemble_rks', False):
        fock0 = cp.asarray(mf.get_fock())
    elif td.nobeta:
        dma, dmb = mf.make_rdm1()
        dm0 = 0.5 * (cp.asarray(dma) + cp.asarray(dmb))
        fock = mf.get_fock(dm=cp.stack((dm0, dm0)))
        fock0 = 0.5 * (cp.asarray(fock.focka) + cp.asarray(fock.fockb))
    else:
        fock = mf.get_fock()
        fock0 = 0.5 * (cp.asarray(fock.focka) + cp.asarray(fock.fockb))
    # The derivative of the same electronic frame needs both objects again.
    # Keep them device-resident so the gradient/NAC driver does not rebuild a
    # spin-GGA kernel and an open-shell Fock after the Davidson solve.
    td._nttda_gpu_fxc_ref = fxc_ref
    td._nttda_gpu_fock0_fockz = (fock0, fockz)

    fock_coco0 = orbos.T @ (fock0 - fockz) @ orbos
    fock_coco1 = orbcs.T @ (fock0 + fockz) @ orbcs
    fock_coco2 = orbcs.T @ fockz @ orbcs
    fock_cocv = orbos.T @ (fock0 - fockz) @ orbvs
    fock_cooo0 = orbos.T @ (fock0 + fockz) @ orbcs
    fock_cooo1 = orbos.T @ (fock0 - fockz) @ orbcs
    fock_cvcv0 = orbvs.T @ (fock0 - fockz) @ orbvs
    fock_cvcv1 = fock_coco1
    fock_cvcv2 = orbvs.T @ fockz @ orbvs
    fock_cvcv3 = fock_coco2
    fock_cvoo = orbvs.T @ fockz @ orbcs
    fock_cvov = fock_cooo0
    fock_oooo0 = fock_coco0
    fock_oooo1 = orbos.T @ (fock0 + fockz) @ orbos
    fock_ooov0 = fock_cocv
    fock_ooov1 = orbos.T @ (fock0 + fockz) @ orbvs
    fock_ovov0 = fock_cvcv0
    fock_ovov1 = fock_oooo1
    fock_ovov2 = fock_cvcv2

    hdiag_co = fock_coco0.diagonal()[None, :] - fock_coco1.diagonal()[:, None]
    hdiag_co = hdiag_co - fock_coco2.diagonal()[:, None] * 2 / (2 * s - 1)
    hdiag_cv = fock_cvcv0.diagonal()[None, :] - fock_cvcv1.diagonal()[:, None]
    hdiag_cv = hdiag_cv - (fock_cvcv2.diagonal()[None, :] / s
                           + fock_cvcv3.diagonal()[:, None] / s)
    hdiag_oo = fock_oooo0.diagonal()[None, :] - fock_oooo1.diagonal()[:, None]
    hdiag_ov = fock_ovov0.diagonal()[None, :] - fock_ovov1.diagonal()[:, None]
    hdiag_ov = hdiag_ov - fock_ovov2.diagonal()[None, :] * 2 / (2 * s - 1)
    hdiag = cp.concatenate((
        cp.concatenate((hdiag_co, hdiag_cv), axis=1),
        cp.concatenate((hdiag_oo, hdiag_ov), axis=1),
    ), axis=0).ravel()
    open_diag = np.diag_indices(nos)

    def vind(zs):
        zs = cp.asarray(zs).reshape(-1, nocc, nvir)
        zs_co = zs[:, core_rows, open_cols]
        zs_cv = zs[:, core_rows, virt_cols]
        zs_oo = zs[:, open_rows, open_cols]
        zs_ov = zs[:, open_rows, virt_cols]
        dms_co = contract('xov,pv->xpo', zs_co, orbos)
        dms_co = contract('xpo,qo->xpq', dms_co, orbcs)
        dms_cv = contract('xov,pv->xpo', zs_cv, orbvs)
        dms_cv = contract('xpo,qo->xpq', dms_cv, orbcs)
        dms_oo = contract('xov,pv->xpo', zs_oo, orbos)
        dms_oo = contract('xpo,qo->xpq', dms_oo, orbos)
        dms_ov = contract('xov,pv->xpo', zs_ov, orbvs)
        dms_ov = contract('xpo,qo->xpq', dms_ov, orbos)
        v1ao_co, v1ao_cv, v1ao_oo, v1ao_ov = vresp(
            dms_co, dms_cv, dms_oo, dms_ov,
        )
        v1mo_co = contract('xpq,qo->xpo', v1ao_co, orbcs)
        v1mo_co = contract('xpo,pv->xov', v1mo_co, orbos)
        v1mo_cv = contract('xpq,qo->xpo', v1ao_cv, orbcs)
        v1mo_cv = contract('xpo,pv->xov', v1mo_cv, orbvs)
        v1mo_oo = contract('xpq,qo->xpo', v1ao_oo, orbos)
        v1mo_oo = contract('xpo,pv->xov', v1mo_oo, orbos)
        v1mo_ov = contract('xpq,qo->xpo', v1ao_ov, orbos)
        v1mo_ov = contract('xpo,pv->xov', v1mo_ov, orbvs)

        if use_mo_grid_fxc1:
            denom = 2 * s - 1
            in_blocks = {
                'co': (zs_co, 'c', 'o'),
                'ov': (zs_ov, 'o', 'v'),
            }
            out_blocks = {
                'co': ('c', 'o'),
                'ov': ('o', 'v'),
            }
            terms = (
                ('co', 'co', 1.0 / denom),
                ('ov', 'co', -1.0 / denom),
                ('co', 'ov', -1.0 / denom),
                ('ov', 'ov', 1.0 / denom),
            )
            vref1_mo = nr_rks_fxc1_mo(
                mf, mo_blocks, in_blocks, out_blocks, terms, fxc_ref,
            )
            v1mo_co += vref1_mo['co']
            v1mo_ov += vref1_mo['ov']

        v1mo_co += contract('uv,xiv->xiu', fock_coco0, zs_co)
        v1mo_co -= contract('ji,xju->xiu', fock_coco1, zs_co)
        v1mo_co -= contract('ji,xju->xiu', fock_coco2, zs_co) * 2 / (2 * s - 1)
        v1mo_co += contract('ub,xib->xiu', fock_cocv, zs_cv) \
            * np.sqrt((2 * s + 1) / 2 / s)
        v1mo_co -= contract('wi,xwu->xiu', fock_cooo0, zs_oo) \
            * np.sqrt(2 * s / (2 * s - 1))
        v1mo_co += cp.einsum('ui,xvv->xiu', fock_cooo1, zs_oo) \
            / np.sqrt(2 * s * (2 * s - 1))

        v1mo_cv += contract('av,xiv->xia', fock_cocv.T, zs_co) \
            * np.sqrt((2 * s + 1) / 2 / s)
        v1mo_cv += contract('ab,xib->xia', fock_cvcv0, zs_cv)
        v1mo_cv -= contract('ji,xja->xia', fock_cvcv1, zs_cv)
        v1mo_cv -= contract('ab,xib->xia', fock_cvcv2, zs_cv) / s
        v1mo_cv -= contract('ji,xja->xia', fock_cvcv3, zs_cv) / s
        v1mo_cv -= cp.einsum('ai,xvv->xia', fock_cvoo, zs_oo) \
            / s * np.sqrt((2 * s + 1) / (2 * s - 1))
        v1mo_cv -= contract('vi,xva->xia', fock_cvov, zs_ov) \
            * np.sqrt((2 * s + 1) / 2 / s)

        v1mo_oo -= contract('ju,xjt->xut', fock_cooo0.T, zs_co) \
            * np.sqrt(2 * s / (2 * s - 1))
        v1mo_oo[:, open_diag[0], open_diag[1]] += (
            contract('jv,xjv->x', fock_cooo1.T, zs_co)
            / np.sqrt(2 * s * (2 * s - 1))
        )[:, None]
        v1mo_oo[:, open_diag[0], open_diag[1]] -= (
            contract('jb,xjb->x', fock_cvoo.T, zs_cv)
            / s * np.sqrt((2 * s + 1) / (2 * s - 1))
        )[:, None]
        v1mo_oo += contract('tv,xuv->xut', fock_oooo0, zs_oo)
        v1mo_oo -= contract('wu,xwt->xut', fock_oooo1, zs_oo)
        v1mo_oo += contract('tb,xub->xut', fock_ooov0, zs_ov) \
            * np.sqrt(2 * s / (2 * s - 1))
        v1mo_oo[:, open_diag[0], open_diag[1]] -= (
            contract('vb,xvb->x', fock_ooov1, zs_ov)
            / np.sqrt(2 * s * (2 * s - 1))
        )[:, None]

        v1mo_ov -= contract('ju,xja->xua', fock_cvov.T, zs_cv) \
            * np.sqrt((2 * s + 1) / 2 / s)
        v1mo_ov += contract('av,xuv->xua', fock_ooov0.T, zs_oo) \
            * np.sqrt(2 * s / (2 * s - 1))
        v1mo_ov -= cp.einsum('au,xvv->xua', fock_ooov1.T, zs_oo) \
            / np.sqrt(2 * s * (2 * s - 1))
        v1mo_ov += contract('ab,xub->xua', fock_ovov0, zs_ov)
        v1mo_ov -= contract('vu,xva->xua', fock_ovov1, zs_ov)
        v1mo_ov -= contract('ab,xub->xua', fock_ovov2, zs_ov) * 2 / (2 * s - 1)

        v1mo = cp.zeros_like(zs)
        v1mo[:, core_rows, open_cols] = v1mo_co
        v1mo[:, core_rows, virt_cols] = v1mo_cv
        v1mo[:, open_rows, open_cols] = v1mo_oo
        v1mo[:, open_rows, virt_cols] = v1mo_ov
        return v1mo.reshape(len(v1mo), -1)

    return vind, hdiag


class NTTDA(lib.StreamObject):
    '''GPU NTTDA excitation energies for ROKS or EnsembleRKS references.

    The production ``deltaS = -1`` and ``deltaS = 0`` channels are native;
    the spin-lowered zero-energy reference is filtered only for ``deltaS=-1``.
    '''

    deltaS = -1
    nobeta = False
    reference_overlap_tol = 0.8
    reference_energy_tol = 1e-6

    def __init__(self, mf):
        self._scf = mf
        self.mol = mf.mol
        self.verbose = mf.verbose
        self.stdout = mf.stdout
        self.max_memory = mf.max_memory
        self.nstates = 3
        self.conv_tol = 1e-6
        self.lindep = 1e-12
        self.max_cycle = 100
        self.converged = None
        self.e = None
        self.xy = None
        self.reference_root_index = None
        self.reference_root_energy = None
        self.reference_root_overlap = None
        self._nttda_solver_stats = {}

    gen_vind_sfd = gen_vind_sfd
    gen_vind_sc = gen_vind_sc

    def get_precond(self, hdiag):
        def precond(x, e, *args):
            x = cp.asarray(x)
            e = cp.asarray(e)
            if x.ndim == 1:
                diag = hdiag - e
            else:
                diag = hdiag[None, :] - e.reshape(-1, 1)
            diag = cp.where(
                cp.abs(diag) < 1e-8,
                cp.where(diag < 0, -1e-8, 1e-8),
                diag,
            )
            return x / diag
        return precond

    def init_guess(self, hdiag, nstates=None):
        if nstates is None:
            nstates = self.nstates
        n_init = min(nstates + 3, hdiag.size)
        idx = cp.argsort(hdiag)[:n_init]
        x0 = cp.zeros((n_init, hdiag.size))
        x0[cp.arange(n_init), idx] = 1.0
        return x0

    def kernel(self, x0=None, nstates=None):
        log = logger.new_logger(self)
        t0 = log.init_timer()
        if self.deltaS not in (-1, 0):
            raise NotImplementedError(
                'GPU NTTDA currently implements deltaS=-1 and deltaS=0'
            )
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        nroots = nstates + (self.deltaS == -1)

        if self.deltaS == -1:
            vind_orig, hdiag = self.gen_vind_sfd()
        else:
            vind_orig, hdiag = self.gen_vind_sc()
        precond = self.get_precond(hdiag)
        warm_start = x0 is not None
        if x0 is None:
            x0 = self.init_guess(hdiag, nstates)
        else:
            x0 = cp.asarray(x0).reshape(-1, hdiag.size)
            # Keep diagonal guesses alongside cross-geometry warm starts.
            # The latter contain only the physical roots because the
            # spin-lowered zero-energy reference is filtered after each
            # frame; diagonal guesses ensure that reference root and any new
            # crossing root remain discoverable.
            x0 = cp.concatenate((x0, self.init_guess(hdiag, nstates)), axis=0)
        initial_subspace_width = int(x0.shape[0])

        # --- Low-overhead solver profiler (Task 1) ---
        # `counted_vind` wraps the real vind to record batch widths without
        # any GPU synchronization.  `vind_widths` is a plain Python list of
        # ints (one per Davidson iteration), so the overhead is negligible.
        vind_widths = []
        def counted_vind(zs):
            vind_widths.append(int(zs.shape[0]))
            return vind_orig(zs)

        # NTTDA_PROFILE=1 enables per-iteration Davidson callback for full
        # residual/energy recording.  Default (unset) avoids the cp.asnumpy
        # sync inside the callback, keeping timing clean.
        profile_mode = os.environ.get('NTTDA_PROFILE', '0') != '0'
        davidson_iterations = []
        callback = None
        if profile_mode:
            def on_iter(info):
                davidson_iterations.append(info)
            callback = on_iter

        def all_eigs(w, v, nroots, envs):
            return w, v, np.arange(w.size)

        converged, energies, x1 = lr_eigh(
            counted_vind, x0, precond,
            tol_residual=self.conv_tol,
            # ``lr_eigh`` compares the squared norm of a preconditioned trial
            # vector against ``lindep``.  Keep a small margin below the target
            # residual squared so the last useful correction is not discarded
            # before a tightly converged root reaches ``conv_tol``.
            lindep=min(self.lindep, 1e-2 * self.conv_tol ** 2),
            nroots=nroots,
            pick=all_eigs,
            max_cycle=self.max_cycle,
            max_memory=self.max_memory,
            verbose=log,
            callback=callback,
        )
        energies = np.asarray(cp.asnumpy(cp.asarray(energies)))
        converged = np.atleast_1d(
            cp.asnumpy(cp.asarray(converged)),
        ).astype(bool, copy=False)
        if len(converged) != len(energies):
            raise RuntimeError(
                'Davidson convergence flags are inconsistent with roots'
            )
        if len(x1) != len(energies):
            raise RuntimeError(
                'Davidson eigenvectors are inconsistent with roots'
            )

        if self.deltaS == -1:
            csidx, osidx, vsidx = _orbital_indices(self._scf)
            nocc = len(csidx) + len(osidx)
            nvir = len(osidx) + len(vsidx)
            reference = cp.asarray(
                _spin_lowered_reference_vector(
                    nocc, nvir, len(osidx),
                )
            )
            vectors = cp.stack([cp.asarray(xi).ravel() for xi in x1])
            vector_norms = cp.linalg.norm(vectors, axis=1)
            if bool(cp.any(vector_norms <= np.finfo(float).eps)):
                raise RuntimeError(
                    'NTTDA Davidson returned a zero-norm eigenvector'
                )
            reference_overlaps = cp.asnumpy(
                cp.abs(vectors.conj() @ reference) / vector_norms
            )
            reference_index, physical_order = _select_physical_root_order(
                energies,
                reference_overlaps,
                nstates,
                overlap_tol=self.reference_overlap_tol,
                energy_tol=max(
                    self.reference_energy_tol, 100.0 * self.conv_tol,
                ),
            )

            self.reference_root_index = reference_index
            self.reference_root_energy = float(energies[reference_index])
            self.reference_root_overlap = float(
                reference_overlaps[reference_index]
            )
            self.e = energies[physical_order]
            self.xy = [
                (cp.asarray(x1[index]).reshape(nocc, nvir), 0)
                for index in physical_order
            ]
            self.converged = converged[physical_order]
        else:
            self.reference_root_index = None
            self.reference_root_energy = None
            self.reference_root_overlap = None
            self.e = energies[:nstates]
            self.xy = [
                (cp.asarray(x1[index]).ravel(), 0)
                for index in range(min(nstates, len(x1)))
            ]
            self.converged = converged[:nstates]
        self.nstates = len(self.e)
        # Solver stats consumed by compute_frame's record_stats and the
        # FSSH profile script.  Fields:
        #   vind_calls              — number of Davidson iterations
        #   vind_widths             — batch width per iteration (Python list)
        #   total_vector_applications — sum(vind_widths)
        #   davidson_iterations     — per-iteration info (only if NTTDA_PROFILE=1)
        #   initial_subspace_width  — x0 width after warm-start concatenation
        #   warm_start              — whether x0 was provided (cross-frame)
        #   final_residuals        — last-iteration residual norms (profile only)
        self._nttda_solver_stats = {
            'vind_calls': len(vind_widths),
            'vind_widths': vind_widths,
            'total_vector_applications': sum(vind_widths),
            'davidson_iterations': len(davidson_iterations),
            'initial_subspace_width': initial_subspace_width,
            'warm_start': warm_start,
            'nroots': int(nroots),
            'final_residuals': (
                davidson_iterations[-1]['residuals'].tolist()
                if davidson_iterations else None
            ),
            'converged': self.converged.tolist(),
        }
        log.timer('GPU NTTDA', *t0)
        return self.e, self.xy

    def run(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.kernel()
        return self

    def nuc_grad_method(self):
        from gpu4pyscf.grad.nttda import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method

    def nac_method(self):
        from gpu4pyscf.nac.nttda import NAC
        return NAC(self)

    NAC = nac_method
