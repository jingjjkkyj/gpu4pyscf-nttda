# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import ctypes
import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, ndarray, transpose_sum, get_avail_mem)
from gpu4pyscf.df.grad.rhf import (
    _split_l_ctr_pattern, get_ao_pair_loc, libvhf_rys, Int3c2eOpt, int2c2e,
    int3c2e_scheme, _gen_metric_solver, _factorize_dm)
from gpu4pyscf.df import df
from gpu4pyscf.tdscf import rhf as tdrhf
from gpu4pyscf.grad import tdrhf as tdrhf_grad

__all__ = ['Gradients']

from gpu4pyscf.grad.nttda_params import PARAMS as _NTTDA_PARAMS

DM_BLOCK = _NTTDA_PARAMS['dm_block']

def _jk_energy_per_atom(int3c2e_opt, dms, j_factor=None, k_factor=None, hermi=0,
                        verbose=None):
    '''
    Computes the first-order derivatives of J/K contributions from multiple
    density matrices and adds up the results.
    '''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    if k_factor is None:
        return _j_energy_per_atom(int3c2e_opt, dms, j_factor, hermi, verbose)

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm_factor_l, dm_factor_r = _factorize_dm(mol, dms, hermi)
    n_dm, nao, nocc = dm_factor_l.shape
    assert len(k_factor) == n_dm
    # TODO: if nocc is large, memory might not be enough to store a tensor of
    # shape (n_dm, naux, nocc, nocc). Split dms into several sub tensors and
    # process separately.
    log.debug1('dm_factor shape %s', dm_factor_l.shape)

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    naux = auxmol.nao

    mem_free = get_avail_mem(exclude_memory_pool=True)
    mem_avail = mem_free - n_dm*naux*nocc**2*8 - n_dm*nao**2*8
    _mem_frac = _NTTDA_PARAMS['df_mem_fraction']
    _batch_factor = _NTTDA_PARAMS['df_batch_factor']
    _blk_factor = _NTTDA_PARAMS['df_blk_factor']
    batch_size = max(1, min(naux, int(mem_avail*_mem_frac/(nao_pair*8)*_batch_factor)))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, int(mem_avail*.4/(nao*nao*2*8))//8*8 * int(_blk_factor)))
    log.debug('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
              mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    buf1 = cp.empty((blksize, nocc, nao))
    j3c_oo = [cp.empty((naux, nocc, nocc)) for i in range(n_dm)]
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
            tmp = ndarray((nocc, nao, dk), buffer=buf1)
            for i in range(n_dm):
                contract('pqr,pi->iqr', j3c, dm_factor_r[i], out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor_l[i], out=j3c_oo[i][aux0:aux1])
    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
    t0 = log.timer_debug1('contract dm', *t0)

    aux_coeff = cp.asarray(auxmol.ctr_coeff)
    aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
    aux_coeff[aux_sorting] = tmp
    tmp = None

    j2c = int2c2e(auxmol)
    if mol.omega <= 0 and not auxmol.mol.cart:
        metric = aux_coeff.dot(cp.linalg.solve(j2c, aux_coeff.T))
    else:
        metric = aux_coeff.dot(_gen_metric_solver(j2c, 'ED')(aux_coeff.T))
    j2c = aux_coeff = None
    dm_oo = []
    buf = None
    for i in range(n_dm):
        dm_oo.append(contract('uv,vij->uij', metric, j3c_oo[i], out=buf))
        buf = j3c_oo[i]
    metric = j3c_oo = buf = None
    if j_factor is not None:
        auxvec = cp.empty((n_dm, naux))
        for i in range(n_dm):
            dm_oo[i].trace(axis1=1, axis2=2, out=auxvec[i])
        auxvec_jfac = cp.asarray(j_factor)[:,None] * auxvec

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(mol.omega, 54)
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*4)
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache)
    aux_loc = auxmol.ao_loc

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    l_ctr_aux_counts = l_ctr_aux_offsets[1:] - l_ctr_aux_offsets[:-1]

    if j_factor is not None:
        dms = contract('npi,nqi->npq', dm_factor_l, dm_factor_r)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.sum_ejk_int3c2e_ip1
    l = np.arange(laux+1)
    nf = (l + 1) * (l + 2) // 2
    aux0 = aux1 = 0
    buf = cp.empty((nao_pair*batch_size))
    buf1 = cp.empty((blksize, nao, nao))
    buf2 = cp.empty((blksize, nao, nao))
    ejk = cp.zeros((mol.natm, 3))
    ejk_aux = cp.zeros((mol.natm, 3))
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * l_ctr_aux_counts[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        compressed = ndarray((nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao,nao,dk), buffer=buf1)
            tmp = ndarray((nocc,nao,dk), buffer=buf2)
            if j_factor is None:
                dm_tensor[:] = 0
            else:
                contract('npq,nr->pqr', dms, auxvec_jfac[:,aux0:aux1], out=dm_tensor)
            for i in range(n_dm):
                contract('rji,qj->iqr', dm_oo[i][aux0:aux1], dm_factor_l[i], out=tmp)
                contract('iqr,pi->pqr', tmp, dm_factor_r[i], -.5*k_factor[i], 1, out=dm_tensor)
            if hermi == 1:
                cp.take(dm_tensor.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
            else:
                dm_tensor1 = ndarray((nao,nao,dk), buffer=buf2)
                dm_tensor1[:] = dm_tensor.transpose(1,0,2)
                dm_tensor1[:] += dm_tensor
                cp.take(dm_tensor1.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
            ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.c_int(1),
            ctypes.byref(int3c2e_envs),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(nao), ctypes.c_int(nao_pair),
            ctypes.c_int(naux_in_batch), ctypes.c_int(mol.natm))
        if err != 0:
            raise RuntimeError('int3c2e_ejk_ip1 failed')
    buf = buf1 = buf2 = compressed = dm_tensor = dm_tensor1 = tmp = None
    if hermi == 1:
        ejk *= 2
        ejk_aux *= 2
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    if j_factor is None:
        dm_aux = cp.zeros((naux,naux))
    else:
        dm_aux = auxvec.T.dot(auxvec_jfac)
    for i in range(n_dm):
        contract('rij,sji->rs', dm_oo[i], dm_oo[i], -.5*k_factor[i], 1, out=dm_aux)
    dm_aux = dm_aux[aux_sorting[:,None], aux_sorting]
    ejk_aux -= cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    dm_aux = None

    ejk += ejk_aux
    ejk = ejk.get()
    return ejk

def _j_energy_per_atom(int3c2e_opt, dms, j_factor, hermi=0, verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    if dms.ndim == 2:
        dms = dms[None]
    dms = mol.apply_C_mat_CT(dms)
    if hermi != 1:
        dms = transpose_sum(dms, inplace=True)
        dms[:] *= .5
    auxvec = int3c2e_opt.contract_dm(dms, hermi=1)
    auxvec = auxmol.apply_CT_dot(auxvec, axis=1)
    t0 = log.timer_debug1('contract dm', *t0)
    j2c = int2c2e(auxmol)

    n_dm = len(dms)
    assert len(j_factor) == n_dm
    if mol.omega <= 0 and not auxmol.mol.cart:
        auxvec = cp.linalg.solve(j2c, auxvec.T).T
    else:
        auxvec = _gen_metric_solver(j2c, 'ED')(auxvec.T).T
    auxvec = cp.asarray(auxmol.apply_C_dot(auxvec, axis=1), order='C')
    auxvec_jfac = auxvec * cp.asarray(j_factor)[:,None]
    naux = auxvec.shape[1]
    j2c = None

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(mol.omega, 54)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*16)
    ksh_offsets_cpu = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    kern = libvhf_rys.sum_ejk_int3c2e_ip1
    ej = cp.zeros((mol.natm, 3))
    ej_aux = cp.zeros_like(ej)

    err = kern(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p),
        ctypes.cast(ej_aux.data.ptr, ctypes.c_void_p),
        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec_jfac.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n_dm),
        ctypes.byref(int3c2e_envs),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(shl_pair_offsets) - 1),
        ctypes.c_int(len(ksh_offsets_cpu) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        lib.c_null_ptr(),
        ctypes.c_int(0),
        ctypes.c_int(0), ctypes.c_int(0),
        ctypes.c_int(naux), ctypes.c_int(mol.natm))
    if err != 0:
        raise RuntimeError('int3c2e_ejk_ip1 failed')
    ej *= 2
    ej_aux *= 2
    ej = ej.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    #ej_aux += .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
    dm_aux = auxvec.T.dot(auxvec_jfac)
    ej_aux -= cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
    ej += ej_aux.get()
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

def _jk_energies_per_atom(int3c2e_opt, dm_pairs, j_factor=None, k_factor=None,
                          sum_results=False, verbose=None, stats_sink=None,
                          output_group_indices=None, output_group_count=None):
    '''
    Computes a set of first-order derivatives of J/K contributions for each
    element (density matrix or a pair of density matrices) in dm_pairs.
    This method is similar _jk_energy_per_atom, but instead of summing all
    contributions into a single result, it returns derivatives for each
    individual set.

    This function supports evaluating multiple sets of energy derivatives in a
    single call. Additionally, for each set, the two density matrices for the
    four-index Coulomb integrals can be different.

    Args:
        dm_pairs:
            A list of density-matrix-pairs [[dm, dm], [dm, dm], ...].
            Each element corresponds to one set of energy derivative.
        j_factor:
            A list of factors for Coulomb (J) term
        k_factor:
            A list of factors for Coulomb (K) term
        hermi:
            No effects
        sum_results : bool
            If True, aggregate all sets of derivatives into a single result.
        stats_sink : list or None
            If not None, per-split layer timing dicts are appended here.
        output_group_indices : sequence of int or None
            Optional output group for each input task. Already weighted
            compressed tensors with the same group index are summed before
            the derivative CUDA kernel.
        output_group_count : int or None
            Total number of output groups. Required with
            ``output_group_indices`` when groups may be absent from a split.

    Returns:
        An numpy ndarray of shape (*, Natm, 3)
    '''
    n_dm = len(dm_pairs)
    assert j_factor is None or len(j_factor) == n_dm
    assert k_factor is None or len(k_factor) == n_dm
    if output_group_indices is not None:
        if sum_results:
            raise ValueError(
                'output_group_indices and sum_results are mutually exclusive'
            )
        output_group_indices = np.asarray(
            output_group_indices, dtype=np.int32,
        )
        if output_group_indices.shape != (n_dm,):
            raise ValueError('one output group index is required per DM pair')
        if output_group_count is None:
            output_group_count = (
                int(output_group_indices.max()) + 1 if n_dm else 0
            )
        if (
                output_group_count < 1
                or np.any(output_group_indices < 0)
                or np.any(output_group_indices >= output_group_count)):
            raise ValueError('invalid output group indices/count')
    if k_factor is None or all(x == 0 for x in k_factor):
        result = _j_energies_per_atom(
            int3c2e_opt, dm_pairs, j_factor, sum_results, verbose,
        )
        if output_group_indices is None:
            return result
        grouped = np.zeros(
            (output_group_count,) + result.shape[1:], dtype=result.dtype,
        )
        np.add.at(grouped, output_group_indices, result)
        return grouped

    mol = int3c2e_opt.mol
    factor_cache = {}
    dm_factors = [
        _factorize_multiple_dm(
            mol, dm1_dm2, hermi=0, factor_cache=factor_cache,
        )
        for dm1_dm2 in dm_pairs
    ]

    splits = [0, n_dm]
    if n_dm > 2:
        mem_avail = get_avail_mem(exclude_memory_pool=True) * .6
        dm1_noccs = [x[0].shape[1] for x in dm_factors]
        dm2_noccs = [x[2].shape[1] for x in dm_factors]
        nao = int3c2e_opt.mol.nao
        naux = int3c2e_opt.auxmol.nao
        splits = [0]
        cost = 0
        for i, (x, y) in enumerate(zip(dm1_noccs, dm2_noccs)):
            if cost > mem_avail:
                splits.append(i)
                cost = 0
            elif i > 0 and (i - splits[-1]) % DM_BLOCK == 0:
                batches = (i - splits[-1]) // DM_BLOCK
                if cost/batches*(batches+1.2) > mem_avail:
                    # memory is not sufficient to include the next entire batch
                    splits.append(i)
                    cost = 0
            cost += (2*x*y*naux + 2*nao**2) * 8
        splits.append(n_dm)
        if len(splits) > 2:
            logger.debug(mol, 'Partition %d DMs into %d tasks', n_dm, len(splits)-1)

    out = []
    grouped_out = None
    j_factor_batch = None
    for p0, p1 in zip(splits[:-1], splits[1:]):
        if j_factor is not None:
            j_factor_batch = j_factor[p0:p1]
        result = _jk_energies_by_dm_factors(
            int3c2e_opt, dm_factors[p0:p1], j_factor_batch, k_factor[p0:p1],
            sum_results, verbose, stats_sink=stats_sink,
            output_group_indices=(
                None if output_group_indices is None
                else output_group_indices[p0:p1]
            ),
            output_group_count=output_group_count,
        )
        if output_group_indices is None:
            out.append(result)
        elif grouped_out is None:
            grouped_out = result
        else:
            grouped_out += result
    if output_group_indices is not None:
        return grouped_out
    if sum_results:
        return sum(out)
    else:
        return np.vstack(out)


def _factor_contraction_groups(factors):
    """Group exact CuPy factor views for shared three-center contractions."""
    groups = {}
    for index, factor in enumerate(factors):
        key = (
            int(factor.data.ptr),
            factor.shape,
            factor.strides,
            factor.dtype.str,
        )
        groups.setdefault(key, []).append(index)
    return tuple(tuple(indices) for indices in groups.values())


def _slot_rank_batched_plan(dm_factors, j_factor, k_factor,
                            output_group_indices):
    """Group pure-K rank-(2,R)/(R,2) tasks by output slot and ranks."""
    candidates = {}
    for index, (factors, kfac, output_group) in enumerate(zip(
            dm_factors, k_factor, output_group_indices)):
        rank1 = factors[0].shape[1]
        rank2 = factors[2].shape[1]
        if (
                kfac != 0
                and (j_factor is None or j_factor[index] == 0)
                and (rank1 == 2 or rank2 == 2)):
            key = (int(output_group), int(rank1), int(rank2))
            candidates.setdefault(key, []).append(index)

    groups = []
    mask = np.zeros(len(dm_factors), dtype=bool)
    for (output_group, rank1, rank2), indices in candidates.items():
        if len(indices) < 2:
            continue
        indices = tuple(indices)
        groups.append((output_group, rank1, rank2, indices))
        mask[list(indices)] = True
    return tuple(groups), mask


def _jk_energies_by_dm_factors(int3c2e_opt, dm_factors, j_factor, k_factor,
                               sum_results, verbose, stats_sink=None,
                               output_group_indices=None,
                               output_group_count=None):
    '''Batched per-atom DF J/K derivative kernel.

    Evaluates first-derivative J/K contributions for ``n_dm`` density-matrix
    factor pairs in a single call.  The kernel has four GPU stages:

    1. **contract_dm** — three-center integral evaluation + factor contraction
       (builds ``j3c_o1o2`` / ``j3c_o2o1`` and J auxiliary vectors).
    2. **metric_transform** — 2c2e metric solve + j3c metric back-transform.
    3. **int2c2e_ip1** — pseudo-density formation + int2c2e_ip1 per atom.
    4. **ejk_kernel** — Python loop that builds the ``compressed`` tensor
       (dm_tensor = J part + K part, symmetrized, pair-addressed) followed by
       the CUDA ``ejk_int3c2e_ip1`` derivative kernel call.

    When ``stats_sink`` is not None, per-split layer timing dicts are appended
    for profiling.  Each dict contains keys ``contract_dm``, ``metric_transform``,
    ``int2c2e_ip1``, ``ejk_kernel`` (wall seconds), ``compressed_build``,
    ``kern_call`` (sub-timing of ``ejk_kernel``), and ``n_dm``/``nao``/``naux``.

    The ``compressed_build`` loop skips K contractions when ``k_factor[i] == 0``
    (J-only DMs), avoiding ~4 wasted GPU contractions per J pair.  This was
    validated to reduce DF integral time by ~10% on azobenzene (Task 5).

    Memory fractions and batch multipliers are read from
    :mod:`gpu4pyscf.grad.nttda_params` (env-var driven, defaults calibrated
    on RTX 4060 8 GB; A100 sweep pending).
    '''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    import time as _time
    n_dm = len(dm_factors)
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    _layer_t = {}  # per-split layer timing dict, appended to stats_sink
    compressed_backend = 'legacy'
    if output_group_indices is not None:
        if sum_results:
            raise ValueError(
                'output_group_indices and sum_results are mutually exclusive'
            )
        output_group_indices = np.asarray(
            output_group_indices, dtype=np.int32,
        )
        if output_group_indices.shape != (n_dm,):
            raise ValueError('one output group index is required per DM pair')
        if output_group_count is None:
            output_group_count = (
                int(output_group_indices.max()) + 1 if n_dm else 0
            )
        if (
                output_group_count < 1
                or np.any(output_group_indices < 0)
                or np.any(output_group_indices >= output_group_count)):
            raise ValueError('invalid output group indices/count')
    slot_aware = output_group_indices is not None
    effective_compressed_backend = (
        'slot_grouped' if slot_aware else compressed_backend
    )
    compressed_profile = _NTTDA_PARAMS['df_compressed_profile']
    rank_bucket_events = {}
    derivative_kernel_events = []
    slot_rank_2xr_groups = ()
    slot_rank_2xr_mask = np.zeros(n_dm, dtype=bool)
    if slot_aware:
        slot_rank_2xr_groups, slot_rank_2xr_mask = _slot_rank_batched_plan(
            dm_factors, j_factor, k_factor, output_group_indices)
    _layer_t['compressed_backend'] = compressed_backend
    _layer_t['effective_compressed_backend'] = (
        effective_compressed_backend
    )
    _layer_t['output_backend'] = 'slot_aware' if slot_aware else 'legacy'
    _layer_t['input_dm'] = n_dm
    _layer_t['kernel_dm'] = (
        int(output_group_count) if slot_aware else n_dm
    )
    _layer_t['compressed_profile'] = compressed_profile
    _layer_t['legacy_dm'] = n_dm
    slot_rank_2x2_groups = [
        group for group in slot_rank_2xr_groups
        if group[1] == 2 and group[2] == 2
    ]
    slot_rank_2x2_tasks = sum(len(group[3]) for group in slot_rank_2x2_groups)
    slot_rank_2xr_tasks = int(slot_rank_2xr_mask.sum())
    _layer_t['slot_rank_2x2_groups'] = len(slot_rank_2x2_groups)
    _layer_t['slot_rank_2x2_tasks'] = slot_rank_2x2_tasks
    _layer_t['slot_rank_2x2_tiles'] = 0
    _layer_t['slot_rank_2xr_groups'] = len(slot_rank_2xr_groups)
    _layer_t['slot_rank_2xr_tasks'] = slot_rank_2xr_tasks
    _layer_t['slot_rank_2xr_mixed_groups'] = (
        len(slot_rank_2xr_groups) - len(slot_rank_2x2_groups)
    )
    _layer_t['slot_rank_2xr_mixed_tasks'] = (
        slot_rank_2xr_tasks - slot_rank_2x2_tasks
    )
    _layer_t['slot_rank_2xr_tiles'] = 0
    _layer_t['slot_rank_2xr_rank_buckets'] = [
        {
            'output_group': output_group,
            'rank_left': rank1,
            'rank_right': rank2,
            'task_count': len(indices),
        }
        for output_group, rank1, rank2, indices in slot_rank_2xr_groups
    ]
    if slot_rank_2xr_groups:
        _layer_t['legacy_dm'] = n_dm - slot_rank_2xr_tasks

    dm1_factor_l, dm1_factor_r, dm2_factor_l, dm2_factor_r = zip(*dm_factors)
    dm1_noccs = [x.shape[1] for x in dm1_factor_l]
    dm2_noccs = [x.shape[1] for x in dm2_factor_l]
    dm2_right_groups = _factor_contraction_groups(dm2_factor_r)
    dm1_right_groups = _factor_contraction_groups(dm1_factor_r)
    nao = mol.nao
    nocc_max = max(max(dm1_noccs), max(dm2_noccs))
    _layer_t['dm1_noccs'] = list(dm1_noccs)
    _layer_t['dm2_noccs'] = list(dm2_noccs)
    _layer_t['dm1_ranks'] = [x.shape[-1] for x in dm1_factor_l]
    _layer_t['dm2_ranks'] = [x.shape[-1] for x in dm2_factor_l]
    _layer_t['contract_first_raw'] = 2 * n_dm
    _layer_t['contract_first_unique'] = (
        len(dm2_right_groups) + len(dm1_right_groups)
    )
    _layer_t['contract_first_reused'] = (
        2 * n_dm - _layer_t['contract_first_unique']
    )
    log.debug1('nao=%d dm1_noccs=%s dm2_noccs=%s', nao, dm1_noccs, dm2_noccs)

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    naux = auxmol.nao

    if j_factor is not None:
        dm1 = cp.empty((n_dm, nao, nao))
        dm2 = cp.empty((n_dm, nao, nao))
        for i in range(n_dm):
            dm1_factor_l[i].dot(dm1_factor_r[i].T, out=dm1[i])
            dm2_factor_l[i].dot(dm2_factor_r[i].T, out=dm2[i])
        auxvec1 = cp.empty((n_dm, naux))
        auxvec2 = cp.empty((n_dm, naux))

    mem_free = get_avail_mem(exclude_memory_pool=True)
    mem_avail = mem_free - 2*naux*np.dot(dm1_noccs, dm2_noccs)*8 - 2*n_dm*nao**2*8
    _mem_frac = _NTTDA_PARAMS['df_mem_fraction']
    _batch_factor = _NTTDA_PARAMS['df_batch_factor']
    _blk_factor = _NTTDA_PARAMS['df_blk_factor']
    batch_size = int(mem_avail * _mem_frac / (n_dm*nao_pair*8) * _batch_factor)
    laux = auxmol.uniq_l_ctr[:,0].max()
    if batch_size <= (laux+1)*(laux+2)//2:
        raise RuntimeError('Insufficient memory for storing intermediates')
    batch_size = min(naux, batch_size)
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, int(mem_avail*.45/(nao*nao*2*8))//8*8 * int(_blk_factor)))
    _layer_t['batch_size'] = int(batch_size)
    _layer_t['blksize'] = int(blksize)
    _layer_t['mem_free_gib'] = round(mem_free * 1e-9, 3)
    _layer_t['mem_est_gib'] = round(
        (2 * naux * np.dot(dm1_noccs, dm2_noccs) * 8 + 2 * n_dm * nao**2 * 8) * 1e-9, 3)
    log.debug('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
              mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    aux0 = aux1 = 0
    j3c_full = cp.zeros((nao, nao, blksize))
    buf = cp.empty((batch_size, nao_pair))
    buf1 = cp.empty((blksize, nocc_max, nao))
    j3c_o2o1 = [cp.empty((naux, n2, n1)) for n1, n2 in zip(dm1_noccs, dm2_noccs)]
    j3c_o1o2 = [cp.empty((naux, n1, n2)) for n1, n2 in zip(dm1_noccs, dm2_noccs)]
    _t_contract_dm = _time.perf_counter()
    for kbatch in range(aux_batches):
        compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
        naux_in_batch = compressed.shape[1]
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
            for group in dm2_right_groups:
                first = group[0]
                nocc = dm2_noccs[first]
                tmp = ndarray((nocc, nao, dk), buffer=buf1)
                contract('pqr,pi->iqr', j3c, dm2_factor_r[first], out=tmp)
                for i in group:
                    contract('iqr,qj->rij', tmp, dm1_factor_l[i],
                             out=j3c_o2o1[i][aux0:aux1])
            for group in dm1_right_groups:
                first = group[0]
                nocc = dm1_noccs[first]
                tmp = ndarray((nocc, nao, dk), buffer=buf1)
                contract('pqr,pi->iqr', j3c, dm1_factor_r[first], out=tmp)
                for i in group:
                    contract('iqr,qj->rij', tmp, dm2_factor_l[i],
                             out=j3c_o1o2[i][aux0:aux1])
            if j_factor is not None:
                auxvec1[:,aux0:aux1] = cp.einsum('pqr,nqp->nr', j3c, dm1)
                auxvec2[:,aux0:aux1] = cp.einsum('pqr,nqp->nr', j3c, dm2)
    j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
    _layer_t['contract_dm'] = _time.perf_counter() - _t_contract_dm
    t0 = log.timer_debug1('contract dm', *t0)

    _t_metric = _time.perf_counter()
    aux_coeff = cp.asarray(auxmol.ctr_coeff)
    aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
    aux_coeff[aux_sorting] = tmp
    tmp = None

    j2c = int2c2e(auxmol)
    if mol.omega <= 0 and not auxmol.mol.cart:
        metric = aux_coeff.dot(cp.linalg.solve(j2c, aux_coeff.T))
    else:
        metric = aux_coeff.dot(_gen_metric_solver(j2c, 'ED')(aux_coeff.T))
    j2c = aux_coeff = None
    for i in range(n_dm):
        j3c_o2o1[i] = contract('uv,vij->uij', metric, j3c_o2o1[i])
        j3c_o1o2[i] = contract('uv,vij->uij', metric, j3c_o1o2[i])
    cp.get_default_memory_pool().free_all_blocks()

    if j_factor is not None:
        j_factor = cp.asarray(j_factor)
        auxvec1 = cp.einsum('uv,nv->nu', metric, auxvec1)
        auxvec2 = cp.einsum('uv,nv->nu', metric, auxvec2)
        auxvec1_jfac = j_factor[:,None] * auxvec1
        auxvec2_jfac = j_factor[:,None] * auxvec2
    metric = None
    _layer_t['metric_transform'] = _time.perf_counter() - _t_metric

    # (d/dX P|Q) contributions
    _t_int2c2e = _time.perf_counter()
    if sum_results:
        dm_aux = cp.zeros((naux, naux))
        buf = cp.empty_like(dm_aux)
        if j_factor is not None:
            for i in range(n_dm):
                dm_aux += cp.multiply(auxvec1[i,:,None], auxvec2_jfac[i], out=buf)
        for i in range(n_dm):
            contract('rij,sji->rs', j3c_o1o2[i], j3c_o2o1[i], -.5*k_factor[i],
                     1, out=dm_aux)
        # needs to scale by *.5, applied at the end of this function
        dm_aux = transpose_sum(dm_aux, inplace=True)
        dm_aux = dm_aux[aux_sorting[:,None], aux_sorting]
        ejk_aux = -cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
        int2c2e_kernel_dm = 1
    elif slot_aware:
        dm_aux = cp.empty((naux, naux))
        grouped_dm_aux = cp.empty_like(dm_aux)
        ejk_aux = []
        int2c2e_kernel_dm = 0
        for group in range(output_group_count):
            members = np.flatnonzero(output_group_indices == group)
            if not len(members):
                ejk_aux.append(np.zeros((mol.natm, 3)))
                continue
            grouped_dm_aux.fill(0)
            for i in members:
                if j_factor is None:
                    beta = 0
                else:
                    cp.multiply(
                        auxvec1[i,:,None], auxvec2_jfac[i], out=dm_aux,
                    )
                    beta = 1
                contract(
                    'rij,sji->rs',
                    j3c_o1o2[i], j3c_o2o1[i], -.5*k_factor[i],
                    beta, out=dm_aux,
                )
                grouped_dm_aux += dm_aux
            # Symmetrization, auxiliary reordering, and the derivative
            # contraction are linear, so one call per output slot is exact.
            transpose_sum(grouped_dm_aux, inplace=True)
            sorted_dm_aux = grouped_dm_aux[
                aux_sorting[:,None], aux_sorting
            ]
            ejk_aux.append(
                -int2c2e_ip1_per_atom(auxmol, sorted_dm_aux)
            )
            int2c2e_kernel_dm += 1
        ejk_aux = cp.asarray(np.stack(ejk_aux))
        grouped_dm_aux = sorted_dm_aux = None
    else:
        dm_aux = cp.empty((naux, naux))
        ejk_aux = []
        for i in range(n_dm):
            if j_factor is None:
                beta = 0
            else:
                cp.multiply(auxvec1[i,:,None], auxvec2_jfac[i], out=dm_aux)
                beta = 1
            contract('rij,sji->rs', j3c_o1o2[i], j3c_o2o1[i], -.5*k_factor[i],
                     beta, out=dm_aux)
            # needs to scale by *.5, applied at the end of this function
            dm_aux = transpose_sum(dm_aux, inplace=True)
            dm_aux = dm_aux[aux_sorting[:,None], aux_sorting]
            ejk_aux.append(-int2c2e_ip1_per_atom(auxmol, dm_aux))
        ejk_aux = cp.asarray(np.stack(ejk_aux))
        int2c2e_kernel_dm = n_dm
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    _layer_t['int2c2e_ip1'] = _time.perf_counter() - _t_int2c2e
    _layer_t['int2c2e_input_dm'] = n_dm
    _layer_t['int2c2e_kernel_dm'] = int2c2e_kernel_dm
    auxvec1 = auxvec2 = dm_aux = None

    # contract the derivatives and the pseudo DM/rho
    _t_ejk_kernel = _time.perf_counter()
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(mol.omega, 54)
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*4)
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache)
    aux_loc = auxmol.ao_loc

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    l_ctr_aux_counts = l_ctr_aux_offsets[1:] - l_ctr_aux_offsets[:-1]

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    l = np.arange(laux+1)
    nf = (l + 1) * (l + 2) // 2
    aux0 = aux1 = 0
    buf1 = cp.empty((blksize, nao, nao))
    buf2 = cp.empty((blksize, nao, nao))
    if sum_results:
        kern = libvhf_rys.sum_ejk_int3c2e_ip1
        ejk = cp.zeros((mol.natm, 3))
        buf = cp.empty((nao_pair*batch_size))
        kernel_dm = n_dm
    elif slot_aware:
        kern = libvhf_rys.ejk_int3c2e_ip1
        kernel_dm = int(output_group_count)
        ejk = cp.zeros((kernel_dm, mol.natm, 3))
        buf = cp.empty((kernel_dm*nao_pair*batch_size))
    else:
        kern = libvhf_rys.ejk_int3c2e_ip1
        kernel_dm = n_dm
        ejk = cp.zeros((n_dm, mol.natm, 3))
        buf = cp.empty((n_dm*nao_pair*batch_size))

    # --- Stage 4: ejk_kernel ---
    # This stage consists of two sub-phases:
    #   (a) compressed_build: Python loop over n_dm that assembles the
    #       `compressed` tensor = J outer-product + K contraction, then
    #       symmetrizes and pair-addresses it.  Each iteration issues ~9
    #       async cupy calls; with n_dm=118 this is ~1000 kernel launches.
    #   (b) kern_call: a single ctypes dispatch of `ejk_int3c2e_ip1` to GPU.
    #
    # Timing note: cupy/ctypes calls are asynchronous, so wall time cannot
    # separate these phases. In particular, `compressed_build` can block on
    # the preceding derivative kernel when buffers are reused. The optional
    # NTTDA_DF_COMPRESSED_PROFILE path records CUDA events without per-pair
    # synchronization and reports rank_bucket_gpu plus
    # derivative_kernel_gpu_ms.
    #
    # Optimization (Task 5): K contractions are skipped when k_factor[i]==0
    # (J-only DMs).  This avoids 4 wasted cupy contract calls per J pair.
    slot_rank_2xr_data = []
    for output_group, rank1, rank2, indices in slot_rank_2xr_groups:
        scale = cp.asarray(
            [-.5*k_factor[index] for index in indices],
        )[:, None, None]
        slot_rank_2xr_data.append((
            output_group, rank1, rank2,
            cp.stack([j3c_o1o2[index] for index in indices]),
            cp.stack([j3c_o2o1[index] for index in indices]),
            cp.stack([dm1_factor_l[index] for index in indices]),
            cp.stack([dm2_factor_l[index] for index in indices]),
            cp.stack([dm2_factor_r[index] for index in indices]) * scale,
            cp.stack([dm1_factor_r[index] for index in indices]) * scale,
        ))
    _t_compressed_build = 0.0
    _t_slot_rank_2xr = 0.0
    _t_legacy = 0.0
    _t_kern_call = 0.0
    for kbatch, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        naux_in_batch = nf[lk] * l_ctr_aux_counts[kbatch]
        aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
        if sum_results:
            compressed = cp.zeros((nao_pair, naux_in_batch))
        elif slot_aware:
            compressed = ndarray(
                (kernel_dm, nao_pair, naux_in_batch), buffer=buf,
            )
            compressed[:] = 0
        else:
            compressed = ndarray((n_dm, nao_pair, naux_in_batch), buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            _t_cb = _time.perf_counter()

            _t_slot_rank_start = _time.perf_counter()
            for (
                    output_group, rank1, rank2, j3c_12, j3c_21,
                    factor1_l, factor2_l,
                    factor2_r_scaled, factor1_r_scaled,
            ) in slot_rank_2xr_data:
                group_size = j3c_12.shape[0]
                workspace_rank = max(rank1, rank2)
                max_tile = max(
                    1,
                    min(
                        group_size,
                        nao * blksize // (workspace_rank * dk),
                    ),
                )
                dm_tensor = ndarray((nao, nao, dk), buffer=buf2)
                dm_tensor[:] = 0
                for p0, p1 in lib.prange(0, group_size, max_tile):
                    tile = p1 - p0
                    tmp = ndarray(
                        (tile, rank2, nao, dk), buffer=buf1,
                    )
                    contract(
                        'brji,bqj->biqr',
                        j3c_12[p0:p1, aux0:aux1],
                        factor1_l[p0:p1],
                        out=tmp,
                    )
                    contract(
                        'biqr,bpi->pqr',
                        tmp, factor2_r_scaled[p0:p1],
                        1, 1, out=dm_tensor,
                    )
                    tmp = ndarray(
                        (tile, rank1, nao, dk), buffer=buf1,
                    )
                    contract(
                        'brji,bqj->biqr',
                        j3c_21[p0:p1, aux0:aux1],
                        factor2_l[p0:p1],
                        out=tmp,
                    )
                    contract(
                        'biqr,bpi->pqr',
                        tmp, factor1_r_scaled[p0:p1],
                        1, 1, out=dm_tensor,
                    )
                    _layer_t['slot_rank_2xr_tiles'] += 1
                    if rank1 == 2 and rank2 == 2:
                        _layer_t['slot_rank_2x2_tiles'] += 1
                dm_tensor1 = ndarray((nao, nao, dk), buffer=buf1)
                dm_tensor1[:] = dm_tensor.transpose(1, 0, 2)
                dm_tensor1 += dm_tensor
                pair_compressed = ndarray(
                    (nao_pair, dk), buffer=buf2,
                )
                cp.take(
                    dm_tensor1.reshape(-1, dk),
                    pair_addresses,
                    axis=0,
                    out=pair_compressed,
                )
                compressed[output_group, :, k0:k1] += pair_compressed
            _t_slot_rank_2xr += (
                _time.perf_counter() - _t_slot_rank_start
            )

            _t_legacy_start = _time.perf_counter()
            for i in range(n_dm):
                if slot_rank_2xr_mask[i]:
                    continue
                profile_events = None
                if compressed_profile:
                    start_event = cp.cuda.Event()
                    stop_event = cp.cuda.Event()
                    start_event.record()
                    profile_events = (start_event, stop_event)
                dm_tensor = ndarray((nao,nao,dk), buffer=buf2)
                dm_tensor1 = ndarray((nao,nao,dk), buffer=buf1)
                if j_factor is None:
                    dm_tensor[:] = 0
                else:
                    cp.multiply(dm1[i][:,:,None], auxvec2_jfac[i,None,None,aux0:aux1], out=dm_tensor)
                    cp.multiply(dm2[i][:,:,None], auxvec1_jfac[i,None,None,aux0:aux1], out=dm_tensor1)
                    dm_tensor += dm_tensor1
                # K contraction: skip when k_factor[i]==0 (J-only DM).
                # Saves 4 cupy contract calls per J pair (~10% DF time).
                if k_factor[i] != 0:
                    tmp = ndarray((dm2_noccs[i],nao,dk), buffer=buf1)
                    contract('rji,qj->iqr', j3c_o1o2[i][aux0:aux1], dm1_factor_l[i], out=tmp)
                    contract('iqr,pi->pqr', tmp, dm2_factor_r[i], -.5*k_factor[i], 1, out=dm_tensor)
                    tmp = ndarray((dm1_noccs[i],nao,dk), buffer=buf1)
                    contract('rji,qj->iqr', j3c_o2o1[i][aux0:aux1], dm2_factor_l[i], out=tmp)
                    contract('iqr,pi->pqr', tmp, dm1_factor_r[i], -.5*k_factor[i], 1, out=dm_tensor)
                dm_tensor1[:] = dm_tensor.transpose(1,0,2)
                dm_tensor1[:] += dm_tensor
                if sum_results:
                    compressed[:,k0:k1] += cp.take(
                        dm_tensor1.reshape(-1,dk), pair_addresses, axis=0,
                        out=ndarray((nao_pair, dk), buffer=buf))
                elif slot_aware:
                    pair_compressed = ndarray(
                        (nao_pair, dk), buffer=buf2,
                    )
                    cp.take(
                        dm_tensor1.reshape(-1, dk),
                        pair_addresses,
                        axis=0,
                        out=pair_compressed,
                    )
                    compressed[
                        output_group_indices[i], :, k0:k1
                    ] += pair_compressed
                else:
                    cp.take(dm_tensor1.reshape(-1,dk), pair_addresses, axis=0,
                            out=compressed[i,:,k0:k1])
                if profile_events is not None:
                    profile_events[1].record()
                    operator = 'J' if k_factor[i] == 0 else 'K'
                    key = (operator, dm1_noccs[i], dm2_noccs[i])
                    rank_bucket_events.setdefault(key, []).append(
                        profile_events,
                    )
            _t_legacy += _time.perf_counter() - _t_legacy_start
            _t_compressed_build += _time.perf_counter() - _t_cb
            _t_kc = _time.perf_counter()
        kernel_profile_events = None
        if compressed_profile:
            start_event = cp.cuda.Event()
            stop_event = cp.cuda.Event()
            start_event.record()
            kernel_profile_events = (start_event, stop_event)
        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
            ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.c_int(kernel_dm),
            ctypes.byref(int3c2e_envs),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(nao), ctypes.c_int(nao_pair),
            ctypes.c_int(naux_in_batch), ctypes.c_int(mol.natm))
        if err != 0:
            raise RuntimeError('int3c2e_ejk_ip1 failed')
        if kernel_profile_events is not None:
            kernel_profile_events[1].record()
            derivative_kernel_events.append(kernel_profile_events)
        _t_kern_call += _time.perf_counter() - _t_kc
    ejk += ejk_aux
    ejk *= .5
    ejk = ejk.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)
    _layer_t['ejk_kernel'] = _time.perf_counter() - _t_ejk_kernel
    _layer_t['compressed_build'] = _t_compressed_build
    _layer_t['slot_rank_2xr_build'] = _t_slot_rank_2xr
    _layer_t['legacy_build'] = _t_legacy
    _layer_t['kern_call'] = _t_kern_call
    if compressed_profile:
        bucket_task_counts = {}
        for index in range(n_dm):
            if slot_rank_2xr_mask[index]:
                continue
            operator = 'J' if k_factor[index] == 0 else 'K'
            key = (
                operator, dm1_noccs[index], dm2_noccs[index],
            )
            bucket_task_counts[key] = bucket_task_counts.get(key, 0) + 1
        bucket_stats = []
        for key, events in rank_bucket_events.items():
            gpu_ms = sum(
                cp.cuda.get_elapsed_time(start, stop)
                for start, stop in events
            )
            task_count = bucket_task_counts[key]
            bucket_stats.append({
                'operator': key[0],
                'rank_left': int(key[1]),
                'rank_right': int(key[2]),
                'task_count': task_count,
                'event_count': len(events),
                'gpu_ms': float(gpu_ms),
                'gpu_ms_per_task': float(gpu_ms / task_count),
            })
        bucket_stats.sort(key=lambda item: item['gpu_ms'], reverse=True)
        _layer_t['rank_bucket_gpu'] = bucket_stats
        _layer_t['compressed_build_gpu_ms'] = float(sum(
            item['gpu_ms'] for item in bucket_stats
        ))
        _layer_t['derivative_kernel_gpu_ms'] = float(sum(
            cp.cuda.get_elapsed_time(start, stop)
            for start, stop in derivative_kernel_events
        ))
    _layer_t['n_dm'] = n_dm
    _layer_t['naux'] = int(naux)
    _layer_t['nao'] = int(nao)
    if stats_sink is not None:
        stats_sink.append(dict(_layer_t))
    return ejk

def _j_energies_per_atom(int3c2e_opt, dm_pairs, j_factor,
                         sum_results=False, verbose=None):
    '''
    Computes first-order derivatives of Coulomb energy for multiple sets of
    density matrix pairs.
    '''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    n_dm = len(dm_pairs)

    if j_factor is None or all(x == 0 for x in j_factor):
        if sum_results:
            return np.zeros((mol.natm, 3))
        else:
            return np.zeros((n_dm, mol.natm, 3))

    assert len(j_factor) == n_dm
    nao = mol.mol.nao
    dms = cp.empty((2, n_dm, nao, nao))
    for i, dm1_dm2 in enumerate(dm_pairs):
        if isinstance(dm1_dm2, cp.ndarray) and dm1_dm2.ndim == 2:
            dms[0,i] = dms[1,i] = dm1_dm2
        else:
            dms[0,i] = dm1_dm2[0]
            dms[1,i] = dm1_dm2[1]
    dms = mol.apply_C_mat_CT(dms.reshape(2*n_dm,nao,nao))
    dms = transpose_sum(dms, inplace=True)
    dms[:] *= .5
    auxvec = int3c2e_opt.contract_dm(dms, hermi=1)
    auxvec = auxmol.apply_CT_dot(auxvec, axis=1)
    t0 = log.timer_debug1('contract dm', *t0)
    j2c = int2c2e(auxmol)

    if mol.omega <= 0 and not auxmol.mol.cart:
        auxvec = cp.linalg.solve(j2c, auxvec.T).T
    else:
        auxvec = _gen_metric_solver(j2c, 'ED')(auxvec.T).T
    auxvec = cp.asarray(auxmol.apply_C_dot(auxvec, axis=1), order='C')
    naux = auxvec.shape[1]
    # Swap the output of dm1 and dm2 in auxvec. They are cross-contracted in
    # ejk_int3c2e_ip1, i.e. dm1*auxvec2 + dm2*auxvec1
    auxvec = auxvec.reshape(2, n_dm, naux)
    auxvec_jfac = auxvec * cp.array(j_factor)[None,:,None]
    auxvec21 = auxvec_jfac[[1, 0]].reshape(2*n_dm, naux)
    j2c = None

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(mol.omega, 54)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*16)
    ksh_offsets_cpu = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    if sum_results:
        kern = libvhf_rys.sum_ejk_int3c2e_ip1
        ej = cp.zeros((mol.natm, 3))
        ej_aux = cp.zeros_like(ej)
    else:
        kern = libvhf_rys.ejk_int3c2e_ip1
        ej = cp.zeros((2, n_dm, mol.natm, 3))
        ej_aux = cp.zeros_like(ej)

    err = kern(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p),
        ctypes.cast(ej_aux.data.ptr, ctypes.c_void_p),
        ctypes.cast(dms.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec21.data.ptr, ctypes.c_void_p),
        ctypes.c_int(2*n_dm),
        ctypes.byref(int3c2e_envs),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(shl_pair_offsets) - 1),
        ctypes.c_int(len(ksh_offsets_cpu) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        lib.c_null_ptr(), ctypes.c_int(0),
        ctypes.c_int(dms.shape[-1]), ctypes.c_int(0),
        ctypes.c_int(naux), ctypes.c_int(mol.natm))
    if err != 0:
        raise RuntimeError('int3c2e_ejk_ip1 failed')

    if not sum_results:
        ej = ej[0] + ej[1]
        ej_aux = ej_aux[0] + ej_aux[1]
    ej = ej.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    #ej_aux += .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
    if sum_results:
        dm_aux = cp.zeros((naux, naux))
        for i in range(n_dm):
            dm_aux += auxvec[0,i,:,None] * auxvec_jfac[1,i]
            dm_aux += auxvec[1,i,:,None] * auxvec_jfac[0,i]
        ej_aux -= .5 * cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
    else:
        for i in range(n_dm):
            dm_aux  = auxvec[0,i,:,None] * auxvec_jfac[1,i]
            dm_aux += auxvec[1,i,:,None] * auxvec_jfac[0,i]
            ej_aux[i] -= .5 * cp.asarray(int2c2e_ip1_per_atom(auxmol, dm_aux))
    ej += ej_aux.get()
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

def _factorize_multiple_dm(mol, dm_pair, hermi, factor_cache=None):
    def factorize(dm):
        if factor_cache is None:
            return _factorize_dm(mol, dm, hermi)
        key = id(dm)
        cached = factor_cache.get(key)
        if cached is not None and cached[0] is dm:
            return cached[1]
        result = _factorize_dm(mol, dm, hermi)
        factor_cache[key] = (dm, result)
        return result

    dm1_factor_r = dm2_factor_r = None
    if isinstance(dm_pair, cp.ndarray):
        if dm_pair.ndim == 2: # dm pair employs two identical density matrices
            res = factorize(dm_pair)
            dm1_factor_l, dm1_factor_r = dm2_factor_l, dm2_factor_r = res
        else:
            res = _factorize_dm(mol, dm_pair, hermi)
            dm1_factor_l, dm2_factor_l = res[0]
            dm1_factor_r, dm2_factor_r = res[1]
    else:
        dm1_factor_l, dm1_factor_r = factorize(dm_pair[0])
        dm2_factor_l, dm2_factor_r = factorize(dm_pair[1])
    return dm1_factor_l, dm1_factor_r, dm2_factor_l, dm2_factor_r

class Gradients(tdrhf_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)
        assert isinstance(self.base, tdrhf.TDHF) or isinstance(self.base, tdrhf.TDA)

    def get_veff(self, mol, dm, j_factor=1, k_factor=1, omega=0,
                 hermi=0, verbose=None):
        ejk = self.jk_energy_per_atom(
            dm, j_factor, k_factor, omega, hermi, verbose)
        return ejk * .5

    def jk_energy_per_atom(self, dms, j_factor=None, k_factor=None, omega=0,
                           hermi=0, verbose=None):
        '''
        Computes the sum of first-order derivatives of J/K contributions for
        multiple density matrices.

        Args:
            dms:
                A list of density-matrices
            j_factor :
                A list of factors for Coulomb (J) term
            k_factor :
                A list of factors for Coulomb (K) term
            hermi :
                An overall symmetry code for all density matrices

        Returns:
            An array of shape (Natm, 3).
        '''
        return self.jk_energies_per_atom(dms, j_factor, k_factor, omega,
                                         sum_results=True, verbose=verbose)

    def jk_energies_per_atom(self, dm_list, j_factor=None, k_factor=None, omega=0,
                             hermi=0, sum_results=False, verbose=None):
        '''
        Computes a set of first-order derivatives of J/K contributions for each
        element (density matrix or a pair of density matrices) in dm_pairs.

        This function supports evaluating multiple sets of energy derivatives in a
        single call. Additionally, for each set, the two density matrices for the
        four-index Coulomb integrals can be different.

        Args:
            dm_list :
                A list of density-matrix-pairs [[dm, dm], [dm, dm], ...].
                Each element corresponds to one set of energy derivative.
            j_factor :
                A list of factors for Coulomb (J) term
            k_factor :
                A list of factors for Coulomb (K) term
            hermi :
                An integer or a list of integer to indicate whether the density
                matrices are symmetric for each set . If an integer is specified,
                the same symmetry code is applied to all density matrices.
            sum_results : bool
                If True, aggregate all sets of derivatives into a single result.

        Returns:
            An array of shape (*, Natm, 3) if sum_results is False; otherwise,
            an array of shape (Natm, 3).
        '''
        assert self.auxbasis_response
        mf = self.base._scf
        mol = mf.with_df.mol
        auxmol = mf.with_df.auxmol
        mf.with_df.reset() # Release GPU memory
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            int3c2e_opt = Int3c2eOpt(mol, auxmol).build()

        if (sum_results and
            # When the input is a list, each density matrix is applied twice in
            # a symmetric manner for computing the J and K contributions.
            isinstance(dm_list, cp.ndarray) and dm_list.ndim < 4):
            if not isinstance(hermi, int):
                hermi = all(x == 1 for x in hermi)
            return _jk_energy_per_atom(
                int3c2e_opt, dm_list, j_factor, k_factor, hermi, verbose=verbose)

        ejk = _jk_energies_per_atom(
            int3c2e_opt, dm_list, j_factor, k_factor, sum_results, verbose=verbose)
        return ejk

Grad = Gradients
