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

"""GPU derivative-ledger execution, independent of the NTTDA method."""
import os
import time
import numpy as np
import cupy as cp
from gpu4pyscf.grad.tdrhf import _jk_energies_per_atom
from gpu4pyscf.grad.nttda_params import PARAMS as _NTTDA_PARAMS
from gpu4pyscf.scf.jk import _VHFOpt

def _density_view_key(density):
    """Identity of one exact NumPy density view without hashing its values."""
    if isinstance(density, cp.ndarray):
        interface = density.__cuda_array_interface__
        location = 'device'
    else:
        density = np.asarray(density)
        interface = density.__array_interface__
        location = 'host'
    return (
        location, interface['data'][0], density.shape, density.strides,
        density.dtype.str,
    )

class _DensityCombination:
    """Linear density combination retaining its source-factor recipe."""

    def __init__(self, densities, coefficients, matrix):
        self.densities = tuple(densities)
        self.coefficients = np.asarray(coefficients)
        self.matrix = matrix

    def __array__(self, dtype=None, copy=None):
        matrix = np.asarray(cp.asnumpy(self.matrix), dtype=dtype)
        return matrix.copy() if copy else matrix

def _density_cache_key(value):
    """Stable key for original density views and temporary combinations."""
    if not isinstance(value, _DensityCombination):
        return ('view', _density_view_key(value))
    coefficients = np.ascontiguousarray(value.coefficients)
    return (
        'linear_combination',
        tuple(_density_view_key(item) for item in value.densities),
        coefficients.shape,
        coefficients.dtype.str,
        coefficients.tobytes(),
    )

def _linear_combination(densities, coefficients):
    dtype = np.result_type(
        *(density.dtype for density in densities), coefficients.dtype,
    )
    output = cp.zeros(densities[0].shape, dtype=dtype)
    for coefficient, density in zip(coefficients, densities):
        if coefficient != 0.0:
            output += coefficient * cp.asarray(density)
    return _DensityCombination(densities, coefficients, output)

def _compress_bilinear_pairs(pairs, factors):
    """Exactly factor bilinear density contractions by numerical-rank SVD."""
    if not pairs:
        return pairs, factors, {
            'input_pairs': 0, 'output_pairs': 0,
            'coefficient_rank': 0, 'compressed': False,
        }
    factors_array = np.asarray(factors)
    left_densities = []
    right_densities = []
    left_indices = {}
    right_indices = {}
    pair_indices = []
    for left, right in pairs:
        left_key = _density_view_key(left)
        right_key = _density_view_key(right)
        if left_key not in left_indices:
            left_indices[left_key] = len(left_densities)
            left_densities.append(left)
        if right_key not in right_indices:
            right_indices[right_key] = len(right_densities)
            right_densities.append(right)
        pair_indices.append((left_indices[left_key], right_indices[right_key]))

    coefficients = np.zeros(
        (len(left_densities), len(right_densities)),
        dtype=factors_array.dtype,
    )
    for (left_index, right_index), factor in zip(
            pair_indices, factors_array):
        coefficients[left_index, right_index] += factor
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        coefficients, full_matrices=False,
    )
    if singular_values.size:
        tolerance = (
            np.finfo(singular_values.dtype).eps
            * max(coefficients.shape) * singular_values[0]
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
    else:
        rank = 0
    details = {
        'input_pairs': len(pairs),
        'output_pairs': rank,
        'coefficient_rank': rank,
        'compressed': rank < len(pairs),
    }
    if rank >= len(pairs):
        return pairs, factors, details

    compressed = []
    for index in range(rank):
        root = np.sqrt(singular_values[index])
        left = _linear_combination(
            left_densities, left_vectors[:, index] * root,
        )
        right = _linear_combination(
            right_densities, right_vectors[index] * root,
        )
        compressed.append((left, right))
    return compressed, np.ones(rank), details

def _expanded_slot_pair_groups(items):
    """Map J/K terms to bilinear pairs without mixing output slots."""
    groups = {}
    for operator, term in items:
        pairs, factors = groups.setdefault(
            (operator, term.slot), ([], []),
        )
        if operator == 'j':
            pairs.append((term.left, term.right))
            factors.append(2.0 * term.scale)
        elif operator == 'k':
            pairs.extend((
                (term.left, term.right.T),
                (term.left.T, term.right),
            ))
            factors.extend((-2.0 * term.scale, -2.0 * term.scale))
        else:
            raise ValueError(f'unknown J/K ledger operator {operator!r}')
    return groups

class DFLedgerBackend:
    '''Density-fitted GPU evaluation of the native ``_JKDerivativeLedger``.

    Same term mapping as :class:`LedgerBackend` (the DF per-atom kernels
    implement the same pair-energy derivative semantics), evaluated with
    the ``df.grad`` machinery so the auxiliary-basis response is included
    -- the result is the exact derivative of the DF energy surface.
    '''

    def __init__(self, gmf, compress_slots=True):
        from gpu4pyscf.df.df_jk import _DFHF

        assert isinstance(gmf, _DFHF)
        self._gmf = gmf
        self._opt = {}
        self.compress_slots = bool(compress_slots)
        self.output_backend = _NTTDA_PARAMS['df_output_backend']
        self.stats = {
            'calls': 0,
            'terms': 0,
            'density_uploads': 0,
            'density_reuses': 0,
            'omega_batches': 0,
            'raw_pairs': 0,
            'candidate_pairs': 0,
            'selected_pairs': 0,
            'compression_groups': 0,
            'compression_accepted': 0,
            'compression_rejected': 0,
            'zero_rank_pairs_skipped': 0,
            'compression_seconds': 0.0,
            'integral_seconds': 0.0,
            'integral_layer_stats': [],
            'lower_bound_gates': 0,
            'lower_bound_rejected': 0,
            'compression_enabled': self.compress_slots,
            'compression_details': [],
            'output_backend': self.output_backend,
            'output_input_tasks': 0,
            'output_kernel_tasks': 0,
        }

    def _get_opt(self, omega):
        key = float(omega or 0.0)
        if key not in self._opt:
            from gpu4pyscf.df.grad.rhf import Int3c2eOpt

            with_df = self._gmf.with_df
            mol = with_df.mol
            auxmol = with_df.auxmol
            if auxmol is None:
                with_df.build()
                auxmol = with_df.auxmol
            with_df.reset()
            with mol.with_range_coulomb(key), \
                    auxmol.with_range_coulomb(key):
                self._opt[key] = Int3c2eOpt(mol, auxmol).build()
        return self._opt[key]

    def __call__(self, terms, mol, atoms, slots=()):
        from gpu4pyscf.df.df_jk import (
            _make_factorized_dm,
            _tag_factorize_dm,
            _transpose_dm,
        )
        from gpu4pyscf.df.grad.tdrhf import (
            _jk_energies_per_atom as _df_jk_energies_per_atom,
        )

        atoms = list(atoms)
        shape = (len(atoms), 3)
        gradients = {slot: cp.zeros(shape) for slot in slots}
        groups = {}
        self.stats['calls'] += 1
        for operator in ('j', 'k'):
            for term in terms[operator]:
                gradients.setdefault(term.slot, cp.zeros(shape))
                groups.setdefault(float(term.omega or 0.0), []).append(
                    (operator, term),
                )
                self.stats['terms'] += 1
        self.stats['omega_batches'] += len(groups)

        # A frame ledger contains many repeated views of the same AO density.
        # Upload and factorize each unique view only once.  The factor tags are
        # consumed directly by df.grad.tdrhf, avoiding another SVD for every
        # J/K pair.  Transposition preserves and swaps those factors.
        density_cache = {}

        def gpu_density(value, factorize):
            combination = (
                value if isinstance(value, _DensityCombination) else None
            )
            array = cp.asarray(value.matrix if isinstance(value, _DensityCombination) else value)
            key = (_density_cache_key(value), bool(factorize))
            density = density_cache.get(key)
            if density is None:
                transpose = None
                if combination is None:
                    transpose_key = (
                        _density_cache_key(array.T), bool(factorize),
                    )
                    transpose = density_cache.get(transpose_key)
                if transpose is not None:
                    density = _transpose_dm(transpose)
                elif combination is not None and factorize:
                    factor_l = []
                    factor_r = []
                    for coefficient, component in zip(
                            combination.coefficients,
                            combination.densities):
                        tagged = gpu_density(component, factorize=True)
                        if tagged.factor_l.shape[-1] == 0:
                            continue
                        factor_l.append(tagged.factor_l * coefficient)
                        factor_r.append(tagged.factor_r)
                    if factor_l:
                        factor_l = cp.hstack(factor_l)
                        factor_r = cp.hstack(factor_r)
                        q_left, r_left = cp.linalg.qr(
                            factor_l, mode='reduced',
                        )
                        q_right, r_right = cp.linalg.qr(
                            factor_r, mode='reduced',
                        )
                        core = r_left @ r_right.T
                        u, singular_values, vh = cp.linalg.svd(
                            core, full_matrices=False,
                        )
                        tolerance = (
                            cp.finfo(singular_values.dtype).eps
                            * max(core.shape) * singular_values[0]
                        )
                        mask = singular_values > tolerance
                        stable_l = q_left @ u[:, mask]
                        stable_r = q_right @ (
                            vh[mask].T * singular_values[mask][None]
                        )
                        density = _make_factorized_dm(
                            stable_l, stable_r, symmetrize=0,
                        )
                    else:
                        density = _tag_factorize_dm(
                            cp.zeros_like(cp.asarray(array)), hermi=0,
                        )
                else:
                    density = cp.asarray(array)
                    if factorize:
                        density = _tag_factorize_dm(density, hermi=0)
                    self.stats['density_uploads'] += 1
                density_cache[key] = density
            else:
                self.stats['density_reuses'] += 1
            return density

        def gpu_pairs(cpu_pairs, factorize):
            return [
                [gpu_density(left, factorize), gpu_density(right, factorize)]
                for left, right in cpu_pairs
            ]

        def pair_cost(pairs, opt):
            nao = opt.mol.nao
            naux = opt.auxmol.nao
            return sum(
                2 * left.factor_l.shape[-1]
                * right.factor_l.shape[-1] * naux
                + 2 * nao**2
                for left, right in pairs
            )

        for omega, items in groups.items():
            factorize = any(operator == 'k' for operator, _term in items)
            opt = self._get_opt(omega)
            nao = opt.mol.nao
            naux = opt.auxmol.nao
            if factorize:
                # Seed factors in the original orientation.  K pairs below
                # then obtain transposes by swapping factor tags, matching the
                # established backend and avoiding a second/zero-rank SVD.
                for _operator, term in items:
                    gpu_density(term.left, factorize=True)
                    gpu_density(term.right, factorize=True)
            pairs = []
            j_factors = []
            k_factors = []
            slot_index = []
            compression_started = time.perf_counter()
            for (operator, slot), (raw_pairs, raw_factors) in (
                    _expanded_slot_pair_groups(items).items()):
                if self.compress_slots:
                    candidate, candidate_factors, details = (
                        _compress_bilinear_pairs(raw_pairs, raw_factors)
                    )
                else:
                    candidate, candidate_factors = raw_pairs, raw_factors
                    details = {'compressed': False}
                self.stats['compression_groups'] += 1
                self.stats['raw_pairs'] += len(raw_pairs)
                self.stats['candidate_pairs'] += len(candidate)
                selected = raw_pairs
                selected_factors = raw_factors
                selected_gpu = None
                if details['compressed'] and not factorize:
                    selected = candidate
                    selected_factors = candidate_factors
                elif details['compressed']:
                    raw_gpu = gpu_pairs(raw_pairs, factorize=True)
                    raw_cost = pair_cost(raw_gpu, opt)
                    candidate_min_rank = details.get(
                        'coefficient_rank', 1,
                    )
                    # Safe lower-bound gate (Task 4):
                    # Estimate the candidate's theoretical minimum cost from
                    # its coefficient_rank before constructing expensive
                    # factorized candidate_gpu.  Only construct the candidate
                    # if its lower bound is below the raw cost.  This gate
                    # can only reject (never accept) — it is conservative.
                    candidate_lower_bound = (
                        2 * candidate_min_rank * candidate_min_rank * naux
                        + 2 * nao**2
                    ) * len(candidate)
                    self.stats['lower_bound_gates'] += 1
                    if candidate_lower_bound >= raw_cost:
                        self.stats['lower_bound_rejected'] += 1
                        selected_gpu = raw_gpu
                    else:
                        candidate_gpu = gpu_pairs(candidate, factorize=True)
                        if pair_cost(candidate_gpu, opt) < raw_cost:
                            selected = candidate
                            selected_factors = candidate_factors
                            selected_gpu = candidate_gpu
                        else:
                            selected_gpu = raw_gpu
                if details['compressed'] and selected is candidate:
                    self.stats['compression_accepted'] += 1
                elif details['compressed']:
                    self.stats['compression_rejected'] += 1
                if selected_gpu is None:
                    selected_gpu = gpu_pairs(selected, factorize)
                if factorize:
                    keep = [
                        index for index, pair in enumerate(selected_gpu)
                        if (pair[0].factor_l.shape[-1] > 0
                            and pair[1].factor_l.shape[-1] > 0)
                    ]
                    self.stats['zero_rank_pairs_skipped'] += (
                        len(selected_gpu) - len(keep)
                    )
                    selected_gpu = [selected_gpu[index] for index in keep]
                    selected_factors = [
                        selected_factors[index] for index in keep
                    ]
                pairs.extend(selected_gpu)
                self.stats['selected_pairs'] += len(selected_gpu)
                self.stats['compression_details'].append({
                    'operator': operator,
                    'slot': repr(slot),
                    'raw_pairs': len(raw_pairs),
                    'candidate_pairs': len(candidate),
                    'selected_pairs': len(selected_gpu),
                    'accepted': bool(
                        details['compressed'] and selected is candidate
                    ),
                })
                slot_index.extend((slot,) * len(selected_gpu))
                if operator == 'j':
                    j_factors.extend(selected_factors)
                    k_factors.extend((0.0,) * len(selected_gpu))
                else:
                    j_factors.extend((0.0,) * len(selected_gpu))
                    k_factors.extend(selected_factors)
            self.stats['compression_seconds'] += (
                time.perf_counter() - compression_started
            )
            if not pairs:
                continue
            integral_started = time.perf_counter()
            layer_stats = []
            output_group_indices = None
            output_group_keys = None
            if self.output_backend == 'slot_aware':
                output_group_keys = []
                output_group_lookup = {}
                output_group_indices = []
                for slot in slot_index:
                    group = output_group_lookup.get(slot)
                    if group is None:
                        group = len(output_group_keys)
                        output_group_lookup[slot] = group
                        output_group_keys.append(slot)
                    output_group_indices.append(group)
            energies = _df_jk_energies_per_atom(
                opt, pairs,
                j_factor=j_factors, k_factor=k_factors, sum_results=False,
                stats_sink=layer_stats,
                output_group_indices=output_group_indices,
                output_group_count=(
                    None if output_group_keys is None
                    else len(output_group_keys)
                ),
            )
            energies = cp.asarray(energies)
            self.stats['integral_seconds'] += (
                time.perf_counter() - integral_started
            )
            self.stats['integral_layer_stats'].extend(layer_stats)
            self.stats['output_input_tasks'] += len(slot_index)
            if output_group_keys is None:
                energy_slots = slot_index
            else:
                energy_slots = output_group_keys
            self.stats['output_kernel_tasks'] += len(energy_slots)
            for row, slot in zip(energies, energy_slots):
                gradients[slot] += row[atoms]
        return gradients

class LedgerBackend:
    '''Batched GPU evaluation of the native ``_JKDerivativeLedger``.

    Implements the seam ``nttda_jk_ledger_backend(terms, mol, atoms,
    slots)``: every term of every slot is pushed into one
    ``_jk_energies_per_atom`` call per range-separation parameter.
    '''

    def __init__(self, gmf):
        self._gmf = gmf
        self._vhfopt = {}
        self.stats = {
            'calls': 0,
            'terms': 0,
            'omega_batches': 0,
        }

    def _get_vhfopt(self, omega):
        key = float(omega or 0.0)
        if key not in self._vhfopt:
            mol = self._gmf.mol
            tol = self._gmf.direct_scf_tol
            if key == 0.0:
                self._vhfopt[key] = _VHFOpt(mol, tol, tile=1).build()
            else:
                with mol.with_range_coulomb(key):
                    self._vhfopt[key] = _VHFOpt(mol, tol, tile=1).build()
        return self._vhfopt[key]

    def __call__(self, terms, mol, atoms, slots=()):
        atoms = list(atoms)
        shape = (len(atoms), 3)
        gradients = {slot: cp.zeros(shape) for slot in slots}
        groups = {}
        self.stats['calls'] += 1
        for operator in ('j', 'k'):
            for term in terms[operator]:
                gradients.setdefault(term.slot, cp.zeros(shape))
                groups.setdefault(float(term.omega or 0.0), []).append(
                    (operator, term),
                )
                self.stats['terms'] += 1
        self.stats['omega_batches'] += len(groups)
        for omega, items in groups.items():
            pairs = []
            j_factors = []
            k_factors = []
            slot_index = []
            for operator, term in items:
                left = cp.asarray(term.left)
                right = cp.asarray(term.right)
                if operator == 'j':
                    pairs.append([left, right])
                    j_factors.append(2.0 * term.scale)
                    k_factors.append(0.0)
                    slot_index.append(term.slot)
                else:
                    pairs.append([left, cp.ascontiguousarray(right.T)])
                    pairs.append([cp.ascontiguousarray(left.T), right])
                    j_factors.extend((0.0, 0.0))
                    k_factors.extend((-2.0 * term.scale, -2.0 * term.scale))
                    slot_index.extend((term.slot, term.slot))
            if not pairs:
                continue
            energies = _jk_energies_per_atom(
                self._get_vhfopt(omega), pairs,
                j_factor=j_factors, k_factor=k_factors, sum_results=False,
            )
            energies = cp.asarray(energies)
            for row, slot in zip(energies, slot_index):
                gradients[slot] += row[atoms]
        return gradients
