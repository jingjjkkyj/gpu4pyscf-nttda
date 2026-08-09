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

'''GPU driver for NTTDA ``deltaS = -1`` excited-state gradients.

Orchestration and scientific formulas reuse the validated CPU forge
implementation (pyscf-forge NTTDA, importable side by side with
gpu4pyscf); expensive integral and GGA grid work run on GPU:

- response J/K builds (M matrix, Fock builds, Z-vector iterations) are
  routed from the CPU twin objects to a GPU ROKS/UKS backend;
- the J/K derivative ledger is evaluated with the batched per-atom
  ``_jk_energies_per_atom`` kernels through the ledger backend seam
  (empirically calibrated mapping, machine-precision on J/K and
  range-separated variants: ``cpu_J = 2*gpu(L, R)``,
  ``cpu_K = -2*[gpu(L, R^T) + gpu(L^T, R)]``);
- the ground-state ROKS gradient uses the native GPU implementation.

For GGA functionals, response, Fock-Z, post-Z derivative contractions, and
the iterative UKS fxc action are evaluated by one geometry-fixed GPU XC
backend.  The CPU forge remains the formula and orchestration reference.
'''

import importlib
import inspect
import os
import sys
import time

import numpy as np
import cupy as cp

from gpu4pyscf.grad.tdrhf import _jk_energies_per_atom
from gpu4pyscf.grad.nttda_params import PARAMS as _NTTDA_PARAMS
from gpu4pyscf.scf.jk import _VHFOpt


def _assert_module_under_root(module, root, label):
    '''Reject an already-loaded forge module from another checkout.'''
    module_file = getattr(module, '__file__', None)
    if not module_file:
        raise ImportError(f'CPU forge {label} module has no source path')
    root = os.path.realpath(root)
    module_file = os.path.realpath(module_file)
    try:
        matches = os.path.commonpath((root, module_file)) == root
    except ValueError:
        matches = False
    if not matches:
        raise ImportError(
            f'CPU forge {label} module was loaded from {module_file}, '
            f'outside requested NTTDA_FORGE_PATH {root}'
        )


def _prepend_path(package, path):
    path = os.path.realpath(path)
    existing = [os.path.realpath(item) for item in package.__path__]
    if path in existing:
        package.__path__.remove(package.__path__[existing.index(path)])
    package.__path__.insert(0, path)


def _validate_forge_capabilities(forge_grad, forge_roks):
    parameters = inspect.signature(
        forge_grad.Gradients._analytic_components,
    ).parameters
    if 'response_cache' not in parameters:
        raise ImportError(
            'Loaded CPU forge NTTDA lacks the response_cache accelerator seam'
        )
    if not hasattr(forge_roks, 'ResponseCache'):
        raise ImportError(
            'Loaded CPU forge NTTDA lacks the ResponseCache accelerator seam'
        )
    frame_stage = getattr(forge_roks, '_stage_frame_jk', None)
    if (
            frame_stage is None
            or 'frame_xc_backend' not in frame_stage.__code__.co_consts):
        raise ImportError(
            'Loaded CPU forge NTTDA lacks the GPU frame-XC backend seam'
        )
    delta = importlib.import_module(
        'pyscf.grad.nttda.delta_s_minus_one',
    )
    contract = delta._JKDerivativeLedger.contract
    code = contract.__code__
    if (
            'nttda_jk_ledger_backend' not in code.co_names
            and 'nttda_jk_ledger_backend' not in code.co_consts):
        raise ImportError(
            'Loaded CPU forge NTTDA lacks the derivative-ledger backend seam'
        )


def _import_forge():
    '''Import one verified CPU forge checkout for the hybrid driver.'''
    requested_root = os.environ.get('NTTDA_FORGE_PATH')
    if requested_root:
        requested_root = os.path.realpath(requested_root)
        path = os.path.join(requested_root, 'pyscf')
        required = (
            path,
            os.path.join(path, 'grad', 'nttda'),
            os.path.join(path, 'nac', 'nttda.py'),
            os.path.join(path, 'sftda', 'nttda.py'),
        )
        if not all(os.path.exists(item) for item in required):
            raise ImportError(
                'NTTDA_FORGE_PATH does not contain the required optimized '
                f'forge modules: {requested_root}'
            )

        import pyscf
        _prepend_path(pyscf, path)
        import pyscf.grad
        import pyscf.nac
        import pyscf.sftda

        for package, sub in (
                (pyscf.grad, os.path.join(path, 'grad')),
                (pyscf.nac, os.path.join(path, 'nac')),
                (pyscf.sftda, os.path.join(path, 'sftda'))):
            _prepend_path(package, sub)

        for name, label in (
                ('pyscf.grad.nttda', 'gradient'),
                ('pyscf.nac.nttda', 'NAC'),
                ('pyscf.sftda.nttda', 'solver')):
            if name in sys.modules:
                _assert_module_under_root(
                    sys.modules[name], requested_root, label,
                )
        importlib.invalidate_caches()

    try:
        forge_grad = importlib.import_module('pyscf.grad.nttda')
        importlib.import_module('pyscf.nac.nttda')
        forge_solver = importlib.import_module('pyscf.sftda.nttda')
        forge_roks = importlib.import_module('pyscf.grad.nttda.roks')
    except ImportError as error:
        raise ImportError(
            'The optimized CPU pyscf-forge NTTDA package is required for '
            'the hybrid GPU driver. Set NTTDA_FORGE_PATH to its checkout root.'
        ) from error

    _validate_forge_capabilities(forge_grad, forge_roks)
    if requested_root:
        for module, label in (
                (forge_grad, 'gradient'),
                (forge_roks, 'gradient response'),
                (forge_solver, 'solver'),
                (sys.modules['pyscf.nac.nttda'], 'NAC'),
                (
                    sys.modules['pyscf.grad.nttda.delta_s_minus_one'],
                    'gradient ledger',
                )):
            _assert_module_under_root(
                module, requested_root, label,
            )
    return forge_grad, forge_roks, forge_solver


def _is_gpu_object(obj):
    return type(obj).__module__.startswith('gpu4pyscf')


def _validate_supported_reference(gmf):
    '''Fail before mixing energy and derivative models we cannot preserve.'''
    if bool(getattr(gmf, 'only_dfj', False)):
        raise NotImplementedError(
            'NTTDA gradients/NAC/FSSH do not yet support only_dfj because '
            'the conventional-K derivative ledger is not implemented'
        )
    for marker, attribute, label in (
            ('_Solvation', 'with_solvent', 'solvent'),
            ('_QMMM', 'mm_mol', 'QM/MM')):
        wrapped = getattr(gmf, attribute, None) is not None
        istype = getattr(gmf, 'istype', None)
        if callable(istype):
            wrapped = wrapped or bool(istype(marker))
        if wrapped:
            raise NotImplementedError(
                f'NTTDA gradients/NAC/FSSH do not support {label} wrappers'
            )
    if getattr(gmf, 'disp', None) not in (None, False, ''):
        raise NotImplementedError(
            'NTTDA gradients/NAC/FSSH do not support dispersion corrections'
        )
    has_nlc = getattr(gmf, 'nlc', None) not in (None, False, '')
    do_nlc = getattr(gmf, 'do_nlc', None)
    if callable(do_nlc):
        has_nlc = has_nlc or bool(do_nlc())
    if has_nlc:
        raise NotImplementedError(
            'NTTDA gradients/NAC/FSSH do not support nonlocal correlation'
        )
    numint = getattr(gmf, '_numint', None)
    if numint is not None:
        from gpu4pyscf.dft import numint as gpu_numint

        if type(numint) is not gpu_numint.NumInt:
            raise NotImplementedError(
                'NTTDA gradients/NAC/FSSH require the standard GPU NumInt'
            )


def build_cpu_twin(gpu_td):
    '''CPU forge NTTDA twin of a converged GPU NTTDA calculation.

    Orbitals, occupations, energies, amplitudes, and the (sorted) grid
    are copied so the twin never runs SCF, Davidson, or grid building of
    its own.
    '''
    from pyscf import dft

    _forge_grad, _forge_roks, forge_solver = _import_forge()
    gmf = gpu_td._scf
    _validate_supported_reference(gmf)
    mol = gmf.mol
    if getattr(gmf, 'is_ensemble_rks', False):
        from pyscf.sftda import EnsembleRKS

        cpu_mf = EnsembleRKS(mol, nopen=gmf.nopen)
    else:
        cpu_mf = dft.ROKS(mol)
    cpu_mf.xc = gmf.xc
    omega = getattr(gmf, 'omega', None)
    if omega is not None:
        cpu_mf.omega = omega
    cpu_mf.verbose = 0
    cpu_mf.max_memory = gmf.max_memory
    cpu_mf.mo_coeff = cp.asnumpy(cp.asarray(gmf.mo_coeff))
    cpu_mf.mo_occ = cp.asnumpy(cp.asarray(gmf.mo_occ))
    cpu_mf.mo_energy = cp.asnumpy(cp.asarray(gmf.mo_energy))
    cpu_mf.converged = True
    cpu_mf.grids.coords = cp.asnumpy(cp.asarray(gmf.grids.coords))
    cpu_mf.grids.weights = cp.asnumpy(cp.asarray(gmf.grids.weights))
    cpu_mf.grids.non0tab = None

    cpu_td = forge_solver.NTTDA(cpu_mf)
    cpu_td.deltaS = gpu_td.deltaS
    cpu_td.nobeta = gpu_td.nobeta
    cpu_td.nstates = len(gpu_td.e)
    cpu_td.verbose = 0
    cpu_td.e = np.asarray(gpu_td.e)
    cpu_td.xy = [
        (cp.asnumpy(cp.asarray(x)), 0) for x, _y in gpu_td.xy
    ]
    converged = gpu_td.converged
    if converged is None:
        converged = []
    cpu_td.converged = cp.asnumpy(cp.asarray(converged))
    for name in ('_nttda_gpu_fxc_ref', '_nttda_gpu_fock0_fockz'):
        if hasattr(gpu_td, name):
            setattr(cpu_td, name, getattr(gpu_td, name))
    return cpu_td


def route_jk_to_gpu(cpu_mf, gmf):
    '''Route a CPU SCF object's J/K builds to a GPU backend (in place).

    Instance-level overrides so the CPU response/Fock machinery calls the
    GPU integral engines transparently (numpy in / numpy out).  The
    range-separated J uses the erf-kernel VHFOpt path because the stock
    GPU J engine ignores ``omega``.
    '''
    if getattr(cpu_mf, '_nttda_gpu_routed', False):
        return cpu_mf
    from gpu4pyscf.df.df_jk import _DFHF

    nao = cpu_mf.mol.nao_nr()
    route_stats = {
        'j_calls': 0,
        'k_calls': 0,
        'jk_calls': 0,
        'combined_jk_calls': 0,
        'density_matrices': 0,
    }

    def _shape_back(value, dm):
        value = cp.asnumpy(cp.asarray(value))
        return value.reshape(np.asarray(dm).shape)

    def get_j(mol=None, dm=None, hermi=0, omega=None, **_kw):
        dms = cp.asarray(dm).reshape(-1, nao, nao)
        route_stats['j_calls'] += 1
        route_stats['density_matrices'] += len(dms)
        if omega:
            from gpu4pyscf.sftda.nttda import _get_j_range_separated

            # _get_j_range_separated dispatches internally: per-omega
            # cderi for DF references, erf-kernel VHFOpt otherwise.
            vj = _get_j_range_separated(gmf, dms, hermi, omega)
        else:
            vj = gmf.get_j(gmf.mol, dms, hermi)
        return _shape_back(vj, dm)

    def get_k(mol=None, dm=None, hermi=0, omega=None, **_kw):
        dms = cp.asarray(dm).reshape(-1, nao, nao)
        route_stats['k_calls'] += 1
        route_stats['density_matrices'] += len(dms)
        if omega:
            vk = gmf.get_k(gmf.mol, dms, hermi, omega=omega)
        else:
            vk = gmf.get_k(gmf.mol, dms, hermi)
        return _shape_back(vk, dm)

    def get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None, **_kw):
        route_stats['jk_calls'] += 1
        if with_j and with_k and (not omega or isinstance(gmf, _DFHF)):
            # DF J and K share the same resident three-centre tensors.  Calling
            # the combined entry is materially cheaper than two independent
            # passes, especially for the block-RHS response action.
            dms = cp.asarray(dm).reshape(-1, nao, nao)
            vj, vk = gmf.get_jk(
                gmf.mol, dms, hermi, with_j=True, with_k=True,
                omega=omega,
            )
            route_stats['combined_jk_calls'] += 1
            route_stats['density_matrices'] += len(dms)
            return _shape_back(vj, dm), _shape_back(vk, dm)
        vj = get_j(mol, dm, hermi, omega) if with_j else None
        vk = get_k(mol, dm, hermi, omega) if with_k else None
        return vj, vk

    cpu_mf.get_j = get_j
    cpu_mf.get_k = get_k
    cpu_mf.get_jk = get_jk
    cpu_mf._nttda_jk_route_stats = route_stats
    cpu_mf._nttda_gpu_routed = True
    return cpu_mf


def _density_view_key(density):
    """Identity of one exact NumPy density view without hashing its values."""
    density = np.asarray(density)
    interface = density.__array_interface__
    return (
        interface['data'][0], density.shape, density.strides,
        density.dtype.str,
    )


class _DensityCombination:
    """Linear density combination retaining its source-factor recipe."""

    def __init__(self, densities, coefficients, matrix):
        self.densities = tuple(densities)
        self.coefficients = np.asarray(coefficients)
        self.matrix = matrix

    def __array__(self, dtype=None, copy=None):
        matrix = np.asarray(self.matrix, dtype=dtype)
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
    output = np.zeros(densities[0].shape, dtype=dtype)
    for coefficient, density in zip(coefficients, densities):
        if coefficient != 0.0:
            output += coefficient * density
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
        left = np.asarray(left)
        right = np.asarray(right)
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
    """Map forge J/K terms to bilinear pairs without mixing output slots."""
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
    '''Density-fitted GPU evaluation of a CPU ``_JKDerivativeLedger``.

    Same term mapping as :class:`LedgerBackend` (the DF per-atom kernels
    implement the same pair-energy derivative semantics), evaluated with
    the ``df.grad`` machinery so the auxiliary-basis response is included
    -- the result is the exact derivative of the DF energy surface.
    '''

    def __init__(self, gmf, compress_slots=None):
        from gpu4pyscf.df.df_jk import _DFHF

        assert isinstance(gmf, _DFHF)
        self._gmf = gmf
        self._opt = {}
        if compress_slots is None:
            compress_slots = os.environ.get(
                'NTTDA_COMPRESS_DF_LEDGER', '1',
            ) != '0'
        self.compress_slots = bool(compress_slots)
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
        gradients = {slot: np.zeros(shape) for slot in slots}
        groups = {}
        self.stats['calls'] += 1
        for operator in ('j', 'k'):
            for term in terms[operator]:
                gradients.setdefault(term.slot, np.zeros(shape))
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
            array = np.asarray(value)
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
            energies = _df_jk_energies_per_atom(
                opt, pairs,
                j_factor=j_factors, k_factor=k_factors, sum_results=False,
                stats_sink=layer_stats,
            )
            energies = cp.asnumpy(cp.asarray(energies))
            self.stats['integral_seconds'] += (
                time.perf_counter() - integral_started
            )
            self.stats['integral_layer_stats'].extend(layer_stats)
            for row, slot in zip(energies, slot_index):
                gradients[slot] += row[atoms]
        return gradients


class LedgerBackend:
    '''Batched GPU evaluation of a CPU ``_JKDerivativeLedger``.

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
        gradients = {slot: np.zeros(shape) for slot in slots}
        groups = {}
        self.stats['calls'] += 1
        for operator in ('j', 'k'):
            for term in terms[operator]:
                gradients.setdefault(term.slot, np.zeros(shape))
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
            energies = cp.asnumpy(cp.asarray(energies))
            for row, slot in zip(energies, slot_index):
                gradients[slot] += row[atoms]
        return gradients


def make_gpu_xc_backend(cpu_td, gmf):
    '''Build the geometry-fixed GPU XC backend where it is supported.'''
    xctype = gmf._numint._xc_type(gmf.xc)
    if xctype not in ('GGA', 'MGGA'):
        return None
    # The nobeta equal-spin common-Fock correction has a separate XC
    # functional derivative.  Keep MGGA entirely on the reference CPU path
    # until that correction is migrated instead of mixing backends silently.
    if (
            xctype == 'MGGA'
            and bool(getattr(cpu_td, 'nobeta', False))
            and not getattr(gmf, 'is_ensemble_rks', False)):
        return None
    from gpu4pyscf.grad.nttda_xc import GPUXCFrameBackend

    return GPUXCFrameBackend(gmf, cpu_td)


def make_gpu_response_cache(cpu_td, gmf, xc_backend=None):
    '''Forge ResponseCache with GPU-native GGA response and GPU J/K.'''
    _forge_grad, forge_roks, _forge_solver = _import_forge()
    if xc_backend is None:
        xc_backend = make_gpu_xc_backend(cpu_td, gmf)

    class _GPUResponseCache(forge_roks.ResponseCache):
        def __init__(self, tdobj):
            super().__init__(tdobj)
            self.frame_xc_backend = xc_backend
            cached_focks = getattr(
                tdobj, '_nttda_gpu_fock0_fockz', None,
            )
            if cached_focks is not None:
                self.extra['fock0_fockz'] = tuple(
                    cp.asnumpy(cp.asarray(value)) for value in cached_focks
                )
            elif xc_backend is not None:
                self.extra['fock0_fockz'] = (
                    xc_backend.spin_lowering_fock0_fockz()
                )

        def reference(self):
            reference = super().reference()
            if reference is not self._tdobj._scf:
                route_jk_to_gpu(reference, gmf)
            return reference

        def response(self, hermi):
            if getattr(gmf, 'is_ensemble_rks', False):
                if hermi not in self._responses:
                    from gpu4pyscf.scf import _response_functions

                    gpu_response = _response_functions._gen_rhf_response(
                        gmf, hermi=hermi,
                    )
                    self.stats['response_builds'] += 1

                    def counted_ensemble_response(density):
                        density = np.asarray(density)
                        width = (
                            1 if density.ndim == 2
                            else int(np.prod(density.shape[:-2]))
                        )
                        self.stats['response_calls'] += 1
                        self.stats['response_rhs'] += width
                        self.stats['response_batch_widths'].append(width)
                        return cp.asnumpy(gpu_response(cp.asarray(density)))

                    self._responses[hermi] = counted_ensemble_response
                return self._responses[hermi]
            if xc_backend is None:
                return super().response(hermi)
            if hermi not in self._responses:
                response = xc_backend.response(hermi)
                self.stats['response_builds'] += 1

                def counted_response(*args, **kwargs):
                    density = np.asarray(args[0])
                    if density.ndim <= 3:
                        width = 1
                    elif density.shape[0] == 2:
                        width = int(np.prod(density.shape[1:-2]))
                    else:
                        width = int(np.prod(density.shape[:-2]))
                    self.stats['response_calls'] += 1
                    self.stats['response_rhs'] += width
                    self.stats['response_batch_widths'].append(width)
                    return response(*args, **kwargs)

                self._responses[hermi] = counted_response
            return self._responses[hermi]

        def fxc_ref(self):
            if xc_backend is not None:
                raise RuntimeError(
                    'GPU-native GGA frame unexpectedly requested the CPU '
                    'spin-flip fxc reference'
                )
            return super().fxc_ref()

    return _GPUResponseCache(cpu_td)


def make_frame_cache():
    """Persistent AO Z-vector guesses for consecutive dynamics frames."""
    _forge_grad, forge_roks, _forge_solver = _import_forge()
    return forge_roks.ZVectorFrameCache()


def _resolve_input(td):
    '''Return the private CPU twin and GPU reference for a GPU NTTDA.'''
    if not _is_gpu_object(td):
        raise TypeError(
            'The hybrid derivative driver accepts only a GPU NTTDA object; '
            'CPU forge inputs are intentionally unsupported'
        )
    gmf = td._scf
    _validate_supported_reference(gmf)
    cpu_td = build_cpu_twin(td)
    route_jk_to_gpu(cpu_td._scf, gmf)
    return cpu_td, gmf


def _make_gradients_class():
    forge_grad, _forge_roks, _forge_solver = _import_forge()

    class Gradients(forge_grad.Gradients):
        '''GPU-accelerated NTTDA gradients (CPU formulas, GPU integrals).'''

        def __init__(self, td):
            from gpu4pyscf.df.df_jk import _DFHF

            cpu_td, gmf = _resolve_input(td)
            super().__init__(cpu_td)
            self._gmf = gmf
            self._with_df = isinstance(gmf, _DFHF)
            if self._with_df:
                self.nttda_jk_ledger_backend = DFLedgerBackend(gmf)
            else:
                self.nttda_jk_ledger_backend = LedgerBackend(gmf)

        def _analytic_components(self, xy, atmlst, response_cache=None):
            if response_cache is None:
                response_cache = getattr(
                    self, 'shared_response_cache', None,
                )
            if response_cache is None:
                response_cache = make_gpu_response_cache(
                    self.base, self._gmf,
                )
            return super()._analytic_components(
                xy, atmlst, response_cache=response_cache,
            )

        def grad_nuc(self, atmlst=None):
            if self._gmf.grids.coords is None:
                self._gmf.grids.build(sort_grids=True)
            if self._with_df:
                if getattr(self._gmf, 'is_ensemble_rks', False):
                    from gpu4pyscf.df.grad.rks import Gradients as DFGrad
                else:
                    from gpu4pyscf.df.grad.roks import Gradients as DFGrad
                driver = DFGrad(self._gmf)
            else:
                driver = self._gmf.nuc_grad_method()
            driver.verbose = 0
            value = np.asarray(driver.kernel())
            if atmlst is not None:
                value = value[list(atmlst)]
            return value

    return Gradients


_GRADIENTS_CLASS = None


def Gradients(td):
    '''Build the hybrid GPU NTTDA gradient driver for ``td``.

    ``td`` must be a converged ``gpu4pyscf.sftda.NTTDA``.  A private CPU
    formula twin is created internally without mutating a user-owned object.
    '''
    if not _is_gpu_object(td):
        raise TypeError(
            'The hybrid derivative driver accepts only a GPU NTTDA object'
        )
    global _GRADIENTS_CLASS
    if _GRADIENTS_CLASS is None:
        _GRADIENTS_CLASS = _make_gradients_class()
    return _GRADIENTS_CLASS(td)


Grad = Gradients


def compute_frame(td, active_state, nac_pairs=(), cphf_conv_tol=1e-10,
                  cphf_max_cycle=None, use_etfs=True, frame_cache=None):
    '''One dynamics frame: gradient of the active state plus NAC pairs.

    All geometry-fixed intermediates -- the spin-flip reference kernel,
    UKS response closures, spin Fock pair, F0/Fz, and the J/K derivative
    engines (VHFOpt / Int3c2eOpt) -- are built once and shared across the
    gradient and every NAC pair.

    ``frame_cache`` may be a forge ``ZVectorFrameCache`` owned by a dynamics
    driver.  It is updated only after every requested property succeeds.

    Returns ``{'grad': (natm, 3), 'nac': {(i, j): (natm, 3)}}``.  NAC
    entries are derivative couplings (the energy-scaled numerator divided
    by the state gap); ``use_etfs=False`` includes the moving-CSF term,
    while ``use_etfs=True`` retains the ETF/Hellmann--Feynman term.
    '''
    from gpu4pyscf.nac.nttda import NAC as make_nac

    frame_started = time.perf_counter()
    gpu_mem_start = cp.get_default_memory_pool().used_bytes()
    nstates = len(td.e)
    if (
            isinstance(active_state, (bool, np.bool_))
            or not isinstance(active_state, (int, np.integer))
            or not 1 <= active_state <= nstates):
        raise ValueError(
            'active_state must be in [1, %d]' % nstates,
        )
    active_state = int(active_state)
    parsed_pairs = []
    for pair in nac_pairs:
        if len(pair) != 2:
            raise ValueError('each NAC pair must contain two state indices')
        state_i, state_j = pair
        if any(
                isinstance(state, (bool, np.bool_))
                or not isinstance(state, (int, np.integer))
                for state in (state_i, state_j)):
            raise ValueError('NAC states must be integer root indices')
        parsed_pairs.append((int(state_i), int(state_j)))
    nac_pairs = tuple(parsed_pairs)
    for state_i, state_j in nac_pairs:
        if not 1 <= state_i <= nstates or not 1 <= state_j <= nstates:
            raise ValueError(
                'NAC states must be in [1, %d]' % nstates,
            )
        if state_i == state_j:
            raise ValueError('NAC pairs require two distinct states')

    driver_started = time.perf_counter()
    grad = Gradients(td)
    grad.verbose = 0
    grad.cphf_conv_tol = cphf_conv_tol
    grad.cphf_max_cycle = cphf_max_cycle
    xctype = grad.base._scf._numint._xc_type(grad.base._scf.xc)
    gpu_xc_backend = make_gpu_xc_backend(grad.base, grad._gmf)
    cache = make_gpu_response_cache(
        grad.base, grad._gmf, xc_backend=gpu_xc_backend,
    )
    grad.shared_response_cache = cache
    backend = grad.nttda_jk_ledger_backend

    delta = importlib.import_module(
        'pyscf.grad.nttda.delta_s_minus_one',
    )
    forge_roks = importlib.import_module('pyscf.grad.nttda.roks')
    atmlst = tuple(range(td.mol.natm))
    task_keys = [('grad', active_state)]
    nac = None
    nac_gradient = None
    if nac_pairs:
        nac = make_nac(td)
        nac.verbose = 0
        nac.cphf_conv_tol = cphf_conv_tol
        nac.cphf_max_cycle = cphf_max_cycle
        nac.use_etfs = use_etfs
        nac.shared_response_cache = cache
        nac.nttda_jk_ledger_backend = backend
        nac_gradient = nac._gradient_driver(verbose=0)
    driver_seconds = time.perf_counter() - driver_started

    xc_started = time.perf_counter()
    grad_xc_terms = None
    nac_xc_terms = [None] * len(nac_pairs)
    if xctype in ('GGA', 'MGGA') and gpu_xc_backend is not None:
        grad_channel, spaces, grad_pz = delta.gradient_xc_request(
            grad.base, grad.base.xy[active_state - 1],
        )
        channels = [grad_channel]
        pz_batch = [grad_pz]
        for state_i, state_j in nac_pairs:
            request = delta.cross_xc_request(
                nac.base,
                nac.base.xy[state_i - 1],
                nac.base.xy[state_j - 1],
            )
            channels.extend(request[0])
            pz_batch.append(request[2])
        response_builder = getattr(
            gpu_xc_backend, xctype.lower() + '_response_terms_batch',
        )
        fockz_builder = getattr(
            gpu_xc_backend, xctype.lower() + '_fockz_terms_batch',
        )
        response_xc = response_builder(
            grad, grad.base, channels, atmlst=atmlst,
        )
        fockz_xc = fockz_builder(
            grad, grad.base, spaces, pz_batch, atmlst=atmlst,
        )
        grad_xc_terms = (response_xc[0], fockz_xc[0])
        for index in range(len(nac_pairs)):
            offset = 1 + 2 * index
            nac_xc_terms[index] = (
                response_xc[offset],
                response_xc[offset + 1],
                fockz_xc[index + 1],
            )
    xc_seconds = time.perf_counter() - xc_started

    prepare_started = time.perf_counter()
    prepared = [delta.prepare_grad_elec(
        grad,
        grad.base,
        grad.base.xy[active_state - 1],
        atmlst=atmlst,
        tolerance=cphf_conv_tol,
        max_cycle=cphf_max_cycle,
        cache=cache,
        xc_terms=grad_xc_terms,
    )]
    for index, (state_i, state_j) in enumerate(nac_pairs):
        task_keys.append(('nac', state_i, state_j, bool(use_etfs)))
        prepared.append(delta.prepare_grad_elec_cross(
            nac_gradient,
            nac.base,
            nac.base.xy[state_i - 1],
            nac.base.xy[state_j - 1],
            atmlst=atmlst,
            tolerance=cphf_conv_tol,
            max_cycle=cphf_max_cycle,
            cache=cache,
            xc_terms=nac_xc_terms[index],
        ))
    prepare_seconds = time.perf_counter() - prepare_started

    pairs = forge_roks.canonical_pairs(grad.base, compact=True)
    initial = None
    cache_hits = 0
    if frame_cache is not None:
        initial, cache_hits = frame_cache.project(
            grad.base, pairs, task_keys,
        )
    finish_started = time.perf_counter()
    components = forge_roks.finish_prepared_gradients(
        prepared, initial=initial,
    )
    finish_seconds = time.perf_counter() - finish_started
    grad.nttda_details = components[0]
    nuclear_started = time.perf_counter()
    result = {
        'grad': np.asarray(grad.grad_nuc()) + components[0].total,
        'nac': {},
    }
    nuclear_seconds = time.perf_counter() - nuclear_started

    def record_stats(nac_postprocess_seconds=0.0):
        '''Collect per-frame stats from all backends into ``td._nttda_frame_stats``.

        Called once at the end of ``compute_frame`` (after all gradient/NAC
        work succeeds).  The stats dict is read by the FSSH profile script
        and by ``frame_cache.last_stats``.  Fields:

        - ``timings`` — phase-level wall seconds (drivers, xc_batch, prepare,
          zvector_and_derivatives, nuclear, nac_postprocess, total).
        - ``nttda_solver`` — Davidson vind/Davidson stats from
          ``NTTDA._nttda_solver_stats``.
        - ``scf`` — SCF cycle count and convergence.
        - ``gpu_memory`` — cupy memory pool usage at frame start/end.
        - ``runtime_params`` — snapshot of ``nttda_params.PARAMS``.
        - ``jk_backend`` — DFLedgerBackend stats (pair counts, compression,
          layer timing, lower-bound gate counters).
        - ``xc_backend`` / ``response_cache`` / ``response_jk`` — backend stats.
        '''
        scf_obj = getattr(td, '_scf', None)
        scf_stats = {
            'cycles': int(getattr(scf_obj, 'cycles', 0)),
            'converged': bool(getattr(scf_obj, 'converged', False)),
        } if scf_obj is not None else {}
        stats = {
            'active_state': active_state,
            'nac_pairs': len(nac_pairs),
            'zvector_batch_width': len(prepared),
            'zvector_cache_hits': int(np.count_nonzero(cache_hits)),
            'xc_type': xctype,
            'xc_response_channels': (
                1 + 2 * len(nac_pairs)
                if gpu_xc_backend is not None else 0
            ),
            'xc_fockz_tasks': (
                1 + len(nac_pairs)
                if gpu_xc_backend is not None else 0
            ),
            'timings': {
                'drivers': driver_seconds,
                'xc_batch': xc_seconds,
                'prepare': prepare_seconds,
                'zvector_and_derivatives': finish_seconds,
                'nuclear': nuclear_seconds,
                'nac_postprocess': nac_postprocess_seconds,
                'total': time.perf_counter() - frame_started,
            },
            'nttda_solver': dict(getattr(td, '_nttda_solver_stats', {})),
            'scf': scf_stats,
            'gpu_memory': {
                'start_bytes': int(gpu_mem_start),
                'end_bytes': int(cp.get_default_memory_pool().used_bytes()),
            },
            'runtime_params': dict(_NTTDA_PARAMS),
            'jk_backend': dict(getattr(backend, 'stats', {})),
            'xc_backend': dict(getattr(gpu_xc_backend, 'stats', {})),
            'response_cache': dict(getattr(cache, 'stats', {})),
            'response_jk': dict(getattr(
                cache.reference(), '_nttda_jk_route_stats', {},
            )),
        }
        td._nttda_frame_stats = stats
        if frame_cache is not None:
            frame_cache.last_stats = stats

    if not nac_pairs:
        if frame_cache is not None:
            frame_cache.commit(
                grad.base,
                pairs,
                task_keys,
                np.asarray([components[0].zvector]),
            )
        record_stats()
        return result

    forge_nac = importlib.import_module('pyscf.nac.nttda')
    csf_started = time.perf_counter()
    for (state_i, state_j), item in zip(nac_pairs, components[1:]):
        gap = float(nac.base.e[state_j - 1] - nac.base.e[state_i - 1])
        if abs(gap) < nac.gap_tol:
            raise ZeroDivisionError(
                'NTTDA state gap %.6e is below gap_tol %.6e'
                % (gap, nac.gap_tol)
            )
        numerator = np.asarray(item.total)
        if not use_etfs:
            csf = forge_nac.nac_csf_components(
                nac,
                nac.base.xy[state_i - 1],
                nac.base.xy[state_j - 1],
                cache=cache,
            )
            numerator = numerator + gap * np.asarray(csf.total)
        result['nac'][(state_i, state_j)] = np.real_if_close(
            numerator / gap,
        )
    if frame_cache is not None:
        frame_cache.commit(
            grad.base,
            pairs,
            task_keys,
            np.asarray([item.zvector for item in components]),
        )
    record_stats(time.perf_counter() - csf_started)
    return result
