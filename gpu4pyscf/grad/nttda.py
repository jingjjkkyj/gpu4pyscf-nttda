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

'''Hybrid GPU driver for NTTDA ``deltaS = -1`` excited-state gradients.

Orchestration and XC quadrature reuse the validated CPU forge
implementation (pyscf-forge NTTDA, importable side by side with
gpu4pyscf); every four-center integral task runs on GPU:

- response J/K builds (M matrix, Fock builds, Z-vector iterations) are
  routed from the CPU twin objects to a GPU ROKS/UKS backend;
- the J/K derivative ledger is evaluated with the batched per-atom
  ``_jk_energies_per_atom`` kernels through the ledger backend seam
  (empirically calibrated mapping, machine-precision on J/K and
  range-separated variants: ``cpu_J = 2*gpu(L, R)``,
  ``cpu_K = -2*[gpu(L, R^T) + gpu(L^T, R)]``);
- the ground-state ROKS gradient uses the native GPU implementation.

XC derivative quadrature (the forge ``xc`` backend) stays on CPU in this
version; for hybrid functionals the four-center work dominates.
'''

import importlib
import inspect
import os
import sys

import numpy as np
import cupy as cp

from gpu4pyscf.grad.tdrhf import _jk_energies_per_atom
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
    from gpu4pyscf.sftda.nttda import _get_j_range_separated

    nao = cpu_mf.mol.nao_nr()

    def _shape_back(value, dm):
        value = cp.asnumpy(cp.asarray(value))
        return value.reshape(np.asarray(dm).shape)

    def get_j(mol=None, dm=None, hermi=0, omega=None, **_kw):
        dms = cp.asarray(dm).reshape(-1, nao, nao)
        if omega:
            # _get_j_range_separated dispatches internally: per-omega
            # cderi for DF references, erf-kernel VHFOpt otherwise.
            vj = _get_j_range_separated(gmf, dms, hermi, omega)
        else:
            vj = gmf.get_j(gmf.mol, dms, hermi)
        return _shape_back(vj, dm)

    def get_k(mol=None, dm=None, hermi=0, omega=None, **_kw):
        dms = cp.asarray(dm).reshape(-1, nao, nao)
        if omega:
            vk = gmf.get_k(gmf.mol, dms, hermi, omega=omega)
        else:
            vk = gmf.get_k(gmf.mol, dms, hermi)
        return _shape_back(vk, dm)

    def get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None, **_kw):
        vj = get_j(mol, dm, hermi, omega) if with_j else None
        vk = get_k(mol, dm, hermi, omega) if with_k else None
        return vj, vk

    cpu_mf.get_j = get_j
    cpu_mf.get_k = get_k
    cpu_mf.get_jk = get_jk
    cpu_mf._nttda_gpu_routed = True
    return cpu_mf


class DFLedgerBackend:
    '''Density-fitted GPU evaluation of a CPU ``_JKDerivativeLedger``.

    Same term mapping as :class:`LedgerBackend` (the DF per-atom kernels
    implement the same pair-energy derivative semantics), evaluated with
    the ``df.grad`` machinery so the auxiliary-basis response is included
    -- the result is the exact derivative of the DF energy surface.
    '''

    def __init__(self, gmf):
        from gpu4pyscf.df.df_jk import _DFHF

        assert isinstance(gmf, _DFHF)
        self._gmf = gmf
        self._opt = {}

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
        from gpu4pyscf.df.grad.tdrhf import (
            _jk_energies_per_atom as _df_jk_energies_per_atom,
        )

        atoms = list(atoms)
        shape = (len(atoms), 3)
        gradients = {slot: np.zeros(shape) for slot in slots}
        groups = {}
        for operator in ('j', 'k'):
            for term in terms[operator]:
                gradients.setdefault(term.slot, np.zeros(shape))
                groups.setdefault(float(term.omega or 0.0), []).append(
                    (operator, term),
                )
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
            energies = _df_jk_energies_per_atom(
                self._get_opt(omega), pairs,
                j_factor=j_factors, k_factor=k_factors, sum_results=False,
            )
            energies = cp.asnumpy(cp.asarray(energies))
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
        for operator in ('j', 'k'):
            for term in terms[operator]:
                gradients.setdefault(term.slot, np.zeros(shape))
                groups.setdefault(float(term.omega or 0.0), []).append(
                    (operator, term),
                )
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


def make_gpu_response_cache(cpu_td, gmf):
    '''Forge ResponseCache whose UKS response view is GPU-JK-routed.'''
    _forge_grad, forge_roks, _forge_solver = _import_forge()

    class _GPUResponseCache(forge_roks.ResponseCache):
        def reference(self):
            reference = super().reference()
            if reference is not self._tdobj._scf:
                route_jk_to_gpu(reference, gmf)
            return reference

    return _GPUResponseCache(cpu_td)


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
                from gpu4pyscf.df.grad.roks import Gradients as DFRoksGrad

                driver = DFRoksGrad(self._gmf)
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
                  cphf_max_cycle=None, use_etfs=True):
    '''One dynamics frame: gradient of the active state plus NAC pairs.

    All geometry-fixed intermediates -- the spin-flip reference kernel,
    UKS response closures, spin Fock pair, F0/Fz, and the J/K derivative
    engines (VHFOpt / Int3c2eOpt) -- are built once and shared across the
    gradient and every NAC pair.

    Returns ``{'grad': (natm, 3), 'nac': {(i, j): (natm, 3)}}``.  NAC
    entries are derivative couplings (the energy-scaled numerator divided
    by the state gap); ``use_etfs=False`` includes the moving-CSF term,
    while ``use_etfs=True`` retains the ETF/Hellmann--Feynman term.
    '''
    from gpu4pyscf.nac.nttda import NAC as make_nac

    nstates = len(td.e)
    if not 1 <= active_state <= nstates:
        raise ValueError(
            'active_state must be in [1, %d]' % nstates,
        )
    nac_pairs = tuple((int(i), int(j)) for i, j in nac_pairs)
    for state_i, state_j in nac_pairs:
        if not 1 <= state_i <= nstates or not 1 <= state_j <= nstates:
            raise ValueError(
                'NAC states must be in [1, %d]' % nstates,
            )
        if state_i == state_j:
            raise ValueError('NAC pairs require two distinct states')

    grad = Gradients(td)
    grad.verbose = 0
    grad.cphf_conv_tol = cphf_conv_tol
    grad.cphf_max_cycle = cphf_max_cycle
    cache = make_gpu_response_cache(grad.base, grad._gmf)
    grad.shared_response_cache = cache
    backend = grad.nttda_jk_ledger_backend

    result = {'grad': np.asarray(grad.kernel(state=active_state)),
              'nac': {}}
    if nac_pairs:
        nac = make_nac(td)
        nac.verbose = 0
        nac.cphf_conv_tol = cphf_conv_tol
        nac.cphf_max_cycle = cphf_max_cycle
        nac.use_etfs = use_etfs
        nac.shared_response_cache = cache
        nac.nttda_jk_ledger_backend = backend
        for state_i, state_j in nac_pairs:
            # ``td.xy`` already carries one globally aligned phase per root.
            # The forge NAC driver's two-slot history is intended for one
            # fixed pair across geometries; reusing it across different pairs
            # would compare unrelated roots and can introduce a spurious sign.
            nac.reset_phase()
            value = nac.kernel(
                state_I=state_i, state_J=state_j, ediff=True,
            )
            result['nac'][(state_i, state_j)] = np.asarray(value)
    return result
