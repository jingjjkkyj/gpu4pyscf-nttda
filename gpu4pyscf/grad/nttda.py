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

import os

import numpy as np
import cupy as cp

from gpu4pyscf.grad.tdrhf import _jk_energies_per_atom
from gpu4pyscf.scf.jk import _VHFOpt


def _import_forge():
    '''Import the CPU forge NTTDA packages, injecting paths if needed.'''
    try:
        import pyscf.grad.nttda  # noqa: F401
        import pyscf.nac.nttda  # noqa: F401
        import pyscf.sftda.nttda  # noqa: F401
    except ImportError:
        root = os.environ.get('NTTDA_FORGE_PATH')
        if not root:
            raise ImportError(
                'The CPU pyscf-forge NTTDA package is required for the '
                'hybrid GPU driver.  Load it via PYSCF_EXT_PATH or set '
                'NTTDA_FORGE_PATH to the forge checkout root.'
            )
        import pyscf

        path = os.path.join(root, 'pyscf')
        if path not in pyscf.__path__:
            pyscf.__path__.insert(0, path)
        import pyscf.grad
        import pyscf.nac
        import pyscf.sftda

        for package, sub in (
                (pyscf.grad, os.path.join(path, 'grad')),
                (pyscf.nac, os.path.join(path, 'nac')),
                (pyscf.sftda, os.path.join(path, 'sftda'))):
            if sub not in package.__path__:
                package.__path__.insert(0, sub)
    from pyscf.grad import nttda as forge_grad
    from pyscf.grad.nttda import roks as forge_roks
    from pyscf.sftda import nttda as forge_solver

    return forge_grad, forge_roks, forge_solver


def _is_gpu_object(obj):
    return type(obj).__module__.startswith('gpu4pyscf')


def build_cpu_twin(gpu_td):
    '''CPU forge NTTDA twin of a converged GPU NTTDA calculation.

    Orbitals, occupations, energies, amplitudes, and the (sorted) grid
    are copied so the twin never runs SCF, Davidson, or grid building of
    its own.
    '''
    from pyscf import dft

    _forge_grad, _forge_roks, forge_solver = _import_forge()
    gmf = gpu_td._scf
    mol = gmf.mol
    cpu_mf = dft.ROKS(mol)
    cpu_mf.xc = gmf.xc
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
    '''Return (cpu_td, gmf) for a GPU NTTDA or an already-CPU NTTDA.'''
    if _is_gpu_object(td):
        cpu_td = build_cpu_twin(td)
        gmf = td._scf
    else:
        from gpu4pyscf.dft import roks as gpu_roks

        _import_forge()
        cpu_td = td
        cpu_mf = td._scf
        gmf = gpu_roks.ROKS(cpu_mf.mol, xc=cpu_mf.xc)
        gmf.verbose = 0
        gmf.mo_coeff = cp.asarray(cpu_mf.mo_coeff)
        gmf.mo_occ = cp.asarray(cpu_mf.mo_occ)
        gmf.mo_energy = cp.asarray(cpu_mf.mo_energy)
        gmf.converged = True
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

    ``td`` may be a converged ``gpu4pyscf.sftda.NTTDA`` (a CPU twin is
    created internally) or a CPU forge NTTDA (a GPU integral backend is
    attached).
    '''
    global _GRADIENTS_CLASS
    if _GRADIENTS_CLASS is None:
        _GRADIENTS_CLASS = _make_gradients_class()
    return _GRADIENTS_CLASS(td)


Grad = Gradients
