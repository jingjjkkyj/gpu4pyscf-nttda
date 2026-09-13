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

"""Verified CPU forge imports, private twins and GPU integral routing."""
import importlib
import inspect
import os
import sys
import numpy as np
import cupy as cp

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
    response = importlib.import_module('pyscf.grad.nttda.response')
    if not hasattr(response, 'ResponseCache'):
        raise ImportError(
            'Loaded CPU forge NTTDA lacks the ResponseCache accelerator seam'
        )
    frame_stage = getattr(response, '_stage_frame_jk', None)
    if (
            frame_stage is None
            or 'frame_xc_backend' not in frame_stage.__code__.co_consts):
        raise ImportError(
            'Loaded CPU forge NTTDA lacks the GPU frame-XC backend seam'
        )
    delta = importlib.import_module(
        'pyscf.grad.nttda.derivative_jk',
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
    if requested_root:
        for name in ('pyscf.grad.nttda.response', 'pyscf.grad.nttda.derivative_jk',
                     'pyscf.sftda.nttda_methods'):
            _assert_module_under_root(importlib.import_module(name), requested_root, name)
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

def build_cpu_twin(gpu_td, forge=None):
    '''CPU forge NTTDA twin of a converged GPU NTTDA calculation.

    Orbitals, occupations, energies, amplitudes, and the (sorted) grid are
    copied so the twin never runs SCF, Davidson, or grid building of its own.
    The concrete EnsembleRKS type is preserved because the CPU formula layer
    uses that marker to choose the occupation-difference orbital Hessian and
    the equal-spin response probes.
    '''
    from pyscf import dft

    _forge_grad, _forge_roks, forge_solver = _import_forge() if forge is None else forge
    from pyscf.sftda import nttda_methods as methods
    method = methods.bind_method(gpu_td, derivative=True)
    gmf = gpu_td._scf
    _validate_supported_reference(gmf)
    mol = gmf.mol
    if (
            method.reference_kind == 'ensemble_roks'):
        from pyscf.sftda import EnsembleROKS

        cpu_mf = EnsembleROKS(mol, nopen=gmf.nopen)
    elif method.reference_kind == 'ensemble_rks':
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
    methods.copy_method(gpu_td, cpu_td)
    return cpu_td

def route_jk_to_gpu(cpu_mf, gmf, *, array_backend=None):
    '''Route a CPU SCF object's J/K builds to a GPU backend (in place).

    Instance-level overrides so the CPU response/Fock machinery calls the
    GPU integral engines transparently (numpy in / numpy out).  The
    range-separated J uses the erf-kernel VHFOpt path because the stock
    GPU J engine ignores ``omega``.
    '''
    xp = cp if array_backend is None else array_backend
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
        value = xp.asnumpy(xp.asarray(value))
        return value.reshape(np.asarray(dm).shape)

    def get_j(mol=None, dm=None, hermi=0, omega=None, **_kw):
        dms = xp.asarray(dm).reshape(-1, nao, nao)
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
        dms = xp.asarray(dm).reshape(-1, nao, nao)
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
            dms = xp.asarray(dm).reshape(-1, nao, nao)
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

def rebuild_reference(gmf, mol, fixed_grid=False):
    '''Rebuild a displaced reference of the source's own concrete type.

    The displaced reference used by finite differences must use the same
    integral model as the differentiated calculation, so the concrete
    reference type (ROKS, EnsembleRKS or EnsembleROKS) must be preserved and
    the model settings carried over.  When ``gmf`` is density fitted the
    auxiliary basis is carried over so the displaced reference energy is
    density fitted too; the quadrature is frozen when ``fixed_grid`` is set.
    '''
    from pyscf import dft

    from gpu4pyscf.dft import roks as gpu_roks
    from gpu4pyscf.sftda import EnsembleRKS as GpuEnsembleRKS
    from gpu4pyscf.sftda import EnsembleROKS as GpuEnsembleROKS

    if isinstance(gmf, GpuEnsembleROKS):
        reference = GpuEnsembleROKS(
            mol, xc=gmf.xc, nopen=getattr(gmf, 'nopen', None))
    elif isinstance(gmf, GpuEnsembleRKS):
        reference = GpuEnsembleRKS(
            mol, xc=gmf.xc, nopen=getattr(gmf, 'nopen', None))
    elif isinstance(gmf, dft.KohnShamDFT):
        reference = gpu_roks.ROKS(mol, xc=gmf.xc)
    else:
        reference = gmf.__class__(mol)
    with_df = getattr(gmf, 'with_df', None)
    if with_df is not None:
        reference = reference.density_fit(
            auxbasis=getattr(with_df, 'auxbasis', None),
        )
    for name in (
            'conv_tol', 'conv_tol_grad', 'max_cycle', 'max_memory',
            'level_shift', 'damp', 'direct_scf_tol', 'small_rho_cutoff',
            'nlc', 'disp', 'disp_with_3body'):
        if hasattr(gmf, name):
            setattr(reference, name, getattr(gmf, name))
    if getattr(gmf, 'omega', None) is not None:
        reference.omega = gmf.omega
    reference.grids.level = gmf.grids.level
    reference.grids.prune = gmf.grids.prune
    if fixed_grid and gmf.grids.coords is not None:
        reference.grids.coords = cp.asarray(gmf.grids.coords).copy()
        reference.grids.weights = cp.asarray(gmf.grids.weights).copy()
        reference.grids.non0tab = None
    reference.verbose = 0
    return reference

def _host_normalized_amplitude(xy):
    '''Normalize a device or host ``(x, y)`` amplitude on the host.'''
    part = xy[0] if isinstance(xy, (list, tuple)) else xy
    vector = np.asarray(cp.asnumpy(cp.asarray(part))).ravel()
    return vector / np.linalg.norm(vector)

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


def import_methods():
    _import_forge()
    return importlib.import_module('pyscf.sftda.nttda_methods')
