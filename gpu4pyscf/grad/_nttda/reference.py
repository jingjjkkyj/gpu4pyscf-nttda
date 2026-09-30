"""Reference validation and displaced GPU references."""

import numpy as np
import cupy as cp


def _validate_supported_reference(gmf):
    """Fail before mixing energy and derivative models we cannot preserve."""
    if bool(getattr(gmf, 'only_dfj', False)):
        raise NotImplementedError(
            'NTTDA gradients/NAC/FSSH do not yet support only_dfj because '
            'the conventional-K derivative ledger is not implemented'
        )
    for marker, attribute, label in (('_Solvation', 'with_solvent', 'solvent'), ('_QMMM', 'mm_mol', 'QM/MM')):
        wrapped = getattr(gmf, attribute, None) is not None
        istype = getattr(gmf, 'istype', None)
        if callable(istype):
            wrapped = wrapped or bool(istype(marker))
        if wrapped:
            raise NotImplementedError(f'NTTDA gradients/NAC/FSSH do not support {label} wrappers')
    if getattr(gmf, 'disp', None) not in (None, False, ''):
        raise NotImplementedError('NTTDA gradients/NAC/FSSH do not support dispersion corrections')
    has_nlc = getattr(gmf, 'nlc', None) not in (None, False, '')
    do_nlc = getattr(gmf, 'do_nlc', None)
    if callable(do_nlc):
        has_nlc = has_nlc or bool(do_nlc())
    if has_nlc:
        raise NotImplementedError('NTTDA gradients/NAC/FSSH do not support nonlocal correlation')
    numint = getattr(gmf, '_numint', None)
    if numint is not None:
        from gpu4pyscf.dft import numint as gpu_numint

        if type(numint) is not gpu_numint.NumInt:
            raise NotImplementedError('NTTDA gradients/NAC/FSSH require the standard GPU NumInt')


def rebuild_reference(gmf, mol, fixed_grid=False):
    """Rebuild a displaced reference of the source's own concrete type.

    The displaced reference used by finite differences must use the same
    integral model as the differentiated calculation, so the concrete
    reference type (ROKS, EnsembleRKS or EnsembleROKS) must be preserved and
    the model settings carried over.  When ``gmf`` is density fitted the
    auxiliary basis is carried over so the displaced reference energy is
    density fitted too; the quadrature is frozen when ``fixed_grid`` is set.
    """
    from gpu4pyscf import dft

    from gpu4pyscf.dft import roks as gpu_roks
    from gpu4pyscf.sftda import EnsembleRKS as GpuEnsembleRKS
    from gpu4pyscf.sftda import EnsembleROKS as GpuEnsembleROKS

    if isinstance(gmf, GpuEnsembleROKS):
        reference = GpuEnsembleROKS(mol, xc=gmf.xc, nopen=getattr(gmf, 'nopen', None))
    elif isinstance(gmf, GpuEnsembleRKS):
        reference = GpuEnsembleRKS(mol, xc=gmf.xc, nopen=getattr(gmf, 'nopen', None))
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
        'conv_tol',
        'conv_tol_grad',
        'max_cycle',
        'max_memory',
        'level_shift',
        'damp',
        'direct_scf_tol',
        'small_rho_cutoff',
        'nlc',
        'disp',
        'disp_with_3body',
    ):
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
    """Normalize a device or host ``(x, y)`` amplitude on the host."""
    part = xy[0] if isinstance(xy, (list, tuple)) else xy
    vector = np.asarray(cp.asnumpy(cp.asarray(part))).ravel()
    return vector / np.linalg.norm(vector)
