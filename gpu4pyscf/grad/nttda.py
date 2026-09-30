"""Analytic nuclear gradients for :mod:`gpu4pyscf.sftda.nttda`."""

import numpy as np

from pyscf import lib
from gpu4pyscf import dft
from gpu4pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger
from gpu4pyscf.sftda.ensemble_rks import EnsembleRKS
from gpu4pyscf.sftda.nttda import NTTDA
from gpu4pyscf.sftda import nttda_methods as methods

from ._nttda.reference import rebuild_reference
from ._nttda.response import ResponseCache
import cupy as cp


def _normalized_amplitude(xy):
    vector = cp.asnumpy(cp.asarray(xy[0])).ravel()
    return vector / np.linalg.norm(vector)


def _copy_td_settings(source, target):
    for name in ('deltaS', 'nobeta', 'nstates', 'conv_tol', 'lindep', 'max_cycle', 'max_memory'):
        setattr(target, name, getattr(source, name))
    methods.copy_method(source, target)
    target.verbose = 0
    return target


def _reference_energy(mf):
    """Return the reference zero selected by ``mf`` (legacy ``e_tot`` fallback)."""
    selector = getattr(mf, 'reference_energy', None)
    if selector is None:
        return float(mf.e_tot)
    return float(selector())


class Gradients(rhf_grad.GradientsBase):
    """Analytic NTTDA gradients for ``deltaS=-1`` and ``deltaS=0``."""

    _keys = rhf_grad.GradientsBase._keys | {
        'state',
        'method',
        'step',
        'fixed_grid',
        'root_overlap_tol',
        'cphf_conv_tol',
        'cphf_max_cycle',
        'reference_z_solver_diagnostics',
    }

    def __init__(self, tdobj, *, context=None):
        super().__init__(tdobj)
        self.state = 1
        self.method = 'analytic'
        self.step = 1e-3
        self.fixed_grid = isinstance(tdobj._scf, dft.KohnShamDFT) and not isinstance(tdobj._scf, EnsembleRKS)
        self.root_overlap_tol = 0.5
        self.cphf_conv_tol = 1e-12
        self.cphf_max_cycle = None
        self.nttda_details = None
        self.reference_z_solver_diagnostics = None
        self._context = ResponseCache(tdobj) if context is None else context
        self._gmf = tdobj._scf
        self._with_df = self._context.with_df

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** NTTDA nuclear gradients ********')
        log.info('state = %d', self.state)
        log.info('deltaS = %d', self.base.deltaS)
        log.info('NTTDA method = %s', methods.resolve_method(self.base).id)
        log.info('method = %s', self.method)
        log.info('fixed_grid = %s', self.fixed_grid)
        if self.method == 'finite_diff':
            log.info('finite-difference step = %.6g Bohr', self.step)
        return self

    def grad_nuc(self, atmlst=None):
        return self.reference_gradient(atmlst)

    def reference_gradient(self, atmlst=None):
        self._context.validate()
        if self._gmf.grids.coords is None:
            self._gmf.grids.build(sort_grids=True)
        if self._with_df:
            # Selected-reference objects (non-stationary reference energy)
            # must use their own gradient driver.  EnsembleROKS inherits
            # is_ensemble_rks=True, so that flag alone would silently
            # select the ordinary DF-RKS gradient and bypass the reference
            # response.
            selected_reference = self._context.method.reference_kind == 'ensemble_roks'
            if selected_reference:
                driver = self._gmf.nuc_grad_method()
                driver.nttda_xc_backend = self._context.xc_backend
            elif self._context.method.reference_kind == 'ensemble_rks':
                from gpu4pyscf.df.grad.rks import Gradients as DFGrad

                driver = DFGrad(self._gmf)
            else:
                from gpu4pyscf.df.grad.roks import Gradients as DFGrad

                driver = DFGrad(self._gmf)
        else:
            # The selected-ROKS reference binds nuc_grad_method() to the
            # T06 gpu4pyscf.grad.ensemble_roks.ReferenceGradients driver;
            # legacy ROKS/EnsembleRKS references keep their own gradient.
            driver = self._gmf.nuc_grad_method()
        driver.verbose = 0
        value = np.asarray(driver.kernel())
        diagnostics = getattr(driver, 'z_solver_diagnostics', None)
        self.reference_z_solver_diagnostics = diagnostics.as_dict() if diagnostics is not None else None
        self.reference_z_df_cache_stats = getattr(driver, 'z_df_cache_stats', None)
        self.reference_zb_backend = getattr(driver, 'zb_backend', None)
        self.reference_zb_skeleton_stats = getattr(
            driver,
            'z_b_skeleton_stats',
            None,
        )
        self.reference_gradient_calls = (
            getattr(
                self,
                'reference_gradient_calls',
                0,
            )
            + 1
        )
        if atmlst is not None:
            value = value[list(atmlst)]
        return value

    def _analytic_components(self, xy, atmlst, response_cache=None):
        self._context.validate()
        if response_cache is None:
            response_cache = self._context
        method = methods.bind_method(self.base, derivative=True)
        return methods.gradient_components(method, self, xy, atmlst, response_cache)

    def grad_elec(self, xy, atmlst=None, response_cache=None):
        """Return the analytic excitation-energy derivative ``d omega/dR``."""
        if atmlst is None:
            atmlst = range(self.mol.natm)
        components = self._analytic_components(
            xy,
            tuple(atmlst),
            response_cache=response_cache,
        )
        self.nttda_details = components
        return cp.asnumpy(cp.asarray(components.total))

    def _energy_at(self, coords, reference_amplitude):
        mol = self.mol.copy()
        mol.set_geom_(coords, unit='Bohr')
        mf = rebuild_reference(self.base._scf, mol, self.fixed_grid)
        mf.kernel(dm0=self.base._scf.make_rdm1())
        if not mf.converged:
            raise RuntimeError('displaced NTTDA reference did not converge')
        tdobj = _copy_td_settings(self.base, NTTDA(mf))
        tdobj.kernel()
        overlaps = np.asarray([abs(np.vdot(reference_amplitude, _normalized_amplitude(xy))) for xy in tdobj.xy])
        root = int(np.argmax(overlaps))
        if overlaps[root] < self.root_overlap_tol:
            raise RuntimeError(
                'NTTDA state tracking overlap %.6f is below %.6f' % (overlaps[root], self.root_overlap_tol)
            )
        return tdobj.reference_energy() + tdobj.e[root]

    def _finite_difference(self, atmlst):
        coords0 = self.mol.atom_coords()
        reference_amplitude = _normalized_amplitude(
            self.base.xy[self.state - 1],
        )
        result = np.zeros((len(atmlst), 3))
        for index, atom in enumerate(atmlst):
            for xyz in range(3):
                coords_plus = coords0.copy()
                coords_minus = coords0.copy()
                coords_plus[atom, xyz] += self.step
                coords_minus[atom, xyz] -= self.step
                energy_plus = self._energy_at(
                    coords_plus,
                    reference_amplitude,
                )
                energy_minus = self._energy_at(
                    coords_minus,
                    reference_amplitude,
                )
                result[index, xyz] = (energy_plus - energy_minus) / (2.0 * self.step)
        return result

    def kernel(self, state=None, atmlst=None, method=None, step=None):
        """Return ``d(E_reference + omega_state)/dR`` in Eh/Bohr."""
        self._context.validate()
        if state is not None:
            self.state = state
        if method is not None:
            self.method = method
        if step is not None:
            self.step = step
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst
        if atmlst is None:
            atmlst = range(self.mol.natm)
        atmlst = tuple(atmlst)

        if self.state == 0:
            return self.grad_nuc(atmlst=atmlst)
        if self.base.xy is None:
            self.base.run()
        if not 1 <= self.state <= len(self.base.xy):
            raise ValueError('state must be in [1, %d]' % len(self.base.xy))
        if self.verbose >= logger.INFO:
            self.dump_flags()

        if self.method == 'analytic':
            excitation = self.grad_elec(
                self.base.xy[self.state - 1],
                atmlst=atmlst,
            )
            result = self.grad_nuc(atmlst=atmlst) + excitation
        elif self.method == 'finite_diff':
            result = self._finite_difference(atmlst)
        else:
            raise ValueError('unknown NTTDA gradient method %s' % self.method)
        self.de = result
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name='grad')


Grad = Gradients

from ._nttda.frame import compute_frame, make_frame_cache

__all__ = ['Grad', 'Gradients', 'compute_frame', 'make_frame_cache']


def _gradients_from_context(td, context):
    return Gradients(td, context=context)
