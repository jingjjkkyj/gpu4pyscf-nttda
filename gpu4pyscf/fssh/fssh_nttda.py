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

'''FSSH adapter for GPU NTTDA (deltaS = -1) surfaces.

Per electronic step: GPU ROKS SCF (previous-density warm start) -> GPU
NTTDA -> energy-ordered roots with same-index phase alignment against the
previous frame -> one joint frame evaluation (shared response cache and
derivative engines) for the active-state force and all NAC pairs.  A
character-following Hungarian assignment remains available as the explicit
``state_ordering='overlap'`` research mode.

``states`` are 1-based NTTDA root indices (the zero-energy reference
root is already filtered by the solver).

Every completed trajectory frame also atomically replaces
``<trajectory>.checkpoint.pkl``.  ``restore(trajectory)`` reloads the nuclear
state, NumPy RNG, MO/NTTDA gauge, and AO Z-vector guesses so ``kernel()`` can
continue from the next step without recomputing the checkpoint frame.
'''

import copy
import os
import pickle
import time
from pathlib import Path

import cupy as cp
import h5py
import numpy as np
from pyscf import gto

from gpu4pyscf.fssh.fssh import FSSH, PES
from gpu4pyscf.grad.nttda import (
    _validate_supported_reference,
    compute_frame,
    make_frame_cache,
    rebuild_reference,
)
from gpu4pyscf.grad.nttda_params import PARAMS as _NTTDA_PARAMS
from gpu4pyscf.sftda.nttda import NTTDA


def _asnumpy(value):
    return cp.asnumpy(cp.asarray(value))


class FSSH_NTTDA(FSSH):

    def __init__(self, td, states, scf_conv_tol=1e-10, td_conv_tol=1e-8,
                 cphf_conv_tol=1e-9, cphf_max_cycle=None,
                 root_overlap_tol=0.4, use_etfs=True,
                 root_tracking_buffer=1, state_ordering='energy',
                 scf_mom_overlap_tol=0.5):
        if not isinstance(states, (list, tuple)) or len(states) < 2:
            raise ValueError('at least two NTTDA states must be specified')
        if any(
                isinstance(state, (bool, np.bool_))
                or not isinstance(state, (int, np.integer))
                or state < 1
                for state in states):
            raise ValueError('states are 1-based NTTDA root indices')
        states = [int(state) for state in states]
        if len(set(states)) != len(states):
            raise ValueError('NTTDA state indices must be unique')
        if (not isinstance(root_tracking_buffer, (int, np.integer))
                or isinstance(root_tracking_buffer, (bool, np.bool_))
                or root_tracking_buffer < 0):
            raise ValueError('root_tracking_buffer must be a non-negative integer')
        if not hasattr(td, '_scf') or not hasattr(td._scf, 'mol'):
            raise TypeError('td must be a GPU NTTDA object with a reference SCF')
        if getattr(td, 'deltaS', None) != -1:
            raise ValueError('FSSH_NTTDA supports only NTTDA deltaS=-1')
        _validate_supported_reference(td._scf)
        if (cphf_max_cycle is not None
                and (
                    isinstance(cphf_max_cycle, (bool, np.bool_))
                    or not isinstance(cphf_max_cycle, (int, np.integer))
                    or cphf_max_cycle < 1
                )):
            raise ValueError('cphf_max_cycle must be a positive integer or None')
        if state_ordering not in ('energy', 'overlap'):
            raise ValueError(
                "state_ordering must be either 'energy' or 'overlap'"
            )
        if (
                isinstance(scf_mom_overlap_tol, (bool, np.bool_))
                or not isinstance(
                    scf_mom_overlap_tol,
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(scf_mom_overlap_tol)
                or not 0.0 < scf_mom_overlap_tol <= 1.0):
            raise ValueError(
                'scf_mom_overlap_tol must be in the interval (0, 1]'
            )

        super().__init__(td._scf.mol, states)
        self.tddft = td
        self.scf_conv_tol = scf_conv_tol
        self.td_conv_tol = td_conv_tol
        self.cphf_conv_tol = cphf_conv_tol
        if cphf_max_cycle is not None:
            self.cphf_max_cycle = int(cphf_max_cycle)
        else:
            self.cphf_max_cycle = _NTTDA_PARAMS.get('cphf_max_cycle')
        self.root_overlap_tol = root_overlap_tol
        self.state_ordering = state_ordering
        self.scf_mom_overlap_tol = float(scf_mom_overlap_tol)
        self.use_etfs = bool(use_etfs)
        self.root_tracking_buffer = int(root_tracking_buffer)
        self.nstates_solver = max(states) + self.root_tracking_buffer
        self.Nstates = len(states)
        self.nac_idx = [
            (i, j)
            for i in range(len(states))
            for j in range(i + 1, len(states))
        ]
        scf_ready = (
            bool(getattr(td._scf, 'converged', False))
            and float(getattr(td._scf, 'conv_tol', np.inf))
            <= self.scf_conv_tol
        )
        td_ready = (
            self._td_solution_ready(td)
            and float(getattr(td, 'conv_tol', np.inf)) <= self.td_conv_tol
        )
        self._last_mf = td._scf if scf_ready else None
        self._last_td = td if scf_ready and td_ready else None
        self._prev = None  # (mol, C, occ, xy_list), all arrays on the CPU
        if self._last_mf is not None and self._last_td is not None:
            self._prev = self._make_snapshot(
                self._last_td.mol, self._last_mf, self._last_td.xy,
            )
        self._initial_frame_available = bool(
            self._prev is not None and len(self._last_td.e) >= self.nstates_solver
        )
        self._frame_cache = make_frame_cache()
        self._reused_initial_reference = False
        self.root_assignment = None
        self.root_overlaps = None
        self.scf_mom_retry = False
        self.scf_mom_diagnostics = None
        self._step_phase_timing = {}
        self._restored_pes = None

    @staticmethod
    def _copy_setting(source, target, name):
        if not hasattr(source, name):
            return
        value = getattr(source, name)
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            value = copy.deepcopy(value)
        setattr(target, name, value)

    @classmethod
    def _copy_grid_settings(cls, source, target):
        for name in (
                'atom_grid', 'atomic_radii', 'radii_adjust', 'radi_method',
                'becke_scheme', 'prune', 'level', 'alignment', 'cutoff'):
            cls._copy_setting(source, target, name)

    def _new_scf(self, mol):
        '''Rebuild the concrete NTTDA reference with matching settings.'''
        from gpu4pyscf.df.df_jk import _DFHF

        template = self.tddft._scf
        mf = rebuild_reference(template, mol)
        for name in (
                'conv_tol_grad', 'max_cycle', 'max_memory', 'direct_scf_tol',
                'init_guess', 'level_shift', 'damp', 'diis_space',
                'diis_damp', 'diis_start_cycle', 'diis_space_rollback',
                'conv_check', 'disp', 'nlc', 'small_rho_cutoff', 'DIIS'):
            self._copy_setting(template, mf, name)
        if getattr(template, 'omega', None) is not None:
            self._copy_setting(template, mf, 'omega')
        self._copy_setting(template, mf, 'disp_with_3body')
        if isinstance(getattr(template, 'diis', None), (bool, int)):
            mf.diis = template.diis
        self._copy_grid_settings(template.grids, mf.grids)
        if hasattr(template, 'nlcgrids') and hasattr(mf, 'nlcgrids'):
            self._copy_grid_settings(template.nlcgrids, mf.nlcgrids)
        if hasattr(template, 'cphf_grids') and hasattr(mf, 'cphf_grids'):
            self._copy_grid_settings(template.cphf_grids, mf.cphf_grids)

        if isinstance(template, _DFHF):
            mf = mf.density_fit(
                auxbasis=copy.deepcopy(template.with_df.auxbasis),
                only_dfj=bool(getattr(template, 'only_dfj', False)),
            )
            for name in ('max_memory', 'use_gpu_memory'):
                self._copy_setting(template.with_df, mf.with_df, name)
            self._copy_setting(template, mf, 'screen_tol')

        mf.conv_tol = self.scf_conv_tol
        mf.verbose = 0
        mf.chkfile = None
        return mf

    def _new_td(self, mf):
        from pyscf.sftda import nttda_methods as methods

        td = NTTDA(mf)
        for name in ('deltaS', 'nobeta', 'lindep', 'max_cycle', 'max_memory'):
            self._copy_setting(self.tddft, td, name)
        methods.copy_method(self.tddft, td)
        td.nstates = self.nstates_solver
        td.conv_tol = self.td_conv_tol
        td.verbose = 0
        return td

    @staticmethod
    def _td_solution_ready(td):
        if getattr(td, 'e', None) is None or getattr(td, 'xy', None) is None:
            return False
        if len(td.e) != len(td.xy) or getattr(td, 'converged', None) is None:
            return False
        converged = np.atleast_1d(_asnumpy(td.converged)).astype(
            bool, copy=False,
        )
        return len(converged) == len(td.e) and bool(np.all(converged))

    def _validate_td_solution(self, td):
        if len(td.e) < max(self.states):
            raise RuntimeError(
                'GPU NTTDA returned %d roots, but state %d was requested'
                % (len(td.e), max(self.states))
            )
        if td.converged is None:
            raise RuntimeError('GPU NTTDA did not return convergence flags')
        converged = np.atleast_1d(_asnumpy(td.converged)).astype(
            bool, copy=False,
        )
        if len(converged) != len(td.e):
            raise RuntimeError(
                'GPU NTTDA convergence flags are inconsistent with roots'
            )
        if not np.all(converged):
            failed = np.flatnonzero(~converged) + 1
            raise RuntimeError(
                'GPU NTTDA roots did not converge: %s' % failed.tolist()
            )

    @staticmethod
    def _make_snapshot(mol, mf, xy):
        return (
            mol.copy(),
            _asnumpy(mf.mo_coeff),
            _asnumpy(mf.mo_occ),
            [_asnumpy(x) for x, _y in xy],
        )

    @staticmethod
    def _array_or_none(value):
        return None if value is None else _asnumpy(value)

    def _checkpoint_path(self, trajectory_file=None):
        trajectory = Path(
            self.filename if trajectory_file is None else trajectory_file
        )
        return Path(str(trajectory) + '.checkpoint.pkl')

    def _checkpoint_signature(self):
        from pyscf.sftda import nttda_methods as methods

        mf = self.tddft._scf
        mol = mf.mol
        with_df = getattr(mf, 'with_df', None)
        return {
            'method_id': methods.resolve_method(self.tddft).id,
            'symbols': tuple(mol.atom_symbol(i) for i in range(mol.natm)),
            'charge': int(mol.charge),
            'spin': int(mol.spin),
            'nao': int(mol.nao_nr()),
            'cart': bool(mol.cart),
            'basis': repr(mol._basis),
            'ecp': repr(mol._ecp),
            'xc': str(getattr(mf, 'xc', 'HF')),
            'omega': getattr(mf, 'omega', None),
            'with_df': with_df is not None,
            'auxbasis': repr(getattr(with_df, 'auxbasis', None)),
            'states': tuple(self.states),
            'nstates_solver': self.nstates_solver,
            'state_ordering': self.state_ordering,
            'root_tracking_buffer': self.root_tracking_buffer,
            'root_overlap_tol': self.root_overlap_tol,
            'scf_mom_overlap_tol': self.scf_mom_overlap_tol,
            'use_etfs': self.use_etfs,
            'delta_s': int(getattr(self.tddft, 'deltaS', -1)),
            'nobeta': bool(getattr(self.tddft, 'nobeta', False)),
            'grid_level': int(getattr(mf.grids, 'level', 3)),
            'scf_conv_tol': self.scf_conv_tol,
            'td_conv_tol': self.td_conv_tol,
            'cphf_conv_tol': self.cphf_conv_tol,
            'cphf_max_cycle': self.cphf_max_cycle,
        }

    def _pack_frame_cache(self):
        cache = self._frame_cache
        return {
            'signature': cache._signature,
            'entries': {
                key: (
                    np.array(alpha, copy=True),
                    np.array(beta, copy=True),
                )
                for key, (alpha, beta) in cache._entries.items()
            },
            'last_stats': copy.deepcopy(cache.last_stats),
        }

    def _write_checkpoint(
            self, step, position, velocity, pes, coeffs, cur_state):
        mf = self._last_mf
        td = self._last_td
        if mf is None or td is None:
            raise RuntimeError(
                'cannot checkpoint before an electronic frame is available'
            )
        payload = {
            'schema_version': 1,
            'signature': self._checkpoint_signature(),
            'trajectory_file': str(Path(self.filename).resolve()),
            'step': int(step),
            'dt': float(self.dt),
            'decoherence': bool(self.decoherence),
            'alpha': float(self.alpha),
            'coupling_method': self.coupling_method,
            'save_force': bool(self.save_force),
            'position': np.array(position, dtype=float),
            'velocity': np.array(velocity, dtype=float),
            'coefficient': np.array(coeffs, dtype=np.complex128),
            'cur_state': int(cur_state),
            'energy': np.array(pes.energy, dtype=float),
            'force': np.array(pes.force, dtype=float),
            'nacv': np.array(pes.nacv, dtype=float),
            'rng_state': np.random.get_state(),
            'reused_initial_reference': bool(
                self._reused_initial_reference
            ),
            'electronic': {
                'mo_coeff': _asnumpy(mf.mo_coeff),
                'mo_occ': _asnumpy(mf.mo_occ),
                'mo_energy': self._array_or_none(mf.mo_energy),
                'e_tot': float(_asnumpy(mf.e_tot)),
                'scf_cycles': int(getattr(mf, 'cycles', 0)),
                'td_e': _asnumpy(td.e),
                'td_xy': [
                    (
                        _asnumpy(x),
                        self._array_or_none(y),
                    )
                    for x, y in td.xy
                ],
                'td_converged': _asnumpy(td.converged).astype(bool),
                'solver_stats': copy.deepcopy(
                    getattr(td, '_nttda_solver_stats', None)
                ),
                'root_assignment': copy.deepcopy(self.root_assignment),
                'root_overlaps': self._array_or_none(self.root_overlaps),
                'scf_mom_retry': bool(self.scf_mom_retry),
                'scf_mom_diagnostics': copy.deepcopy(
                    self.scf_mom_diagnostics
                ),
                'frame_cache': self._pack_frame_cache(),
            },
        }
        checkpoint = self._checkpoint_path()
        temporary = Path(str(checkpoint) + '.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint)

    def write_trajectory(
            self, step, position, velocity, pes, coeffs, cur_state,
            **kwargs):
        super().write_trajectory(
            step, position, velocity, pes, coeffs, cur_state, **kwargs,
        )
        group = self._h5trajfile[str(step)]
        if 'velocity' in group:
            del group['velocity']
        group['velocity'] = np.asarray(velocity)
        self._h5trajfile.flush()
        self._write_checkpoint(
            step, position, velocity, pes, coeffs, cur_state,
        )

    def _tracking_data(self, mol, mf):
        C = _asnumpy(mf.mo_coeff)
        occ = _asnumpy(mf.mo_occ)
        data = {'C': C, 'occ': occ}
        if self._prev is None:
            return data

        mol_p, C_p, occ_p, xy_p = self._prev
        s_cross = gto.intor_cross('int1e_ovlp', mol_p, mol)
        s_mo = C_p.T @ s_cross @ C
        rows_p = np.flatnonzero(occ_p > 0)
        rows = np.flatnonzero(occ > 0)
        cols_p = np.flatnonzero(occ_p < 2)
        cols = np.flatnonzero(occ < 2)
        data.update({
            'xy_p': xy_p,
            's_occ': s_mo[np.ix_(rows_p, rows)],
            's_vir': s_mo[np.ix_(cols_p, cols)],
        })
        return data

    @staticmethod
    def _mom_occupation(overlap, reference_occ):
        '''Assign fixed 2/1/0 occupations by maximum subspace overlap.

        The occupied projector is selected first.  The closed-shell projector
        is then selected inside that occupied set, which guarantees that the
        resulting beta-like occupied space remains a subset of the alpha-like
        occupied space and preserves the exact closed/open/virtual counts.
        '''
        overlap = cp.asarray(overlap)
        reference_occ = cp.asarray(reference_occ)
        if overlap.ndim != 2 or overlap.shape[0] != reference_occ.size:
            raise ValueError(
                'MOM overlap must have one row per reference orbital'
            )
        valid = (
            (reference_occ == 0)
            | (reference_occ == 1)
            | (reference_occ == 2)
        )
        if not bool(cp.all(valid)):
            raise ValueError('MOM reference occupations must contain 0/1/2')

        nclosed = int(cp.count_nonzero(reference_occ == 2))
        nopen = int(cp.count_nonzero(reference_occ == 1))
        noccupied = nclosed + nopen
        nmo = overlap.shape[1]
        if noccupied > nmo:
            raise RuntimeError(
                'MOM reference has more occupied orbitals than candidates'
            )

        mo_occ = cp.zeros(nmo, dtype=cp.float64)
        if noccupied == 0:
            return mo_occ

        occupied_scores = cp.sum(
            cp.abs(overlap[reference_occ > 0]) ** 2, axis=0,
        ).real
        occupied_order = cp.argsort(-occupied_scores, kind='stable')
        occupied = occupied_order[:noccupied]
        mo_occ[occupied] = 1

        if nclosed:
            closed_scores = cp.sum(
                cp.abs(overlap[reference_occ == 2]) ** 2, axis=0,
            ).real
            restricted_scores = cp.full(nmo, -cp.inf)
            restricted_scores[occupied] = closed_scores[occupied]
            closed = cp.argsort(
                -restricted_scores, kind='stable',
            )[:nclosed]
            mo_occ[closed] = 2
        return mo_occ

    def _scf_occupation_continuity(self, mol, mf, previous=None):
        if previous is None:
            previous = self._last_mf
        if previous is None:
            return None, None

        previous_coeff = cp.asarray(previous.mo_coeff)
        previous_occ = cp.asarray(previous.mo_occ)
        current_coeff = cp.asarray(mf.mo_coeff)
        current_occ = cp.asarray(mf.mo_occ)
        if previous_occ.ndim != 1 or current_occ.ndim != 1:
            raise RuntimeError(
                'MOM recovery requires one-dimensional 0/1/2 occupations'
            )
        if previous_coeff.shape[1] != previous_occ.size:
            raise RuntimeError(
                'previous SCF orbitals and occupations are inconsistent'
            )
        if current_coeff.shape[1] != current_occ.size:
            raise RuntimeError(
                'current SCF orbitals and occupations are inconsistent'
            )

        s_cross = cp.asarray(gto.intor_cross(
            'int1e_ovlp', previous.mol, mol,
        ))
        overlap = previous_coeff.conj().T @ s_cross @ current_coeff
        proposed_occ = self._mom_occupation(overlap, previous_occ)

        def minimum_singular_value(previous_mask, current_mask):
            nprevious = int(cp.count_nonzero(previous_mask))
            ncurrent = int(cp.count_nonzero(current_mask))
            if nprevious != ncurrent:
                raise RuntimeError(
                    'SCF occupation class counts changed across FSSH frames'
                )
            if nprevious == 0:
                return 1.0
            block = overlap[previous_mask][:, current_mask]
            singular_values = cp.linalg.svd(
                block, compute_uv=False,
            )
            return float(cp.min(singular_values))

        occupied_min = minimum_singular_value(
            previous_occ > 0, current_occ > 0,
        )
        closed_min = minimum_singular_value(
            previous_occ == 2, current_occ == 2,
        )
        current_occ_cpu = _asnumpy(current_occ)
        proposed_occ_cpu = _asnumpy(proposed_occ)
        changed = np.flatnonzero(current_occ_cpu != proposed_occ_cpu)
        diagnostics = {
            'minimum_occupied_singular_value': occupied_min,
            'minimum_closed_singular_value': closed_min,
            'minimum_class_singular_value': min(occupied_min, closed_min),
            'current_occupation': current_occ_cpu.tolist(),
            'mom_occupation': proposed_occ_cpu.tolist(),
            'changed_orbitals': [
                {
                    'orbital_index': int(index),
                    'current_occupation': float(current_occ_cpu[index]),
                    'mom_occupation': float(proposed_occ_cpu[index]),
                }
                for index in changed
            ],
        }
        return diagnostics, s_cross

    def _install_mom_occupation(self, mf, s_cross):
        previous_coeff = cp.asarray(self._last_mf.mo_coeff)
        previous_occ = cp.asarray(self._last_mf.mo_occ)
        overlap_left = previous_coeff.conj().T @ cp.asarray(s_cross)

        def get_occ(mo_energy=None, mo_coeff=None):
            if mo_coeff is None:
                mo_coeff = mf.mo_coeff
            overlap = overlap_left @ cp.asarray(mo_coeff)
            return self._mom_occupation(overlap, previous_occ)

        mf.get_occ = get_occ
        return mf

    def _ensemble_newton_recovery(self, mol, failed=None, previous=None):
        from gpu4pyscf.sftda.ensemble_soscf import project_orbitals

        mf = self._new_scf(mol)
        if previous is None:
            previous = self._last_mf
        if previous is None:
            if failed is None:
                raise RuntimeError(
                    'second-order recovery requires an orbital guess'
                )
            coeff, occ = failed.mo_coeff, failed.mo_occ
        else:
            coeff = project_orbitals(mf, previous)
            occ = cp.asarray(previous.mo_occ)
        solver = mf.newton()
        solver.max_cycle = min(mf.max_cycle, 50)
        cycles = [0]

        def count_cycles(env):
            cycles[0] = int(env['imacro']) + 1

        solver.callback = count_cycles
        solver.kernel(coeff, occ)
        for name in ('mo_coeff', 'mo_occ', 'mo_energy', 'e_tot', 'converged'):
            setattr(mf, name, getattr(solver, name))
        mf.cycles = cycles[0]
        return mf

    def _ensemble_newton_continuation(self, mol):
        '''Transport a difficult reference through intermediate geometries.

        Every accepted electronic substep uses the same occupation-continuity
        predicate as a normal FSSH frame. This can follow a rapidly rotating
        stationary solution without weakening the configured guard.
        '''
        previous = self._last_mf
        if previous is None:
            return None, []
        origin = previous.mol.atom_coords()
        target = mol.atom_coords()
        fraction = 0.0
        increment = 0.25
        path = []
        while fraction < 1.0:
            next_fraction = min(1.0, fraction + increment)
            bridge = mol.set_geom_(
                origin + next_fraction * (target - origin),
                unit='Bohr', inplace=False,
            )
            candidate = self._ensemble_newton_recovery(
                bridge, previous=previous,
            )
            diagnostics, _ = self._scf_occupation_continuity(
                bridge, candidate, previous=previous,
            )
            accepted = bool(
                candidate.converged
                and not (
                    diagnostics['changed_orbitals']
                    and diagnostics['minimum_class_singular_value']
                    < self.scf_mom_overlap_tol
                )
            )
            path.append({
                'fraction': next_fraction,
                'converged': bool(candidate.converged),
                'minimum_class_singular_value': diagnostics[
                    'minimum_class_singular_value'
                ],
                'changed_orbitals': diagnostics['changed_orbitals'],
                'accepted': accepted,
            })
            if accepted:
                previous = candidate
                fraction = next_fraction
                increment = min(0.25, increment * 1.5)
            else:
                increment *= 0.5
                if increment < 1e-4:
                    return None, path
        return previous, path

    def _run_scf_with_mom_recovery(self, mol, dm0=None):
        self.scf_mom_retry = False
        self.scf_mom_diagnostics = None

        mf = self._new_scf(mol)
        mf.kernel(dm0=dm0) if dm0 is not None else mf.kernel()
        normal_converged = bool(mf.converged)
        normal_cycles = int(getattr(mf, 'cycles', 0))
        ensemble_reference = bool(
            getattr(mf, 'is_ensemble_rks', False)
        )
        newton_diagnostics = {}
        if not mf.converged and ensemble_reference:
            mf = self._ensemble_newton_recovery(mol, mf)
            newton_diagnostics = {
                'newton_scf_converged': bool(mf.converged),
                'newton_scf_cycles': int(mf.cycles),
            }
            self.scf_mom_diagnostics = {
                'normal_scf_converged': normal_converged,
                'normal_scf_cycles': normal_cycles,
                **newton_diagnostics,
            }
        if not mf.converged:
            if ensemble_reference:
                continued, path = self._ensemble_newton_continuation(mol)
                self.scf_mom_diagnostics.update({
                    'newton_continuation_attempted': True,
                    'newton_continuation_converged': continued is not None,
                    'newton_continuation_path': path,
                })
                if continued is not None:
                    endpoint, _ = self._scf_occupation_continuity(
                        mol, continued,
                    )
                    endpoint.update(self.scf_mom_diagnostics)
                    endpoint['continuity_via_substeps'] = True
                    self.scf_mom_diagnostics = endpoint
                    return continued
            raise RuntimeError('GPU ROKS SCF did not converge')

        diagnostics, s_cross = self._scf_occupation_continuity(mol, mf)
        if diagnostics is None:
            return mf
        diagnostics['overlap_tolerance'] = self.scf_mom_overlap_tol
        diagnostics['normal_scf_cycles'] = normal_cycles
        diagnostics['normal_scf_converged'] = normal_converged
        diagnostics.update(newton_diagnostics)
        retry = bool(
            diagnostics['changed_orbitals']
            and diagnostics['minimum_class_singular_value']
            < self.scf_mom_overlap_tol
        )
        diagnostics['retry_triggered'] = retry
        self.scf_mom_diagnostics = diagnostics
        if not retry:
            return mf
        if self.state_ordering == 'energy':
            # Energy-ordered dynamics is adiabatic: orbital character may
            # change at a reference-state crossing. Record the switch, but do
            # not force the old diabatic occupation map back with MOM.
            diagnostics['adiabatic_branch_switch_accepted'] = True
            return mf
        if ensemble_reference:
            continued, path = self._ensemble_newton_continuation(mol)
            diagnostics['newton_continuation_attempted'] = True
            diagnostics['newton_continuation_path'] = path
            diagnostics['newton_continuation_converged'] = (
                continued is not None
            )
            if continued is not None:
                endpoint, _ = self._scf_occupation_continuity(
                    mol, continued,
                )
                endpoint.update({
                    'overlap_tolerance': self.scf_mom_overlap_tol,
                    'normal_scf_cycles': normal_cycles,
                    'normal_scf_converged': normal_converged,
                    **newton_diagnostics,
                    'retry_triggered': True,
                    'newton_continuation_attempted': True,
                    'newton_continuation_converged': True,
                    'newton_continuation_path': path,
                    'continuity_via_substeps': True,
                })
                self.scf_mom_diagnostics = endpoint
                return continued

        recovered = self._install_mom_occupation(
            self._new_scf(mol), s_cross,
        )
        recovered.kernel(dm0=dm0) if dm0 is not None else recovered.kernel()
        self.scf_mom_retry = True
        diagnostics['mom_scf_cycles'] = int(
            getattr(recovered, 'cycles', 0)
        )
        diagnostics['mom_scf_converged'] = bool(recovered.converged)
        if not recovered.converged:
            raise RuntimeError('GPU ROKS SCF MOM retry did not converge')

        recovered_diagnostics, _ = self._scf_occupation_continuity(
            mol, recovered,
        )
        diagnostics['recovered_occupation'] = recovered_diagnostics[
            'current_occupation'
        ]
        diagnostics['recovered_minimum_occupied_singular_value'] = (
            recovered_diagnostics['minimum_occupied_singular_value']
        )
        diagnostics['recovered_minimum_closed_singular_value'] = (
            recovered_diagnostics['minimum_closed_singular_value']
        )
        diagnostics['recovered_minimum_class_singular_value'] = (
            recovered_diagnostics['minimum_class_singular_value']
        )
        diagnostics['recovered_changed_orbitals'] = (
            recovered_diagnostics['changed_orbitals']
        )
        if (
                recovered_diagnostics['changed_orbitals']
                or recovered_diagnostics['minimum_class_singular_value']
                < self.scf_mom_overlap_tol):
            raise RuntimeError(
                'GPU ROKS SCF MOM retry did not restore orbital continuity'
            )
        return recovered

    @staticmethod
    def _project_previous_roots(tracking):
        if 'xy_p' not in tracking:
            return None
        s_occ = tracking['s_occ']
        s_vir = tracking['s_vir']
        guesses = []
        for x_p in tracking['xy_p']:
            guess = s_occ.T @ x_p @ s_vir
            norm = np.linalg.norm(guess)
            if norm > 1e-12:
                guesses.append((guess / norm).ravel())
        if not guesses:
            return None
        return cp.asarray(np.asarray(guesses))

    def _track_roots(self, mol, td, tracking):
        '''Phase-align energy roots, or opt into character-following order.'''
        C = tracking['C']
        occ = tracking['occ']
        xy = [_asnumpy(x) for x, _ in td.xy]
        self.root_assignment = list(range(len(xy)))
        self.root_overlaps = np.ones(len(xy))
        if 'xy_p' in tracking:
            xy_p = tracking['xy_p']
            s_occ = tracking['s_occ']
            s_vir = tracking['s_vir']
            n_p, n = len(xy_p), len(xy)
            n_track = min(n_p, n)
            overlaps = np.empty((n_track, n))
            for i, x_p in enumerate(xy_p[:n_track]):
                left = x_p.T @ s_occ
                for j, x in enumerate(xy):
                    overlaps[i, j] = np.trace(left @ x @ s_vir.T)

            if self.state_ordering == 'overlap':
                from scipy.optimize import linear_sum_assignment
                rows_a, cols_a = linear_sum_assignment(-np.abs(overlaps))
            else:
                rows_a = np.arange(n_track)
                cols_a = np.arange(n_track)
            order = [None] * n
            signs = np.ones(n)
            for i, j in zip(rows_a, cols_a):
                order[i] = int(j)
                signs[i] = -1.0 if overlaps[i, j] < 0 else 1.0
                self.root_overlaps[i] = abs(overlaps[i, j])
                if (self.state_ordering == 'overlap'
                        and i + 1 in self.states
                        and self.root_overlaps[i] < self.root_overlap_tol):
                    raise RuntimeError(
                        'NTTDA root tracking overlap %.3f below %.3f '
                        'for state %d' % (
                            self.root_overlaps[i],
                            self.root_overlap_tol, i + 1))
            assigned = {index for index in order if index is not None}
            unused = iter(index for index in range(n) if index not in assigned)
            order = [
                next(unused) if index is None else index for index in order
            ]
            self.root_assignment = order
            td.e = np.asarray(td.e)[order]
            td.xy = [
                (cp.asarray(xy[j] * signs[i]), 0)
                for i, j in enumerate(order)
            ]
            xy = [xy[j] * signs[i] for i, j in enumerate(order)]
            if td.converged is not None:
                converged = np.asarray(td.converged, dtype=bool)
                if len(converged) != n:
                    raise RuntimeError(
                        'NTTDA convergence flags are inconsistent with roots'
                    )
                td.converged = converged[order]
        return (mol.copy(), C, occ, xy)

    def calc_electronic(self, position, cur_state=None, with_nacv=True,
                        with_frame=True):
        '''Evaluate electronic structure for one FSSH step.

        Executes four phases, each timed separately into ``self._step_phase_timing``:

        1. **scf** — GPU ROKS SCF kernel (warm-started from previous frame density).
        2. **nttda** — NTTDA kernel (tracking data prep + root projection + solve).
        3. **root_tracking** — phase-aligned energy ordering or Hungarian assignment.
        4. **frame** — ``compute_frame`` call (gradient + NAC pairs).

        When ``with_frame=False`` (solver-only mode), phase 4 is skipped
        entirely and force/nacv are returned as zeros.  This is used by the
        profiler to isolate SCF+NTTDA timing from gradient/NAC overhead.

        The phase timing dict is read by
        ``tests/profile_fssh_nttda_steady_state.py`` after each step.
        '''
        position = np.asarray(position, dtype=float).reshape(-1, 3)
        if len(position) != self.tddft.mol.natm:
            raise ValueError('position must contain one xyz row per atom')
        if cur_state is None:
            cur_state = self.cur_state
        if cur_state not in self.states:
            raise ValueError('cur_state must be one of the configured states')
        reuse_initial = (
            self._initial_frame_available
            and np.allclose(
                position, self.tddft.mol.atom_coords(unit='Bohr'),
                rtol=0.0, atol=1e-12,
            )
        )
        phase_timing = {}
        if reuse_initial:
            mol = self.tddft.mol
            mf = self.tddft._scf
            td = self.tddft
            snapshot = self._prev
            self.root_assignment = list(range(len(td.e)))
            self.root_overlaps = np.ones(len(td.e))
        else:
            mol = self.tddft.mol.set_geom_(
                position, unit='Bohr', inplace=False,
            )
            dm0 = None
            if self._last_mf is not None:
                dm0 = cp.asarray(self._last_mf.make_rdm1())
            t = time.perf_counter()
            mf = self._run_scf_with_mom_recovery(mol, dm0=dm0)
            phase_timing['scf'] = time.perf_counter() - t

            t = time.perf_counter()
            tracking = self._tracking_data(mol, mf)
            td = self._new_td(mf)
            td.kernel(x0=self._project_previous_roots(tracking))
            phase_timing['nttda'] = time.perf_counter() - t

            t = time.perf_counter()
            snapshot = self._track_roots(mol, td, tracking)
            phase_timing['root_tracking'] = time.perf_counter() - t
        self._validate_td_solution(td)

        total_energies = np.asarray(td.e_tot, dtype=float)
        energy = total_energies[np.asarray(self.states) - 1]

        pairs = []
        if with_nacv:
            pairs = [
                (self.states[i], self.states[j]) for i, j in self.nac_idx
            ]
        if with_frame:
            t = time.perf_counter()
            frame = compute_frame(
                td, active_state=cur_state, nac_pairs=pairs,
                cphf_conv_tol=self.cphf_conv_tol,
                cphf_max_cycle=self.cphf_max_cycle,
                use_etfs=self.use_etfs,
                frame_cache=self._frame_cache,
            )
            phase_timing['frame'] = time.perf_counter() - t
            force = -frame['grad']
            nacv = np.zeros((self.Nstates, self.Nstates, mol.natm, 3))
            for (i, j), (si, sj) in zip(self.nac_idx, pairs):
                value = frame['nac'][(si, sj)]
                nacv[i, j] = value
                nacv[j, i] = -value
        else:
            force = np.zeros((mol.natm, 3))
            nacv = np.zeros((self.Nstates, self.Nstates, mol.natm, 3))

        phase_timing['total'] = sum(phase_timing.values())
        self._step_phase_timing = phase_timing

        # Commit cross-frame warm-start and gauge state only after the entire
        # electronic frame has completed successfully.
        self._prev = snapshot
        self._last_mf = mf
        self._last_td = td
        self._initial_frame_available = False
        self._reused_initial_reference = (
            self._reused_initial_reference or reuse_initial
        )
        return energy, force, nacv

    def evaluate_pes(self, position, cur_state, with_nacv=True, with_frame=True):
        if self._restored_pes is not None:
            if with_nacv:
                raise RuntimeError(
                    'restored PES may only initialize the resumed trajectory'
                )
            if cur_state != self.cur_state or not np.allclose(
                    position, self.position, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    'restored PES does not match the requested state/geometry'
                )
            pes = self._restored_pes
            self._restored_pes = None
            self._step_phase_timing = {
                'restart_reuse': True,
                'total': 0.0,
            }
            return PES(
                energy=np.array(pes.energy, copy=True),
                force=np.array(pes.force, copy=True),
                nacv=np.array(pes.nacv, copy=True),
            )
        energy, force, nacv = self.calc_electronic(
            position, cur_state=cur_state, with_nacv=with_nacv,
            with_frame=with_frame,
        )
        return PES(energy=energy, force=force, nacv=nacv)

    def restore(self, trajectory_file):
        """Restore the last atomically checkpointed NTTDA/FSSH frame.

        Construct this driver with the same electronic model and state options,
        call ``restore()``, set ``nsteps`` to the desired final step number,
        then call ``kernel()`` without position, velocity, or coefficients.
        """
        trajectory = Path(trajectory_file).resolve()
        checkpoint = self._checkpoint_path(trajectory)
        if not trajectory.is_file():
            raise FileNotFoundError(f'trajectory not found: {trajectory}')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'checkpoint not found: {checkpoint}')
        with checkpoint.open('rb') as handle:
            payload = pickle.load(handle)
        if payload.get('schema_version') != 1:
            raise ValueError(
                'unsupported FSSH_NTTDA checkpoint schema %r'
                % payload.get('schema_version')
            )
        signature = copy.deepcopy(payload.get('signature'))
        if isinstance(signature, dict):
            signature.setdefault('scf_mom_overlap_tol', 0.5)
        expected = self._checkpoint_signature()
        if signature != expected:
            raise ValueError(
                'checkpoint electronic model does not match this driver: '
                f'{signature!r} != {expected!r}'
            )

        position = np.asarray(payload['position'], dtype=float)
        velocity = np.asarray(payload['velocity'], dtype=float)
        coefficient = np.asarray(
            payload['coefficient'], dtype=np.complex128,
        )
        step = int(payload['step'])
        if position.shape != (self.mol.natm, 3):
            raise ValueError(
                'checkpoint position shape %s does not match (%d, 3)'
                % (position.shape, self.mol.natm)
            )
        if velocity.shape != position.shape:
            raise ValueError('checkpoint velocity shape is inconsistent')
        if coefficient.shape != (self.Nstates,):
            raise ValueError('checkpoint coefficient shape is inconsistent')
        energy = np.asarray(payload['energy'], dtype=float)
        force = np.asarray(payload['force'], dtype=float)
        nacv = np.asarray(payload['nacv'], dtype=float)
        if energy.shape != (self.Nstates,):
            raise ValueError('checkpoint energy shape is inconsistent')
        if force.shape != position.shape:
            raise ValueError('checkpoint force shape is inconsistent')
        if nacv.shape != (
                self.Nstates, self.Nstates, self.mol.natm, 3):
            raise ValueError('checkpoint NAC shape is inconsistent')
        for name, value in (
                ('position', position), ('velocity', velocity),
                ('coefficient', coefficient),
                ('energy', energy), ('force', force), ('nacv', nacv)):
            if not np.all(np.isfinite(value)):
                raise ValueError(f'checkpoint {name} contains non-finite data')

        electronic = payload['electronic']
        mol = self.tddft.mol.set_geom_(
            position, unit='Bohr', inplace=False,
        )
        mf = self._new_scf(mol)
        mf.mo_coeff = cp.asarray(electronic['mo_coeff'])
        mf.mo_occ = cp.asarray(electronic['mo_occ'])
        if electronic['mo_energy'] is not None:
            mf.mo_energy = cp.asarray(electronic['mo_energy'])
        mf.e_tot = float(electronic['e_tot'])
        mf.converged = True
        mf.cycles = int(electronic['scf_cycles'])

        td = self._new_td(mf)
        td.e = np.asarray(electronic['td_e'], dtype=float)
        td.xy = []
        for x, y in electronic['td_xy']:
            y = np.asarray(y)
            td.xy.append((
                cp.asarray(x),
                y.item() if y.ndim == 0 else cp.asarray(y),
            ))
        td.converged = np.asarray(
            electronic['td_converged'], dtype=bool,
        )
        if electronic['solver_stats'] is not None:
            td._nttda_solver_stats = copy.deepcopy(
                electronic['solver_stats']
            )
        self._validate_td_solution(td)

        cache_payload = electronic['frame_cache']
        cache = make_frame_cache()
        cache._mol = mol.copy()
        cache._signature = cache_payload['signature']
        cache._entries = {
            key: (
                np.array(alpha, copy=True),
                np.array(beta, copy=True),
            )
            for key, (alpha, beta) in cache_payload['entries'].items()
        }
        cache.last_stats = copy.deepcopy(cache_payload['last_stats'])

        with h5py.File(trajectory, 'r+') as handle:
            frame_ids = sorted(
                int(key) for key in handle if key.isdigit()
            )
            if step not in frame_ids:
                raise ValueError(
                    f'trajectory does not contain checkpoint frame {step}'
                )
            if not np.array_equal(
                    np.asarray(handle[f'{step}/position']), position):
                raise ValueError(
                    'checkpoint position does not match its trajectory frame'
                )
            if int(handle[f'{step}/cur_state'][()]) != int(
                    payload['cur_state']):
                raise ValueError(
                    'checkpoint state does not match its trajectory frame'
                )
            for frame_id in frame_ids:
                if frame_id > step:
                    del handle[str(frame_id)]
            handle['velocity'][:] = velocity
            handle.flush()

        self.filename = str(trajectory)
        self.dt = float(payload['dt'])
        self.decoherence = bool(payload['decoherence'])
        self.alpha = float(payload['alpha'])
        self.coupling_method = payload['coupling_method']
        self.save_force = bool(payload['save_force'])
        self.cur_step = step
        self.cur_state = int(payload['cur_state'])
        self.position = position
        self.velocity = velocity
        self.coefficient = coefficient
        self.energy = energy
        self.mol = mol
        self._last_mf = mf
        self._last_td = td
        self._prev = self._make_snapshot(mol, mf, td.xy)
        self._frame_cache = cache
        self.root_assignment = copy.deepcopy(
            electronic['root_assignment']
        )
        self.root_overlaps = self._array_or_none(
            electronic['root_overlaps']
        )
        self.scf_mom_retry = bool(
            electronic.get('scf_mom_retry', False)
        )
        self.scf_mom_diagnostics = copy.deepcopy(
            electronic.get('scf_mom_diagnostics')
        )
        self._initial_frame_available = False
        self._reused_initial_reference = bool(
            payload.get('reused_initial_reference', False)
        )
        self._restored_pes = PES(
            energy=energy,
            force=force,
            nacv=nacv,
        )
        self._step_phase_timing = {'restart_loaded': True}
        np.random.set_state(payload['rng_state'])
        self.seed = None
        return self
