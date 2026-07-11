# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""GPU-accelerated spin-flip TDDFT fewest-switches dynamics."""

from __future__ import annotations

import copy
import csv
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cupy as cp
import numpy as np

from .fssh import FSSH
from .nac_monitor import NACMonitor
from .sf_excited_analysis import append_excited_state_analysis_csv
from .sf_gauge import GaugeTracker
from .sf_state_selection import append_state_selection_csv, select_sf_singlet_manifold

logger = logging.getLogger(__name__)

FS2AUTIME = 41.34137        # Conversion factor: femtoseconds to atomic time units
A2BOHR = 1.889726           # Conversion factor: Angstrom to Bohr radius
AMU2AU = 1822.8884858012984  # Conversion factor: amu to atomic mass units


def _asnumpy(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)


def _array_or_zero_to_numpy(value):
    if isinstance(value, (int, float)) and value == 0:
        return 0
    return np.array(_asnumpy(value), copy=True)


def _array_or_zero_to_cupy(value):
    if isinstance(value, (int, float)) and value == 0:
        return 0
    return cp.asarray(value)


def _vector_to_numpy(vector):
    x, y = vector
    return _array_or_zero_to_numpy(x), _array_or_zero_to_numpy(y)


def _vector_to_cupy(vector):
    x, y = vector
    return _array_or_zero_to_cupy(x), _array_or_zero_to_cupy(y)


class FSSH_SF(FSSH):
    """
    一个用于Spin-Flip TDDFT的FSSH实现。
    它继承自通用的FSSH类，并重写了calc_electronic方法以正确处理
    SF-TDDFT的能量计算和scanner调用。
    """
    def __init__(self, tddft, states: List[int], **kwargs):
        """Initialize a GPU SF-TDA/SF-TDDFT FSSH trajectory.

        ``states`` are the 1-based singlet labels used by the BSCC driver.  They
        are mapped to GPU4PySCF's zero-based raw SF roots at every geometry.
        ``dt`` is in fs and initial velocities passed to :meth:`kernel` retain
        the BSCC convention of Angstrom/ps.
        """
        if not isinstance(states, (list, tuple)) or len(states) < 2:
            raise ValueError("At least two electronic states must be specified")
        if any(not isinstance(state, int) or state <= 0 for state in states):
            raise ValueError(
                "All FSSH-SF state labels must be 1-based positive integers"
            )
        if not hasattr(tddft, "Gradients") or not hasattr(tddft, "NAC"):
            raise TypeError(
                "tddft must be a GPU4PySCF spin-flip TD object with Gradients() and NAC()"
            )

        self.tddft = tddft
        self.sf_method_cls = type(tddft)
        self.sf_method_name = self.sf_method_cls.__name__
        self.tdgrad = self.tddft.Gradients()
        self.tdnac = self.tddft.NAC()

        self.states = list(states)
        self.Nstates = len(states)
        self.cur_state = states[0]
        self.mass = (
            np.asarray(self.tddft.mol.atom_mass_list(True), dtype=float).reshape(-1, 1)
            * AMU2AU
        )
        self.nac_idx = [
            (i, j)
            for i in range(self.Nstates - 1)
            for j in range(i + 1, self.Nstates)
        ]

        self.dt = 0.5 * FS2AUTIME
        self.nsteps = 1
        self.output_dir = Path(".")
        self.alpha = 0.1
        self.scf_conv_tol = 1e-6
        self.scf_max_cycle = 200
        self.scf_max_nonconverged_steps = 4
        self.scf_nonconverged_streak = 0
        self.sf_selection_n_lowest = max(4, max(self.states) + 1)
        self.triplet_zero_threshold_ev = 0.3
        self.mo_align_threshold = 0.4
        self.analysis_top_n = 5
        self.analysis_coeff_threshold = 0.1
        self.analysis_max_states = 4
        self.resume = False

        for key, value in kwargs.items():
            if key == "dt":
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError("Time step must be positive")
                self.dt = float(value) * FS2AUTIME
            elif key == "nsteps":
                if not isinstance(value, int) or value <= 0:
                    raise ValueError("Number of steps must be positive")
                self.nsteps = value
            elif key == "output_dir":
                self.output_dir = Path(value)
            elif key != "cphf_options":
                setattr(self, key, value)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sf_selection_n_lowest = max(
            int(self.sf_selection_n_lowest), max(self.states) + 1
        )
        self.tddft.nstates = max(
            int(getattr(self.tddft, "nstates", 0)), self.sf_selection_n_lowest
        )
        self.gauge_tracker = GaugeTracker(self.mo_align_threshold)
        self.prev_wavefunctions_flat = None
        self.prev_ci_vectors = None
        self.prev_mo_coeff = None
        self.prev_mo_occ = None
        self.prev_mol_for_mo_align = None
        self._last_mf = None

        cphf_options = kwargs.get("cphf_options", {})
        self.cphf_max_cycle = int(cphf_options.get("max_cycle", 100))
        self.cphf_conv_tol = float(cphf_options.get("conv_tol", 1e-7))
        for derivative in (self.tdgrad, self.tdnac):
            if hasattr(derivative, "cphf_max_cycle"):
                derivative.cphf_max_cycle = self.cphf_max_cycle
            if hasattr(derivative, "cphf_conv_tol"):
                derivative.cphf_conv_tol = self.cphf_conv_tol

        # 8. 计时统计相关属性 (Timing Statistics)
        self.timing_data = []  # 存储每一步的计时数据
        self.timing_csv_filename = 'timing_statistics.csv'  # CSV文件名
        self._current_step_timing = {}  # 用于在calc_electronic中记录子步骤耗时
        self.state_selection_csv_filename = 'state_selection.csv'
        self.excited_state_analysis_csv_filename = 'excited_state_analysis.csv'
        self.electronic_diagnostics_csv_filename = 'electronic_diagnostics.csv'
        self.checkpoint_filename = 'checkpoint.pkl'
        self._current_step_index = 0
        self._current_time_fs = 0.0
        self._last_state_selection = None
        self._last_scf_converged = True
        self._last_scf_energy = None
        self._last_scf_s2 = None
        self.nac_monitor = NACMonitor(self.output_dir, self.states)

        # 9. 日志 (Logging)
        logger.info(
            "GPU FSSH-SF initialized with %d states, dt=%.3f fs, %d steps (%s)",
            self.Nstates,
            self.dt / FS2AUTIME,
            self.nsteps,
            self.sf_method_name,
        )

    def _init_timing_csv(self) -> None:
        """
        初始化计时统计CSV文件，写入表头。
        """
        filepath = self.output_dir / self.timing_csv_filename
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入CSV表头 - 聚焦于calc_electronic内部的子步骤
            header = [
                'step',                      # 动力学步数
                'time_fs',                   # 模拟时间 (fs)
                # calc_electronic 内部子步骤
                'mol_setup_s',               # 分子对象设置 (秒)
                'scf_s',                     # SCF计算 (秒)
                'tddft_s',                   # TDDFT计算 (秒)
                'state_analysis_s',          # 态分析与筛选 (秒)
                'gradient_s',                # 梯度计算 (秒)
                'phase_correction_s',        # 相位校正 (秒)
                'nac_total_s',               # NAC总计算时间 (秒)
                'nac_pair_details',          # NAC各态对耗时详情 (字符串)
                'calc_electronic_total_s',   # calc_electronic总耗时 (秒)
                # 其他
                'other_steps_s',             # 外部其他步骤总耗时 (秒)
                'step_total_s',              # 本步总耗时 (秒)
                'current_state',             # 当前态
                'hop_occurred',              # 是否发生跃迁
            ]
            writer.writerow(header)

    def _write_timing_row(self, timing_dict: Dict) -> None:
        """
        将单步的计时数据写入CSV文件。
        
        Args:
            timing_dict: 包含该步骤所有计时数据的字典
        """
        filepath = self.output_dir / self.timing_csv_filename
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                timing_dict.get('step', 0),
                timing_dict.get('time_fs', 0.0),
                # calc_electronic 内部子步骤
                timing_dict.get('mol_setup_s', 0.0),
                timing_dict.get('scf_s', 0.0),
                timing_dict.get('tddft_s', 0.0),
                timing_dict.get('state_analysis_s', 0.0),
                timing_dict.get('gradient_s', 0.0),
                timing_dict.get('phase_correction_s', 0.0),
                timing_dict.get('nac_total_s', 0.0),
                timing_dict.get('nac_pair_details', ''),
                timing_dict.get('calc_electronic_total_s', 0.0),
                # 其他
                timing_dict.get('other_steps_s', 0.0),
                timing_dict.get('step_total_s', 0.0),
                timing_dict.get('current_state', 0),
                timing_dict.get('hop_occurred', False),
            ]
            writer.writerow(row)

    def _reset_run_outputs(self) -> None:
        for filename in (
            'trajectory.xyz',
            self.state_selection_csv_filename,
            self.excited_state_analysis_csv_filename,
            self.electronic_diagnostics_csv_filename,
            self.timing_csv_filename,
        ):
            path = self.output_dir / filename
            if path.exists():
                path.unlink()
        self.nac_monitor = NACMonitor(self.output_dir, self.states)
        self.nac_monitor.reset_files()

    @staticmethod
    def _block_norm2(block) -> float:
        if not isinstance(block, (int, float)):
            array = _asnumpy(block)
            return float(np.vdot(array, array).real)
        return 0.0

    def _xy_norm_metrics(self, xy) -> Dict[str, float]:
        x, y = xy
        x2 = self._block_norm2(x)
        y2 = self._block_norm2(y)
        return {
            'x_norm': float(np.sqrt(max(x2, 0.0))),
            'y_norm': float(np.sqrt(max(y2, 0.0))),
            'x2_minus_y2': float(x2 - y2),
        }

    def _format_root_xy_summary(self, mftd, selection) -> str:
        parts = []
        by_root = {entry.root: entry for entry in selection.roots}
        for root in sorted(by_root):
            metrics = self._xy_norm_metrics(mftd.xy[root])
            role = by_root[root].role
            parts.append(
                f"root{root}:{role}:x={metrics['x_norm']:.8g}:"
                f"y={metrics['y_norm']:.8g}:x2-y2={metrics['x2_minus_y2']:.8g}"
            )
        return ';'.join(parts)

    def _write_electronic_diagnostics(
        self,
        mftd,
        selection,
        energy: np.ndarray,
        force: np.ndarray,
        nacv: np.ndarray,
    ) -> None:
        filepath = self.output_dir / self.electronic_diagnostics_csv_filename
        write_header = not filepath.exists()
        state_pos = {state: idx for idx, state in enumerate(self.states)}
        active_idx = state_pos[self.cur_state]
        active_root = selection.state_map[self.cur_state]
        by_root = {entry.root: entry for entry in selection.roots}
        active_entry = by_root.get(active_root)
        active_xy = self._xy_norm_metrics(mftd.xy[active_root])

        selected_gaps = []
        for i in range(len(energy) - 1):
            for j in range(i + 1, len(energy)):
                selected_gaps.append(abs(float(energy[j] - energy[i])))
        min_gap_ha = min(selected_gaps) if selected_gaps else 0.0

        atom_force_norms = np.linalg.norm(force, axis=1)
        nac_l2 = float(np.linalg.norm(nacv.ravel()))
        nac_rms = float(np.sqrt(np.mean(nacv ** 2))) if nacv.size else 0.0
        nac_max_abs = float(np.max(np.abs(nacv))) if nacv.size else 0.0

        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'step',
                    'time_fs',
                    'td_class',
                    'scf_converged',
                    'scf_nonconverged_streak',
                    'scf_energy',
                    'scf_s2',
                    'active_state',
                    'active_root',
                    'active_role',
                    'active_energy_ha',
                    'active_excitation_ev',
                    'active_s2',
                    'min_selected_gap_ha',
                    'min_selected_gap_ev',
                    'active_x_norm',
                    'active_y_norm',
                    'active_x2_minus_y2',
                    'force_l2',
                    'max_atom_force',
                    'max_atom_force_index',
                    'nac_l2',
                    'nac_rms',
                    'nac_max_abs',
                    'root_xy_norms',
                    'state_map',
                    'gauge_overlaps',
                    'gauge_flipped',
                    'min_mo_overlap',
                    'danger',
                    'danger_reason',
                ])
            writer.writerow([
                int(self._current_step_index),
                float(self._current_time_fs),
                type(mftd).__name__,
                bool(self._last_scf_converged),
                int(self.scf_nonconverged_streak),
                '' if self._last_scf_energy is None else float(self._last_scf_energy),
                '' if self._last_scf_s2 is None else float(self._last_scf_s2),
                int(self.cur_state),
                int(active_root),
                active_entry.role if active_entry else '',
                float(energy[active_idx]),
                float(active_entry.excitation_ev) if active_entry else '',
                float(active_entry.s2) if active_entry else '',
                float(min_gap_ha),
                float(min_gap_ha * 27.211386245988),
                active_xy['x_norm'],
                active_xy['y_norm'],
                active_xy['x2_minus_y2'],
                float(np.linalg.norm(force.ravel())),
                float(np.max(atom_force_norms)) if atom_force_norms.size else 0.0,
                int(np.argmax(atom_force_norms)) if atom_force_norms.size else -1,
                nac_l2,
                nac_rms,
                nac_max_abs,
                self._format_root_xy_summary(mftd, selection),
                ';'.join(f'{state}->{root}' for state, root in sorted(selection.state_map.items())),
                self._current_step_timing.get('gauge_overlaps', ''),
                self._current_step_timing.get('gauge_flipped', ''),
                self._current_step_timing.get('min_mo_overlap', ''),
                bool(selection.danger),
                selection.danger_reason,
            ])

    def _checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_filename

    def _write_checkpoint(
        self,
        step: int,
        time_fs: float,
        position: np.ndarray,
        velocity: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        force: np.ndarray,
        nacv: np.ndarray,
    ) -> None:
        checkpoint_path = self._checkpoint_path()
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
        payload = {
            'step': int(step),
            'time_fs': float(time_fs),
            'position': np.array(position, dtype=float),
            'velocity': np.array(velocity, dtype=float),
            'coefficient': np.array(coefficient, dtype=np.complex128),
            'cur_state': int(self.cur_state),
            'energy': np.array(energy, dtype=float),
            'force': np.array(force, dtype=float),
            'nacv': np.array(nacv, dtype=float),
            'rng_state': np.random.get_state(),
            'gauge_tracker': self.gauge_tracker,
            'nac_monitor': self.nac_monitor,
            'timing_data': self.timing_data,
            'scf_nonconverged_streak': int(self.scf_nonconverged_streak),
        }
        with tmp_path.open('wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, checkpoint_path)

    def _load_checkpoint(self) -> Dict:
        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        with checkpoint_path.open('rb') as handle:
            loaded = pickle.load(handle)
        self.cur_state = int(loaded['cur_state'])
        np.random.set_state(loaded['rng_state'])
        self.gauge_tracker = loaded['gauge_tracker']
        self.nac_monitor = loaded['nac_monitor']
        self.timing_data = list(loaded['timing_data'])
        self.prev_ci_vectors = self.gauge_tracker.prev_ci_vectors
        self.prev_mo_coeff = self.gauge_tracker.prev_mo_coeff
        self.prev_mo_occ = self.gauge_tracker.prev_mo_occ
        self.prev_mol_for_mo_align = self.gauge_tracker.prev_mol
        self.scf_nonconverged_streak = int(loaded.get('scf_nonconverged_streak', 0))
        return loaded

    def _copy_scf(self, mol):
        template = self.tddft._scf
        mf = template.copy() if hasattr(template, "copy") else copy.copy(template)
        mf.reset(mol)
        mf.conv_tol = float(self.scf_conv_tol)
        mf.max_cycle = int(self.scf_max_cycle)
        return mf

    def _new_td(self, mf):
        td = self.sf_method_cls(mf)
        keys = set(getattr(self.tddft, "_keys", ()))
        keys.update(
            {
                "extype",
                "collinear",
                "collinear_samples",
                "conv_tol",
                "lindep",
                "max_cycle",
                "max_memory",
                "positive_eig_threshold",
                "deg_eia_thresh",
                "verbose",
            }
        )
        for key in keys:
            if hasattr(self.tddft, key):
                setattr(td, key, getattr(self.tddft, key))
        td.nstates = max(int(getattr(td, "nstates", 0)), self.sf_selection_n_lowest)
        return td

    def _configure_derivative(self, derivative):
        if hasattr(derivative, "cphf_max_cycle"):
            derivative.cphf_max_cycle = self.cphf_max_cycle
        if hasattr(derivative, "cphf_conv_tol"):
            derivative.cphf_conv_tol = self.cphf_conv_tol
        return derivative

    def calc_electronic(
        self, position: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate GPU UKS/SF-TD energies, active force, and pairwise NACVs."""
        calc_start = time.perf_counter()
        position = np.asarray(position, dtype=float).reshape(self.tddft.mol.natm, 3)

        t0 = time.perf_counter()
        current_mol = self.tddft.mol.set_geom_(
            position, unit="Bohr", inplace=False
        )
        mf = self._copy_scf(current_mol)
        self._current_step_timing["mol_setup_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dm0 = None
        dm_source = self._last_mf
        if dm_source is None and bool(getattr(self.tddft._scf, "converged", False)):
            dm_source = self.tddft._scf
        if dm_source is not None:
            try:
                dm0 = dm_source.make_rdm1()
            except Exception:
                dm0 = None
        mf.kernel(dm0=dm0) if dm0 is not None else mf.kernel()
        ref_energy = float(_asnumpy(mf.e_tot).reshape(-1)[0])
        self._last_scf_converged = bool(getattr(mf, "converged", False))
        self._last_scf_energy = ref_energy
        try:
            self._last_scf_s2 = float(
                _asnumpy(mf.spin_square()[0]).reshape(-1)[0]
            )
        except Exception:
            self._last_scf_s2 = None
        if self._last_scf_converged:
            self.scf_nonconverged_streak = 0
        else:
            self.scf_nonconverged_streak += 1
            logger.warning(
                "GPU SCF did not converge at step %s, time %.3f fs "
                "(streak %d/%d, e_tot %.12f)",
                self._current_step_index,
                self._current_time_fs,
                self.scf_nonconverged_streak,
                self.scf_max_nonconverged_steps,
                ref_energy,
            )
        self._current_step_timing["scf_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        mftd = self._new_td(mf)
        mftd.kernel()
        converged = np.asarray(_asnumpy(mftd.converged), dtype=bool)
        if not np.all(converged[: self.sf_selection_n_lowest]):
            raise RuntimeError("GPU spin-flip TD calculation did not converge")
        self._current_step_timing["tddft_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        selection = select_sf_singlet_manifold(
            mf,
            mftd,
            states=self.states,
            n_lowest=self.sf_selection_n_lowest,
            triplet_zero_threshold_ev=self.triplet_zero_threshold_ev,
            tdtype=self.sf_method_name,
        )
        state_map = selection.state_map
        self._last_state_selection = selection
        append_state_selection_csv(
            self.output_dir / self.state_selection_csv_filename,
            self._current_step_index,
            self._current_time_fs,
            selection,
        )
        append_excited_state_analysis_csv(
            self.output_dir / self.excited_state_analysis_csv_filename,
            self._current_step_index,
            self._current_time_fs,
            self.states,
            selection,
            mftd,
            top_n=int(self.analysis_top_n),
            coeff_threshold=float(self.analysis_coeff_threshold),
            max_states=int(self.analysis_max_states),
        )
        energy = selection.energies_for_states(self.states)
        self._current_step_timing["state_analysis_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        current_ci_vectors = {
            state: _vector_to_numpy(mftd.xy[state_map[state]])
            for state in self.states
        }
        gauge_result = self.gauge_tracker.correct(
            current_mol, mf, current_ci_vectors, mftd.extype
        )
        current_ci_vectors = gauge_result.ci_vectors
        for state in self.states:
            mftd.xy[state_map[state]] = _vector_to_cupy(current_ci_vectors[state])
        self._current_step_timing["gauge_overlaps"] = ";".join(
            f"{state}:{overlap:.6f}"
            for state, overlap in gauge_result.overlaps.items()
        )
        self._current_step_timing["gauge_flipped"] = ";".join(
            str(state)
            for state, flipped in gauge_result.flipped.items()
            if flipped
        )
        self._current_step_timing["min_mo_overlap"] = gauge_result.min_mo_overlap
        self._current_step_timing["phase_correction_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        active_root = state_map[self.cur_state] + 1
        self.tdgrad = self._configure_derivative(mftd.Gradients())
        force = -_asnumpy(self.tdgrad.kernel(state=active_root))
        self._current_step_timing["gradient_s"] = time.perf_counter() - t0

        self.tdnac = self._configure_derivative(mftd.NAC())
        nac_start = time.perf_counter()
        nac_pair_times = []
        nacv = np.zeros((self.Nstates, self.Nstates, current_mol.natm, 3))
        for i, j in self.nac_idx:
            pair_start = time.perf_counter()
            root_i = state_map[self.states[i]] + 1
            root_j = state_map[self.states[j]] + 1
            _, _, _, de_etf_scaled = self.tdnac.kernel(states=[root_i, root_j])
            orientation = 1.0 if root_i < root_j else -1.0
            pair_nac = orientation * _asnumpy(de_etf_scaled)
            nacv[i, j] = pair_nac
            nacv[j, i] = -pair_nac
            nac_pair_times.append(
                f"({self.states[i]}-{self.states[j]}:"
                f"{time.perf_counter() - pair_start:.3f}s)"
            )
        self._current_step_timing["nac_total_s"] = time.perf_counter() - nac_start
        self._current_step_timing["nac_pair_details"] = ";".join(nac_pair_times)

        self.prev_ci_vectors = current_ci_vectors
        self.prev_mo_coeff = self.gauge_tracker.prev_mo_coeff
        self.prev_mo_occ = self.gauge_tracker.prev_mo_occ
        self.prev_mol_for_mo_align = self.gauge_tracker.prev_mol
        self._last_mf = mf
        self.last_td = mftd
        self._write_electronic_diagnostics(mftd, selection, energy, force, nacv)

        max_bad_scf_steps = int(self.scf_max_nonconverged_steps)
        if (
            not self._last_scf_converged
            and self.scf_nonconverged_streak >= max_bad_scf_steps
        ):
            raise RuntimeError(
                "GPU SCF failed to converge for "
                f"{self.scf_nonconverged_streak} consecutive electronic steps "
                f"(limit {max_bad_scf_steps}) at step {self._current_step_index}, "
                f"time {self._current_time_fs:.3f} fs"
            )

        self._current_step_timing["calc_electronic_total_s"] = (
            time.perf_counter() - calc_start
        )
        return energy, force, nacv

    def exp_propagator(self, c: np.ndarray, Veff: np.ndarray, dt: float) -> np.ndarray:
        """
        Propagate quantum coefficients using matrix exponential.
        dc/dt = -i V_eff(R,P) c

        The matrix exponential is computed efficiently using eigenvalue decomposition:
        exp(-i * V_eff * dt) = U * diag(exp(-i * λ_k * dt)) * U†

        Args:
            c (np.ndarray): Current quantum coefficients (Nstates,)
            Veff (np.ndarray): Effective Hamiltonian matrix (Nstates * Nstates)
            dt (float): Time step in atomic units
        
        Returns:
            np.ndarray: Updated quantum coefficients (Nstates,)
        
        Note:
            The effective Hamiltonian includes both diagonal energies and
            off-diagonal nonadiabatic coupling terms.        
        """
        # Diagonalize the effective Hamiltonian
        diags, coeff = np.linalg.eigh(Veff)

        # Compute the matrix exponential
        U = coeff @ np.diag(np.exp(-1j * diags * dt)) @ coeff.T.conj()

        # Apply propagator to coefficients
        c_new = np.dot(U, c)

        # Normalize coefficients
        c_new = c_new / np.linalg.norm(c_new)

        return c_new

    def update_coefficient(self, coeffs: np.ndarray, 
                           energy: np.ndarray, 
                           nact: np.ndarray) -> np.ndarray:
        """
        Update quantum coefficients using the effective Hamiltonian.
        
        The effective Hamiltonian in the FSSH method combines:
        1. Diagonal electronic energies: E_ii(R)
        2. Off-diagonal nonadiabatic coupling: -i * κ_ij
        
        V_eff = E(R) - i * d(R) * P/m = diag(E) - i * κTDC
        
        Args:
            coeffs (np.ndarray): Current quantum coefficients (Nstates,)
            energy (np.ndarray): Electronic energies (Nstates,)
            nact (np.ndarray): κTDC coupling matrix (Nstates * Nstates)
        
        Returns:
            np.ndarray: Updated quantum coefficients (Nstates,)
        """
        # Construct effective Hamiltonian
        Veff = np.diag(energy) - 1j * nact

        # Propagate coefficients
        c_new = self.exp_propagator(coeffs, Veff, self.dt)

        return c_new
    
    def compute_hopping_probability(self, 
                                    coeffs: np.ndarray, 
                                    nact: np.ndarray) -> np.ndarray:
        """
        Calculate surface hopping probabilities using Tully's formula.
        
        The hopping probability from the current state i to state j is:
        g_ij = (2 * Re(κ_ij * c_i* * c_j) - 2 / ħ * Im(V_ij * c_i* * c_j)) * dt / |c_i|²
        p_ij = max(0, g_ij)
        p_ij = min(1, p_ij)

        Args:
            coeffs (np.ndarray): Current quantum coefficients (Nstates,)
            nact (np.ndarray): κTDC coupling matrix (Nstates * Nstates)
        
        Returns:
            np.ndarray: Hopping probabilities from current state (Nstates,)
        """

        # Get index of current state in the states list
        state_idx = self.states.index(self.cur_state)

        # Current state coefficient
        c_i = coeffs[state_idx]

        # Calculate hopping probabilities
        g_ij = 2 * (nact[state_idx] * c_i.conj() * coeffs).real * self.dt / (np.abs(c_i)**2)

        # Adjust hopping probabilities
        p_ij = np.where(g_ij < 0, 0, g_ij)
        p_ij = np.where(p_ij > 1, 1, p_ij)
        
        return p_ij
    
    def check_hop(self, r: float, p_ij: np.ndarray) -> int:
        """
        Determine if a surface hop occurs.

        The hopping decision is made by comparing a random number r ∈ [0,1)
        with cumulative probabilities. A hop to state k occurs if:
        Σ_{j=0}^{k-1} p_j < r ≤ Σ_{j=0}^{k} p_j
        
        Args:
            r (float): Random number between 0 and 1
            p_ij (np.ndarray): Hopping probabilities (Nstates,)
        
        Returns:
            int: Index of target state (-1 if no hop occurs)
        
        Note:
            Returns -1 if no hop occurs (r falls in the "stay" probability region)
        """
        # Calculate cumulative probabilities
        cumu_p_ij = np.cumsum(p_ij)

        # Check each state for hopping condition
        for k, u_bound in enumerate(cumu_p_ij):
            l_bound = 0.0 if k == 0 else cumu_p_ij[k-1]

            if l_bound < r <= u_bound:
                return k
            
        return -1
    
    def rescale_velocity(self, 
                         hop_index: int,
                         energy: np.ndarray,
                         velocity: np.ndarray,
                         d_vec: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Rescale nuclear velocities to conserve total energy after surface hopping.
        
        When a surface hop occurs, the nuclear kinetic energy must be adjusted to
        compensate for the change in electronic energy. This is achieved by solving
        the energy conservation equation:
        
        1/2m(v')² + E_new = 1/2mv² + E_old

        The new velocity is:
        v' = v - gamma * d_vec / mass

        if delta > 0:
            gamma = (b +- sqrt(b^2 - 4ac)) / 2a
            a = sum_i (d_i^2 / 2m_i)
            b = sum_i (v_i * d_i)
            c = E_new - E_old
        else:
            gamma = b / a

        Args:
            hop_index (int): Index of target state in states list
            energy (np.ndarray): Electronic energies for all states
            velocity (np.ndarray): Current nuclear velocities (Natoms × 3)
            d_vec (np.ndarray): Difference vector for velocity adjustment
        
        Returns:
            Tuple[bool, np.ndarray]: 
                - hop_allowed: Whether the hop is energetically allowed
                - velocity: Updated nuclear velocities
        """

        # To conserve energy, the new velocity v' = v - gamma * d_vec / mass must satisfy 
        # the energy conservation equation, which leads to a quadratic equation for the 
        # scaling factor gamma:
        #     a*gamma^2 - b*gamma + c = 0
        # where:
        #     a = sum_i (d_i^2 / 2m_i)
        #     b = sum_i (v_i * d_i)
        #     c = E_new - E_old

        # Get index of current state in the states list
        state_idx = self.states.index(self.cur_state)

        # Coefficients for the quadratic equation
        a = np.sum(d_vec**2 / (2 * self.mass))
        b = np.sum(velocity * d_vec)
        c = energy[hop_index] - energy[state_idx]

        # Discriminant of the quadratic equation
        delta = b**2 - 4 * a * c

        if delta >= 0:
            gamma = (b + np.sqrt(delta)) / (2 * a) if b < 0 else (b - np.sqrt(delta)) / (2 * a)
            velocity -= gamma * d_vec / self.mass
            return True, velocity
        else:
            gamma = b / a
            velocity -= gamma * d_vec / self.mass
            return False, velocity
    
    # NOT TESTED YET!!!!(去相干校正，未测试，不想用的话注释掉就行)
    def decoherence(self,
                    coeffs: np.ndarray,
                    velocity: np.ndarray,
                    energy: np.ndarray) -> np.ndarray:
        """
        Decoherence.

        c_j = c_j * exp(-dt / tau_ji)
        c_i = c_i * sqrt((1 - sum_j(j!=i) |c_j|**2) / |c_i|**2)
        tau_ji = ħ / |E_jj - E_ii| * (1 + a / E_kin)
        """

        E_kin = (0.5 * self.mass * np.sum(velocity ** 2)).sum()
        cumu_sum = 0
        cur_idx = self.states.index(self.cur_state)
        
        for i in range(len(coeffs)):
            if i != cur_idx:
                tau_ji = 1 / np.abs(energy[i] - energy[cur_idx]) * (1 + self.alpha / E_kin)
                coeffs[i] = coeffs[i] * np.exp(-self.dt / tau_ji)
                cumu_sum += np.abs(coeffs[i]) ** 2
        
        coeffs[cur_idx] = np.sqrt((1 - cumu_sum) / np.abs(coeffs[cur_idx]) ** 2) * coeffs[cur_idx]
        return coeffs

    def write_trajectory(self, 
                         step: int, 
                         position: np.ndarray, 
                         velocity: np.ndarray,
                         energy: np.ndarray, 
                         coeffs: np.ndarray, 
                         filename: str = 'trajectory.xyz') -> None:
        """
        Write current trajectory frame to XYZ file with comprehensive metadata.
        
        Args:
            step (int): Current simulation step
            position (np.ndarray): Nuclear coordinates in Bohr
            velocity (np.ndarray): Nuclear velocities in atomic units
            energy (np.ndarray): Electronic energies in Hartree
            coeffs (np.ndarray): Quantum coefficients
            filename (str): Output filename
        """
        filepath = self.output_dir / filename
        mode = 'w' if step == 0 else 'a'
        
        with open(filepath, mode) as f:
            # Write number of atoms
            f.write(f'{self.tddft.mol.natm}\n')
            
            # Write comment line with simulation data
            time_fs = step * self.dt / FS2AUTIME
            current_energy = energy[self.states.index(self.cur_state)]
            
            comment = (f'Step {step}, Time {time_fs:.3f} fs, '
                       f'State {self.cur_state}, Energy {current_energy:.8f} Ha, '
                       f'Coefficient {coeffs}')
            f.write(comment + '\n')
            
            # Write atomic coordinates
            for i, coord in enumerate(position):
                symbol = self.tddft.mol.atom_pure_symbol(i)
                x, y, z = coord / A2BOHR  # Convert to Angstrom
                f.write(f'{symbol:4.2s} {x:12.6f} {y:12.6f} {z:12.6f}\n')
    
    def print_step_info(self, 
                        step: int, 
                        total_time: float, 
                        energy: np.ndarray,
                        coeffs: np.ndarray, 
                        ) -> None:
        """
        Print detailed information about the current simulation step.
        
        Args:
            step (int): Current step number
            total_time (float): Total simulation time in fs
            energy (np.ndarray): Electronic energies
            coeffs (np.ndarray): Quantum coefficients
            nact (np.ndarray): κTDC coupling matrix
            hop_occurred (bool): Whether a hop occurred in this step
        """
        
        current_idx = self.states.index(self.cur_state)
        current_energy = energy[current_idx]
        populations = np.abs(coeffs)**2
            
        # Format output
        logger.info(f"Step {step:4d}: Time {total_time:8.3f} fs, State {self.cur_state:2d}, "
              f"Energy {current_energy:12.8f} Ha, Populations: {populations}")

    def kernel(self, 
               position: Optional[np.ndarray] = None, 
               velocity: Optional[np.ndarray] = None, 
               coefficient: Optional[np.ndarray] = None,
               resume: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the main FSSH trajectory simulation.
        
        This method implements the complete FSSH algorithm using the velocity Verlet
        integration scheme.

        Integration Frame Ref:
            Nonadiabatic Field on Quantum Phase Space: A Century after Ehrenfest
            Baihua Wu, Xin He, and Jian Liu
            The Journal of Physical Chemistry Letters 2024 15 (2), 644-658
            DOI: 10.1021/acs.jpclett.3c03385

        Args:
            position (Optional[np.ndarray]): Initial nuclear coordinates in Bohr
                If None, uses equilibrium geometry from TDDFT object
            velocity (Optional[np.ndarray]): Initial nuclear velocities in Angstrom/ps
                Must be provided for dynamics simulation
            coefficient (Optional[np.ndarray]): Initial quantum coefficients
                If None, starts in the first specified state
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Final nuclear positions in Angstrom
                - Final nuclear velocities in Angstrom/ps
                - Final quantum coefficients
        """
    
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"Starting FSSH trajectory simulation at {now_str}")
        start_time = time.time()
        
        if resume is None:
            resume = bool(getattr(self, 'resume', False))

        if resume:
            checkpoint = self._load_checkpoint()
            start_step = int(checkpoint['step'])
            total_time = float(checkpoint['time_fs'])
            position = np.array(checkpoint['position'], dtype=float)
            velocity = np.array(checkpoint['velocity'], dtype=float)
            coefficient = np.array(checkpoint['coefficient'], dtype=np.complex128)
            energy = np.array(checkpoint['energy'], dtype=float)
            force = np.array(checkpoint['force'], dtype=float)
            nacv = np.array(checkpoint['nacv'], dtype=float)
            logger.info(f"Resuming from checkpoint step {start_step}, time {total_time:.3f} fs")
        else:
            self._reset_run_outputs()
            start_step = 0
            total_time = 0.0
            # Initialize or validate input parameters
            if position is None:
                position = self.tddft.mol.atom_coords(unit='Bohr')
            else:
                position = np.asarray(position, dtype=float)

            if velocity is None:
                raise ValueError("Initial velocity must be provided for a fresh FSSH run.")
            velocity = velocity*A2BOHR/ (FS2AUTIME * 1e3) # (Natoms, 3) Angstrom/ps -> Bohr/a.u.time

            if coefficient is None:
                coefficient = np.zeros(self.Nstates, dtype=complex)
                coefficient[self.states.index(self.cur_state)] = 1.0
            norm = np.linalg.norm(coefficient)
            if norm == 0:
                raise ValueError("Initial coefficient norm must be non-zero.")
            coefficient = coefficient / norm

            self._current_step_index = 0
            self._current_time_fs = 0.0
            energy, force, nacv = self.calc_electronic(position)
            self.write_trajectory(0, position, velocity, energy, coefficient)
            self.nac_monitor.record(
                0,
                0.0,
                self.cur_state,
                nacv,
                danger=bool(self._last_state_selection and self._last_state_selection.danger),
            )
            self._write_checkpoint(0, 0.0, position, velocity, coefficient, energy, force, nacv)
            self._init_timing_csv()
        
        # Main simulation loop
        logger.info(f"Starting main simulation loop for {self.nsteps} steps")
        
        for i in range(start_step, self.nsteps):
            # 本步开始时间
            step_start = time.perf_counter()
            
            # 重置当前步的计时字典
            self._current_step_timing = {}
            
            # 初始化本步计时字典
            timing_dict = {
                'step': i + 1,
                'hop_occurred': False,
            }
            
            # 1. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass
            
            # 2. update the nuclear coordinate within a full-time step
            position = position + self.dt * velocity
            
            # 3. calculte new energy, force, and nacv
            # calc_electronic 内部会自动记录各子步骤耗时到 self._current_step_timing
            next_step = i + 1
            next_time_fs = next_step * self.dt / FS2AUTIME
            self._current_step_index = next_step
            self._current_time_fs = next_time_fs
            energy, force, nacv = self.calc_electronic(position)
            
            # 4. update the electronic amplitude within a full-time step
            nact = np.einsum('ijnd,nd->ij', nacv, velocity)
            coefficient = self.update_coefficient(coefficient, energy, nact)
            
            # 5. evaluate the switching probability
            p_ij = self.compute_hopping_probability(coefficient, nact)
            r = np.random.rand()
            hop_index = self.check_hop(r, p_ij)

            logger.debug(f"Switching probability: {p_ij}, Random number: {r}")
            
            # 6. adjust nuclear velocity
            cur_idx = self.states.index(self.cur_state)
            if hop_index != -1 and hop_index != cur_idx:
          
                # Attempt velocity rescaling
                d_vec = nacv[cur_idx, hop_index]
                hop_allowed, velocity = self.rescale_velocity(hop_index, energy, velocity, d_vec)
                
                if hop_allowed:
                    old_state = self.cur_state
                    self.cur_state = self.states[hop_index]
                    timing_dict['hop_occurred'] = True
                    
                    logger.info(f"Hop: {old_state} → {self.cur_state} at step {i + 1}")

                else:
                    logger.debug(f"Hop to state {self.states[hop_index]} rejected "
                                 f"due to insufficient kinetic energy")
            
            # 7. update nuclear velocity within a half time step
            velocity = velocity + 0.5 * self.dt * force / self.mass
            
            # 8. update total time
            total_time = next_time_fs
            timing_dict['time_fs'] = total_time

            # 9. decoherence
            coefficient = self.decoherence(coefficient, velocity, energy)
            
            # 10. 写入轨迹
            self.write_trajectory(i + 1, position, velocity, energy, coefficient)   
            self.nac_monitor.record(
                i + 1,
                total_time,
                self.cur_state,
                nacv,
                danger=bool(self._last_state_selection and self._last_state_selection.danger),
            )
            self._write_checkpoint(i + 1, total_time, position, velocity, coefficient, energy, force, nacv)
            
            # 记录当前态
            timing_dict['current_state'] = self.cur_state
            
            # 本步总耗时
            step_total = time.perf_counter() - step_start
            timing_dict['step_total_s'] = step_total
            
            # 从 calc_electronic 内部获取详细计时
            timing_dict['mol_setup_s'] = self._current_step_timing.get('mol_setup_s', 0.0)
            timing_dict['scf_s'] = self._current_step_timing.get('scf_s', 0.0)
            timing_dict['tddft_s'] = self._current_step_timing.get('tddft_s', 0.0)
            timing_dict['state_analysis_s'] = self._current_step_timing.get('state_analysis_s', 0.0)
            timing_dict['gradient_s'] = self._current_step_timing.get('gradient_s', 0.0)
            timing_dict['phase_correction_s'] = self._current_step_timing.get('phase_correction_s', 0.0)
            timing_dict['nac_total_s'] = self._current_step_timing.get('nac_total_s', 0.0)
            timing_dict['nac_pair_details'] = self._current_step_timing.get('nac_pair_details', '')
            timing_dict['calc_electronic_total_s'] = self._current_step_timing.get('calc_electronic_total_s', 0.0)
            
            # 计算外部其他步骤耗时 = 总耗时 - calc_electronic耗时
            timing_dict['other_steps_s'] = step_total - timing_dict['calc_electronic_total_s']
            
            # 将本步计时数据写入CSV
            self._write_timing_row(timing_dict)
            
            # 同时将数据保存到内存中（可选，用于后续分析）
            self.timing_data.append(timing_dict.copy())
            
            self.print_step_info(i + 1, total_time, energy, coefficient)
        
        # Simulation completed successfully
        elapsed_time = time.time() - start_time
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logger.info(f"FSSH simulation completed successfully at {now_str}")
        logger.info(f"Total simulation time: {elapsed_time:.2f} s")
        logger.info(f"Timing statistics saved to: {self.output_dir / self.timing_csv_filename}")
            
        # Convert results back to user units
        final_position = position / A2BOHR  # Angstrom
        final_velocity = velocity * (FS2AUTIME * 1e3) / A2BOHR  # Angstrom/ps
        
        return final_position, final_velocity, coefficient


FSSH_SFTDA = FSSH_SF
FSSH_SFTDDFT = FSSH_SF
