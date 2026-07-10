# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import cupy as cp
import numpy as np
from pyscf import gto
from scipy.optimize import linear_sum_assignment

from gpu4pyscf.lib import logger
from gpu4pyscf.md.fssh import FSSH, PES

EV_PER_HARTREE = 27.211386245988


def _asnumpy(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)


def _array_or_zero_to_numpy(value):
    if isinstance(value, (int, float)) and value == 0:
        return 0
    return _asnumpy(value)


def _array_or_zero_to_cupy(value):
    if isinstance(value, (int, float)) and value == 0:
        return 0
    return cp.asarray(value)


def _vector_to_numpy(vec):
    x, y = vec
    return (_array_or_zero_to_numpy(x), _array_or_zero_to_numpy(y))


def _vector_to_cupy(vec):
    x, y = vec
    return (_array_or_zero_to_cupy(x), _array_or_zero_to_cupy(y))


def _dot_block(block1, block2) -> float:
    if isinstance(block1, np.ndarray) and isinstance(block2, np.ndarray):
        return float(np.dot(block1.ravel(), block2.ravel()))
    return 0.0


def _sf_vector_dot(vec1, vec2) -> float:
    x1, y1 = vec1
    x2, y2 = vec2
    return _dot_block(x1, x2) - _dot_block(y1, y2)


def _scale_block(block, factor):
    if isinstance(block, np.ndarray):
        return block * factor
    return block


def _scale_sf_vector(vec, factor: float):
    x, y = vec
    return (_scale_block(x, factor), _scale_block(y, factor))


@dataclass
class SFStateRoot:
    root: int
    role: str
    excitation_ha: float
    excitation_ev: float
    total_energy_ha: float
    s2: float


@dataclass
class SFStateSelection:
    roots: list[SFStateRoot]
    state_map: dict[int, int]
    danger: bool
    danger_reason: str
    triplet_root: int
    triplet_excitation_ev: float

    def energies_for_states(self, states: Sequence[int]) -> np.ndarray:
        by_root = {entry.root: entry.total_energy_ha for entry in self.roots}
        return np.array([by_root[self.state_map[state]] for state in states], dtype=float)


def _spin_square_root(td, root: int) -> float:
    # gpu4pyscf.tdscf.uhf.SpinFlipTDA.spin_square uses 0-based root indices.
    return float(_asnumpy(td.spin_square(state=root)).reshape(-1)[0])


def select_sf_singlet_manifold(
    mf,
    td,
    states: Sequence[int],
    n_lowest: int = 4,
    triplet_zero_threshold_ev: float = 0.3,
) -> SFStateSelection:
    if not states:
        raise ValueError("states must not be empty")
    if any(state < 1 for state in states):
        raise ValueError("SFTDA-FSSH states are 1-based singlet labels")
    if max(states) > n_lowest - 1:
        raise ValueError(
            f"states={list(states)} requires more singlet candidates than "
            f"available from n_lowest={n_lowest}"
        )
    if len(td.xy) < n_lowest or len(td.e) < n_lowest:
        raise RuntimeError(
            f"SFTDA returned fewer than {n_lowest} roots; cannot select "
            "the low-root singlet manifold."
        )

    roots = []
    for root in range(n_lowest):
        exc_ha = float(_asnumpy(td.e[root]))
        roots.append(
            SFStateRoot(
                root=root,
                role="unassigned",
                excitation_ha=exc_ha,
                excitation_ev=exc_ha * EV_PER_HARTREE,
                total_energy_ha=float(mf.e_tot + exc_ha),
                s2=_spin_square_root(td, root),
            )
        )

    triplet = max(roots, key=lambda item: item.s2)
    singlets = sorted(
        (item for item in roots if item.root != triplet.root),
        key=lambda item: item.excitation_ha,
    )

    triplet.role = "T_candidate"
    for idx, item in enumerate(singlets):
        item.role = f"S{idx}"

    state_map = {state: singlets[state - 1].root for state in states}
    danger = abs(triplet.excitation_ev) > triplet_zero_threshold_ev
    danger_reason = ""
    if danger:
        danger_reason = (
            f"abs(T_candidate excitation energy) = "
            f"{abs(triplet.excitation_ev):.6f} eV > "
            f"{triplet_zero_threshold_ev:.6f} eV"
        )

    return SFStateSelection(
        roots=sorted(roots, key=lambda item: item.root),
        state_map=state_map,
        danger=bool(danger),
        danger_reason=danger_reason,
        triplet_root=triplet.root,
        triplet_excitation_ev=triplet.excitation_ev,
    )


def _match_and_reorder_mos(
    s12_ao: np.ndarray,
    mo_coeff_ref: np.ndarray,
    mo_coeff_cur: np.ndarray,
    threshold: float,
):
    if mo_coeff_ref.shape != mo_coeff_cur.shape:
        raise ValueError("Reference and current MO coefficient arrays must match")
    if s12_ao.shape[0] != s12_ao.shape[1] or s12_ao.shape[0] != mo_coeff_ref.shape[0]:
        raise ValueError("AO overlap matrix shape does not match MO coefficients")

    mo_overlap = mo_coeff_ref.T @ s12_ao @ mo_coeff_cur
    abs_overlap = np.abs(mo_overlap)
    cost = -abs_overlap
    cost[abs_overlap < threshold] = mo_coeff_ref.shape[1] + 1
    row_ind, col_ind = linear_sum_assignment(cost)
    chosen = abs_overlap[row_ind, col_ind]
    sign = np.sign(mo_overlap[row_ind, col_ind])
    sign[sign == 0] = 1.0
    return col_ind, sign, chosen


def _spin_alignment(
    s12_ao: np.ndarray,
    ref_coeff: np.ndarray,
    ref_occ: np.ndarray,
    cur_coeff: np.ndarray,
    cur_occ: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    ref_occ_idx = np.where(ref_occ > 0)[0]
    ref_vir_idx = np.where(ref_occ == 0)[0]
    cur_occ_idx = np.where(cur_occ > 0)[0]
    cur_vir_idx = np.where(cur_occ == 0)[0]

    if len(ref_occ_idx) != len(cur_occ_idx) or len(ref_vir_idx) != len(cur_vir_idx):
        raise ValueError("Occupation pattern changed; cannot align SF-TD vectors")

    if len(cur_occ_idx):
        occ_perm, occ_sign, occ_overlap = _match_and_reorder_mos(
            s12_ao,
            ref_coeff[:, ref_occ_idx],
            cur_coeff[:, cur_occ_idx],
            threshold,
        )
    else:
        occ_perm = np.array([], dtype=int)
        occ_sign = np.array([], dtype=float)
        occ_overlap = np.array([], dtype=float)

    if len(cur_vir_idx):
        vir_perm, vir_sign, vir_overlap = _match_and_reorder_mos(
            s12_ao,
            ref_coeff[:, ref_vir_idx],
            cur_coeff[:, cur_vir_idx],
            threshold,
        )
    else:
        vir_perm = np.array([], dtype=int)
        vir_sign = np.array([], dtype=float)
        vir_overlap = np.array([], dtype=float)

    return {
        "occ_perm": occ_perm,
        "occ_sign": occ_sign,
        "occ_overlap": occ_overlap,
        "vir_perm": vir_perm,
        "vir_sign": vir_sign,
        "vir_overlap": vir_overlap,
    }


def _transform_block(block, row_perm, col_perm, row_sign, col_sign):
    if not isinstance(block, np.ndarray):
        return block
    transformed = np.array(block, copy=True)
    if transformed.size == 0:
        return transformed
    transformed = transformed[np.asarray(row_perm, dtype=int), :]
    transformed = transformed[:, np.asarray(col_perm, dtype=int)]
    transformed = transformed * np.asarray(row_sign, dtype=float)[:, None]
    transformed = transformed * np.asarray(col_sign, dtype=float)[None, :]
    return transformed


def _transform_sf_vector_to_reference_gauge(vec, alignment, extype: int):
    x, y = vec
    if extype == 0:
        x_new = _transform_block(
            x,
            alignment["beta"]["occ_perm"],
            alignment["alpha"]["vir_perm"],
            alignment["beta"]["occ_sign"],
            alignment["alpha"]["vir_sign"],
        )
        y_new = _transform_block(
            y,
            alignment["alpha"]["occ_perm"],
            alignment["beta"]["vir_perm"],
            alignment["alpha"]["occ_sign"],
            alignment["beta"]["vir_sign"],
        )
    elif extype == 1:
        x_new = _transform_block(
            x,
            alignment["alpha"]["occ_perm"],
            alignment["beta"]["vir_perm"],
            alignment["alpha"]["occ_sign"],
            alignment["beta"]["vir_sign"],
        )
        y_new = _transform_block(
            y,
            alignment["beta"]["occ_perm"],
            alignment["alpha"]["vir_perm"],
            alignment["beta"]["occ_sign"],
            alignment["alpha"]["vir_sign"],
        )
    else:
        raise ValueError(f"Unsupported spin-flip excitation type extype={extype}")
    return (x_new, y_new)


@dataclass
class GaugeCorrectionResult:
    ci_vectors: dict[int, object]
    overlaps: dict[int, float]
    flipped: dict[int, bool]
    min_mo_overlap: Optional[float]


class GaugeTracker:
    def __init__(self, mo_align_threshold: float = 0.4):
        self.mo_align_threshold = mo_align_threshold
        self.prev_ci_vectors = None
        self.prev_mo_coeff = None
        self.prev_mo_occ = None
        self.prev_mol = None

    def correct(self, mol, mf, ci_vectors: Mapping[int, object], extype: int):
        corrected = {state: copy.deepcopy(vec) for state, vec in ci_vectors.items()}
        overlaps = {}
        flipped = {state: False for state in corrected}
        min_mo_overlap = None

        if self.prev_ci_vectors is not None and self.prev_mol is not None:
            alignment = self._build_alignment(mol, mf)
            overlap_arrays = [
                alignment[spin][key]
                for spin in ("alpha", "beta")
                for key in ("occ_overlap", "vir_overlap")
            ]
            nonempty = [arr for arr in overlap_arrays if arr.size]
            if nonempty:
                min_mo_overlap = float(min(arr.min() for arr in nonempty))

            for state, vec in corrected.items():
                if state not in self.prev_ci_vectors:
                    continue
                aligned = _transform_sf_vector_to_reference_gauge(
                    vec, alignment, extype
                )
                overlap = _sf_vector_dot(self.prev_ci_vectors[state], aligned)
                overlaps[state] = float(overlap)
                if overlap < 0:
                    corrected[state] = _scale_sf_vector(vec, -1.0)
                    flipped[state] = True

        self.prev_ci_vectors = {state: copy.deepcopy(vec) for state, vec in corrected.items()}
        self.prev_mo_coeff = (
            np.array(_asnumpy(mf.mo_coeff[0]), copy=True),
            np.array(_asnumpy(mf.mo_coeff[1]), copy=True),
        )
        self.prev_mo_occ = (
            np.array(_asnumpy(mf.mo_occ[0]), copy=True),
            np.array(_asnumpy(mf.mo_occ[1]), copy=True),
        )
        self.prev_mol = mol.copy()
        return GaugeCorrectionResult(corrected, overlaps, flipped, min_mo_overlap)

    def _build_alignment(self, mol, mf):
        if self.prev_mo_coeff is None or self.prev_mo_occ is None:
            raise RuntimeError("Previous MO data is not available")
        s12_ao = gto.intor_cross("int1e_ovlp", self.prev_mol, mol)
        cur_coeff = (_asnumpy(mf.mo_coeff[0]), _asnumpy(mf.mo_coeff[1]))
        cur_occ = (_asnumpy(mf.mo_occ[0]), _asnumpy(mf.mo_occ[1]))
        return {
            "alpha": _spin_alignment(
                s12_ao,
                self.prev_mo_coeff[0],
                self.prev_mo_occ[0],
                cur_coeff[0],
                cur_occ[0],
                self.mo_align_threshold,
            ),
            "beta": _spin_alignment(
                s12_ao,
                self.prev_mo_coeff[1],
                self.prev_mo_occ[1],
                cur_coeff[1],
                cur_occ[1],
                self.mo_align_threshold,
            ),
        }


class FSSH_SFTDA(FSSH):
    """FSSH driver for GPU spin-flip TDA/TDDFT electronic structure.

    The public ``states`` argument uses the CPU-reference convention: 1-based
    singlet labels after the low-root SF manifold selection. Internally, each
    label is mapped to the current GPU TD root before evaluating gradients and
    excited-state NACs.
    """

    diagnostic_files = ("state_selection.csv", "electronic_diagnostics.csv")

    def __init__(
        self,
        td,
        states: Sequence[int],
        n_lowest: int = 4,
        triplet_zero_threshold_ev: float = 0.3,
        mo_align_threshold: float = 0.4,
        write_diagnostics: bool = True,
    ):
        if any(state < 1 for state in states):
            raise ValueError("SFTDA-FSSH states must be 1-based positive singlet labels")
        if len(states) < 2:
            raise ValueError("At least two SFTDA-FSSH states are required")

        self.td_template = td
        self.td_class = td.__class__
        self.scf_template = td._scf
        self.n_lowest = max(int(n_lowest), max(states) + 1)
        self.triplet_zero_threshold_ev = float(triplet_zero_threshold_ev)
        self.gauge_tracker = GaugeTracker(mo_align_threshold)
        self.write_diagnostics = bool(write_diagnostics)
        self._diagnostics_initialized = False
        self._pes_eval_count = 0
        self._last_mf = None
        self.last_td = None
        self.last_selection = None
        super().__init__(td.mol, list(states))
        self.verbose = getattr(td, "verbose", self.verbose)

    def _copy_scf(self, mol):
        if hasattr(self.scf_template, "copy"):
            mf = self.scf_template.copy()
        else:
            mf = copy.copy(self.scf_template)
        mf.reset(mol)
        return mf

    def _new_td(self, mf):
        td = self.td_class(mf)
        copied_keys = set(getattr(self.td_template, "_keys", ()))
        copied_keys.update(
            {
                "nstates",
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
        for key in copied_keys:
            if hasattr(self.td_template, key):
                setattr(td, key, getattr(self.td_template, key))
        td.nstates = max(int(getattr(td, "nstates", self.n_lowest)), self.n_lowest)
        return td

    def _run_scf(self, mf):
        dm0 = None
        if self._last_mf is not None:
            try:
                dm0 = self._last_mf.make_rdm1()
            except Exception:
                dm0 = None
        if dm0 is None:
            mf.kernel()
        else:
            mf.kernel(dm0=dm0)
        if not bool(getattr(mf, "converged", False)):
            raise RuntimeError("GPU SCF did not converge in SFTDA-FSSH electronic step")

    def _run_td(self, mf):
        td = self._new_td(mf)
        td.kernel()
        converged = np.asarray(_asnumpy(td.converged), dtype=bool)
        if not all(converged[: self.n_lowest]):
            raise RuntimeError("GPU spin-flip TD calculation did not converge")
        return td

    def _diagnostics_dir(self):
        return Path(self.filename).expanduser().resolve().parent

    def _initialize_diagnostics(self):
        if self._diagnostics_initialized or not self.write_diagnostics:
            return
        out_dir = self._diagnostics_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in self.diagnostic_files:
            path = out_dir / name
            if path.exists():
                path.unlink()
        self._diagnostics_initialized = True

    def _write_state_selection(self, step: int, time_fs: float, selection: SFStateSelection):
        path = self._diagnostics_dir() / "state_selection.csv"
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(
                    [
                        "step",
                        "time_fs",
                        "root",
                        "role",
                        "excitation_ha",
                        "excitation_ev",
                        "total_energy_ha",
                        "s2",
                        "danger",
                        "danger_reason",
                    ]
                )
            for entry in selection.roots:
                writer.writerow(
                    [
                        int(step),
                        float(time_fs),
                        int(entry.root),
                        entry.role,
                        float(entry.excitation_ha),
                        float(entry.excitation_ev),
                        float(entry.total_energy_ha),
                        float(entry.s2),
                        bool(selection.danger),
                        selection.danger_reason,
                    ]
                )

    def _xy_norm_metrics(self, xy):
        x, y = _vector_to_numpy(xy)
        x2 = float(np.vdot(x, x).real) if isinstance(x, np.ndarray) else 0.0
        y2 = float(np.vdot(y, y).real) if isinstance(y, np.ndarray) else 0.0
        return {
            "x_norm": float(np.sqrt(max(x2, 0.0))),
            "y_norm": float(np.sqrt(max(y2, 0.0))),
            "x2_minus_y2": float(x2 - y2),
        }

    def _write_electronic_diagnostics(
        self,
        step: int,
        time_fs: float,
        td,
        selection: SFStateSelection,
        active_state: int,
        energy: np.ndarray,
        force: np.ndarray,
        nacv: Optional[np.ndarray],
        gauge_result: GaugeCorrectionResult,
    ):
        path = self._diagnostics_dir() / "electronic_diagnostics.csv"
        write_header = not path.exists()
        state_pos = {state: idx for idx, state in enumerate(self.states)}
        active_idx = state_pos[active_state]
        active_root = selection.state_map[active_state]
        by_root = {entry.root: entry for entry in selection.roots}
        active_entry = by_root[active_root]
        active_xy = self._xy_norm_metrics(td.xy[active_root])

        selected_gaps = []
        for i in range(len(energy) - 1):
            for j in range(i + 1, len(energy)):
                selected_gaps.append(abs(float(energy[j] - energy[i])))
        min_gap_ha = min(selected_gaps) if selected_gaps else 0.0

        atom_force_norms = np.linalg.norm(force, axis=1)
        if nacv is None:
            nac_l2 = nac_rms = nac_max_abs = 0.0
        else:
            nac_l2 = float(np.linalg.norm(nacv.ravel()))
            nac_rms = float(np.sqrt(np.mean(nacv**2))) if nacv.size else 0.0
            nac_max_abs = float(np.max(np.abs(nacv))) if nacv.size else 0.0

        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if write_header:
                writer.writerow(
                    [
                        "step",
                        "time_fs",
                        "td_class",
                        "active_state",
                        "active_root",
                        "active_role",
                        "active_energy_ha",
                        "active_excitation_ev",
                        "active_s2",
                        "min_selected_gap_ha",
                        "min_selected_gap_ev",
                        "active_x_norm",
                        "active_y_norm",
                        "active_x2_minus_y2",
                        "force_l2",
                        "max_atom_force",
                        "max_atom_force_index",
                        "nac_l2",
                        "nac_rms",
                        "nac_max_abs",
                        "state_map",
                        "gauge_overlaps",
                        "gauge_flipped",
                        "min_mo_overlap",
                        "danger",
                        "danger_reason",
                    ]
                )
            writer.writerow(
                [
                    int(step),
                    float(time_fs),
                    type(td).__name__,
                    int(active_state),
                    int(active_root),
                    active_entry.role,
                    float(energy[active_idx]),
                    float(active_entry.excitation_ev),
                    float(active_entry.s2),
                    float(min_gap_ha),
                    float(min_gap_ha * EV_PER_HARTREE),
                    active_xy["x_norm"],
                    active_xy["y_norm"],
                    active_xy["x2_minus_y2"],
                    float(np.linalg.norm(force.ravel())),
                    float(np.max(atom_force_norms)) if atom_force_norms.size else 0.0,
                    int(np.argmax(atom_force_norms)) if atom_force_norms.size else -1,
                    nac_l2,
                    nac_rms,
                    nac_max_abs,
                    ";".join(
                        f"{state}->{root}"
                        for state, root in sorted(selection.state_map.items())
                    ),
                    ";".join(
                        f"{state}:{overlap:.6f}"
                        for state, overlap in sorted(gauge_result.overlaps.items())
                    ),
                    ";".join(
                        str(state)
                        for state, flipped in sorted(gauge_result.flipped.items())
                        if flipped
                    ),
                    "" if gauge_result.min_mo_overlap is None else gauge_result.min_mo_overlap,
                    bool(selection.danger),
                    selection.danger_reason,
                ]
            )

    def _write_diagnostics(
        self,
        step: int,
        td,
        selection: SFStateSelection,
        active_state: int,
        energy: np.ndarray,
        force: np.ndarray,
        nacv: Optional[np.ndarray],
        gauge_result: GaugeCorrectionResult,
    ):
        if not self.write_diagnostics:
            return
        self._initialize_diagnostics()
        time_fs = step * self.timestep_fs
        self._write_state_selection(step, time_fs, selection)
        self._write_electronic_diagnostics(
            step, time_fs, td, selection, active_state, energy, force, nacv, gauge_result
        )

    def _correct_gauge(self, mol, mf, td, selection):
        current_vectors = {
            state: _vector_to_numpy(td.xy[selection.state_map[state]])
            for state in self.states
        }
        gauge_result = self.gauge_tracker.correct(mol, mf, current_vectors, td.extype)
        for state, vec in gauge_result.ci_vectors.items():
            td.xy[selection.state_map[state]] = _vector_to_cupy(vec)
        return gauge_result

    def evaluate_pes(self, position, cur_state, with_nacv=True):
        mol = self.mol.set_geom_(np.asarray(position).reshape(self.mol.natm, 3),
                                 unit="Bohr", inplace=False)
        mf = self._copy_scf(mol)
        self._run_scf(mf)
        td = self._run_td(mf)
        selection = select_sf_singlet_manifold(
            mf,
            td,
            states=self.states,
            n_lowest=self.n_lowest,
            triplet_zero_threshold_ev=self.triplet_zero_threshold_ev,
        )
        gauge_result = self._correct_gauge(mol, mf, td, selection)

        active_root = selection.state_map[cur_state]
        tdgrad = td.Gradients()
        force = -_asnumpy(tdgrad.kernel(state=active_root + 1))
        energy = selection.energies_for_states(self.states)

        nacv = None
        if with_nacv:
            tdnac = td.NAC()
            natm = mol.natm
            nstates = len(self.states)
            nacv = np.zeros((nstates, nstates, natm, 3))
            for i in range(nstates - 1):
                for j in range(i + 1, nstates):
                    state_i = self.states[i]
                    state_j = self.states[j]
                    root_i = selection.state_map[state_i] + 1
                    root_j = selection.state_map[state_j] + 1
                    _, _, _, de_etf_scaled = tdnac.kernel(states=[root_i, root_j])
                    de_etf_scaled = _asnumpy(de_etf_scaled)
                    nacv[i, j] = de_etf_scaled
                    nacv[j, i] = -de_etf_scaled

        step = self._pes_eval_count
        self._pes_eval_count += 1
        self._write_diagnostics(
            step, td, selection, cur_state, energy, force, nacv, gauge_result
        )

        self._last_mf = mf
        self.last_td = td
        self.last_selection = selection
        log = logger.new_logger(self.mol, self.verbose)
        if selection.danger:
            log.warn(selection.danger_reason)

        return PES(energy=energy, force=force, nacv=nacv)


FSSH = FSSH_SFTDA
