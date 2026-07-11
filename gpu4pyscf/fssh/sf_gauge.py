# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import cupy as cp
from pyscf import gto
from scipy.optimize import linear_sum_assignment


def asnumpy(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)


def _dot_block(block1, block2) -> float:
    if isinstance(block1, np.ndarray) and isinstance(block2, np.ndarray):
        return float(np.dot(block1.ravel(), block2.ravel()))
    return 0.0


def vector_dot(vec1, vec2) -> float:
    x1, y1 = vec1
    x2, y2 = vec2
    return _dot_block(x1, x2) - _dot_block(y1, y2)


def _scale_block(block, factor):
    if isinstance(block, np.ndarray):
        return block * factor
    return block


def scale_vector(vec, factor: float = -1.0):
    x, y = vec
    return (_scale_block(x, factor), _scale_block(y, factor))


def match_and_reorder_mos(
    s12_ao: np.ndarray,
    mo_coeff_ref: np.ndarray,
    mo_coeff_cur: np.ndarray,
    threshold: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mo_coeff_ref.shape != mo_coeff_cur.shape:
        raise ValueError("Reference and current MO coefficient arrays must have the same shape.")
    if s12_ao.shape[0] != s12_ao.shape[1] or s12_ao.shape[0] != mo_coeff_ref.shape[0]:
        raise ValueError("AO overlap matrix shape does not match MO coefficients.")

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
) -> Dict[str, np.ndarray]:
    ref_occ_idx = np.where(ref_occ > 0)[0]
    ref_vir_idx = np.where(ref_occ == 0)[0]
    cur_occ_idx = np.where(cur_occ > 0)[0]
    cur_vir_idx = np.where(cur_occ == 0)[0]

    if len(ref_occ_idx) != len(cur_occ_idx) or len(ref_vir_idx) != len(cur_vir_idx):
        raise ValueError("Occupation pattern changed; cannot safely align MOs.")

    if len(cur_occ_idx):
        occ_perm, occ_sign, occ_overlap = match_and_reorder_mos(
            s12_ao, ref_coeff[:, ref_occ_idx], cur_coeff[:, cur_occ_idx], threshold=threshold
        )
    else:
        occ_perm = np.array([], dtype=int)
        occ_sign = np.array([], dtype=float)
        occ_overlap = np.array([], dtype=float)

    if len(cur_vir_idx):
        vir_perm, vir_sign, vir_overlap = match_and_reorder_mos(
            s12_ao, ref_coeff[:, ref_vir_idx], cur_coeff[:, cur_vir_idx], threshold=threshold
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


def transform_excitation_block(block, row_perm, col_perm, row_sign, col_sign):
    if not isinstance(block, np.ndarray):
        return block
    transformed = np.array(block, copy=True)
    if transformed.size == 0:
        return transformed
    transformed = transformed[np.asarray(row_perm, dtype=int), :]
    transformed = transformed[:, np.asarray(col_perm, dtype=int)]
    transformed = transformed * np.asarray(row_sign, dtype=float)[:, np.newaxis]
    transformed = transformed * np.asarray(col_sign, dtype=float)[np.newaxis, :]
    return transformed


def transform_ci_vector_to_reference_gauge(vec, alignment, extype: int):
    x, y = vec

    if extype == 0:
        x_new = transform_excitation_block(
            x,
            alignment["beta"]["occ_perm"],
            alignment["alpha"]["vir_perm"],
            alignment["beta"]["occ_sign"],
            alignment["alpha"]["vir_sign"],
        )
        y_new = transform_excitation_block(
            y,
            alignment["alpha"]["occ_perm"],
            alignment["beta"]["vir_perm"],
            alignment["alpha"]["occ_sign"],
            alignment["beta"]["vir_sign"],
        )
    elif extype == 1:
        x_new = transform_excitation_block(
            x,
            alignment["alpha"]["occ_perm"],
            alignment["beta"]["vir_perm"],
            alignment["alpha"]["occ_sign"],
            alignment["beta"]["vir_sign"],
        )
        y_new = transform_excitation_block(
            y,
            alignment["beta"]["occ_perm"],
            alignment["alpha"]["vir_perm"],
            alignment["beta"]["occ_sign"],
            alignment["alpha"]["vir_sign"],
        )
    else:
        raise ValueError(f"Unsupported SF excitation type extype={extype}.")

    return (x_new, y_new)


@dataclass
class GaugeCorrectionResult:
    ci_vectors: Dict[int, object]
    overlaps: Dict[int, float]
    flipped: Dict[int, bool]
    min_mo_overlap: Optional[float]


class GaugeTracker:
    def __init__(self, mo_align_threshold: float = 0.4):
        self.mo_align_threshold = mo_align_threshold
        self.prev_ci_vectors = None
        self.prev_mo_coeff = None
        self.prev_mo_occ = None
        self.prev_mol = None

    def correct(self, mol, mf, ci_vectors: Mapping[int, object], extype: int) -> GaugeCorrectionResult:
        corrected = {state: copy.deepcopy(vec) for state, vec in ci_vectors.items()}
        overlaps: Dict[int, float] = {}
        flipped: Dict[int, bool] = {state: False for state in corrected}
        min_mo_overlap: Optional[float] = None

        if self.prev_ci_vectors is not None and self.prev_mol is not None:
            alignment = self._build_alignment(mol, mf)
            overlap_arrays = [
                alignment[spin][key]
                for spin in ("alpha", "beta")
                for key in ("occ_overlap", "vir_overlap")
                if alignment[spin][key].size
            ]
            if overlap_arrays:
                min_mo_overlap = float(min(np.min(values) for values in overlap_arrays))

            for state, vec in list(corrected.items()):
                if state not in self.prev_ci_vectors:
                    continue
                vec_in_prev_gauge = transform_ci_vector_to_reference_gauge(vec, alignment, extype)
                overlap = vector_dot(self.prev_ci_vectors[state], vec_in_prev_gauge)
                overlaps[state] = float(overlap)
                if overlap < 0:
                    corrected[state] = scale_vector(vec, -1.0)
                    flipped[state] = True

        self.update_reference(mol, mf, corrected)
        return GaugeCorrectionResult(corrected, overlaps, flipped, min_mo_overlap)

    def update_reference(self, mol, mf, ci_vectors: Mapping[int, object]) -> None:
        self.prev_ci_vectors = {state: copy.deepcopy(vec) for state, vec in ci_vectors.items()}
        self.prev_mo_coeff = (
            np.array(asnumpy(mf.mo_coeff[0]), copy=True),
            np.array(asnumpy(mf.mo_coeff[1]), copy=True),
        )
        self.prev_mo_occ = (
            np.array(asnumpy(mf.mo_occ[0]), copy=True),
            np.array(asnumpy(mf.mo_occ[1]), copy=True),
        )
        self.prev_mol = mol.copy()

    def _build_alignment(self, mol, mf):
        if self.prev_mo_coeff is None or self.prev_mo_occ is None:
            raise RuntimeError("Gauge reference is incomplete.")
        s12_ao = gto.intor_cross("int1e_ovlp", self.prev_mol, mol)
        cur_coeff = (
            np.array(asnumpy(mf.mo_coeff[0]), copy=True),
            np.array(asnumpy(mf.mo_coeff[1]), copy=True),
        )
        cur_occ = (
            np.array(asnumpy(mf.mo_occ[0]), copy=True),
            np.array(asnumpy(mf.mo_occ[1]), copy=True),
        )
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
