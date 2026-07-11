# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import cupy as cp

from .sf_state_selection import SFStateSelection


def _asnumpy(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)


def _format_component(extype: int, nocc_a: int, nocc_b: int, occ_idx: int, vir_idx: int, coeff: float) -> str:
    if extype == 1:
        occ_label = f"{occ_idx + 1}a"
        vir_label = f"{vir_idx + 1 + nocc_b}b"
    elif extype == 0:
        occ_label = f"{occ_idx + 1}b"
        vir_label = f"{vir_idx + 1 + nocc_a}a"
    else:
        occ_label = f"{occ_idx + 1}"
        vir_label = f"{vir_idx + 1}"
    return f"{occ_label}->{vir_label}:{coeff:+.5f}(|c|^2={abs(coeff) ** 2:.5f})"


def _top_x_components(mftd, root: int, top_n: int, coeff_threshold: float) -> str:
    x, _ = mftd.xy[root]
    if isinstance(x, (int, float)) or x is None:
        return ""
    x = _asnumpy(x)
    if x.size == 0:
        return ""

    flat_abs = np.abs(x).ravel()
    nonzero = np.flatnonzero(flat_abs > 0)
    if nonzero.size == 0:
        return ""

    order = nonzero[np.argsort(flat_abs[nonzero])[::-1]]
    selected = [idx for idx in order if flat_abs[idx] >= coeff_threshold][:top_n]
    if not selected:
        selected = order[: min(top_n, order.size)].tolist()

    mo_occ = mftd._scf.mo_occ
    nocc_a = int(np.count_nonzero(_asnumpy(mo_occ[0]) > 0))
    nocc_b = int(np.count_nonzero(_asnumpy(mo_occ[1]) > 0))

    components = []
    for flat_idx in selected:
        occ_idx, vir_idx = np.unravel_index(int(flat_idx), x.shape)
        coeff = float(x[occ_idx, vir_idx])
        components.append(_format_component(mftd.extype, nocc_a, nocc_b, occ_idx, vir_idx, coeff))
    return "; ".join(components)


def _selected_analysis_entries(selection: SFStateSelection, max_states: int):
    triplets = [entry for entry in selection.roots if entry.role == "T_candidate"]
    singlets = sorted(
        (entry for entry in selection.roots if entry.role.startswith("S")),
        key=lambda entry: int(entry.role[1:]),
    )
    return (triplets + singlets)[:max_states]


def _state_label(role: str) -> str:
    if role == "T_candidate":
        return "T1"
    return role


def _oscillator_strengths_by_root(mftd, roots: Sequence[int]) -> Dict[int, float]:
    if not hasattr(mftd, "oscillator_strength") or len(roots) < 2:
        return {}

    ref_root = roots[0] + 1
    target_roots = [root + 1 for root in roots[1:]]
    if not target_roots:
        return {}

    try:
        f_values = np.atleast_1d(
            _asnumpy(mftd.oscillator_strength(ref=ref_root, state=target_roots))
        )
    except Exception:
        return {}
    return {root: float(fosc) for root, fosc in zip(roots[1:], f_values)}


def append_excited_state_analysis_csv(
    path: Path,
    step: int,
    time_fs: float,
    states: Sequence[int],
    selection: SFStateSelection,
    mftd,
    top_n: int = 5,
    coeff_threshold: float = 0.1,
    max_states: int = 4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    selected_entries = _selected_analysis_entries(selection, max_states=max_states)
    selected_roots = [entry.root for entry in selected_entries]
    oscillator_strengths = _oscillator_strengths_by_root(mftd, selected_roots)
    ref_label = _state_label(selected_entries[0].role) if len(selected_entries) >= 2 else None

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "step",
                    "time_fs",
                    "state",
                    "root",
                    "role",
                    "excitation_ha",
                    "excitation_ev",
                    "total_energy_ha",
                    "s2",
                    "x_norm_sq",
                    "y_norm_sq",
                    "osc_ref_state",
                    "oscillator_strength_from_ref",
                    "top_x_components",
                ]
            )

        for entry in selected_entries:
            root = entry.root
            x, y = mftd.xy[root]
            x_norm_sq = (
                float(np.sum(np.abs(_asnumpy(x)) ** 2))
                if not isinstance(x, (int, float))
                else 0.0
            )
            y_norm_sq = (
                float(np.sum(np.abs(_asnumpy(y)) ** 2))
                if not isinstance(y, (int, float))
                else 0.0
            )
            writer.writerow(
                [
                    int(step),
                    float(time_fs),
                    _state_label(entry.role),
                    int(root),
                    entry.role,
                    float(entry.excitation_ha),
                    float(entry.excitation_ev),
                    float(entry.total_energy_ha),
                    float(entry.s2),
                    x_norm_sq,
                    y_norm_sq,
                    "" if entry.root == selected_roots[0] else ref_label,
                    "" if entry.root == selected_roots[0] else oscillator_strengths.get(entry.root, ""),
                    _top_x_components(mftd, root, top_n=top_n, coeff_threshold=coeff_threshold),
                ]
            )
