# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import cupy as cp

EV_PER_HARTREE = 27.211386245988


def _asnumpy(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)


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
    roots: List[SFStateRoot]
    state_map: Dict[int, int]
    danger: bool
    danger_reason: str
    triplet_root: int
    triplet_excitation_ev: float

    def energies_for_states(self, states: Sequence[int]) -> np.ndarray:
        by_root = {entry.root: entry.total_energy_ha for entry in self.roots}
        return np.array([by_root[self.state_map[state]] for state in states], dtype=float)


def _default_s2_calculator(mf, mftd, root: int, tdtype: str) -> float:
    if not hasattr(mftd, "spin_square"):
        raise AttributeError("SF-TD object does not provide spin_square(state=...).")
    # GPU4PySCF spin-flip solvers use zero-based roots here.
    return float(_asnumpy(mftd.spin_square(state=root)).reshape(-1)[0])


def select_sf_singlet_manifold(
    mf,
    mftd,
    states: Sequence[int] = (1, 2),
    n_lowest: int = 4,
    triplet_zero_threshold_ev: float = 0.3,
    tdtype: str = "TDDFT",
    s2_calculator: Optional[Callable[[object, object, int, str], float]] = None,
) -> SFStateSelection:
    if not states:
        raise ValueError("states must not be empty")
    if any(state < 1 for state in states):
        raise ValueError("SF singlet state labels must be 1-based")
    if max(states) > n_lowest - 1:
        raise ValueError(
            f"states={list(states)} requires more singlet candidates than available from n_lowest={n_lowest}"
        )
    if len(mftd.xy) < n_lowest or len(mftd.e) < n_lowest:
        raise RuntimeError(
            f"SF-TD returned fewer than {n_lowest} roots; cannot apply low-root S0/S1/T candidate selection."
        )

    calc_s2 = s2_calculator or _default_s2_calculator
    roots: List[SFStateRoot] = []
    for root in range(n_lowest):
        exc_ha = float(_asnumpy(mftd.e[root]).reshape(-1)[0])
        roots.append(
            SFStateRoot(
                root=root,
                role="unassigned",
                excitation_ha=exc_ha,
                excitation_ev=exc_ha * EV_PER_HARTREE,
                total_energy_ha=float(_asnumpy(mf.e_tot).reshape(-1)[0] + exc_ha),
                s2=float(calc_s2(mf, mftd, root, tdtype)),
            )
        )

    triplet = max(roots, key=lambda item: item.s2)
    singlets = sorted([item for item in roots if item.root != triplet.root], key=lambda item: item.excitation_ha)

    triplet.role = "T_candidate"
    for idx, item in enumerate(singlets):
        item.role = f"S{idx}"

    state_map = {state: singlets[state - 1].root for state in states}
    danger = abs(triplet.excitation_ev) > triplet_zero_threshold_ev
    danger_reason = ""
    if danger:
        danger_reason = (
            f"abs(T_candidate excitation energy) = {abs(triplet.excitation_ev):.6f} eV > "
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


def append_state_selection_csv(path: Path, step: int, time_fs: float, selection: SFStateSelection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
