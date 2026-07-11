# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _safe_cosine(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1.0e-14) -> float:
    denom = max(float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b)), eps)
    return float(np.dot(vec_a, vec_b) / denom)


class NACMonitor:
    def __init__(self, output_dir: Path, states, csv_name: str = "nac_smoothness_metrics.csv"):
        self.output_dir = Path(output_dir)
        self.states = list(states)
        self.csv_path = self.output_dir / csv_name
        self.npz_path = self.output_dir / "nac_vectors.npz"
        self.prev_vectors: Dict[Tuple[int, int], np.ndarray] = {}
        self.records: List[Dict[str, object]] = []
        self.vector_frames: List[np.ndarray] = []
        self.steps: List[int] = []
        self.times_fs: List[float] = []
        self.active_states: List[int] = []
        self.danger_flags: List[bool] = []
        self._header_written = False

    def reset_files(self) -> None:
        if self.csv_path.exists():
            self.csv_path.unlink()
        if self.npz_path.exists():
            self.npz_path.unlink()
        self._header_written = False

    def record(self, step: int, time_fs: float, active_state: int, nacv: np.ndarray, danger: bool = False) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vector_frames.append(np.array(nacv, dtype=float, copy=True))
        self.steps.append(int(step))
        self.times_fs.append(float(time_fs))
        self.active_states.append(int(active_state))
        self.danger_flags.append(bool(danger))

        rows = []
        for i in range(len(self.states) - 1):
            for j in range(i + 1, len(self.states)):
                pair = (self.states[i], self.states[j])
                vec = np.array(nacv[i, j], dtype=float, copy=True)
                flat = vec.reshape(-1)
                prev = self.prev_vectors.get(pair)
                if prev is None:
                    cos_prev = np.nan
                    delta_l2 = np.nan
                    delta_rel = np.nan
                else:
                    prev_flat = prev.reshape(-1)
                    cos_prev = _safe_cosine(prev_flat, flat)
                    delta_l2 = float(np.linalg.norm(flat - prev_flat))
                    delta_rel = delta_l2 / max(float(np.linalg.norm(prev_flat)), 1.0e-14)

                atom_norms = np.linalg.norm(vec, axis=1)
                max_atom = int(np.argmax(atom_norms)) if atom_norms.size else -1
                row = {
                    "step": int(step),
                    "time_fs": float(time_fs),
                    "active_state": int(active_state),
                    "state_i": int(pair[0]),
                    "state_j": int(pair[1]),
                    "l2": float(np.linalg.norm(flat)),
                    "rms": float(np.sqrt(np.mean(flat**2))) if flat.size else 0.0,
                    "max_abs": float(np.max(np.abs(flat))) if flat.size else 0.0,
                    "cos_prev": cos_prev,
                    "delta_l2": delta_l2,
                    "delta_rel": delta_rel,
                    "max_atom_index": max_atom,
                    "max_atom_norm": float(atom_norms[max_atom]) if max_atom >= 0 else 0.0,
                    "danger": bool(danger),
                }
                rows.append(row)
                self.records.append(row)
                self.prev_vectors[pair] = vec

        self._append_rows(rows)
        self._write_npz()

    def _append_rows(self, rows: List[Dict[str, object]]) -> None:
        fieldnames = [
            "step",
            "time_fs",
            "active_state",
            "state_i",
            "state_j",
            "l2",
            "rms",
            "max_abs",
            "cos_prev",
            "delta_l2",
            "delta_rel",
            "max_atom_index",
            "max_atom_norm",
            "danger",
        ]
        write_header = not self._header_written and not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        self._header_written = True

    def _write_npz(self) -> None:
        np.savez(
            self.npz_path,
            nacv=np.array(self.vector_frames, dtype=float),
            steps=np.array(self.steps, dtype=int),
            times_fs=np.array(self.times_fs, dtype=float),
            active_states=np.array(self.active_states, dtype=int),
            danger_flags=np.array(self.danger_flags, dtype=bool),
            states=np.array(self.states, dtype=int),
        )
