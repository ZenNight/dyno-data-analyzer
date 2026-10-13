"""Loading a dyno run from CSV.

Expected columns (header names are case insensitive, extra columns are kept):

    time_s        seconds since start of the pull
    rpm           engine speed
    torque_nm     brake torque at the dyno, N m
    egt1_c ...    exhaust gas temperature per cylinder, deg C (any count, egtN_c)
    lambda        air fuel equivalence ratio (optional)
    map_kpa       manifold absolute pressure (optional)

Power is derived, never read from the file, so every run is computed the same way.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_EGT = re.compile(r"^egt(\d+)_c$")


@dataclass
class Run:
    name: str
    time_s: np.ndarray
    rpm: np.ndarray
    torque_nm: np.ndarray
    egt_c: np.ndarray            # shape (n_samples, n_cylinders); may have zero columns
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def power_kw(self) -> np.ndarray:
        """P = T * omega, omega = 2 pi rpm / 60."""
        return self.torque_nm * self.rpm * 2.0 * np.pi / 60.0 / 1000.0

    @property
    def power_hp(self) -> np.ndarray:
        return self.power_kw / 0.7457

    @property
    def n_cylinders(self) -> int:
        return int(self.egt_c.shape[1]) if self.egt_c.ndim == 2 else 0

    def __len__(self) -> int:
        return int(len(self.rpm))


def load_run(path: str | Path, name: str | None = None) -> Run:
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty file")
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        for req in ("time_s", "rpm", "torque_nm"):
            if req not in cols:
                raise ValueError(f"{path}: missing required column '{req}' (have {sorted(cols)})")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")

    def col(key: str) -> np.ndarray:
        return np.array([float(r[cols[key]]) for r in rows], dtype=float)

    egt_keys = sorted((k for k in cols if _EGT.match(k)), key=lambda k: int(_EGT.match(k).group(1)))
    egt = np.column_stack([col(k) for k in egt_keys]) if egt_keys else np.empty((len(rows), 0))
    extra = {k: col(k) for k in cols if k not in {"time_s", "rpm", "torque_nm", *egt_keys}}

    run = Run(name=name or path.stem, time_s=col("time_s"), rpm=col("rpm"), torque_nm=col("torque_nm"),
              egt_c=egt, extra=extra)
    _validate(run, path)
    return run


def _validate(run: Run, path: Path) -> None:
    if np.any(np.diff(run.time_s) <= 0):
        raise ValueError(f"{path}: time_s must be strictly increasing")
    if np.any(run.rpm <= 0):
        raise ValueError(f"{path}: rpm must be positive")
    if np.any(~np.isfinite(run.torque_nm)):
        raise ValueError(f"{path}: torque_nm contains NaN or inf")
