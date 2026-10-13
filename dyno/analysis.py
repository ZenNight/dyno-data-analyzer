"""Curve processing, peaks, EGT analysis and run comparison.

Everything works on rpm binned curves so two runs with different sample rates
or sweep speeds can be compared point by point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .run import Run


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding. window=1 returns a copy."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if window == 1:
        return y.copy()
    pad = window // 2
    padded = np.pad(y, (pad, window - 1 - pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def bin_by_rpm(rpm: np.ndarray, y: np.ndarray, step: float = 100.0, lo: float | None = None,
               hi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Average y inside rpm bins of width `step`. Bins with no samples are NaN."""
    lo = float(np.floor(rpm.min() / step) * step) if lo is None else lo
    hi = float(np.ceil(rpm.max() / step) * step) if hi is None else hi
    edges = np.arange(lo, hi + step, step)
    centers = edges[:-1] + step / 2
    idx = np.digitize(rpm, edges) - 1
    out = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = idx == i
        if m.any():
            out[i] = y[m].mean()
    return centers, out


@dataclass
class Peak:
    value: float
    rpm: float


@dataclass
class Summary:
    name: str
    samples: int
    rpm_min: float
    rpm_max: float
    peak_torque: Peak
    peak_power_kw: Peak
    peak_power_hp: float
    egt_mean_c: list[float]
    egt_spread_c: float          # max of (hottest minus coldest cylinder) over the pull
    egt_outlier: int | None      # 1 based cylinder index deviating most, or None


def summarize(run: Run, smooth: int = 5) -> Summary:
    tq = moving_average(run.torque_nm, smooth)
    pw = tq * run.rpm * 2 * np.pi / 60 / 1000
    it, ip = int(np.argmax(tq)), int(np.argmax(pw))
    egt_mean = [float(x) for x in run.egt_c.mean(axis=0)] if run.n_cylinders else []
    spread = float((run.egt_c.max(axis=1) - run.egt_c.min(axis=1)).max()) if run.n_cylinders else 0.0
    outlier = None
    if run.n_cylinders >= 3:
        dev = np.abs(np.array(egt_mean) - np.median(egt_mean))
        outlier = int(np.argmax(dev)) + 1 if dev.max() > 0 else None
    return Summary(
        name=run.name, samples=len(run), rpm_min=float(run.rpm.min()), rpm_max=float(run.rpm.max()),
        peak_torque=Peak(float(tq[it]), float(run.rpm[it])),
        peak_power_kw=Peak(float(pw[ip]), float(run.rpm[ip])),
        peak_power_hp=float(pw[ip] / 0.7457),
        egt_mean_c=egt_mean, egt_spread_c=spread, egt_outlier=outlier,
    )


@dataclass
class Comparison:
    baseline: str
    candidate: str
    rpm: np.ndarray
    torque_delta_nm: np.ndarray      # candidate minus baseline, NaN where either is missing
    power_delta_kw: np.ndarray
    mean_torque_gain_pct: float
    peak_power_gain_kw: float
    best_gain_rpm: float
    worst_loss_rpm: float


def compare(baseline: Run, candidate: Run, step: float = 100.0, smooth: int = 5) -> Comparison:
    lo = max(baseline.rpm.min(), candidate.rpm.min())
    hi = min(baseline.rpm.max(), candidate.rpm.max())
    if hi - lo < 2 * step:
        raise ValueError("runs do not overlap enough in rpm to compare")
    lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    rb, tb = bin_by_rpm(baseline.rpm, moving_average(baseline.torque_nm, smooth), step, lo, hi)
    _, tc = bin_by_rpm(candidate.rpm, moving_average(candidate.torque_nm, smooth), step, lo, hi)
    td = tc - tb
    pb, pc = tb * rb * 2 * np.pi / 60 / 1000, tc * rb * 2 * np.pi / 60 / 1000
    valid = ~np.isnan(td)
    gain_pct = float(np.nanmean(td[valid] / tb[valid]) * 100) if valid.any() else 0.0
    return Comparison(
        baseline=baseline.name, candidate=candidate.name, rpm=rb,
        torque_delta_nm=td, power_delta_kw=pc - pb,
        mean_torque_gain_pct=gain_pct,
        peak_power_gain_kw=float(np.nanmax(pc) - np.nanmax(pb)),
        best_gain_rpm=float(rb[np.nanargmax(td)]),
        worst_loss_rpm=float(rb[np.nanargmin(td)]),
    )
