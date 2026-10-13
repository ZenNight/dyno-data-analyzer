"""Command line entry point.

    dyno summary data/baseline.csv
    dyno compare data/baseline.csv data/new_exhaust.csv
    dyno plot data/baseline.csv data/new_exhaust.csv --out out/curves.png
    dyno egt data/baseline.csv --out out/egt.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .analysis import compare, moving_average, summarize
from .run import load_run


def _fmt_summary(s) -> str:
    lines = [
        f"run            {s.name}  ({s.samples} samples, {s.rpm_min:.0f} to {s.rpm_max:.0f} rpm)",
        f"peak torque    {s.peak_torque.value:.1f} N m @ {s.peak_torque.rpm:.0f} rpm",
        f"peak power     {s.peak_power_kw.value:.1f} kW ({s.peak_power_hp:.1f} hp) @ {s.peak_power_kw.rpm:.0f} rpm",
    ]
    if s.egt_mean_c:
        cyl = ", ".join(f"cyl{i + 1} {t:.0f}" for i, t in enumerate(s.egt_mean_c))
        lines.append(f"EGT mean (C)   {cyl}")
        lines.append(f"EGT spread     {s.egt_spread_c:.0f} C max between cylinders")
        if s.egt_outlier:
            lines.append(f"EGT outlier    cylinder {s.egt_outlier} deviates most from the median")
    return "\n".join(lines)


def _cmd_summary(a) -> int:
    for p in a.files:
        s = summarize(load_run(p), smooth=a.smooth)
        print(json.dumps(asdict(s), indent=2) if a.json else _fmt_summary(s))
        if len(a.files) > 1 and not a.json:
            print()
    return 0


def _cmd_compare(a) -> int:
    c = compare(load_run(a.baseline), load_run(a.candidate), step=a.step, smooth=a.smooth)
    if a.json:
        d = asdict(c)
        for k in ("rpm", "torque_delta_nm", "power_delta_kw"):
            d[k] = [None if np.isnan(x) else round(float(x), 3) for x in d[k]]
        print(json.dumps(d, indent=2))
        return 0
    print(f"{c.candidate} vs {c.baseline}")
    print(f"mean torque gain   {c.mean_torque_gain_pct:+.2f}%")
    print(f"peak power gain    {c.peak_power_gain_kw:+.2f} kW")
    print(f"best gain at       {c.best_gain_rpm:.0f} rpm ({np.nanmax(c.torque_delta_nm):+.1f} N m)")
    print(f"worst loss at      {c.worst_loss_rpm:.0f} rpm ({np.nanmin(c.torque_delta_nm):+.1f} N m)")
    print("\n   rpm   d torque   d power")
    for r, dt, dp in zip(c.rpm, c.torque_delta_nm, c.power_delta_kw):
        if not np.isnan(dt):
            print(f"{r:6.0f}  {dt:+8.1f}  {dp:+8.2f}")
    return 0


def _cmd_plot(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for p in a.files:
        run = load_run(p)
        tq = moving_average(run.torque_nm, a.smooth)
        ax_t.plot(run.rpm, tq, label=run.name)
        ax_p.plot(run.rpm, tq * run.rpm * 2 * np.pi / 60 / 1000, label=run.name)
    ax_t.set_ylabel("torque (N m)"); ax_p.set_ylabel("power (kW)"); ax_p.set_xlabel("rpm")
    for ax in (ax_t, ax_p):
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Dyno curves")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"saved {a.out}")
    return 0


def _cmd_egt(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = load_run(a.file)
    if run.n_cylinders == 0:
        print("no egtN_c columns in this run", file=sys.stderr)
        return 1
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i in range(run.n_cylinders):
        ax.plot(run.rpm, run.egt_c[:, i], label=f"cyl {i + 1}")
    ax.set_xlabel("rpm"); ax.set_ylabel("EGT (C)"); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title(f"Exhaust gas temperature per cylinder: {run.name}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"saved {a.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dyno", description="Engine dyno log analysis.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary", help="peaks and EGT stats for one or more runs")
    s.add_argument("files", nargs="+"); s.add_argument("--smooth", type=int, default=5); s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_summary)

    c = sub.add_parser("compare", help="candidate minus baseline, binned by rpm")
    c.add_argument("baseline"); c.add_argument("candidate")
    c.add_argument("--step", type=float, default=100.0); c.add_argument("--smooth", type=int, default=5)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=_cmd_compare)

    pl = sub.add_parser("plot", help="torque and power curves to PNG")
    pl.add_argument("files", nargs="+"); pl.add_argument("--out", default="out/curves.png"); pl.add_argument("--smooth", type=int, default=5)
    pl.set_defaults(func=_cmd_plot)

    e = sub.add_parser("egt", help="per cylinder EGT plot to PNG")
    e.add_argument("file"); e.add_argument("--out", default="out/egt.png")
    e.set_defaults(func=_cmd_egt)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
