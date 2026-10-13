"""Generate the two sample runs. Synthetic, not measured on a real engine.

A 600 cc four cylinder style torque curve: rises to a plateau around 9000 rpm,
falls off past 12000. The 'new_exhaust' run gains a few percent mid range and
loses a little at the very top, which is a typical header length trade off.
Cylinder 3 runs hotter in the baseline to give the EGT outlier detector something to find.
"""
import csv
import numpy as np

rng = np.random.default_rng(7)


def torque_curve(rpm, bump=0.0, top_loss=0.0):
    base = 42 * np.exp(-((rpm - 9500) / 3200) ** 2) + 18
    mid = 1 + bump * np.exp(-((rpm - 7500) / 1500) ** 2)
    top = 1 - top_loss / (1 + np.exp(-(rpm - 11500) / 300))
    return base * mid * top - 0.0000012 * np.maximum(rpm - 11000, 0) ** 2


def write(name, bump, top_loss=0.0, hot_cyl=None):
    t = np.arange(0, 12.0, 0.05)
    rpm = 4000 + (13000 - 4000) * t / t[-1] + rng.normal(0, 15, len(t))
    tq = torque_curve(rpm, bump, top_loss) + rng.normal(0, 0.6, len(t))
    egt = np.column_stack([620 + 0.02 * (rpm - 4000) + rng.normal(0, 4, len(t)) for _ in range(4)])
    if hot_cyl is not None:
        egt[:, hot_cyl] += 45
    lam = 0.88 + 0.03 * np.sin(rpm / 900) + rng.normal(0, 0.005, len(t))
    with open(f"data/{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "rpm", "torque_nm", "egt1_c", "egt2_c", "egt3_c", "egt4_c", "lambda"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.2f}", f"{rpm[i]:.0f}", f"{tq[i]:.2f}", *[f"{x:.0f}" for x in egt[i]], f"{lam[i]:.3f}"])


write("baseline", 0.0, hot_cyl=2)
write("new_exhaust", 0.06, top_loss=0.04)
print("wrote data/baseline.csv and data/new_exhaust.csv")
