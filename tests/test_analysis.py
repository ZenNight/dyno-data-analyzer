import numpy as np
from dyno.analysis import bin_by_rpm, compare, moving_average, summarize
from dyno.run import load_run


def test_moving_average_preserves_length_and_constant_signal():
    y = np.full(50, 3.0)
    assert np.allclose(moving_average(y, 7), 3.0) and len(moving_average(y, 7)) == 50
    assert np.array_equal(moving_average(np.arange(5.0), 1), np.arange(5.0))


def test_bin_by_rpm_averages_and_marks_empty_bins():
    rpm = np.array([1010, 1020, 1210, 1220, 1230.0])
    y = np.array([1, 3, 10, 20, 30.0])
    c, b = bin_by_rpm(rpm, y, step=100, lo=1000, hi=1300)
    assert list(c) == [1050, 1150, 1250]
    assert b[0] == 2 and np.isnan(b[1]) and b[2] == 20


def test_summary_finds_peaks_in_plausible_range_and_hot_cylinder():
    s = summarize(load_run("data/baseline.csv"))
    assert 9000 <= s.peak_torque.rpm <= 10500
    assert s.peak_power_kw.rpm > s.peak_torque.rpm       # power peaks after torque
    assert 40 <= s.peak_power_kw.value <= 90
    assert s.egt_outlier == 3                           # cylinder 3 was made 45 C hotter
    assert s.egt_spread_c > 30


def test_compare_detects_midrange_gain_and_top_end_loss():
    c = compare(load_run("data/baseline.csv"), load_run("data/new_exhaust.csv"))
    assert c.mean_torque_gain_pct > 0.5
    assert 6500 <= c.best_gain_rpm <= 8500
    assert c.worst_loss_rpm > 11000
    assert len(c.rpm) == len(c.torque_delta_nm) == len(c.power_delta_kw)
