import pytest
from dyno.run import load_run


def test_sample_runs_load_with_four_egt_columns_and_extras():
    r = load_run("data/baseline.csv")
    assert len(r) == 240 and r.n_cylinders == 4
    assert "lambda" in r.extra


def test_power_is_derived_from_torque_and_rpm():
    r = load_run("data/baseline.csv")
    i = 100
    expected = r.torque_nm[i] * r.rpm[i] * 2 * 3.141592653589793 / 60 / 1000
    assert abs(r.power_kw[i] - expected) < 1e-9


def test_missing_column_and_bad_time_are_rejected(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("time_s,rpm\n0,1000\n1,2000\n")
    with pytest.raises(ValueError, match="missing required column 'torque_nm'"):
        load_run(f)
    f.write_text("time_s,rpm,torque_nm\n0,1000,10\n0,2000,11\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        load_run(f)
