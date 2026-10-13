import json
from dyno.cli import main


def test_summary_json_round_trips(capsys):
    assert main(["summary", "data/baseline.csv", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["name"] == "baseline" and d["egt_outlier"] == 3


def test_compare_text_mentions_gain(capsys):
    assert main(["compare", "data/baseline.csv", "data/new_exhaust.csv"]) == 0
    assert "mean torque gain" in capsys.readouterr().out


def test_plot_and_egt_write_png(tmp_path):
    out = tmp_path / "c.png"
    assert main(["plot", "data/baseline.csv", "data/new_exhaust.csv", "--out", str(out)]) == 0
    assert out.stat().st_size > 10_000
    out2 = tmp_path / "e.png"
    assert main(["egt", "data/baseline.csv", "--out", str(out2)]) == 0
    assert out2.stat().st_size > 10_000
