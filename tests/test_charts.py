"""wqa.charts: spec -> PNG/Chart.js/LaTeX (OPS §3.1), offline (WP4)."""
from __future__ import annotations

import json
from pathlib import Path

from wqa import charts


def _bar_spec() -> dict:
    return charts.spec("eda_balance", "Grade balance", "bar",
                       x=["Stub", "Start", "C", "B", "GA", "FA"],
                       series=[{"name": "n", "y": [150, 150, 150, 150, 150, 150]}],
                       xlabel="grade", ylabel="count")


def test_spec_roundtrips_through_save(tmp_path: Path) -> None:
    s = _bar_spec()
    p = charts.save_spec(s, tmp_path)
    assert p.name == "eda_balance.json"
    assert json.loads(p.read_text())["id"] == "eda_balance"


def test_to_png_writes_300dpi_file(tmp_path: Path) -> None:
    s = _bar_spec()
    out = charts.to_png(s, tmp_path / "eda_balance.png")
    assert out.exists() and out.stat().st_size > 0


def test_grade_bar_uses_grade_colours() -> None:
    s = _bar_spec()
    colors = charts._colors(s)
    assert colors == [charts.GRADE_COLOR[g] for g in s["x"]]


def test_to_chartjs_shape() -> None:
    s = _bar_spec()
    cj = charts.to_chartjs(s)
    assert cj["type"] == "bar"
    assert cj["data"]["labels"] == ["Stub", "Start", "C", "B", "GA", "FA"]
    assert cj["data"]["datasets"][0]["data"] == [150, 150, 150, 150, 150, 150]


def test_to_tex_has_booktabs_rules() -> None:
    tex = charts.to_tex(_bar_spec())
    assert r"\toprule" in tex and r"\bottomrule" in tex and r"\midrule" in tex
    assert "eda_balance" in tex


def test_line_and_hist_kinds_render(tmp_path: Path) -> None:
    line = charts.spec("x1", "t", "line", x=[1, 2, 3], series=[{"name": "a", "y": [1, 2, 3]}])
    hist = charts.spec("x2", "t", "hist", x=[], series=[{"name": "a", "y": [1, 2, 2, 3]}])
    assert charts.to_png(line, tmp_path / "l.png").exists()
    assert charts.to_png(hist, tmp_path / "h.png").exists()
