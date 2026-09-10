"""wqa.eda.run end to end against the committed fixture (WP4, offline)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from wqa import manifest as manifest_mod
from wqa.clean import run as clean_run
from wqa.collect import persist as persist_mod
from wqa.eda import run as eda_run
from wqa.features import build as build_mod

FIXTURE = Path(__file__).parent / "fixtures" / "raw_20pg.jsonl"


def _seed_raw(root: Path) -> None:
    by_grade: dict[str, list[str]] = defaultdict(list)
    for line in FIXTURE.read_text().splitlines():
        rec = json.loads(line)
        by_grade[rec["label"]].append(line)
    raw = root / "data" / "small" / "raw"
    raw.mkdir(parents=True)
    for grade, lines in by_grade.items():
        (raw / f"{grade}.jsonl").write_text("\n".join(lines) + "\n")


def test_eda_run_writes_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(clean_run, "ROOT", tmp_path)
    monkeypatch.setattr(persist_mod, "ROOT", tmp_path)
    monkeypatch.setattr(manifest_mod, "ROOT", tmp_path)
    monkeypatch.setattr(build_mod, "ROOT", tmp_path)
    monkeypatch.setattr(eda_run, "ROOT", tmp_path)
    _seed_raw(tmp_path)
    clean_run.run("small")
    build_mod.run("small")

    result = eda_run.run("small")
    out_dir = tmp_path / "results" / "small" / "eda" / result["run_id"]
    assert (out_dir / "index.json").exists()
    assert (out_dir / "session.json").exists()
    assert (out_dir / "notes.md").exists()
    notes_lines = (out_dir / "notes.md").read_text().strip().splitlines()
    assert 1 < len(notes_lines) <= 6  # heading + <=5 bullets

    stats = pd.read_parquet(out_dir / "feature_stats.parquet")
    summary = pd.read_parquet(out_dir / "feature_summary.parquet")
    assert {"feature", "grade", "count", "mean", "p10", "p50", "p90", "missing"} <= set(stats.columns)
    assert {"feature", "mi", "spearman", "missing_frac", "single_feature_acc"} <= set(summary.columns)

    for cid in ("eda_balance", "eda_corr", "eda_truncation"):
        assert (out_dir / f"{cid}.json").exists()
        assert (out_dir / f"{cid}.png").exists()
        assert (out_dir / "report_tex" / f"{cid}.tex").exists()

    top_feat = summary.iloc[0]["feature"]
    assert (out_dir / f"eda_dist_{top_feat}.png").exists()

    current = tmp_path / "results" / "small" / "eda" / "current"
    assert current.is_symlink()
    assert current.resolve().name == result["run_id"]

    nb_path = tmp_path / "notebooks" / f"eda_small_{result['run_id']}.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text())
    assert len(nb["cells"]) > 1
    assert all("wqa." not in "".join(c.get("source", [])) for c in nb["cells"])
