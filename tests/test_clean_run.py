"""wqa.clean.run against the committed fixture (WP4, offline)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from wqa import manifest as manifest_mod
from wqa.clean import run as clean_run
from wqa.collect import persist as persist_mod

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


def test_clean_run_produces_interim_and_sample(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(clean_run, "ROOT", tmp_path)
    monkeypatch.setattr(persist_mod, "ROOT", tmp_path)
    monkeypatch.setattr(manifest_mod, "ROOT", tmp_path)
    _seed_raw(tmp_path)

    result = clean_run.run("small")
    assert result["rows"] == 120

    full = pd.read_parquet(tmp_path / "data" / "small" / "interim" / "full.parquet")
    assert len(full) == 120
    assert {"pageid", "label", "clean_text", "ref_n", "word_count"} <= set(full.columns)

    sample = pd.read_parquet(tmp_path / "data" / "small" / "interim" / "sample.parquet")
    assert len(sample) <= clean_run.SAMPLE_ROWS
    assert list(sample.columns) == ["pageid", "title", "grade", "bytes", "refs", "edits",
                                    "n_projects", "lead300"]
    assert sample["lead300"].str.len().max() <= 300

    m = manifest_mod.load("small")
    names = {d["name"]: d for d in m["datasets"]}
    assert names["interim"]["rows"] == 120
    assert names["sample"]["parent"] == "interim"


def test_clean_run_strips_grade_leakage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(clean_run, "ROOT", tmp_path)
    monkeypatch.setattr(persist_mod, "ROOT", tmp_path)
    monkeypatch.setattr(manifest_mod, "ROOT", tmp_path)
    _seed_raw(tmp_path)
    clean_run.run("small")
    full = pd.read_parquet(tmp_path / "data" / "small" / "interim" / "full.parquet")
    fa_rows = full[full["label"] == "FA"]
    assert not fa_rows.empty
    joined = " ".join(fa_rows["clean_text"].tolist()).lower()
    assert "featured article" not in joined
