"""wqa.features.splits tests: stratification, freeze/refuse-overwrite, temporal_holdout (WP4)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from wqa import manifest as manifest_mod
from wqa.clean import run as clean_run
from wqa.collect import persist as persist_mod
from wqa.features import build as build_mod
from wqa.features import splits as splits_mod

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


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(clean_run, "ROOT", tmp_path)
    monkeypatch.setattr(persist_mod, "ROOT", tmp_path)
    monkeypatch.setattr(manifest_mod, "ROOT", tmp_path)
    monkeypatch.setattr(build_mod, "ROOT", tmp_path)
    monkeypatch.setattr(splits_mod, "ROOT", tmp_path)
    _seed_raw(tmp_path)
    clean_run.run("small")
    build_mod.run("small")
    return tmp_path


def test_splits_sizes_and_stratification(prepared: Path) -> None:
    sizes = splits_mod.run("small")
    assert sizes["train"] + sizes["val"] + sizes["test"] == 120
    assert sizes["train"] > sizes["val"] and sizes["train"] > sizes["test"]

    train = pd.read_parquet(prepared / "data" / "small" / "splits" / "train.parquet")
    val = pd.read_parquet(prepared / "data" / "small" / "splits" / "val.parquet")
    test = pd.read_parquet(prepared / "data" / "small" / "splits" / "test.parquet")
    assert set(train["label"]) == set(val["label"]) == set(test["label"])
    assert "temporal_holdout" in train.columns
    ids = pd.concat([train["pageid"], val["pageid"], test["pageid"]])
    assert ids.is_unique


def test_splits_are_deterministic(prepared: Path) -> None:
    a = splits_mod._temporal_holdout_flag
    features = pd.read_parquet(prepared / "data" / "small" / "features" / "features.parquet")
    interim = pd.read_parquet(prepared / "data" / "small" / "interim" / "full.parquet",
                              columns=["pageid", "created"])
    df = features.merge(interim, on="pageid", how="left")
    flag1 = a(df, "label", 0.10)
    flag2 = a(df, "label", 0.10)
    assert flag1.equals(flag2)


def test_splits_refuse_overwrite(prepared: Path) -> None:
    splits_mod.run("small")
    with pytest.raises(FileExistsError):
        splits_mod.run("small")
