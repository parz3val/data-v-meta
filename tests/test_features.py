"""Unit + fixture tests for wqa.features (WP4, offline)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wqa import config as config_mod
from wqa import manifest as manifest_mod
from wqa.clean import run as clean_run
from wqa.collect import persist as persist_mod
from wqa.features import build as build_mod
from wqa.features import history, reference, structural

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
def interim_df() -> pd.DataFrame:
    """A tiny synthetic interim-shaped frame for direct feature-function unit tests."""
    return pd.DataFrame({
        "bytes": [1000, 5000], "word_count": [100, 500], "sentence_count": [10, 40],
        "section_l2_n": [2, 8], "section_l3_n": [0, 3], "infobox": [True, False],
        "images_n": [1, 5], "tables_n": [0, 2], "lists_n": [3, 10], "wikilinks_n": [20, 100],
        "extlinks_n": [1, 3], "categories_n": [4, 12], "templates_n": [8, 30],
        "cleanup_tags_n": [1, 0], "lead_text": ["a" * 50, "b" * 300],
        "ref_n": [2, 40], "refs_distinct": [2, 38], "cite_templates_n": [1, 35],
        "doi_n": [0, 5], "isbn_n": [0, 1],
        "edits": [10, 400], "editors": [5, 120], "anon_n": [2, 30], "bot_n": [1, 40],
        "reverted_n": [0, 5], "edits_90d": [1, 20], "pageviews_90d_mean": [3.0, 500.0],
        "created": ["2020-01-01T00:00:00Z", "2010-01-01T00:00:00Z"],
        "last_edit": ["2024-06-01T00:00:00Z", "2024-06-01T00:00:00Z"],
        "fetch_ts": ["2024-06-02T00:00:00Z", "2024-06-02T00:00:00Z"],
    })


def test_structural_features(interim_df: pd.DataFrame) -> None:
    assert list(structural.bytes_(interim_df)) == [1000, 5000]
    assert list(structural.words(interim_df)) == [100, 500]
    assert list(structural.infobox(interim_df)) == [1, 0]
    assert list(structural.lead_length(interim_df)) == [50, 300]


def test_reference_features(interim_df: pd.DataFrame) -> None:
    dens = reference.refs_per_1k_words(interim_df)
    assert dens.iloc[0] == pytest.approx(2 / 100 * 1000)
    mix = reference.cite_template_mix(interim_df)
    assert mix.iloc[0] == pytest.approx(1 / 2)
    assert mix.max() <= 1.0


def test_history_features(interim_df: pd.DataFrame) -> None:
    assert history.edits_per_editor(interim_df).iloc[0] == pytest.approx(2.0)
    assert history.anon_share(interim_df).iloc[0] == pytest.approx(0.2)
    assert history.bot_share(interim_df).iloc[1] == pytest.approx(40 / 400)
    age = history.age_days(interim_df)
    assert age.iloc[0] == pytest.approx((pd.Timestamp("2024-06-02", tz="UTC")
                                         - pd.Timestamp("2020-01-01", tz="UTC")).days, abs=1)


def test_top_editor_share_not_collected(interim_df: pd.DataFrame) -> None:
    assert history.top_editor_share(interim_df).isna().all()


def test_views_90d_log(interim_df: pd.DataFrame) -> None:
    got = history.views_90d_log(interim_df)
    assert got.iloc[0] == pytest.approx(np.log1p(3.0))


def test_build_run_matches_configured_feature_names(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(clean_run, "ROOT", tmp_path)
    monkeypatch.setattr(persist_mod, "ROOT", tmp_path)
    monkeypatch.setattr(manifest_mod, "ROOT", tmp_path)
    monkeypatch.setattr(build_mod, "ROOT", tmp_path)
    _seed_raw(tmp_path)
    clean_run.run("small")

    result = build_mod.run("small")
    assert result["rows"] == 120

    out = pd.read_parquet(tmp_path / "data" / "small" / "features" / "features.parquet")
    cfg = config_mod.load("features")
    named = {n for fam in ("structural", "reference", "history", "contributor", "pageviews")
            for n in cfg[fam]}
    assert named <= set(out.columns)
    assert out["top_editor_share"].isna().all()

    m = manifest_mod.load("small")
    names = {d["name"]: d for d in m["datasets"]}
    assert names["features"]["rows"] == 120
    assert names["features"]["parent"] == "interim"


def test_build_run_missing_feature_name_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(clean_run, "ROOT", tmp_path)
    monkeypatch.setattr(persist_mod, "ROOT", tmp_path)
    monkeypatch.setattr(manifest_mod, "ROOT", tmp_path)
    monkeypatch.setattr(build_mod, "ROOT", tmp_path)
    _seed_raw(tmp_path)
    clean_run.run("small")

    real_load = config_mod.load
    bad_cfg = real_load("features")
    bad_cfg = {**bad_cfg, "structural": [*bad_cfg["structural"], "not_a_real_feature"]}
    monkeypatch.setattr(build_mod.config, "load",
                        lambda name: bad_cfg if name == "features" else real_load(name))
    with pytest.raises(KeyError):
        build_mod.run("small")
